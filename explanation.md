# main.cu Detailed Function Explanation

This document explains every function implemented in `src/main.cu`: what it does, what inputs/outputs it uses, and how it works internally.

## 1) High-level flow

The program builds a full denoising pipeline for grayscale medical images:

1. Parse CLI arguments.
2. Load image (`PGM` directly, other formats via `stb_image` when available).
3. If input is clean, synthetically add Gaussian + optional salt-pepper noise.
4. Estimate noise sigma (or use user-provided sigma).
5. Auto-tune effective denoising parameters.
6. Run CPU and GPU median filtering.
7. Run CPU and GPU NLM filtering (optional Rician bias correction).
8. Optionally sharpen outputs.
9. Save images and print quality/performance metrics.

---

## 2) Utility and helper functions

### `inline int clamp_int(int v, int lo, int hi)`
- What it does: Clamps an integer `v` into `[lo, hi]`.
- Why needed: Prevents out-of-bounds pixel access at image borders.
- How it works: Uses `std::max(lo, std::min(v, hi))`.

### `inline float clamp_float(float v, float lo, float hi)`
- What it does: Clamps a float into `[lo, hi]`.
- Why needed: Keeps pixel intensities valid in `[0, 255]`.
- How it works: Same clamp pattern as integer version.

### `__device__ __forceinline__ int clamp_int_device(int v, int lo, int hi)`
- What it does: GPU-side integer clamp for kernels.
- Why needed: Device code cannot use host-only helpers.
- How it works: Uses CUDA `max/min` intrinsics.

### `std::string to_lower(std::string s)`
- What it does: Converts all characters in a string to lowercase.
- Why needed: Case-insensitive file extension and argument handling.
- How it works: `std::transform` + `std::tolower`.

### `std::string file_extension(const std::string& path)`
- What it does: Extracts file extension without dot, lowercase.
- Why needed: Routes image loading logic based on extension.
- How it works: Finds last dot and lowercases substring after it.

---

## 3) Image I/O functions

### `Image read_pgm(const std::string& path)`
- What it does: Reads `P5` (binary) and `P2` (ASCII) PGM images.
- Inputs: File path.
- Output: `Image {width, height, pixels}`.
- How it works:
  1. Opens file in binary mode.
  2. Uses a local `next_token` lambda that skips comments (`# ...`).
  3. Parses magic, width, height, max value.
  4. Reads either raw bytes (`P5`) or text values (`P2`).
  5. Converts to float pixel vector.
- Important checks: Valid dimensions and `max_val <= 255`.

### `Image read_with_stb_grayscale(const std::string& path)`
- What it does: Loads non-PGM image as single-channel grayscale using `stb_image`.
- Inputs: File path.
- Output: `Image`.
- How it works:
  1. Calls `stbi_load(..., desired_channels=1)`.
  2. Copies bytes into float vector.
  3. Frees stb buffer with `stbi_image_free`.
- Failure behavior: Throws if stb support is unavailable or loading fails.

### `Image read_image(const std::string& path)`
- What it does: Dispatches image loading based on extension.
- Inputs: File path.
- Output: `Image`.
- How it works:
  - `.pgm` -> `read_pgm`
  - `.png/.jpg/.jpeg/.tif/.tiff/.bmp` -> `read_with_stb_grayscale`
  - Else throws unsupported extension error.

### `void write_pgm(const std::string& path, const Image& img)`
- What it does: Saves image as binary `P5` PGM.
- Inputs: Output path + image.
- How it works:
  1. Writes PGM header.
  2. Clamps each float pixel to `[0,255]`.
  3. Rounds and stores as `unsigned char`.
  4. Writes raw data buffer.

---

## 4) Noise estimation and parameter tuning

### `float median_inplace(std::vector<float>& vals)`
- What it does: Returns median value of a vector.
- Input: Mutable vector (modified by `nth_element`).
- Output: Median float.
- How it works: Uses `std::nth_element` at center index.

### `float estimate_noise_sigma_mad(const Image& img)`
- What it does: Estimates noise sigma robustly using MAD of Laplacian response.
- Input: Image.
- Output: Estimated sigma.
- How it works:
  1. For each non-border pixel, computes 4-neighbor Laplacian:
     `lap = left + right + up + down - 4 * center`.
  2. Collects absolute Laplacian responses.
  3. Takes median absolute response.
  4. Converts median absolute response to sigma using
     `sigma = med_abs / (0.67448975 * sqrt(20))`.
- Why robust: Median-based estimator is resistant to outliers.

### `void tune_pipeline_params(const Params& params, float sigma, int* median_radius, int* patch_radius, int* search_radius, float* nlm_h)`
- What it does: Converts user/default settings into effective runtime parameters.
- Inputs: Original params + estimated sigma + output pointers.
- Output: Effective median radius, patch radius, search radius, NLM `h`.
- How it works:
  1. Starts from user-provided values.
  2. If `nlm_h <= 0`, computes `h = max(6.0, nlm_h_factor * sigma)`.
  3. If auto-tune enabled, increases radii based on sigma thresholds.
  4. Applies safety bounds (`median_radius` in `[1,3]`, others >= 1, `h >= 1e-3`).
- Why useful: Makes pipeline adaptive to noise severity.

### `Image add_noise(const Image& clean, float stddev, float salt_pepper_ratio, uint32_t seed = 42)`
- What it does: Creates synthetic noisy image from clean image.
- Inputs: Clean image, Gaussian stddev, impulse ratio, RNG seed.
- Output: Noisy image.
- How it works:
  1. Adds Gaussian random value to each pixel.
  2. With probability `salt_pepper_ratio`, replaces pixel by `0` or `255`.
  3. Clamps to valid range.
- Typical use: Controlled benchmarking when clean reference is available.

---

## 5) CPU filtering functions

### `Image sharpen_unsharp_cpu(const Image& in, float amount)`
- What it does: Optional unsharp-mask enhancement after denoising.
- Inputs: Image + sharpening amount.
- Output: Sharpened image.
- How it works:
  1. Builds blurred value using fixed 3x3 Gaussian-like kernel.
  2. Computes `sharpened = original + amount * (original - blurred)`.
  3. Clamps to `[0,255]`.
- Note: If `amount <= 0`, returns input unchanged.

### `Image median_filter_cpu(const Image& in, int radius)`
- What it does: CPU median filter.
- Inputs: Image + radius (`r`).
- Output: Denoised image.
- How it works:
  1. For each pixel, collects `(2r+1)^2` neighborhood with border clamping.
  2. Uses `nth_element` to pick middle value.
- Strength: Excellent for impulse/outlier noise.

### `Image nlm_filter_cpu(const Image& in, int patch_radius, int search_radius, float h_param, bool rician_bias_correction, float noise_sigma)`
- What it does: CPU Non-Local Means filtering.
- Inputs: Image + NLM radii + strength + optional bias correction flags.
- Output: Denoised image.
- How it works:
  1. For each center pixel `(x,y)`, scans candidate pixels in search window.
  2. Computes patch distance (sum of squared differences).
  3. Converts distance to weight: `w = exp(-patch_dist / h^2)`.
  4. Accumulates weighted mean (and weighted square mean).
  5. If bias correction enabled, reconstructs from second moment:
     `sqrt(max(0, E[I^2] - 2*sigma^2))`.
  6. Else uses normal weighted mean.
- Tradeoff: High quality but computationally expensive on CPU.

---

## 6) CUDA kernels (GPU core)

### `__global__ void median_filter_kernel(...)`
- What it does: Shared-memory tiled median filter kernel.
- Inputs: Device input/output pointers, image dimensions, radius.
- How it works:
  1. Loads tile + halo into shared memory.
  2. Each thread extracts local window values (max 49 for radius <= 3).
  3. Uses partial selection to get median.
  4. Writes one output pixel.
- Why fast: Reuses neighborhood data via shared memory.

### `__global__ void nlm_filter_kernel(...)`
- What it does: Global-memory brute-force NLM kernel.
- Inputs: Device pointers + NLM parameters + bias correction settings.
- How it works: Mirrors CPU NLM logic directly on GPU, but reads from global memory.
- Pros/cons: Simple, general; slower than shared-memory variant due to repeated global reads.

### `__global__ void nlm_filter_kernel_shared(...)`
- What it does: Shared-memory optimized NLM kernel.
- Inputs: Same conceptual inputs as brute-force kernel + `total_radius`.
- How it works:
  1. Loads larger tile covering patch + search extents into shared memory.
  2. Patch comparisons use shared memory instead of repeated global fetches.
  3. Performs weighted accumulation and optional bias correction.
- Why faster: Dramatically reduces global memory traffic.

### `__global__ void mse_partial_reduce_kernel(const float* a, const float* b, size_t n, float* partial)`
- What it does: Parallel reduction kernel for MSE computation.
- Inputs: Two arrays, element count, output partial sums.
- Output: Per-block sum of squared differences.
- How it works:
  1. Each thread computes one squared error.
  2. Block-level reduction in shared memory.
  3. Thread 0 writes block sum to `partial[blockIdx.x]`.

---

## 7) GPU wrapper/orchestration functions

### `Image median_filter_gpu(const Image& in, int radius, double* kernel_ms, double* total_ms)`
- What it does: End-to-end GPU median filtering wrapper.
- Inputs: Host image + radius + optional timing output pointers.
- Output: Host image result.
- How it works:
  1. Allocates device buffers.
  2. Copies input H2D.
  3. Launches `median_filter_kernel` with dynamic shared memory.
  4. Copies result D2H.
  5. Measures kernel and total time using CUDA events.
  6. Frees resources.

### `Image nlm_filter_gpu(const Image& in, int patch_radius, int search_radius, float h_param, bool rician_bias_correction, float noise_sigma, double* kernel_ms, double* total_ms)`
- What it does: End-to-end GPU NLM wrapper.
- Inputs: Host image + NLM params + optional timing outputs.
- Output: Host image result.
- How it works:
  1. Allocates/copies buffers.
  2. Computes required shared memory size.
  3. Queries device shared memory limit.
  4. Chooses kernel path automatically:
     - Shared-memory NLM if memory fits.
     - Else brute-force global-memory NLM.
  5. Copies output back and reports timings.

---

## 8) Metrics and reporting functions

### `double elapsed_ms(...)`
- What it does: Computes elapsed milliseconds between two C++ time points.
- Why needed: CPU timing for stage comparisons.

### `float compute_psnr(const Image& ref, const Image& test)`
- What it does: CPU PSNR calculation.
- How it works:
  1. Computes MSE over all pixels.
  2. Returns infinity for near-zero MSE.
  3. Computes `10*log10((255^2)/MSE)`.

### `float compute_psnr_gpu(const Image& ref, const Image& test)`
- What it does: GPU-accelerated PSNR via parallel MSE reduction.
- How it works:
  1. Copies images to GPU.
  2. Launches `mse_partial_reduce_kernel`.
  3. Copies partial sums to host and finalizes MSE/PSNR.

### `float compute_ssim_global(const Image& ref, const Image& test)`
- What it does: Global SSIM approximation (single global statistics).
- How it works:
  1. Computes global means.
  2. Computes variances/covariance.
  3. Applies SSIM formula with constants `c1`, `c2`.
- Note: This is not local-window SSIM.

### `Metrics evaluate(const Image& ref, const Image& test, bool use_gpu_psnr)`
- What it does: Unified evaluator returning `{psnr, ssim}`.
- How it works: Calls GPU or CPU PSNR path based on flag, always uses CPU SSIM.

### `float compute_sharpness_laplacian_variance(const Image& img)`
- What it does: Computes Laplacian variance sharpness score.
- How it works:
  1. Computes Laplacian at each non-border pixel.
  2. Returns variance of Laplacian responses.
- Interpretation: Higher often means sharper detail.

### `void print_usage(const char* prog)`
- What it does: Prints complete CLI usage and options.
- Why needed: User guidance and error fallback.

### `Params parse_args(int argc, char** argv)`
- What it does: Parses and validates all command-line arguments.
- Output: `Params` structure.
- How it works:
  1. Iterates over argv, consumes values for known flags.
  2. Validates booleans and numeric ranges.
  3. Enforces required fields (`--input`).
  4. Throws descriptive runtime errors on invalid input.

### `void print_report(const std::string& title, const Timing& t, const Metrics& m_cpu, const Metrics& m_gpu)`
- What it does: Prints stage report including timing + quality for CPU and GPU.
- Includes: CPU time, GPU total time, GPU kernel time, speedups, PSNR, SSIM.

### `void print_timing_only_report(const std::string& title, const Timing& t)`
- What it does: Prints timing-only version when no clean reference is available.

---

## 9) Main program function

### `int main(int argc, char** argv)`
- What it does: Orchestrates the complete pipeline end-to-end.
- Detailed flow:
  1. Parse CLI (`parse_args`).
  2. Read input image (`read_image`).
  3. Build noisy image:
     - If clean mode: synthesize noise using `add_noise`.
     - If noisy mode: use input directly; optionally load clean reference.
  4. Determine sigma:
     - Use `--noise-sigma` if provided and positive.
     - Else use known synthetic sigma in clean mode.
     - Else estimate with `estimate_noise_sigma_mad`.
  5. Compute effective parameters using `tune_pipeline_params`.
  6. Run denoising stages:
     - CPU median + GPU median.
     - CPU NLM + GPU NLM.
  7. Optionally sharpen outputs.
  8. Save all output images.
  9. Compute metrics if reference exists.
 10. Print performance and quality reports.
 11. Return success or print error + usage on exception.

---

## 10) 4-way split (calculated contribution split)

The split below is calculated using weighted complexity points per function group so each person has a clear explanation scope.

### Weighted split summary
- Person 1: 21 / 93 points = 22.6%
- Person 2: 21 / 93 points = 22.6%
- Person 3: 27 / 93 points = 29.0%
- Person 4: 24 / 93 points = 25.8%

This is close to an even 25% split, with slightly higher share for GPU kernel owner because that part is most complex.

### Person 1 - Input, utility, and CLI (22.6%)
Explain these functions:
- `clamp_int`, `clamp_float`, `clamp_int_device`
- `to_lower`, `file_extension`
- `read_pgm`, `read_with_stb_grayscale`, `read_image`, `write_pgm`
- `print_usage`, `parse_args`

What this person can claim:
- Implemented robust image I/O and extension handling.
- Built argument parsing and validation logic.
- Ensured safe border/index handling helpers.

### Person 2 - Noise modeling and CPU denoising (22.6%)
Explain these functions:
- `median_inplace`
- `estimate_noise_sigma_mad`
- `tune_pipeline_params`
- `add_noise`
- `sharpen_unsharp_cpu`
- `median_filter_cpu`
- `nlm_filter_cpu`

What this person can claim:
- Implemented adaptive denoising configuration.
- Built CPU denoising reference pipeline.
- Added robust sigma estimator and optional post-sharpening.

### Person 3 - CUDA kernels and GPU filter execution (29.0%)
Explain these functions:
- `median_filter_kernel`
- `nlm_filter_kernel`
- `nlm_filter_kernel_shared`
- `mse_partial_reduce_kernel`
- `median_filter_gpu`
- `nlm_filter_gpu`

What this person can claim:
- Implemented GPU kernels and shared-memory optimization.
- Added automatic kernel-path selection based on shared-memory fit.
- Integrated CUDA event timing and memory transfer orchestration.

### Person 4 - Metrics, reporting, and orchestration (25.8%)
Explain these functions:
- `elapsed_ms`
- `compute_psnr`
- `compute_psnr_gpu`
- `compute_ssim_global`
- `evaluate`
- `compute_sharpness_laplacian_variance`
- `print_report`
- `print_timing_only_report`
- `main`

What this person can claim:
- Implemented evaluation and benchmark reporting.
- Integrated CPU/GPU quality comparisons.
- Assembled complete end-to-end pipeline flow in `main`.

---

## 11) Suggested presentation order (quick viva strategy)

1. Person 1: Data entry and CLI reliability.
2. Person 2: Denoising theory and CPU baseline.
3. Person 3: GPU acceleration and kernel optimization.
4. Person 4: Metrics, results interpretation, and full orchestration.

This ordering matches the runtime path and sounds coherent in a demo.
