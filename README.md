# GPU-Accelerated Medical Image Enhancement (CUDA)

This repository implements a complete CUDA image enhancement pipeline for denoising and quality evaluation, with CPU and GPU versions for fair speedup comparison.

It is designed for MRI-style denoising experiments and supports both synthetic-noise experiments and real noisy input runs.

## 1. What This Project Does

Core features:

- CPU Median filter
- GPU Median filter (shared-memory tiled)
- CPU Non-Local Means (NLM)
- GPU NLM with two paths:
- Brute-force global-memory kernel
- Shared-memory accelerated kernel (auto selected when memory fits)
- Cascaded denoising pipeline: Median output is fed into NLM (CPU and GPU paths)
- Gaussian and Rician noise synthesis
- Optional salt-and-pepper impulse noise
- PSNR and SSIM quality metrics
- Optional GPU-accelerated PSNR using MSE reduction
- Optional unsharp-mask sharpening post-process
- Timing reports for CPU time, GPU total time, and GPU kernel-only time

## 2. Project Structure

- `src/main.cu`: full implementation (I/O, noise, filters, kernels, metrics, CLI, reporting)
- `tools/generate_phantom.py`: deterministic synthetic phantom generator
- `tools/convert_to_pgm.py`: converts common image formats to PGM
- `CMakeLists.txt`: CUDA+C++ build config

## 3. Supported Input Formats

Native support:

- `.pgm` (P5 or P2)

Optional support via `stb_image.h`:

- `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

How optional support works:

- At compile time, `__has_include` checks for `stb_image.h` in common include locations.
- If found, non-PGM formats load directly.
- If not found, non-PGM loads fail with a clear runtime message.

Fallback always available:

- Convert dataset images to PGM using `tools/convert_to_pgm.py`.

## 4. Build Instructions

Requirements:

- NVIDIA GPU
- CUDA toolkit (`nvcc` available)
- CMake >= 3.18
- C++17 compiler
- Optional: `stb_image.h` (for direct PNG/JPG/TIFF reads)

Build:

```bash
cmake -S . -B build
cmake --build build -j
```

Binary:

```bash
./build/medimg_cuda
```

## 5. Command-Line Usage

Basic form:

```bash
./build/medimg_cuda --input <image_path> [options]
```

### 5.1 Arguments

- `--input <path>`
- Required input image.

- `--input-is-clean <0|1>`
- `1` means input is a clean reference and synthetic noise will be added.
- `0` means input is already noisy and will be denoised directly.
- Default: `1`.

- `--ref <path>`
- Optional clean reference image when input is noisy.
- Used for PSNR/SSIM reporting.

- `--output-prefix <name>`
- Prefix for generated result images.
- Default: `output`.

- `--noise-model <gaussian|rician>`
- Used only when `--input-is-clean=1`.
- Default: `gaussian`.

- `--noise-std <float>`
- Standard deviation for Gaussian component.
- Default: `15`.

- `--sp-ratio <float>`
- Salt-and-pepper probability in `[0,1]`.
- Default: `0.01`.

- `--median-radius <int>`
- Median window radius (currently constrained to `1..3`).
- Default: `1`.

- `--nlm-patch <int>`
- NLM patch radius.
- Default: `1`.

- `--nlm-search <int>`
- NLM search radius.
- Default: `3`.

- `--nlm-h <float>`
- NLM filtering strength parameter.
- Default: `12`.

- `--sharpen-amount <float>`
- Unsharp mask amount after denoising.
- `0` disables sharpening.
- Default: `0`.

- `--gpu-metrics <0|1>`
- `1` enables GPU MSE reduction for PSNR.
- Default: `1`.

### 5.2 Example Runs

Clean image input, synthetic Rician noise:

```bash
./build/medimg_cuda \
  --input phantom.pgm \
  --input-is-clean 1 \
  --noise-model rician \
  --noise-std 15 \
  --sp-ratio 0.01 \
  --median-radius 1 \
  --nlm-patch 1 \
  --nlm-search 3 \
  --nlm-h 12 \
  --gpu-metrics 1 \
  --output-prefix run1
```

Real noisy input with clean reference:

```bash
./build/medimg_cuda \
  --input noisy_slice.png \
  --input-is-clean 0 \
  --ref clean_slice.png \
  --median-radius 1 \
  --nlm-patch 1 \
  --nlm-search 3 \
  --nlm-h 12 \
  --output-prefix mri_case
```

## 6. Output Files

Always produced:

- `<prefix>_noisy.pgm`
- `<prefix>_median_cpu.pgm`
- `<prefix>_median_gpu.pgm`
- `<prefix>_nlm_cpu.pgm`
- `<prefix>_nlm_gpu.pgm`

Produced when reference exists:

- `<prefix>_clean.pgm`

Produced when sharpening is enabled:

- `<prefix>_median_cpu_sharp.pgm`
- `<prefix>_median_gpu_sharp.pgm`
- `<prefix>_nlm_cpu_sharp.pgm`
- `<prefix>_nlm_gpu_sharp.pgm`

## 7. End-to-End Pipeline (How The Code Works)

Pipeline topology used by both CPU and GPU branches:

- noisy -> median -> NLM

Execution order in `main`:

1. Parse and validate CLI args.
2. Load input image (PGM or optional stb-backed format).
3. Decide clean/noisy mode.
4. If clean mode: synthesize noise (Gaussian or Rician) + optional impulse noise.
5. Run CPU Median.
6. Run GPU Median and record timings.
7. Run CPU NLM on `median_cpu` output.
8. Run GPU NLM on `median_gpu` output and auto-select shared-memory path if feasible.
9. Optionally apply unsharp sharpening to outputs.
10. Save output images.
11. Compute metrics (PSNR and SSIM) if reference is available.
12. Print performance and quality reports.

## 8. Function-by-Function Reference

This section maps every major function in `src/main.cu` to purpose and behavior.

### 8.1 Utility and Parsing Helpers

- `clamp_int(int v, int lo, int hi)`
- CPU-side integer clamp for boundary-safe indexing.

- `clamp_float(float v, float lo, float hi)`
- CPU-side float clamp, mainly used for output pixel range control.

- `clamp_int_device(int v, int lo, int hi)`
- Device-side integer clamp for kernel-safe boundary handling.

- `to_lower(std::string s)`
- Lowercases text for case-insensitive option parsing and extension handling.

- `file_extension(const std::string& path)`
- Extracts lowercase extension without dot.

### 8.2 Image I/O

- `read_pgm(const std::string& path)`
- Reads binary (`P5`) or ASCII (`P2`) PGM.
- Handles comments in header.
- Validates dimensions and max value.

- `read_with_stb_grayscale(const std::string& path)`
- Uses stb to load image as 1-channel grayscale.
- Available only when `MEDIMG_HAS_STB == 1`.

- `read_image(const std::string& path)`
- Dispatches loading based on extension.
- Uses PGM reader for `.pgm`.
- Uses stb path for common non-PGM formats.

- `write_pgm(const std::string& path, const Image& img)`
- Writes grayscale output as binary `P5` PGM.

### 8.3 Noise and Enhancement

- `add_noise(const Image& clean, const std::string& model, float stddev, float salt_pepper_ratio, uint32_t seed)`
- Adds either Gaussian or Rician noise.
- Optionally injects salt-and-pepper impulse corruption.
- Uses deterministic PRNG seed by default for reproducibility.

- `sharpen_unsharp_cpu(const Image& in, float amount)`
- Applies simple unsharp mask via small Gaussian-like blur kernel.
- Outputs sharpened image with pixel clamping.

### 8.4 CPU Filters

- `median_filter_cpu(const Image& in, int radius)`
- Per-pixel median over `(2r+1)x(2r+1)` neighborhood.
- Uses `std::nth_element` to avoid full sort.

- `nlm_filter_cpu(const Image& in, int patch_radius, int search_radius, float h_param)`
- Brute-force NLM.
- For each pixel, compares local patch with nearby patches and computes weighted average.
- In the current application flow, this function receives the median-filtered image as input.

### 8.5 CUDA Kernels

- `median_filter_kernel(...)`
- Uses shared-memory tile with halo.
- Loads neighborhood into shared memory once per block.
- Uses partial selection to find median value without full sort.

- `nlm_filter_kernel(...)`
- Global-memory brute-force NLM kernel.
- Straightforward but memory intensive.

- `nlm_filter_kernel_shared(...)`
- Shared-memory accelerated NLM kernel.
- Loads a larger tile that covers search and patch radii.
- Reduces redundant global reads significantly.

- `mse_partial_reduce_kernel(...)`
- Parallel block reduction of squared differences.
- Produces per-block partial sums for MSE.

### 8.6 GPU Wrapper Functions

- `median_filter_gpu(...)`
- Allocates GPU buffers.
- Copies input to device.
- Launches median kernel with dynamic shared memory.
- Copies output back.
- Measures kernel-only and total GPU time using CUDA events.

- `nlm_filter_gpu(...)`
- Allocates/copies similarly.
- Computes shared-memory requirement from radii.
- Queries device shared-memory limit.
- Chooses shared or global NLM kernel automatically.
- Returns output and timing.
- In the current application flow, this function receives `median_gpu` output as input.

### 8.7 Metrics and Reporting

- `compute_psnr(...)`
- CPU PSNR computation via MSE.

- `compute_psnr_gpu(...)`
- Computes PSNR using GPU reduction for MSE.
- Useful for large images or batch runs.

- `compute_ssim_global(...)`
- Global SSIM approximation over whole image.
- Not patch-window SSIM.

- `evaluate(...)`
- Returns `Metrics` with PSNR (CPU/GPU path selectable) and SSIM.

- `compute_sharpness_laplacian_variance(...)`
- Returns Laplacian variance as focus/sharpness proxy.

- `print_report(...)`
- Prints timing plus quality metrics for CPU and GPU outputs.

- `print_timing_only_report(...)`
- Used when no clean reference is available.

### 8.8 CLI and Program Control

- `print_usage(...)`
- Prints all command options and defaults.

- `parse_args(...)`
- Parses and validates CLI arguments.
- Enforces value constraints (for example radii and flags).

- `main(...)`
- Orchestrates the complete workflow.

## 9. Mathematical Notes

Median filter:

- For each pixel, output is median of local neighborhood values.

NLM weight for candidate pixel $q$ around center $p$:

$$
w(p, q) = \exp\left(-\frac{\|P(p) - P(q)\|_2^2}{h^2}\right)
$$

Output estimate:

$$
\hat{I}(p) = \frac{\sum_{q \in S(p)} w(p,q)I(q)}{\sum_{q \in S(p)} w(p,q)}
$$

PSNR:

$$
	ext{PSNR} = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

SSIM in this code:

- Global mean/variance/covariance form, not local-window map average.

Laplacian variance sharpness:

- Higher variance often indicates more high-frequency detail.

## 10. Niche Implementation Details You Should Know

- Shared-memory NLM is only used if the required dynamic shared memory fits `sharedMemPerBlock`.
- NLM tile radius is `total_radius = patch_radius + search_radius`.
- Tile dimensions are:
- `tile_w = blockDim.x + 2*total_radius`
- `tile_h = blockDim.y + 2*total_radius`
- Median kernel allows `radius <= 3` because the local fixed array stores up to 49 values.
- Edge handling is clamp-to-border in all CPU and GPU filters.
- GPU timing uses CUDA events and includes:
- Kernel-only timing
- End-to-end GPU timing (allocation + transfer + kernel + transfer back)
- GPU PSNR still copies partial sums to host for final accumulation.
- SSIM remains CPU-side in this version.

## 11. Dataset Workflow (Kaggle MRI Example)

### Option A: Direct non-PGM loading

1. Ensure `stb_image.h` is available in include path.
2. Build project.
3. Run directly on `.png/.jpg/.tif` images.

### Option B: Convert first to PGM

Convert dataset folder:

```bash
python3 tools/convert_to_pgm.py --input-dir DB --output-dir DB_pgm
```

Then run on converted files:

```bash
./build/medimg_cuda --input DB_pgm/sample.pgm --input-is-clean 0 --output-prefix sample_run
```

Notes:

- `tools/convert_to_pgm.py` requires Pillow (`pip install pillow`).

## 12. Interpreting Performance Numbers

The report prints:

- `CPU time (ms)`
- `GPU total time (ms)`
- `GPU kernel time (ms)`
- `End-to-end speedup`
- `Kernel-only speedup`

Interpretation:

- Kernel-only speedup isolates compute acceleration.
- End-to-end speedup reflects practical pipeline cost including memory transfers.
- For small images, transfer overhead can dominate.

## 13. Limitations and Future Work

Current limitations:

- NLM is still expensive for large radii and high-resolution images.
- SSIM is global, not local-window SSIM.
- No batched dataset runner in current code.
- No cuRAND-based GPU-side noise synthesis yet.

Good next improvements:

- Add batch-mode CLI and CSV export for all slices.
- Add local-window SSIM implementation.
- Add cuRAND Rician noise kernel.
- Add CUDA graph or stream-based overlap for transfer/compute.
- Add half precision experiments for memory-bound phases.

## 14. Troubleshooting

`cannot open source file cuda_runtime.h`:

- CUDA toolkit is not installed or include path is not configured.

`Failed to find nvcc` during CMake:

- Install CUDA toolkit and ensure `nvcc` is on `PATH`, or set `CUDAToolkit_ROOT`.

Non-PGM file fails to load:

- Install `stb_image.h` or convert dataset to PGM using the conversion script.

No PSNR/SSIM printed:

- Provide reference image (`--input-is-clean 1` auto creates reference, or pass `--ref` in noisy-input mode).

Very low NLM speedup:

- Reduce radii.
- Check whether shared-memory path is fitting and being selected.
- Profile memory behavior (coalescing, occupancy).

## 15. Quick Start Checklist

1. Build project.
2. Generate or prepare input image.
3. Run with default radii first.
4. Confirm output files are generated.
5. Sweep image sizes and radii for your report.
6. Record CPU/GPU timings and PSNR/SSIM.
