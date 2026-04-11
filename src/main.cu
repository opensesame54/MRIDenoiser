#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include("stb_image.h")
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define MEDIMG_HAS_STB 1
#elif __has_include(<stb_image.h>)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define MEDIMG_HAS_STB 1
#elif __has_include(<stb/stb_image.h>)
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define MEDIMG_HAS_STB 1
#else
#define MEDIMG_HAS_STB 0
#endif

namespace {

constexpr float kPixelMax = 255.0f;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
};

struct Metrics {
    float psnr = 0.0f;
    float ssim = 0.0f;
};

struct Timing {
    double cpu_ms = 0.0;
    double gpu_total_ms = 0.0;
    double gpu_kernel_ms = 0.0;
};

struct Params {
    std::string input_path;
    bool input_is_clean = true;
    std::string ref_path;
    std::string output_prefix = "output";
    float noise_stddev = 15.0f;
    float noise_sigma = -1.0f;
    float salt_pepper_ratio = 0.01f;
    bool auto_tune = true;
    int median_radius = 1;
    int nlm_patch_radius = 1;
    int nlm_search_radius = 3;
    float nlm_h = 0.0f;
    float nlm_h_factor = 0.9f;
    bool rician_bias_correction = true;
    float sharpen_amount = 0.0f;
    bool gpu_metrics = true;
};

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::ostringstream oss__;                                                     \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                  << cudaGetErrorString(err__);                                           \
            throw std::runtime_error(oss__.str());                                        \
        }                                                                                 \
    } while (0)

inline int clamp_int(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

inline float clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

__device__ __forceinline__ int clamp_int_device(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string file_extension(const std::string& path) {
    const size_t pos = path.find_last_of('.');
    if (pos == std::string::npos || pos + 1 >= path.size()) {
        return std::string{};
    }
    return to_lower(path.substr(pos + 1));
}

Image read_pgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path);
    }

    auto next_token = [&](std::istream& is) {
        std::string token;
        while (is >> token) {
            if (!token.empty() && token[0] == '#') {
                std::string discard;
                std::getline(is, discard);
                continue;
            }
            return token;
        }
        return std::string{};
    };

    const std::string magic = next_token(in);
    if (magic != "P5" && magic != "P2") {
        throw std::runtime_error("Unsupported PGM format (expected P5 or P2)");
    }

    const int width = std::stoi(next_token(in));
    const int height = std::stoi(next_token(in));
    const int max_val = std::stoi(next_token(in));
    if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
        throw std::runtime_error("Invalid PGM header values");
    }

    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    if (magic == "P5") {
        in.get();
        std::vector<unsigned char> raw(img.pixels.size());
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (!in) {
            throw std::runtime_error("Failed to read binary PGM payload");
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            img.pixels[i] = static_cast<float>(raw[i]);
        }
    } else {
        for (size_t i = 0; i < img.pixels.size(); ++i) {
            std::string tok = next_token(in);
            if (tok.empty()) {
                throw std::runtime_error("Unexpected EOF while reading ASCII PGM payload");
            }
            img.pixels[i] = static_cast<float>(std::stoi(tok));
        }
    }

    return img;
}

Image read_with_stb_grayscale(const std::string& path) {
#if MEDIMG_HAS_STB
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 1);
    if (!data) {
        throw std::runtime_error("stb_image failed to load: " + path);
    }

    Image img;
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (size_t i = 0; i < img.pixels.size(); ++i) {
        img.pixels[i] = static_cast<float>(data[i]);
    }

    stbi_image_free(data);
    return img;
#else
    (void)path;
    throw std::runtime_error(
        "Non-PGM loading requires stb_image.h. Install libstb-dev or add stb_image.h to include paths.");
#endif
}

Image read_image(const std::string& path) {
    const std::string ext = file_extension(path);
    if (ext == "pgm") {
        return read_pgm(path);
    }
    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "tif" || ext == "tiff" || ext == "bmp") {
        return read_with_stb_grayscale(path);
    }
    throw std::runtime_error("Unsupported image extension: " + ext + " (supported: pgm/png/jpg/jpeg/tif/tiff/bmp)");
}

void write_pgm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    std::vector<unsigned char> raw(img.pixels.size());
    for (size_t i = 0; i < img.pixels.size(); ++i) {
        float v = std::max(0.0f, std::min(kPixelMax, img.pixels[i]));
        raw[i] = static_cast<unsigned char>(std::lround(v));
    }
    out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
}

float median_inplace(std::vector<float>& vals) {
    if (vals.empty()) {
        return 0.0f;
    }
    const size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    return vals[mid];
}

float estimate_noise_sigma_mad(const Image& img) {
    if (img.width < 3 || img.height < 3) {
        return 0.0f;
    }

    const int w = img.width;
    const int h = img.height;
    std::vector<float> abs_residuals;
    abs_residuals.reserve(static_cast<size_t>(w - 2) * static_cast<size_t>(h - 2));

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const float c = img.pixels[static_cast<size_t>(y) * w + x];
            const float l = img.pixels[static_cast<size_t>(y) * w + (x - 1)];
            const float r = img.pixels[static_cast<size_t>(y) * w + (x + 1)];
            const float u = img.pixels[static_cast<size_t>(y - 1) * w + x];
            const float d = img.pixels[static_cast<size_t>(y + 1) * w + x];

            // 4-neighbor Laplacian response is robust for white-noise variance estimation.
            const float lap = l + r + u + d - 4.0f * c;
            abs_residuals.push_back(std::abs(lap));
        }
    }

    const float med_abs = median_inplace(abs_residuals);
    const float normal_quantile_75 = 0.67448975f;
    const float lap_gain = std::sqrt(20.0f);
    const float sigma = med_abs / std::max(1e-6f, normal_quantile_75 * lap_gain);
    return std::max(0.0f, sigma);
}

void tune_pipeline_params(const Params& params, float sigma,
                          int* median_radius, int* patch_radius, int* search_radius, float* nlm_h) {
    if (!median_radius || !patch_radius || !search_radius || !nlm_h) {
        throw std::runtime_error("Internal error: null parameter passed to tune_pipeline_params");
    }

    *median_radius = params.median_radius;
    *patch_radius = params.nlm_patch_radius;
    *search_radius = params.nlm_search_radius;
    *nlm_h = params.nlm_h;

    if (*nlm_h <= 0.0f) {
        *nlm_h = std::max(6.0f, params.nlm_h_factor * sigma);
    }

    if (!params.auto_tune) {
        return;
    }

    if (sigma > 30.0f) {
        *median_radius = std::max(*median_radius, 3);
        *patch_radius = std::max(*patch_radius, 2);
        *search_radius = std::max(*search_radius, 5);
    } else if (sigma > 20.0f) {
        *median_radius = std::max(*median_radius, 2);
        *patch_radius = std::max(*patch_radius, 2);
        *search_radius = std::max(*search_radius, 4);
    } else if (sigma > 14.0f) {
        *search_radius = std::max(*search_radius, 4);
    }

    *median_radius = clamp_int(*median_radius, 1, 3);
    *patch_radius = std::max(1, *patch_radius);
    *search_radius = std::max(1, *search_radius);
    *nlm_h = std::max(1e-3f, *nlm_h);
}

Image add_noise(const Image& clean, float stddev,
                float salt_pepper_ratio, uint32_t seed = 42) {
    Image noisy = clean;
    std::mt19937 gen(seed);
    std::normal_distribution<float> gauss(0.0f, stddev);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (float& p : noisy.pixels) {
        float v = p + gauss(gen);
        if (uni(gen) < salt_pepper_ratio) {
            v = (uni(gen) < 0.5f) ? 0.0f : kPixelMax;
        }
        p = std::max(0.0f, std::min(kPixelMax, v));
    }
    return noisy;
}

Image sharpen_unsharp_cpu(const Image& in, float amount) {
    if (amount <= 0.0f) {
        return in;
    }

    Image out;
    out.width = in.width;
    out.height = in.height;
    out.pixels.resize(in.pixels.size());

    const int w = in.width;
    const int h = in.height;
    constexpr float k[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f},
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float blurred = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int xx = clamp_int(x + kx, 0, w - 1);
                    int yy = clamp_int(y + ky, 0, h - 1);
                    blurred += k[ky + 1][kx + 1] * in.pixels[static_cast<size_t>(yy) * w + xx];
                }
            }
            blurred /= 16.0f;

            float orig = in.pixels[static_cast<size_t>(y) * w + x];
            float sharpened = orig + amount * (orig - blurred);
            out.pixels[static_cast<size_t>(y) * w + x] = clamp_float(sharpened, 0.0f, kPixelMax);
        }
    }

    return out;
}

Image median_filter_cpu(const Image& in, int radius) {
    Image out;
    out.width = in.width;
    out.height = in.height;
    out.pixels.resize(in.pixels.size());

    const int w = in.width;
    const int h = in.height;
    const int window = 2 * radius + 1;
    const int count = window * window;
    std::vector<float> vals(static_cast<size_t>(count));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = clamp_int(x + dx, 0, w - 1);
                    int yy = clamp_int(y + dy, 0, h - 1);
                    vals[static_cast<size_t>(idx++)] = in.pixels[static_cast<size_t>(yy) * w + xx];
                }
            }
            std::nth_element(vals.begin(), vals.begin() + count / 2, vals.end());
            out.pixels[static_cast<size_t>(y) * w + x] = vals[static_cast<size_t>(count / 2)];
        }
    }

    return out;
}

Image nlm_filter_cpu(const Image& in, int patch_radius, int search_radius, float h_param,
                     bool rician_bias_correction, float noise_sigma) {
    Image out;
    out.width = in.width;
    out.height = in.height;
    out.pixels.resize(in.pixels.size());

    const int w = in.width;
    const int h = in.height;
    const float h2 = std::max(1e-6f, h_param * h_param);
    const float noise_bias = 2.0f * noise_sigma * noise_sigma;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float weighted_sum = 0.0f;
            float weighted_sum_sq = 0.0f;
            float weight_acc = 0.0f;

            for (int sy = -search_radius; sy <= search_radius; ++sy) {
                for (int sx = -search_radius; sx <= search_radius; ++sx) {
                    int qx = clamp_int(x + sx, 0, w - 1);
                    int qy = clamp_int(y + sy, 0, h - 1);

                    float patch_dist = 0.0f;
                    for (int py = -patch_radius; py <= patch_radius; ++py) {
                        for (int px = -patch_radius; px <= patch_radius; ++px) {
                            int ax = clamp_int(x + px, 0, w - 1);
                            int ay = clamp_int(y + py, 0, h - 1);
                            int bx = clamp_int(qx + px, 0, w - 1);
                            int by = clamp_int(qy + py, 0, h - 1);
                            float diff = in.pixels[static_cast<size_t>(ay) * w + ax] -
                                         in.pixels[static_cast<size_t>(by) * w + bx];
                            patch_dist += diff * diff;
                        }
                    }

                    float wgt = std::exp(-patch_dist / h2);
                    float sample = in.pixels[static_cast<size_t>(qy) * w + qx];
                    weighted_sum += wgt * sample;
                    weighted_sum_sq += wgt * sample * sample;
                    weight_acc += wgt;
                }
            }

            const float center = in.pixels[static_cast<size_t>(y) * w + x];
            float denoised = center;
            if (weight_acc > 1e-8f) {
                if (rician_bias_correction) {
                    float sq = (weighted_sum_sq / weight_acc) - noise_bias;
                    denoised = std::sqrt(std::max(0.0f, sq));
                } else {
                    denoised = weighted_sum / weight_acc;
                }
            }
            out.pixels[static_cast<size_t>(y) * w + x] = clamp_float(denoised, 0.0f, kPixelMax);
        }
    }

    return out;
}

__global__ void median_filter_kernel(const float* in, float* out, int w, int h, int radius) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * blockDim.x + tx;
    const int gy = blockIdx.y * blockDim.y + ty;

    const int tile_w = blockDim.x + 2 * radius;
    const int tile_h = blockDim.y + 2 * radius;

    for (int y = ty; y < tile_h; y += blockDim.y) {
        for (int x = tx; x < tile_w; x += blockDim.x) {
            int src_x = clamp_int_device(blockIdx.x * blockDim.x + x - radius, 0, w - 1);
            int src_y = clamp_int_device(blockIdx.y * blockDim.y + y - radius, 0, h - 1);
            tile[y * tile_w + x] = in[src_y * w + src_x];
        }
    }
    __syncthreads();

    if (gx >= w || gy >= h) {
        return;
    }

    // Radius is constrained on host to <= 3, so max 49 samples.
    float vals[49];
    int idx = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int lx = tx + radius + dx;
            int ly = ty + radius + dy;
            vals[idx++] = tile[ly * tile_w + lx];
        }
    }

    // Partial selection to find only the median without fully sorting all samples.
    const int mid = idx / 2;
    for (int i = 0; i <= mid; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < idx; ++j) {
            if (vals[j] < vals[min_idx]) {
                min_idx = j;
            }
        }
        float tmp = vals[i];
        vals[i] = vals[min_idx];
        vals[min_idx] = tmp;
    }

    out[gy * w + gx] = vals[mid];
}

__global__ void nlm_filter_kernel(const float* in, float* out, int w, int h, int patch_radius,
                                  int search_radius, float h_param, int rician_bias_correction,
                                  float noise_bias) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    const float h2 = fmaxf(1e-6f, h_param * h_param);
    float weighted_sum = 0.0f;
    float weighted_sum_sq = 0.0f;
    float weight_acc = 0.0f;

    for (int sy = -search_radius; sy <= search_radius; ++sy) {
        for (int sx = -search_radius; sx <= search_radius; ++sx) {
            int qx = clamp_int_device(x + sx, 0, w - 1);
            int qy = clamp_int_device(y + sy, 0, h - 1);

            float patch_dist = 0.0f;
            for (int py = -patch_radius; py <= patch_radius; ++py) {
                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    int ax = clamp_int_device(x + px, 0, w - 1);
                    int ay = clamp_int_device(y + py, 0, h - 1);
                    int bx = clamp_int_device(qx + px, 0, w - 1);
                    int by = clamp_int_device(qy + py, 0, h - 1);
                    float diff = in[ay * w + ax] - in[by * w + bx];
                    patch_dist += diff * diff;
                }
            }

            float wgt = expf(-patch_dist / h2);
            float sample = in[qy * w + qx];
            weighted_sum += wgt * sample;
            weighted_sum_sq += wgt * sample * sample;
            weight_acc += wgt;
        }
    }

    float denoised = in[y * w + x];
    if (weight_acc > 1e-8f) {
        if (rician_bias_correction) {
            const float sq = (weighted_sum_sq / weight_acc) - noise_bias;
            denoised = sqrtf(fmaxf(0.0f, sq));
        } else {
            denoised = weighted_sum / weight_acc;
        }
    }
    out[y * w + x] = fminf(kPixelMax, fmaxf(0.0f, denoised));
}

__global__ void nlm_filter_kernel_shared(const float* in, float* out, int w, int h,
                                         int patch_radius, int search_radius,
                                         int total_radius, float h_param,
                                         int rician_bias_correction, float noise_bias) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * blockDim.x + tx;
    const int gy = blockIdx.y * blockDim.y + ty;

    const int tile_w = blockDim.x + 2 * total_radius;
    const int tile_h = blockDim.y + 2 * total_radius;

    for (int y = ty; y < tile_h; y += blockDim.y) {
        for (int x = tx; x < tile_w; x += blockDim.x) {
            int src_x = clamp_int_device(blockIdx.x * blockDim.x + x - total_radius, 0, w - 1);
            int src_y = clamp_int_device(blockIdx.y * blockDim.y + y - total_radius, 0, h - 1);
            tile[y * tile_w + x] = in[src_y * w + src_x];
        }
    }
    __syncthreads();

    if (gx >= w || gy >= h) {
        return;
    }

    const float h2 = fmaxf(1e-6f, h_param * h_param);
    float weighted_sum = 0.0f;
    float weighted_sum_sq = 0.0f;
    float weight_acc = 0.0f;

    const int center_x = tx + total_radius;
    const int center_y = ty + total_radius;

    for (int sy = -search_radius; sy <= search_radius; ++sy) {
        for (int sx = -search_radius; sx <= search_radius; ++sx) {
            float patch_dist = 0.0f;
            for (int py = -patch_radius; py <= patch_radius; ++py) {
                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    float a = tile[(center_y + py) * tile_w + (center_x + px)];
                    float b = tile[(center_y + sy + py) * tile_w + (center_x + sx + px)];
                    float diff = a - b;
                    patch_dist += diff * diff;
                }
            }

            float wgt = expf(-patch_dist / h2);
            float sample = tile[(center_y + sy) * tile_w + (center_x + sx)];
            weighted_sum += wgt * sample;
            weighted_sum_sq += wgt * sample * sample;
            weight_acc += wgt;
        }
    }

    float denoised = in[gy * w + gx];
    if (weight_acc > 1e-8f) {
        if (rician_bias_correction) {
            const float sq = (weighted_sum_sq / weight_acc) - noise_bias;
            denoised = sqrtf(fmaxf(0.0f, sq));
        } else {
            denoised = weighted_sum / weight_acc;
        }
    }
    out[gy * w + gx] = fminf(kPixelMax, fmaxf(0.0f, denoised));
}

__global__ void mse_partial_reduce_kernel(const float* a, const float* b, size_t n, float* partial) {
    extern __shared__ float sh[];
    const size_t tid = static_cast<size_t>(threadIdx.x);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    float val = 0.0f;
    if (gid < n) {
        float d = a[gid] - b[gid];
        val = d * d;
    }
    sh[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

Image median_filter_gpu(const Image& in, int radius, double* kernel_ms, double* total_ms) {
    Image out;
    out.width = in.width;
    out.height = in.height;
    out.pixels.resize(in.pixels.size());

    const size_t bytes = in.pixels.size() * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_total));

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    int tile_w = static_cast<int>(block.x) + 2 * radius;
    int tile_h = static_cast<int>(block.y) + 2 * radius;
    size_t shmem_bytes = static_cast<size_t>(tile_w) * tile_h * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start_kernel));
    median_filter_kernel<<<grid, block, shmem_bytes>>>(d_in, d_out, in.width, in.height, radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    CUDA_CHECK(cudaMemcpy(out.pixels.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float kernel_elapsed = 0.0f;
    float total_elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_elapsed, start_total, stop_total));

    if (kernel_ms) {
        *kernel_ms = static_cast<double>(kernel_elapsed);
    }
    if (total_ms) {
        *total_ms = static_cast<double>(total_elapsed);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));

    return out;
}

Image nlm_filter_gpu(const Image& in, int patch_radius, int search_radius, float h_param,
                     bool rician_bias_correction, float noise_sigma,
                     double* kernel_ms, double* total_ms) {
    Image out;
    out.width = in.width;
    out.height = in.height;
    out.pixels.resize(in.pixels.size());

    const size_t bytes = in.pixels.size() * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_total));

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    const int total_radius = patch_radius + search_radius;
    const int rician_flag = rician_bias_correction ? 1 : 0;
    const float noise_bias = 2.0f * noise_sigma * noise_sigma;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int tile_w = static_cast<int>(block.x) + 2 * total_radius;
    const int tile_h = static_cast<int>(block.y) + 2 * total_radius;
    const size_t nlm_shmem_bytes = static_cast<size_t>(tile_w) * tile_h * sizeof(float);
    const bool use_shared = nlm_shmem_bytes <= static_cast<size_t>(prop.sharedMemPerBlock);

    CUDA_CHECK(cudaEventRecord(start_kernel));
    if (use_shared) {
        nlm_filter_kernel_shared<<<grid, block, nlm_shmem_bytes>>>(
            d_in, d_out, in.width, in.height, patch_radius, search_radius, total_radius, h_param,
            rician_flag, noise_bias);
    } else {
        nlm_filter_kernel<<<grid, block>>>(d_in, d_out, in.width, in.height, patch_radius,
                                           search_radius, h_param, rician_flag, noise_bias);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    CUDA_CHECK(cudaMemcpy(out.pixels.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float kernel_elapsed = 0.0f;
    float total_elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_elapsed, start_total, stop_total));

    if (kernel_ms) {
        *kernel_ms = static_cast<double>(kernel_elapsed);
    }
    if (total_ms) {
        *total_ms = static_cast<double>(total_elapsed);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));

    return out;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point t0,
                  std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float compute_psnr(const Image& ref, const Image& test) {
    if (ref.width != test.width || ref.height != test.height || ref.pixels.size() != test.pixels.size()) {
        throw std::runtime_error("PSNR: image dimensions mismatch");
    }

    double mse = 0.0;
    for (size_t i = 0; i < ref.pixels.size(); ++i) {
        double d = static_cast<double>(ref.pixels[i]) - static_cast<double>(test.pixels[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(ref.pixels.size());
    if (mse < 1e-12) {
        return std::numeric_limits<float>::infinity();
    }

    double psnr = 10.0 * std::log10((kPixelMax * kPixelMax) / mse);
    return static_cast<float>(psnr);
}

float compute_psnr_gpu(const Image& ref, const Image& test) {
    if (ref.width != test.width || ref.height != test.height || ref.pixels.size() != test.pixels.size()) {
        throw std::runtime_error("PSNR(GPU): image dimensions mismatch");
    }

    const size_t n = ref.pixels.size();
    const size_t bytes = n * sizeof(float);
    float* d_ref = nullptr;
    float* d_test = nullptr;
    float* d_partial = nullptr;

    const int threads = 256;
    const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMalloc(&d_test, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(blocks) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ref, ref.pixels.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test, test.pixels.data(), bytes, cudaMemcpyHostToDevice));

    mse_partial_reduce_kernel<<<blocks, threads, static_cast<size_t>(threads) * sizeof(float)>>>(
        d_ref, d_test, n, d_partial);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> partial(static_cast<size_t>(blocks));
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, partial.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_test));
    CUDA_CHECK(cudaFree(d_partial));

    double mse_sum = 0.0;
    for (float v : partial) {
        mse_sum += static_cast<double>(v);
    }
    const double mse = mse_sum / static_cast<double>(n);
    if (mse < 1e-12) {
        return std::numeric_limits<float>::infinity();
    }

    return static_cast<float>(10.0 * std::log10((kPixelMax * kPixelMax) / mse));
}

float compute_ssim_global(const Image& ref, const Image& test) {
    if (ref.width != test.width || ref.height != test.height || ref.pixels.size() != test.pixels.size()) {
        throw std::runtime_error("SSIM: image dimensions mismatch");
    }

    const double c1 = std::pow(0.01 * kPixelMax, 2.0);
    const double c2 = std::pow(0.03 * kPixelMax, 2.0);
    const double n = static_cast<double>(ref.pixels.size());

    double mu_x = 0.0;
    double mu_y = 0.0;
    for (size_t i = 0; i < ref.pixels.size(); ++i) {
        mu_x += ref.pixels[i];
        mu_y += test.pixels[i];
    }
    mu_x /= n;
    mu_y /= n;

    double sigma_x = 0.0;
    double sigma_y = 0.0;
    double sigma_xy = 0.0;
    for (size_t i = 0; i < ref.pixels.size(); ++i) {
        double x = ref.pixels[i] - mu_x;
        double y = test.pixels[i] - mu_y;
        sigma_x += x * x;
        sigma_y += y * y;
        sigma_xy += x * y;
    }

    sigma_x /= n;
    sigma_y /= n;
    sigma_xy /= n;

    const double num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
    const double den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2);
    if (std::abs(den) < 1e-15) {
        return 1.0f;
    }
    return static_cast<float>(num / den);
}

Metrics evaluate(const Image& ref, const Image& test, bool use_gpu_psnr) {
    Metrics m;
    if (use_gpu_psnr) {
        m.psnr = compute_psnr_gpu(ref, test);
    } else {
        m.psnr = compute_psnr(ref, test);
    }
    m.ssim = compute_ssim_global(ref, test);
    return m;
}

float compute_sharpness_laplacian_variance(const Image& img) {
    if (img.width < 3 || img.height < 3) {
        return 0.0f;
    }

    const int w = img.width;
    const int h = img.height;
    double mean = 0.0;
    double mean_sq = 0.0;
    size_t n = 0;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float c = img.pixels[static_cast<size_t>(y) * w + x];
            float l = img.pixels[static_cast<size_t>(y) * w + (x - 1)];
            float r = img.pixels[static_cast<size_t>(y) * w + (x + 1)];
            float u = img.pixels[static_cast<size_t>(y - 1) * w + x];
            float d = img.pixels[static_cast<size_t>(y + 1) * w + x];
            double lap = static_cast<double>(l + r + u + d - 4.0f * c);
            mean += lap;
            mean_sq += lap * lap;
            ++n;
        }
    }

    if (n == 0) {
        return 0.0f;
    }
    mean /= static_cast<double>(n);
    mean_sq /= static_cast<double>(n);
    double var = mean_sq - mean * mean;
    return static_cast<float>(std::max(0.0, var));
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " --input <input_image> [options]\n\n"
              << "Options:\n"
              << "  --input-is-clean <0|1>    Treat --input as clean image (1, default) or noisy image (0)\n"
              << "  --ref <clean_reference>    Optional clean reference image for quality metrics\n"
              << "  --output-prefix <name>    Output prefix (default: output)\n"
              << "  --noise-std <float>       Gaussian noise stddev (used when --input-is-clean=1, default: 15)\n"
              << "  --noise-sigma <float>     Known noise sigma for denoising; <=0 means auto estimate\n"
              << "  --sp-ratio <float>        Salt-pepper probability (used when --input-is-clean=1, default: 0.01)\n"
              << "  --auto-tune <0|1>         Auto-tune median/NLM params using estimated noise (default: 1)\n"
              << "  --median-radius <int>     Median radius, max 3 (default: 1)\n"
              << "  --nlm-patch <int>         NLM patch radius (default: 1)\n"
              << "  --nlm-search <int>        NLM search radius (default: 3)\n"
              << "  --nlm-h <float>           NLM filtering strength; <=0 means auto from sigma\n"
              << "  --nlm-h-factor <float>    Auto NLM strength factor h=factor*sigma (default: 0.9)\n"
              << "  --rician-bias-correct <0|1>  Bias-correct NLM for MRI-like Rician noise (default: 1)\n"
              << "  --sharpen-amount <float>  Unsharp-mask amount on denoised outputs (default: 0)\n"
              << "  --gpu-metrics <0|1>       Use GPU MSE reduction for PSNR (default: 1)\n";
}

Params parse_args(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto require_value = [&](const std::string& flag) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return std::string(argv[++i]);
        };

        if (a == "--input") {
            p.input_path = require_value(a);
        } else if (a == "--input-is-clean") {
            int v = std::stoi(require_value(a));
            if (v != 0 && v != 1) {
                throw std::runtime_error("--input-is-clean must be 0 or 1");
            }
            p.input_is_clean = (v == 1);
        } else if (a == "--ref") {
            p.ref_path = require_value(a);
        } else if (a == "--output-prefix") {
            p.output_prefix = require_value(a);
        } else if (a == "--noise-std") {
            p.noise_stddev = std::stof(require_value(a));
        } else if (a == "--noise-sigma") {
            p.noise_sigma = std::stof(require_value(a));
        } else if (a == "--sp-ratio") {
            p.salt_pepper_ratio = std::stof(require_value(a));
        } else if (a == "--auto-tune") {
            int v = std::stoi(require_value(a));
            if (v != 0 && v != 1) {
                throw std::runtime_error("--auto-tune must be 0 or 1");
            }
            p.auto_tune = (v == 1);
        } else if (a == "--median-radius") {
            p.median_radius = std::stoi(require_value(a));
        } else if (a == "--nlm-patch") {
            p.nlm_patch_radius = std::stoi(require_value(a));
        } else if (a == "--nlm-search") {
            p.nlm_search_radius = std::stoi(require_value(a));
        } else if (a == "--nlm-h") {
            p.nlm_h = std::stof(require_value(a));
        } else if (a == "--nlm-h-factor") {
            p.nlm_h_factor = std::stof(require_value(a));
        } else if (a == "--rician-bias-correct") {
            int v = std::stoi(require_value(a));
            if (v != 0 && v != 1) {
                throw std::runtime_error("--rician-bias-correct must be 0 or 1");
            }
            p.rician_bias_correction = (v == 1);
        } else if (a == "--sharpen-amount") {
            p.sharpen_amount = std::stof(require_value(a));
        } else if (a == "--gpu-metrics") {
            int v = std::stoi(require_value(a));
            if (v != 0 && v != 1) {
                throw std::runtime_error("--gpu-metrics must be 0 or 1");
            }
            p.gpu_metrics = (v == 1);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }

    if (p.input_path.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (p.median_radius < 1 || p.median_radius > 3) {
        throw std::runtime_error("--median-radius must be in [1, 3]");
    }
    if (p.nlm_patch_radius < 1 || p.nlm_search_radius < 1) {
        throw std::runtime_error("NLM radii must be >= 1");
    }
    if (p.nlm_h_factor <= 0.0f) {
        throw std::runtime_error("--nlm-h-factor must be > 0");
    }
    if (p.sharpen_amount < 0.0f) {
        throw std::runtime_error("--sharpen-amount must be >= 0");
    }

    return p;
}

void print_report(const std::string& title, const Timing& t, const Metrics& m_cpu, const Metrics& m_gpu) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time (ms):           " << t.cpu_ms << "\n";
    std::cout << "GPU total time (ms):     " << t.gpu_total_ms << "\n";
    std::cout << "GPU kernel time (ms):    " << t.gpu_kernel_ms << "\n";
    std::cout << "End-to-end speedup:      " << (t.cpu_ms / std::max(1e-9, t.gpu_total_ms)) << "x\n";
    std::cout << "Kernel-only speedup:     " << (t.cpu_ms / std::max(1e-9, t.gpu_kernel_ms)) << "x\n";
    std::cout << "CPU quality:             PSNR=" << m_cpu.psnr << " dB, SSIM=" << m_cpu.ssim << "\n";
    std::cout << "GPU quality:             PSNR=" << m_gpu.psnr << " dB, SSIM=" << m_gpu.ssim << "\n";
}

void print_timing_only_report(const std::string& title, const Timing& t) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time (ms):           " << t.cpu_ms << "\n";
    std::cout << "GPU total time (ms):     " << t.gpu_total_ms << "\n";
    std::cout << "GPU kernel time (ms):    " << t.gpu_kernel_ms << "\n";
    std::cout << "End-to-end speedup:      " << (t.cpu_ms / std::max(1e-9, t.gpu_total_ms)) << "x\n";
    std::cout << "Kernel-only speedup:     " << (t.cpu_ms / std::max(1e-9, t.gpu_kernel_ms)) << "x\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Params params = parse_args(argc, argv);

        Image input_img = read_image(params.input_path);
        Image clean_ref;
        Image noisy;
        bool has_ref = false;

        if (params.input_is_clean) {
            clean_ref = input_img;
            noisy = add_noise(clean_ref, params.noise_stddev, params.salt_pepper_ratio);
            has_ref = true;
        } else {
            noisy = input_img;
            if (!params.ref_path.empty()) {
                clean_ref = read_image(params.ref_path);
                if (clean_ref.width != noisy.width || clean_ref.height != noisy.height ||
                    clean_ref.pixels.size() != noisy.pixels.size()) {
                    throw std::runtime_error("--ref image dimensions must match --input image dimensions");
                }
                has_ref = true;
            }
        }

        float estimated_sigma = params.noise_sigma;
        if (estimated_sigma <= 0.0f) {
            if (params.input_is_clean) {
                estimated_sigma = params.noise_stddev;
            } else {
                estimated_sigma = estimate_noise_sigma_mad(noisy);
            }
        }
        if (!std::isfinite(estimated_sigma) || estimated_sigma <= 0.0f) {
            estimated_sigma = std::max(5.0f, params.noise_stddev);
        }

        int effective_median_radius = params.median_radius;
        int effective_patch_radius = params.nlm_patch_radius;
        int effective_search_radius = params.nlm_search_radius;
        float effective_nlm_h = params.nlm_h;
        tune_pipeline_params(params, estimated_sigma,
                             &effective_median_radius, &effective_patch_radius,
                             &effective_search_radius, &effective_nlm_h);

        const bool likely_rician = !params.input_is_clean;
        const bool use_rician_bias_correction = params.rician_bias_correction && likely_rician;

        auto t0 = std::chrono::high_resolution_clock::now();
        Image median_cpu = median_filter_cpu(noisy, effective_median_radius);
        auto t1 = std::chrono::high_resolution_clock::now();

        double median_gpu_kernel_ms = 0.0;
        double median_gpu_total_ms = 0.0;
        Image median_gpu = median_filter_gpu(noisy, effective_median_radius,
                                             &median_gpu_kernel_ms, &median_gpu_total_ms);

        auto t2 = std::chrono::high_resolution_clock::now();
        Image nlm_cpu = nlm_filter_cpu(median_cpu, effective_patch_radius,
                           effective_search_radius, effective_nlm_h,
                           use_rician_bias_correction, estimated_sigma);
        auto t3 = std::chrono::high_resolution_clock::now();

        double nlm_gpu_kernel_ms = 0.0;
        double nlm_gpu_total_ms = 0.0;
        Image nlm_gpu = nlm_filter_gpu(median_gpu, effective_patch_radius,
                           effective_search_radius, effective_nlm_h,
                           use_rician_bias_correction, estimated_sigma,
                                       &nlm_gpu_kernel_ms, &nlm_gpu_total_ms);

        bool apply_sharpen = params.sharpen_amount > 0.0f;
        Image median_cpu_sharp;
        Image median_gpu_sharp;
        Image nlm_cpu_sharp;
        Image nlm_gpu_sharp;
        if (apply_sharpen) {
            median_cpu_sharp = sharpen_unsharp_cpu(median_cpu, params.sharpen_amount);
            median_gpu_sharp = sharpen_unsharp_cpu(median_gpu, params.sharpen_amount);
            nlm_cpu_sharp = sharpen_unsharp_cpu(nlm_cpu, params.sharpen_amount);
            nlm_gpu_sharp = sharpen_unsharp_cpu(nlm_gpu, params.sharpen_amount);
        }

        if (has_ref) {
            write_pgm(params.output_prefix + "_clean.pgm", clean_ref);
        }
        write_pgm(params.output_prefix + "_noisy.pgm", noisy);
        write_pgm(params.output_prefix + "_median_cpu.pgm", median_cpu);
        write_pgm(params.output_prefix + "_median_gpu.pgm", median_gpu);
        write_pgm(params.output_prefix + "_nlm_cpu.pgm", nlm_cpu);
        write_pgm(params.output_prefix + "_nlm_gpu.pgm", nlm_gpu);
        if (apply_sharpen) {
            write_pgm(params.output_prefix + "_median_cpu_sharp.pgm", median_cpu_sharp);
            write_pgm(params.output_prefix + "_median_gpu_sharp.pgm", median_gpu_sharp);
            write_pgm(params.output_prefix + "_nlm_cpu_sharp.pgm", nlm_cpu_sharp);
            write_pgm(params.output_prefix + "_nlm_gpu_sharp.pgm", nlm_gpu_sharp);
        }

        Timing median_t;
        median_t.cpu_ms = elapsed_ms(t0, t1);
        median_t.gpu_total_ms = median_gpu_total_ms;
        median_t.gpu_kernel_ms = median_gpu_kernel_ms;

        Timing nlm_t;
        nlm_t.cpu_ms = elapsed_ms(t2, t3);
        nlm_t.gpu_total_ms = nlm_gpu_total_ms;
        nlm_t.gpu_kernel_ms = nlm_gpu_kernel_ms;

        std::cout << "Loaded image: " << noisy.width << "x" << noisy.height << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Estimated sigma: " << estimated_sigma << "\n";
        std::cout << "Effective params: median-radius=" << effective_median_radius
              << ", nlm-patch=" << effective_patch_radius
              << ", nlm-search=" << effective_search_radius
              << ", nlm-h=" << effective_nlm_h
              << ", rician-bias-correct=" << (use_rician_bias_correction ? 1 : 0)
              << "\n";
        if (has_ref) {
            Metrics noisy_m = evaluate(clean_ref, noisy, params.gpu_metrics);
            Metrics median_cpu_m = evaluate(clean_ref, median_cpu, params.gpu_metrics);
            Metrics median_gpu_m = evaluate(clean_ref, median_gpu, params.gpu_metrics);
            Metrics nlm_cpu_m = evaluate(clean_ref, nlm_cpu, params.gpu_metrics);
            Metrics nlm_gpu_m = evaluate(clean_ref, nlm_gpu, params.gpu_metrics);

            std::cout << "Noisy image quality: PSNR=" << noisy_m.psnr << " dB, SSIM=" << noisy_m.ssim << "\n";
            print_report("Median Filter", median_t, median_cpu_m, median_gpu_m);
            print_report("Non-Local Means (after Median)", nlm_t, nlm_cpu_m, nlm_gpu_m);

            if (apply_sharpen) {
                Metrics median_cpu_sharp_m = evaluate(clean_ref, median_cpu_sharp, params.gpu_metrics);
                Metrics median_gpu_sharp_m = evaluate(clean_ref, median_gpu_sharp, params.gpu_metrics);
                Metrics nlm_cpu_sharp_m = evaluate(clean_ref, nlm_cpu_sharp, params.gpu_metrics);
                Metrics nlm_gpu_sharp_m = evaluate(clean_ref, nlm_gpu_sharp, params.gpu_metrics);

                std::cout << "\n=== Sharpening (Unsharp Mask) ===\n";
                std::cout << "Amount:                  " << params.sharpen_amount << "\n";
                std::cout << "Median CPU sharpened:    PSNR=" << median_cpu_sharp_m.psnr
                          << " dB, SSIM=" << median_cpu_sharp_m.ssim << "\n";
                std::cout << "Median GPU sharpened:    PSNR=" << median_gpu_sharp_m.psnr
                          << " dB, SSIM=" << median_gpu_sharp_m.ssim << "\n";
                std::cout << "NLM CPU sharpened:       PSNR=" << nlm_cpu_sharp_m.psnr
                          << " dB, SSIM=" << nlm_cpu_sharp_m.ssim << "\n";
                std::cout << "NLM GPU sharpened:       PSNR=" << nlm_gpu_sharp_m.psnr
                          << " dB, SSIM=" << nlm_gpu_sharp_m.ssim << "\n";
            }
        } else {
            std::cout << "No reference image provided. Reporting timings only.\n";
            print_timing_only_report("Median Filter", median_t);
            print_timing_only_report("Non-Local Means (after Median)", nlm_t);
        }

        if (apply_sharpen) {
            std::cout << "\n=== Sharpness (Laplacian Variance) ===\n";
            std::cout << "Median CPU:              " << compute_sharpness_laplacian_variance(median_cpu)
                      << " -> " << compute_sharpness_laplacian_variance(median_cpu_sharp) << "\n";
            std::cout << "Median GPU:              " << compute_sharpness_laplacian_variance(median_gpu)
                      << " -> " << compute_sharpness_laplacian_variance(median_gpu_sharp) << "\n";
            std::cout << "NLM CPU:                 " << compute_sharpness_laplacian_variance(nlm_cpu)
                      << " -> " << compute_sharpness_laplacian_variance(nlm_cpu_sharp) << "\n";
            std::cout << "NLM GPU:                 " << compute_sharpness_laplacian_variance(nlm_gpu)
                      << " -> " << compute_sharpness_laplacian_variance(nlm_gpu_sharp) << "\n";
        }

        std::cout << "\nSaved output images with prefix: " << params.output_prefix << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
}
