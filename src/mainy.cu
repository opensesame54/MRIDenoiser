#include <cuda_runtime.h>

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static const float kPixelMax = 255.0f;

#define MAX_PATH_LEN 1024
#define MAX_ERR_LEN 1024

typedef struct {
    int width;
    int height;
    size_t size;
    float* pixels;
} Image;

typedef struct {
    float psnr;
    float ssim;
} Metrics;

typedef struct {
    double cpu_ms;
    double gpu_total_ms;
    double gpu_kernel_ms;
} Timing;

typedef struct {
    char input_path[MAX_PATH_LEN];
    bool input_is_clean;
    char ref_path[MAX_PATH_LEN];
    char output_prefix[256];
    float noise_stddev;
    float noise_sigma;
    float salt_pepper_ratio;
    bool auto_tune;
    int median_radius;
    int nlm_patch_radius;
    int nlm_search_radius;
    float nlm_h;
    float nlm_h_factor;
    bool rician_bias_correction;
    float sharpen_amount;
    bool gpu_metrics;
} Params;

typedef struct {
    uint32_t state;
    bool has_spare;
    float spare;
} Rng;

static char g_error[MAX_ERR_LEN] = {0};

static int set_errorf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, args);
    va_end(args);
    return 0;
}

#define CUDA_CHECK_GOTO(call, label)                                                     \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            set_errorf("CUDA error at %s:%d -> %s", __FILE__, __LINE__,                \
                       cudaGetErrorString(err__));                                        \
            goto label;                                                                    \
        }                                                                                  \
    } while (0)

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static float clamp_float(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

__device__ __forceinline__ int clamp_int_device(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

static void params_init(Params* p) {
    memset(p, 0, sizeof(*p));
    p->input_is_clean = true;
    strncpy(p->output_prefix, "output", sizeof(p->output_prefix) - 1);
    p->noise_stddev = 15.0f;
    p->noise_sigma = -1.0f;
    p->salt_pepper_ratio = 0.01f;
    p->auto_tune = true;
    p->median_radius = 1;
    p->nlm_patch_radius = 1;
    p->nlm_search_radius = 3;
    p->nlm_h = 0.0f;
    p->nlm_h_factor = 0.9f;
    p->rician_bias_correction = true;
    p->sharpen_amount = 0.0f;
    p->gpu_metrics = true;
}

static void image_free(Image* img) {
    if (!img) {
        return;
    }
    free(img->pixels);
    img->pixels = NULL;
    img->width = 0;
    img->height = 0;
    img->size = 0;
}

static int image_alloc(Image* img, int width, int height) {
    size_t size = 0;
    if (!img) {
        return set_errorf("Internal error: null image pointer");
    }
    if (width <= 0 || height <= 0) {
        return set_errorf("Invalid image shape %dx%d", width, height);
    }

    size = (size_t)width * (size_t)height;
    if (size / (size_t)width != (size_t)height) {
        return set_errorf("Image size overflow");
    }

    img->pixels = (float*)malloc(size * sizeof(float));
    if (!img->pixels) {
        return set_errorf("Out of memory allocating image %dx%d", width, height);
    }
    img->width = width;
    img->height = height;
    img->size = size;
    return 1;
}

static int image_clone(const Image* src, Image* dst) {
    if (!src || !dst || !src->pixels) {
        return set_errorf("Internal error: null image in clone");
    }
    if (!image_alloc(dst, src->width, src->height)) {
        return 0;
    }
    memcpy(dst->pixels, src->pixels, src->size * sizeof(float));
    return 1;
}

static void to_lower_inplace(char* s) {
    size_t i = 0;
    if (!s) {
        return;
    }
    while (s[i] != '\0') {
        s[i] = (char)tolower((unsigned char)s[i]);
        ++i;
    }
}

static void get_file_extension(const char* path, char* out, size_t out_sz) {
    const char* dot = NULL;
    size_t n = 0;

    if (!out || out_sz == 0) {
        return;
    }
    out[0] = '\0';
    if (!path) {
        return;
    }

    dot = strrchr(path, '.');
    if (!dot || dot[1] == '\0') {
        return;
    }

    strncpy(out, dot + 1, out_sz - 1);
    out[out_sz - 1] = '\0';
    to_lower_inplace(out);

    n = strlen(out);
    while (n > 0 && isspace((unsigned char)out[n - 1])) {
        out[n - 1] = '\0';
        --n;
    }
}

static int read_next_token(FILE* f, char* token, size_t token_sz) {
    int c = 0;
    size_t n = 0;

    if (!f || !token || token_sz == 0) {
        return 0;
    }

    token[0] = '\0';

    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) {
            continue;
        }
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {
            }
            continue;
        }
        break;
    }
    if (c == EOF) {
        return 0;
    }

    while (c != EOF) {
        if (isspace(c)) {
            break;
        }
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {
            }
            break;
        }

        if (n + 1 < token_sz) {
            token[n++] = (char)c;
        }
        c = fgetc(f);
    }

    token[n] = '\0';
    return n > 0;
}

static int parse_int_value(const char* s, const char* flag, int* out) {
    char* end = NULL;
    long v = 0;

    if (!s || !out) {
        return set_errorf("Internal parse error for %s", flag ? flag : "<unknown>");
    }

    v = strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return set_errorf("Invalid integer for %s: %s", flag, s);
    }
    if (v < -2147483647L - 1L || v > 2147483647L) {
        return set_errorf("Integer out of range for %s: %s", flag, s);
    }
    *out = (int)v;
    return 1;
}

static int parse_float_value(const char* s, const char* flag, float* out) {
    char* end = NULL;
    float v = 0.0f;

    if (!s || !out) {
        return set_errorf("Internal parse error for %s", flag ? flag : "<unknown>");
    }

    v = strtof(s, &end);
    if (end == s || *end != '\0') {
        return set_errorf("Invalid float for %s: %s", flag, s);
    }
    *out = v;
    return 1;
}

static int copy_arg_string(char* dst, size_t dst_sz, const char* src, const char* flag) {
    if (!dst || !src || dst_sz == 0) {
        return set_errorf("Internal string copy error for %s", flag ? flag : "<unknown>");
    }
    if (strlen(src) >= dst_sz) {
        return set_errorf("Value too long for %s", flag);
    }
    strcpy(dst, src);
    return 1;
}

static int read_pgm(const char* path, Image* out) {
    FILE* in = NULL;
    char tok[128];
    char magic[8];
    int width = 0;
    int height = 0;
    int max_val = 0;
    int is_binary = 0;
    size_t i = 0;
    unsigned char* raw = NULL;

    if (!path || !out) {
        return set_errorf("Internal error in read_pgm");
    }

    in = fopen(path, "rb");
    if (!in) {
        return set_errorf("Failed to open input file: %s", path);
    }

    if (!read_next_token(in, magic, sizeof(magic))) {
        fclose(in);
        return set_errorf("Failed to read PGM magic from %s", path);
    }
    if (strcmp(magic, "P5") == 0) {
        is_binary = 1;
    } else if (strcmp(magic, "P2") == 0) {
        is_binary = 0;
    } else {
        fclose(in);
        return set_errorf("Unsupported PGM format (expected P5 or P2)");
    }

    if (!read_next_token(in, tok, sizeof(tok)) || !parse_int_value(tok, "width", &width) ||
        !read_next_token(in, tok, sizeof(tok)) || !parse_int_value(tok, "height", &height) ||
        !read_next_token(in, tok, sizeof(tok)) || !parse_int_value(tok, "max", &max_val)) {
        fclose(in);
        if (g_error[0] == '\0') {
            return set_errorf("Invalid PGM header values");
        }
        return 0;
    }

    if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
        fclose(in);
        return set_errorf("Invalid PGM header values");
    }

    if (!image_alloc(out, width, height)) {
        fclose(in);
        return 0;
    }

    if (is_binary) {
        int c = 0;
        do {
            c = fgetc(in);
        } while (c != EOF && isspace(c));
        if (c != EOF) {
            ungetc(c, in);
        }

        raw = (unsigned char*)malloc(out->size);
        if (!raw) {
            fclose(in);
            image_free(out);
            return set_errorf("Out of memory reading binary PGM payload");
        }

        if (fread(raw, 1, out->size, in) != out->size) {
            free(raw);
            fclose(in);
            image_free(out);
            return set_errorf("Failed to read binary PGM payload");
        }

        for (i = 0; i < out->size; ++i) {
            out->pixels[i] = (float)raw[i];
        }
        free(raw);
    } else {
        for (i = 0; i < out->size; ++i) {
            int v = 0;
            if (!read_next_token(in, tok, sizeof(tok)) || !parse_int_value(tok, "P2 pixel", &v)) {
                fclose(in);
                image_free(out);
                if (g_error[0] == '\0') {
                    return set_errorf("Unexpected EOF while reading ASCII PGM payload");
                }
                return 0;
            }
            out->pixels[i] = (float)v;
        }
    }

    fclose(in);
    return 1;
}

static int read_with_stb_grayscale(const char* path, Image* out) {
#if MEDIMG_HAS_STB
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* data = NULL;
    size_t i = 0;

    data = stbi_load(path, &w, &h, &channels, 1);
    if (!data) {
        return set_errorf("stb_image failed to load: %s", path);
    }

    if (!image_alloc(out, w, h)) {
        stbi_image_free(data);
        return 0;
    }

    for (i = 0; i < out->size; ++i) {
        out->pixels[i] = (float)data[i];
    }

    stbi_image_free(data);
    return 1;
#else
    (void)path;
    (void)out;
    return set_errorf(
        "Non-PGM loading requires stb_image.h. Install libstb-dev or add stb_image.h to include paths.");
#endif
}

static int read_image(const char* path, Image* out) {
    char ext[32];
    if (!path || !out) {
        return set_errorf("Internal error in read_image");
    }

    get_file_extension(path, ext, sizeof(ext));
    if (strcmp(ext, "pgm") == 0) {
        return read_pgm(path, out);
    }
    if (strcmp(ext, "png") == 0 || strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0 ||
        strcmp(ext, "tif") == 0 || strcmp(ext, "tiff") == 0 || strcmp(ext, "bmp") == 0) {
        return read_with_stb_grayscale(path, out);
    }
    return set_errorf(
        "Unsupported image extension: %s (supported: pgm/png/jpg/jpeg/tif/tiff/bmp)", ext[0] ? ext : "<none>");
}

static int write_pgm(const char* path, const Image* img) {
    FILE* out = NULL;
    unsigned char* raw = NULL;
    size_t i = 0;

    if (!path || !img || !img->pixels) {
        return set_errorf("Internal error in write_pgm");
    }

    out = fopen(path, "wb");
    if (!out) {
        return set_errorf("Failed to open output file: %s", path);
    }

    fprintf(out, "P5\n%d %d\n255\n", img->width, img->height);

    raw = (unsigned char*)malloc(img->size);
    if (!raw) {
        fclose(out);
        return set_errorf("Out of memory writing PGM: %s", path);
    }

    for (i = 0; i < img->size; ++i) {
        const float v = clamp_float(img->pixels[i], 0.0f, kPixelMax);
        raw[i] = (unsigned char)lroundf(v);
    }

    if (fwrite(raw, 1, img->size, out) != img->size) {
        free(raw);
        fclose(out);
        return set_errorf("Failed to write output file: %s", path);
    }

    free(raw);
    fclose(out);
    return 1;
}

static void swap_float(float* a, float* b) {
    float t = *a;
    *a = *b;
    *b = t;
}

static size_t partition_f32(float* data, size_t left, size_t right, size_t pivot_index) {
    float pivot_value = data[pivot_index];
    size_t store = left;
    size_t i = 0;

    swap_float(&data[pivot_index], &data[right]);
    for (i = left; i < right; ++i) {
        if (data[i] < pivot_value) {
            swap_float(&data[store], &data[i]);
            ++store;
        }
    }
    swap_float(&data[right], &data[store]);
    return store;
}

static float quickselect_f32(float* data, size_t n, size_t k) {
    size_t left = 0;
    size_t right = n - 1;

    while (1) {
        size_t pivot = 0;
        size_t new_pivot = 0;
        if (left == right) {
            return data[left];
        }

        pivot = left + (right - left) / 2;
        new_pivot = partition_f32(data, left, right, pivot);

        if (k == new_pivot) {
            return data[k];
        }
        if (k < new_pivot) {
            if (new_pivot == 0) {
                return data[0];
            }
            right = new_pivot - 1;
        } else {
            left = new_pivot + 1;
        }
    }
}

static float median_inplace(float* vals, size_t n) {
    const size_t mid = n / 2;
    if (!vals || n == 0) {
        return 0.0f;
    }
    return quickselect_f32(vals, n, mid);
}

static float estimate_noise_sigma_mad(const Image* img) {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    size_t idx = 0;
    size_t count = 0;
    float* abs_residuals = NULL;
    float med_abs = 0.0f;
    const float normal_quantile_75 = 0.67448975f;
    const float lap_gain = sqrtf(20.0f);
    float sigma = 0.0f;

    if (!img || !img->pixels) {
        return 0.0f;
    }
    if (img->width < 3 || img->height < 3) {
        return 0.0f;
    }

    w = img->width;
    h = img->height;
    count = (size_t)(w - 2) * (size_t)(h - 2);
    abs_residuals = (float*)malloc(count * sizeof(float));
    if (!abs_residuals) {
        return 0.0f;
    }

    idx = 0;
    for (y = 1; y < h - 1; ++y) {
        for (x = 1; x < w - 1; ++x) {
            const float c = img->pixels[(size_t)y * (size_t)w + (size_t)x];
            const float l = img->pixels[(size_t)y * (size_t)w + (size_t)(x - 1)];
            const float r = img->pixels[(size_t)y * (size_t)w + (size_t)(x + 1)];
            const float u = img->pixels[(size_t)(y - 1) * (size_t)w + (size_t)x];
            const float d = img->pixels[(size_t)(y + 1) * (size_t)w + (size_t)x];
            const float lap = l + r + u + d - 4.0f * c;
            abs_residuals[idx++] = fabsf(lap);
        }
    }

    med_abs = median_inplace(abs_residuals, count);
    free(abs_residuals);

    sigma = med_abs / fmaxf(1e-6f, normal_quantile_75 * lap_gain);
    return fmaxf(0.0f, sigma);
}

static void tune_pipeline_params(const Params* params, float sigma, int* median_radius, int* patch_radius,
                                 int* search_radius, float* nlm_h) {
    *median_radius = params->median_radius;
    *patch_radius = params->nlm_patch_radius;
    *search_radius = params->nlm_search_radius;
    *nlm_h = params->nlm_h;

    if (*nlm_h <= 0.0f) {
        *nlm_h = fmaxf(6.0f, params->nlm_h_factor * sigma);
    }

    if (params->auto_tune) {
        if (sigma > 30.0f) {
            *median_radius = (*median_radius > 3) ? *median_radius : 3;
            *patch_radius = (*patch_radius > 2) ? *patch_radius : 2;
            *search_radius = (*search_radius > 5) ? *search_radius : 5;
        } else if (sigma > 20.0f) {
            *median_radius = (*median_radius > 2) ? *median_radius : 2;
            *patch_radius = (*patch_radius > 2) ? *patch_radius : 2;
            *search_radius = (*search_radius > 4) ? *search_radius : 4;
        } else if (sigma > 14.0f) {
            *search_radius = (*search_radius > 4) ? *search_radius : 4;
        }
    }

    *median_radius = clamp_int(*median_radius, 1, 3);
    if (*patch_radius < 1) {
        *patch_radius = 1;
    }
    if (*search_radius < 1) {
        *search_radius = 1;
    }
    *nlm_h = fmaxf(1e-3f, *nlm_h);
}

static uint32_t rng_next_u32(Rng* rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static float rng_uniform01(Rng* rng) {
    const uint32_t v = rng_next_u32(rng);
    return (float)((v >> 8) * (1.0 / 16777216.0));
}

static float rng_normal01(Rng* rng) {
    if (rng->has_spare) {
        rng->has_spare = false;
        return rng->spare;
    }

    while (1) {
        float u = rng_uniform01(rng);
        float v = rng_uniform01(rng);
        if (u <= 1e-7f) {
            u = 1e-7f;
        }

        {
            const float r = sqrtf(-2.0f * logf(u));
            const float theta = 2.0f * 3.14159265358979323846f * v;
            const float z0 = r * cosf(theta);
            const float z1 = r * sinf(theta);
            rng->spare = z1;
            rng->has_spare = true;
            return z0;
        }
    }
}

static int add_noise(const Image* clean, float stddev, float salt_pepper_ratio, uint32_t seed, Image* noisy) {
    size_t i = 0;
    Rng rng;

    if (!clean || !clean->pixels || !noisy) {
        return set_errorf("Internal error in add_noise");
    }

    if (!image_clone(clean, noisy)) {
        return 0;
    }

    rng.state = seed ? seed : 42u;
    rng.has_spare = false;
    rng.spare = 0.0f;

    for (i = 0; i < noisy->size; ++i) {
        float v = noisy->pixels[i] + stddev * rng_normal01(&rng);
        if (rng_uniform01(&rng) < salt_pepper_ratio) {
            v = (rng_uniform01(&rng) < 0.5f) ? 0.0f : kPixelMax;
        }
        noisy->pixels[i] = clamp_float(v, 0.0f, kPixelMax);
    }

    return 1;
}

static int sharpen_unsharp_cpu(const Image* in, float amount, Image* out) {
    int x = 0;
    int y = 0;
    static const float k[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f},
    };

    if (!in || !in->pixels || !out) {
        return set_errorf("Internal error in sharpen_unsharp_cpu");
    }

    if (amount <= 0.0f) {
        return image_clone(in, out);
    }

    if (!image_alloc(out, in->width, in->height)) {
        return 0;
    }

    for (y = 0; y < in->height; ++y) {
        for (x = 0; x < in->width; ++x) {
            int ky = 0;
            int kx = 0;
            float blurred = 0.0f;
            float orig = 0.0f;
            float sharpened = 0.0f;
            for (ky = -1; ky <= 1; ++ky) {
                for (kx = -1; kx <= 1; ++kx) {
                    const int xx = clamp_int(x + kx, 0, in->width - 1);
                    const int yy = clamp_int(y + ky, 0, in->height - 1);
                    blurred += k[ky + 1][kx + 1] * in->pixels[(size_t)yy * (size_t)in->width + (size_t)xx];
                }
            }
            blurred /= 16.0f;
            orig = in->pixels[(size_t)y * (size_t)in->width + (size_t)x];
            sharpened = orig + amount * (orig - blurred);
            out->pixels[(size_t)y * (size_t)in->width + (size_t)x] = clamp_float(sharpened, 0.0f, kPixelMax);
        }
    }

    return 1;
}

static int median_filter_cpu(const Image* in, int radius, Image* out) {
    int x = 0;
    int y = 0;
    int dx = 0;
    int dy = 0;
    int idx = 0;
    int window = 0;
    int count = 0;
    float* vals = NULL;

    if (!in || !in->pixels || !out) {
        return set_errorf("Internal error in median_filter_cpu");
    }

    if (!image_alloc(out, in->width, in->height)) {
        return 0;
    }

    window = 2 * radius + 1;
    count = window * window;
    vals = (float*)malloc((size_t)count * sizeof(float));
    if (!vals) {
        image_free(out);
        return set_errorf("Out of memory in median_filter_cpu");
    }

    for (y = 0; y < in->height; ++y) {
        for (x = 0; x < in->width; ++x) {
            idx = 0;
            for (dy = -radius; dy <= radius; ++dy) {
                for (dx = -radius; dx <= radius; ++dx) {
                    const int xx = clamp_int(x + dx, 0, in->width - 1);
                    const int yy = clamp_int(y + dy, 0, in->height - 1);
                    vals[idx++] = in->pixels[(size_t)yy * (size_t)in->width + (size_t)xx];
                }
            }
            out->pixels[(size_t)y * (size_t)in->width + (size_t)x] =
                median_inplace(vals, (size_t)count);
        }
    }

    free(vals);
    return 1;
}

static int nlm_filter_cpu(const Image* in, int patch_radius, int search_radius, float h_param,
                          bool rician_bias_correction, float noise_sigma, Image* out) {
    int x = 0;
    int y = 0;
    int sx = 0;
    int sy = 0;
    int px = 0;
    int py = 0;
    const float h2 = fmaxf(1e-6f, h_param * h_param);
    const float noise_bias = 2.0f * noise_sigma * noise_sigma;

    if (!in || !in->pixels || !out) {
        return set_errorf("Internal error in nlm_filter_cpu");
    }

    if (!image_alloc(out, in->width, in->height)) {
        return 0;
    }

    for (y = 0; y < in->height; ++y) {
        for (x = 0; x < in->width; ++x) {
            float weighted_sum = 0.0f;
            float weighted_sum_sq = 0.0f;
            float weight_acc = 0.0f;

            for (sy = -search_radius; sy <= search_radius; ++sy) {
                for (sx = -search_radius; sx <= search_radius; ++sx) {
                    const int qx = clamp_int(x + sx, 0, in->width - 1);
                    const int qy = clamp_int(y + sy, 0, in->height - 1);
                    float patch_dist = 0.0f;

                    for (py = -patch_radius; py <= patch_radius; ++py) {
                        for (px = -patch_radius; px <= patch_radius; ++px) {
                            const int ax = clamp_int(x + px, 0, in->width - 1);
                            const int ay = clamp_int(y + py, 0, in->height - 1);
                            const int bx = clamp_int(qx + px, 0, in->width - 1);
                            const int by = clamp_int(qy + py, 0, in->height - 1);
                            const float diff = in->pixels[(size_t)ay * (size_t)in->width + (size_t)ax] -
                                               in->pixels[(size_t)by * (size_t)in->width + (size_t)bx];
                            patch_dist += diff * diff;
                        }
                    }

                    {
                        const float wgt = expf(-patch_dist / h2);
                        const float sample = in->pixels[(size_t)qy * (size_t)in->width + (size_t)qx];
                        weighted_sum += wgt * sample;
                        weighted_sum_sq += wgt * sample * sample;
                        weight_acc += wgt;
                    }
                }
            }

            {
                float denoised = in->pixels[(size_t)y * (size_t)in->width + (size_t)x];
                if (weight_acc > 1e-8f) {
                    if (rician_bias_correction) {
                        const float sq = (weighted_sum_sq / weight_acc) - noise_bias;
                        denoised = sqrtf(fmaxf(0.0f, sq));
                    } else {
                        denoised = weighted_sum / weight_acc;
                    }
                }
                out->pixels[(size_t)y * (size_t)in->width + (size_t)x] =
                    clamp_float(denoised, 0.0f, kPixelMax);
            }
        }
    }

    return 1;
}

__global__ void median_filter_kernel(const float* in, float* out, int w, int h, int radius) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * blockDim.x + tx;
    const int gy = blockIdx.y * blockDim.y + ty;

    const int tile_w = blockDim.x + 2 * radius;
    const int tile_h = blockDim.y + 2 * radius;

    int y = 0;
    int x = 0;
    for (y = ty; y < tile_h; y += blockDim.y) {
        for (x = tx; x < tile_w; x += blockDim.x) {
            int src_x = clamp_int_device(blockIdx.x * blockDim.x + x - radius, 0, w - 1);
            int src_y = clamp_int_device(blockIdx.y * blockDim.y + y - radius, 0, h - 1);
            tile[y * tile_w + x] = in[src_y * w + src_x];
        }
    }
    __syncthreads();

    if (gx >= w || gy >= h) {
        return;
    }

    {
        float vals[49];
        int idx = 0;
        int dy = 0;
        int dx = 0;
        int mid = 0;
        int i = 0;
        int j = 0;

        for (dy = -radius; dy <= radius; ++dy) {
            for (dx = -radius; dx <= radius; ++dx) {
                int lx = tx + radius + dx;
                int ly = ty + radius + dy;
                vals[idx++] = tile[ly * tile_w + lx];
            }
        }

        mid = idx / 2;
        for (i = 0; i <= mid; ++i) {
            int min_idx = i;
            for (j = i + 1; j < idx; ++j) {
                if (vals[j] < vals[min_idx]) {
                    min_idx = j;
                }
            }
            {
                float tmp = vals[i];
                vals[i] = vals[min_idx];
                vals[min_idx] = tmp;
            }
        }

        out[gy * w + gx] = vals[mid];
    }
}

__global__ void nlm_filter_kernel(const float* in, float* out, int w, int h, int patch_radius,
                                  int search_radius, float h_param, int rician_bias_correction,
                                  float noise_bias) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) {
        return;
    }

    {
        const float h2 = fmaxf(1e-6f, h_param * h_param);
        float weighted_sum = 0.0f;
        float weighted_sum_sq = 0.0f;
        float weight_acc = 0.0f;
        int sy = 0;
        int sx = 0;

        for (sy = -search_radius; sy <= search_radius; ++sy) {
            for (sx = -search_radius; sx <= search_radius; ++sx) {
                const int qx = clamp_int_device(x + sx, 0, w - 1);
                const int qy = clamp_int_device(y + sy, 0, h - 1);
                float patch_dist = 0.0f;
                int py = 0;
                int px = 0;

                for (py = -patch_radius; py <= patch_radius; ++py) {
                    for (px = -patch_radius; px <= patch_radius; ++px) {
                        const int ax = clamp_int_device(x + px, 0, w - 1);
                        const int ay = clamp_int_device(y + py, 0, h - 1);
                        const int bx = clamp_int_device(qx + px, 0, w - 1);
                        const int by = clamp_int_device(qy + py, 0, h - 1);
                        const float diff = in[ay * w + ax] - in[by * w + bx];
                        patch_dist += diff * diff;
                    }
                }

                {
                    const float wgt = expf(-patch_dist / h2);
                    const float sample = in[qy * w + qx];
                    weighted_sum += wgt * sample;
                    weighted_sum_sq += wgt * sample * sample;
                    weight_acc += wgt;
                }
            }
        }

        {
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
    }
}

__global__ void nlm_filter_kernel_shared(const float* in, float* out, int w, int h, int patch_radius,
                                         int search_radius, int total_radius, float h_param,
                                         int rician_bias_correction, float noise_bias) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * blockDim.x + tx;
    const int gy = blockIdx.y * blockDim.y + ty;

    const int tile_w = blockDim.x + 2 * total_radius;
    const int tile_h = blockDim.y + 2 * total_radius;

    int y = 0;
    int x = 0;
    for (y = ty; y < tile_h; y += blockDim.y) {
        for (x = tx; x < tile_w; x += blockDim.x) {
            int src_x = clamp_int_device(blockIdx.x * blockDim.x + x - total_radius, 0, w - 1);
            int src_y = clamp_int_device(blockIdx.y * blockDim.y + y - total_radius, 0, h - 1);
            tile[y * tile_w + x] = in[src_y * w + src_x];
        }
    }
    __syncthreads();

    if (gx >= w || gy >= h) {
        return;
    }

    {
        const float h2 = fmaxf(1e-6f, h_param * h_param);
        float weighted_sum = 0.0f;
        float weighted_sum_sq = 0.0f;
        float weight_acc = 0.0f;
        const int center_x = tx + total_radius;
        const int center_y = ty + total_radius;
        int sy = 0;
        int sx = 0;

        for (sy = -search_radius; sy <= search_radius; ++sy) {
            for (sx = -search_radius; sx <= search_radius; ++sx) {
                float patch_dist = 0.0f;
                int py = 0;
                int px = 0;

                for (py = -patch_radius; py <= patch_radius; ++py) {
                    for (px = -patch_radius; px <= patch_radius; ++px) {
                        const float a = tile[(center_y + py) * tile_w + (center_x + px)];
                        const float b = tile[(center_y + sy + py) * tile_w + (center_x + sx + px)];
                        const float diff = a - b;
                        patch_dist += diff * diff;
                    }
                }

                {
                    const float wgt = expf(-patch_dist / h2);
                    const float sample = tile[(center_y + sy) * tile_w + (center_x + sx)];
                    weighted_sum += wgt * sample;
                    weighted_sum_sq += wgt * sample * sample;
                    weight_acc += wgt;
                }
            }
        }

        {
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
    }
}

__global__ void mse_partial_reduce_kernel(const float* a, const float* b, size_t n, float* partial) {
    extern __shared__ float sh[];
    const size_t tid = (size_t)threadIdx.x;
    const size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + tid;

    float val = 0.0f;
    if (gid < n) {
        const float d = a[gid] - b[gid];
        val = d * d;
    }
    sh[tid] = val;
    __syncthreads();

    {
        unsigned int s = blockDim.x / 2;
        while (s > 0) {
            if (tid < (size_t)s) {
                sh[tid] += sh[tid + (size_t)s];
            }
            __syncthreads();
            s >>= 1;
        }
    }

    if (tid == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

static int median_filter_gpu(const Image* in, int radius, double* kernel_ms, double* total_ms, Image* out) {
    float* d_in = NULL;
    float* d_out = NULL;
    cudaEvent_t start_total = NULL;
    cudaEvent_t stop_total = NULL;
    cudaEvent_t start_kernel = NULL;
    cudaEvent_t stop_kernel = NULL;
    dim3 block;
    dim3 grid;
    int tile_w = 0;
    int tile_h = 0;
    size_t shmem_bytes = 0;
    float kernel_elapsed = 0.0f;
    float total_elapsed = 0.0f;

    if (!in || !in->pixels || !out) {
        return set_errorf("Internal error in median_filter_gpu");
    }
    if (!image_alloc(out, in->width, in->height)) {
        return 0;
    }

    CUDA_CHECK_GOTO(cudaEventCreate(&start_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&stop_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&start_kernel), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&stop_kernel), cleanup);

    CUDA_CHECK_GOTO(cudaEventRecord(start_total), cleanup);

    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_in, in->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_out, in->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(d_in, in->pixels, in->size * sizeof(float), cudaMemcpyHostToDevice), cleanup);

    block = dim3(16, 16);
    grid = dim3((unsigned int)((in->width + (int)block.x - 1) / (int)block.x),
                (unsigned int)((in->height + (int)block.y - 1) / (int)block.y));
    tile_w = (int)block.x + 2 * radius;
    tile_h = (int)block.y + 2 * radius;
    shmem_bytes = (size_t)tile_w * (size_t)tile_h * sizeof(float);

    CUDA_CHECK_GOTO(cudaEventRecord(start_kernel), cleanup);
    median_filter_kernel<<<grid, block, shmem_bytes>>>(d_in, d_out, in->width, in->height, radius);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    CUDA_CHECK_GOTO(cudaEventRecord(stop_kernel), cleanup);

    CUDA_CHECK_GOTO(cudaMemcpy(out->pixels, d_out, in->size * sizeof(float), cudaMemcpyDeviceToHost), cleanup);

    CUDA_CHECK_GOTO(cudaEventRecord(stop_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventSynchronize(stop_total), cleanup);

    CUDA_CHECK_GOTO(cudaEventElapsedTime(&kernel_elapsed, start_kernel, stop_kernel), cleanup);
    CUDA_CHECK_GOTO(cudaEventElapsedTime(&total_elapsed, start_total, stop_total), cleanup);

    if (kernel_ms) {
        *kernel_ms = (double)kernel_elapsed;
    }
    if (total_ms) {
        *total_ms = (double)total_elapsed;
    }

    if (d_in) {
        cudaFree(d_in);
    }
    if (d_out) {
        cudaFree(d_out);
    }
    if (start_total) {
        cudaEventDestroy(start_total);
    }
    if (stop_total) {
        cudaEventDestroy(stop_total);
    }
    if (start_kernel) {
        cudaEventDestroy(start_kernel);
    }
    if (stop_kernel) {
        cudaEventDestroy(stop_kernel);
    }
    return 1;

cleanup:
    if (d_in) {
        cudaFree(d_in);
    }
    if (d_out) {
        cudaFree(d_out);
    }
    if (start_total) {
        cudaEventDestroy(start_total);
    }
    if (stop_total) {
        cudaEventDestroy(stop_total);
    }
    if (start_kernel) {
        cudaEventDestroy(start_kernel);
    }
    if (stop_kernel) {
        cudaEventDestroy(stop_kernel);
    }
    image_free(out);
    return 0;
}

static int nlm_filter_gpu(const Image* in, int patch_radius, int search_radius, float h_param,
                          bool rician_bias_correction, float noise_sigma, double* kernel_ms,
                          double* total_ms, Image* out) {
    float* d_in = NULL;
    float* d_out = NULL;
    cudaEvent_t start_total = NULL;
    cudaEvent_t stop_total = NULL;
    cudaEvent_t start_kernel = NULL;
    cudaEvent_t stop_kernel = NULL;
    dim3 block;
    dim3 grid;
    const int total_radius = patch_radius + search_radius;
    const int rician_flag = rician_bias_correction ? 1 : 0;
    const float noise_bias = 2.0f * noise_sigma * noise_sigma;
    int dev = 0;
    cudaDeviceProp prop;
    int tile_w = 0;
    int tile_h = 0;
    size_t nlm_shmem_bytes = 0;
    bool use_shared = false;
    float kernel_elapsed = 0.0f;
    float total_elapsed = 0.0f;

    if (!in || !in->pixels || !out) {
        return set_errorf("Internal error in nlm_filter_gpu");
    }
    if (!image_alloc(out, in->width, in->height)) {
        return 0;
    }

    memset(&prop, 0, sizeof(prop));

    CUDA_CHECK_GOTO(cudaEventCreate(&start_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&stop_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&start_kernel), cleanup);
    CUDA_CHECK_GOTO(cudaEventCreate(&stop_kernel), cleanup);

    CUDA_CHECK_GOTO(cudaEventRecord(start_total), cleanup);

    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_in, in->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_out, in->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(d_in, in->pixels, in->size * sizeof(float), cudaMemcpyHostToDevice), cleanup);

    block = dim3(16, 16);
    grid = dim3((unsigned int)((in->width + (int)block.x - 1) / (int)block.x),
                (unsigned int)((in->height + (int)block.y - 1) / (int)block.y));

    CUDA_CHECK_GOTO(cudaGetDevice(&dev), cleanup);
    CUDA_CHECK_GOTO(cudaGetDeviceProperties(&prop, dev), cleanup);

    tile_w = (int)block.x + 2 * total_radius;
    tile_h = (int)block.y + 2 * total_radius;
    nlm_shmem_bytes = (size_t)tile_w * (size_t)tile_h * sizeof(float);
    use_shared = nlm_shmem_bytes <= (size_t)prop.sharedMemPerBlock;

    CUDA_CHECK_GOTO(cudaEventRecord(start_kernel), cleanup);
    if (use_shared) {
        nlm_filter_kernel_shared<<<grid, block, nlm_shmem_bytes>>>(
            d_in, d_out, in->width, in->height, patch_radius, search_radius, total_radius, h_param,
            rician_flag, noise_bias);
    } else {
        nlm_filter_kernel<<<grid, block>>>(d_in, d_out, in->width, in->height, patch_radius,
                                           search_radius, h_param, rician_flag, noise_bias);
    }
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    CUDA_CHECK_GOTO(cudaEventRecord(stop_kernel), cleanup);

    CUDA_CHECK_GOTO(cudaMemcpy(out->pixels, d_out, in->size * sizeof(float), cudaMemcpyDeviceToHost), cleanup);

    CUDA_CHECK_GOTO(cudaEventRecord(stop_total), cleanup);
    CUDA_CHECK_GOTO(cudaEventSynchronize(stop_total), cleanup);

    CUDA_CHECK_GOTO(cudaEventElapsedTime(&kernel_elapsed, start_kernel, stop_kernel), cleanup);
    CUDA_CHECK_GOTO(cudaEventElapsedTime(&total_elapsed, start_total, stop_total), cleanup);

    if (kernel_ms) {
        *kernel_ms = (double)kernel_elapsed;
    }
    if (total_ms) {
        *total_ms = (double)total_elapsed;
    }

    if (d_in) {
        cudaFree(d_in);
    }
    if (d_out) {
        cudaFree(d_out);
    }
    if (start_total) {
        cudaEventDestroy(start_total);
    }
    if (stop_total) {
        cudaEventDestroy(stop_total);
    }
    if (start_kernel) {
        cudaEventDestroy(start_kernel);
    }
    if (stop_kernel) {
        cudaEventDestroy(stop_kernel);
    }
    return 1;

cleanup:
    if (d_in) {
        cudaFree(d_in);
    }
    if (d_out) {
        cudaFree(d_out);
    }
    if (start_total) {
        cudaEventDestroy(start_total);
    }
    if (stop_total) {
        cudaEventDestroy(stop_total);
    }
    if (start_kernel) {
        cudaEventDestroy(start_kernel);
    }
    if (stop_kernel) {
        cudaEventDestroy(stop_kernel);
    }
    image_free(out);
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static float compute_psnr(const Image* ref, const Image* test) {
    double mse = 0.0;
    size_t i = 0;

    if (!ref || !test || !ref->pixels || !test->pixels || ref->width != test->width ||
        ref->height != test->height || ref->size != test->size) {
        return -1.0f;
    }

    for (i = 0; i < ref->size; ++i) {
        const double d = (double)ref->pixels[i] - (double)test->pixels[i];
        mse += d * d;
    }
    mse /= (double)ref->size;
    if (mse < 1e-12) {
        return INFINITY;
    }

    return (float)(10.0 * log10((kPixelMax * kPixelMax) / mse));
}

static int compute_psnr_gpu(const Image* ref, const Image* test, float* out_psnr) {
    const int threads = 256;
    int blocks = 0;
    float* d_ref = NULL;
    float* d_test = NULL;
    float* d_partial = NULL;
    float* partial = NULL;
    size_t i = 0;
    double mse_sum = 0.0;
    double mse = 0.0;

    if (!ref || !test || !out_psnr || !ref->pixels || !test->pixels || ref->width != test->width ||
        ref->height != test->height || ref->size != test->size) {
        return set_errorf("PSNR(GPU): image dimensions mismatch");
    }

    blocks = (int)((ref->size + (size_t)threads - 1) / (size_t)threads);

    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_ref, ref->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_test, ref->size * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_partial, (size_t)blocks * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(d_ref, ref->pixels, ref->size * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(d_test, test->pixels, test->size * sizeof(float), cudaMemcpyHostToDevice), cleanup);

    mse_partial_reduce_kernel<<<blocks, threads, (size_t)threads * sizeof(float)>>>(d_ref, d_test, ref->size,
                                                                                      d_partial);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    partial = (float*)malloc((size_t)blocks * sizeof(float));
    if (!partial) {
        set_errorf("Out of memory in compute_psnr_gpu");
        goto cleanup;
    }
    CUDA_CHECK_GOTO(cudaMemcpy(partial, d_partial, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost),
                    cleanup);

    for (i = 0; i < (size_t)blocks; ++i) {
        mse_sum += (double)partial[i];
    }
    mse = mse_sum / (double)ref->size;
    if (mse < 1e-12) {
        *out_psnr = INFINITY;
    } else {
        *out_psnr = (float)(10.0 * log10((kPixelMax * kPixelMax) / mse));
    }

    free(partial);
    if (d_ref) {
        cudaFree(d_ref);
    }
    if (d_test) {
        cudaFree(d_test);
    }
    if (d_partial) {
        cudaFree(d_partial);
    }
    return 1;

cleanup:
    free(partial);
    if (d_ref) {
        cudaFree(d_ref);
    }
    if (d_test) {
        cudaFree(d_test);
    }
    if (d_partial) {
        cudaFree(d_partial);
    }
    return 0;
}

static float compute_ssim_global(const Image* ref, const Image* test) {
    const double c1 = pow(0.01 * kPixelMax, 2.0);
    const double c2 = pow(0.03 * kPixelMax, 2.0);
    const double n = (double)ref->size;
    double mu_x = 0.0;
    double mu_y = 0.0;
    double sigma_x = 0.0;
    double sigma_y = 0.0;
    double sigma_xy = 0.0;
    size_t i = 0;

    if (!ref || !test || !ref->pixels || !test->pixels || ref->width != test->width ||
        ref->height != test->height || ref->size != test->size) {
        return -1.0f;
    }

    for (i = 0; i < ref->size; ++i) {
        mu_x += ref->pixels[i];
        mu_y += test->pixels[i];
    }
    mu_x /= n;
    mu_y /= n;

    for (i = 0; i < ref->size; ++i) {
        const double x = (double)ref->pixels[i] - mu_x;
        const double y = (double)test->pixels[i] - mu_y;
        sigma_x += x * x;
        sigma_y += y * y;
        sigma_xy += x * y;
    }

    sigma_x /= n;
    sigma_y /= n;
    sigma_xy /= n;

    {
        const double num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
        const double den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2);
        if (fabs(den) < 1e-15) {
            return 1.0f;
        }
        return (float)(num / den);
    }
}

static int evaluate(const Image* ref, const Image* test, bool use_gpu_psnr, Metrics* m) {
    if (!m) {
        return set_errorf("Internal error in evaluate");
    }
    if (use_gpu_psnr) {
        if (!compute_psnr_gpu(ref, test, &m->psnr)) {
            return 0;
        }
    } else {
        m->psnr = compute_psnr(ref, test);
        if (m->psnr < 0.0f) {
            return set_errorf("PSNR: image dimensions mismatch");
        }
    }

    m->ssim = compute_ssim_global(ref, test);
    if (m->ssim < 0.0f) {
        return set_errorf("SSIM: image dimensions mismatch");
    }

    return 1;
}

static float compute_sharpness_laplacian_variance(const Image* img) {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    double mean = 0.0;
    double mean_sq = 0.0;
    size_t n = 0;

    if (!img || !img->pixels) {
        return 0.0f;
    }
    if (img->width < 3 || img->height < 3) {
        return 0.0f;
    }

    w = img->width;
    h = img->height;
    for (y = 1; y < h - 1; ++y) {
        for (x = 1; x < w - 1; ++x) {
            const float c = img->pixels[(size_t)y * (size_t)w + (size_t)x];
            const float l = img->pixels[(size_t)y * (size_t)w + (size_t)(x - 1)];
            const float r = img->pixels[(size_t)y * (size_t)w + (size_t)(x + 1)];
            const float u = img->pixels[(size_t)(y - 1) * (size_t)w + (size_t)x];
            const float d = img->pixels[(size_t)(y + 1) * (size_t)w + (size_t)x];
            const double lap = (double)(l + r + u + d - 4.0f * c);
            mean += lap;
            mean_sq += lap * lap;
            ++n;
        }
    }

    if (n == 0) {
        return 0.0f;
    }

    mean /= (double)n;
    mean_sq /= (double)n;
    {
        const double var = mean_sq - mean * mean;
        return (float)fmax(0.0, var);
    }
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --input <input_image> [options]\n\n"
            "Options:\n"
            "  --input-is-clean <0|1>    Treat --input as clean image (1, default) or noisy image (0)\n"
            "  --ref <clean_reference>   Optional clean reference image for quality metrics\n"
            "  --output-prefix <name>    Output prefix (default: output)\n"
            "  --noise-std <float>       Gaussian noise stddev (used when --input-is-clean=1, default: 15)\n"
            "  --noise-sigma <float>     Known noise sigma for denoising; <=0 means auto estimate\n"
            "  --sp-ratio <float>        Salt-pepper probability (used when --input-is-clean=1, default: 0.01)\n"
            "  --auto-tune <0|1>         Auto-tune median/NLM params using estimated noise (default: 1)\n"
            "  --median-radius <int>     Median radius, max 3 (default: 1)\n"
            "  --nlm-patch <int>         NLM patch radius (default: 1)\n"
            "  --nlm-search <int>        NLM search radius (default: 3)\n"
            "  --nlm-h <float>           NLM filtering strength; <=0 means auto from sigma\n"
            "  --nlm-h-factor <float>    Auto NLM strength factor h=factor*sigma (default: 0.9)\n"
            "  --rician-bias-correct <0|1>  Bias-correct NLM for MRI-like Rician noise (default: 1)\n"
            "  --sharpen-amount <float>  Unsharp-mask amount on denoised outputs (default: 0)\n"
            "  --gpu-metrics <0|1>       Use GPU MSE reduction for PSNR (default: 1)\n",
            prog);
}

static int require_value(int argc, char** argv, int* i, const char* flag, const char** value) {
    if (*i + 1 >= argc) {
        return set_errorf("Missing value for %s", flag);
    }
    ++(*i);
    *value = argv[*i];
    return 1;
}

static int parse_args(int argc, char** argv, Params* p) {
    int i = 0;
    if (!p) {
        return set_errorf("Internal error: null params");
    }

    params_init(p);

    for (i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* val = NULL;

        if (strcmp(a, "--input") == 0) {
            if (!require_value(argc, argv, &i, a, &val) ||
                !copy_arg_string(p->input_path, sizeof(p->input_path), val, a)) {
                return 0;
            }
        } else if (strcmp(a, "--input-is-clean") == 0) {
            int v = 0;
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &v)) {
                return 0;
            }
            if (v != 0 && v != 1) {
                return set_errorf("--input-is-clean must be 0 or 1");
            }
            p->input_is_clean = (v == 1);
        } else if (strcmp(a, "--ref") == 0) {
            if (!require_value(argc, argv, &i, a, &val) ||
                !copy_arg_string(p->ref_path, sizeof(p->ref_path), val, a)) {
                return 0;
            }
        } else if (strcmp(a, "--output-prefix") == 0) {
            if (!require_value(argc, argv, &i, a, &val) ||
                !copy_arg_string(p->output_prefix, sizeof(p->output_prefix), val, a)) {
                return 0;
            }
        } else if (strcmp(a, "--noise-std") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_float_value(val, a, &p->noise_stddev)) {
                return 0;
            }
        } else if (strcmp(a, "--noise-sigma") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_float_value(val, a, &p->noise_sigma)) {
                return 0;
            }
        } else if (strcmp(a, "--sp-ratio") == 0) {
            if (!require_value(argc, argv, &i, a, &val) ||
                !parse_float_value(val, a, &p->salt_pepper_ratio)) {
                return 0;
            }
        } else if (strcmp(a, "--auto-tune") == 0) {
            int v = 0;
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &v)) {
                return 0;
            }
            if (v != 0 && v != 1) {
                return set_errorf("--auto-tune must be 0 or 1");
            }
            p->auto_tune = (v == 1);
        } else if (strcmp(a, "--median-radius") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &p->median_radius)) {
                return 0;
            }
        } else if (strcmp(a, "--nlm-patch") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &p->nlm_patch_radius)) {
                return 0;
            }
        } else if (strcmp(a, "--nlm-search") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &p->nlm_search_radius)) {
                return 0;
            }
        } else if (strcmp(a, "--nlm-h") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_float_value(val, a, &p->nlm_h)) {
                return 0;
            }
        } else if (strcmp(a, "--nlm-h-factor") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_float_value(val, a, &p->nlm_h_factor)) {
                return 0;
            }
        } else if (strcmp(a, "--rician-bias-correct") == 0) {
            int v = 0;
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &v)) {
                return 0;
            }
            if (v != 0 && v != 1) {
                return set_errorf("--rician-bias-correct must be 0 or 1");
            }
            p->rician_bias_correction = (v == 1);
        } else if (strcmp(a, "--sharpen-amount") == 0) {
            if (!require_value(argc, argv, &i, a, &val) || !parse_float_value(val, a, &p->sharpen_amount)) {
                return 0;
            }
        } else if (strcmp(a, "--gpu-metrics") == 0) {
            int v = 0;
            if (!require_value(argc, argv, &i, a, &val) || !parse_int_value(val, a, &v)) {
                return 0;
            }
            if (v != 0 && v != 1) {
                return set_errorf("--gpu-metrics must be 0 or 1");
            }
            p->gpu_metrics = (v == 1);
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            return set_errorf("Unknown option: %s", a);
        }
    }

    if (p->input_path[0] == '\0') {
        return set_errorf("--input is required");
    }
    if (p->median_radius < 1 || p->median_radius > 3) {
        return set_errorf("--median-radius must be in [1, 3]");
    }
    if (p->nlm_patch_radius < 1 || p->nlm_search_radius < 1) {
        return set_errorf("NLM radii must be >= 1");
    }
    if (p->nlm_h_factor <= 0.0f) {
        return set_errorf("--nlm-h-factor must be > 0");
    }
    if (p->sharpen_amount < 0.0f) {
        return set_errorf("--sharpen-amount must be >= 0");
    }

    return 1;
}

static void print_report(const char* title, const Timing* t, const Metrics* m_cpu, const Metrics* m_gpu) {
    printf("\n=== %s ===\n", title);
    printf("CPU time (ms):           %.3f\n", t->cpu_ms);
    printf("GPU total time (ms):     %.3f\n", t->gpu_total_ms);
    printf("GPU kernel time (ms):    %.3f\n", t->gpu_kernel_ms);
    printf("End-to-end speedup:      %.3fx\n", t->cpu_ms / fmax(1e-9, t->gpu_total_ms));
    printf("Kernel-only speedup:     %.3fx\n", t->cpu_ms / fmax(1e-9, t->gpu_kernel_ms));
    printf("CPU quality:             PSNR=%.3f dB, SSIM=%.6f\n", m_cpu->psnr, m_cpu->ssim);
    printf("GPU quality:             PSNR=%.3f dB, SSIM=%.6f\n", m_gpu->psnr, m_gpu->ssim);
}

static void print_timing_only_report(const char* title, const Timing* t) {
    printf("\n=== %s ===\n", title);
    printf("CPU time (ms):           %.3f\n", t->cpu_ms);
    printf("GPU total time (ms):     %.3f\n", t->gpu_total_ms);
    printf("GPU kernel time (ms):    %.3f\n", t->gpu_kernel_ms);
    printf("End-to-end speedup:      %.3fx\n", t->cpu_ms / fmax(1e-9, t->gpu_total_ms));
    printf("Kernel-only speedup:     %.3fx\n", t->cpu_ms / fmax(1e-9, t->gpu_kernel_ms));
}

int main(int argc, char** argv) {
    Params params;
    Image input_img;
    Image clean_ref;
    Image noisy;
    Image median_cpu;
    Image median_gpu;
    Image nlm_cpu;
    Image nlm_gpu;
    Image median_cpu_sharp;
    Image median_gpu_sharp;
    Image nlm_cpu_sharp;
    Image nlm_gpu_sharp;
    bool has_ref = false;
    bool apply_sharpen = false;
    float estimated_sigma = 0.0f;
    int effective_median_radius = 0;
    int effective_patch_radius = 0;
    int effective_search_radius = 0;
    float effective_nlm_h = 0.0f;
    bool likely_rician = false;
    bool use_rician_bias_correction = false;
    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;
    double median_gpu_kernel_ms = 0.0;
    double median_gpu_total_ms = 0.0;
    double nlm_gpu_kernel_ms = 0.0;
    double nlm_gpu_total_ms = 0.0;
    Timing median_t;
    Timing nlm_t;

    memset(&params, 0, sizeof(params));
    memset(&input_img, 0, sizeof(input_img));
    memset(&clean_ref, 0, sizeof(clean_ref));
    memset(&noisy, 0, sizeof(noisy));
    memset(&median_cpu, 0, sizeof(median_cpu));
    memset(&median_gpu, 0, sizeof(median_gpu));
    memset(&nlm_cpu, 0, sizeof(nlm_cpu));
    memset(&nlm_gpu, 0, sizeof(nlm_gpu));
    memset(&median_cpu_sharp, 0, sizeof(median_cpu_sharp));
    memset(&median_gpu_sharp, 0, sizeof(median_gpu_sharp));
    memset(&nlm_cpu_sharp, 0, sizeof(nlm_cpu_sharp));
    memset(&nlm_gpu_sharp, 0, sizeof(nlm_gpu_sharp));
    memset(&median_t, 0, sizeof(median_t));
    memset(&nlm_t, 0, sizeof(nlm_t));

    if (!parse_args(argc, argv, &params)) {
        fprintf(stderr, "Error: %s\n\n", g_error[0] ? g_error : "Unknown error");
        print_usage(argv[0]);
        return 1;
    }

    if (!read_image(params.input_path, &input_img)) {
        goto fail;
    }

    if (params.input_is_clean) {
        if (!image_clone(&input_img, &clean_ref)) {
            goto fail;
        }
        if (!add_noise(&clean_ref, params.noise_stddev, params.salt_pepper_ratio, 42u, &noisy)) {
            goto fail;
        }
        has_ref = true;
    } else {
        if (!image_clone(&input_img, &noisy)) {
            goto fail;
        }
        if (params.ref_path[0] != '\0') {
            if (!read_image(params.ref_path, &clean_ref)) {
                goto fail;
            }
            if (clean_ref.width != noisy.width || clean_ref.height != noisy.height ||
                clean_ref.size != noisy.size) {
                set_errorf("--ref image dimensions must match --input image dimensions");
                goto fail;
            }
            has_ref = true;
        }
    }

    estimated_sigma = params.noise_sigma;
    if (estimated_sigma <= 0.0f) {
        if (params.input_is_clean) {
            estimated_sigma = params.noise_stddev;
        } else {
            estimated_sigma = estimate_noise_sigma_mad(&noisy);
        }
    }
    if (!isfinite(estimated_sigma) || estimated_sigma <= 0.0f) {
        estimated_sigma = fmaxf(5.0f, params.noise_stddev);
    }

    effective_median_radius = params.median_radius;
    effective_patch_radius = params.nlm_patch_radius;
    effective_search_radius = params.nlm_search_radius;
    effective_nlm_h = params.nlm_h;
    tune_pipeline_params(&params, estimated_sigma, &effective_median_radius, &effective_patch_radius,
                         &effective_search_radius, &effective_nlm_h);

    likely_rician = !params.input_is_clean;
    use_rician_bias_correction = params.rician_bias_correction && likely_rician;

    t0 = now_ms();
    if (!median_filter_cpu(&noisy, effective_median_radius, &median_cpu)) {
        goto fail;
    }
    t1 = now_ms();

    if (!median_filter_gpu(&noisy, effective_median_radius, &median_gpu_kernel_ms, &median_gpu_total_ms,
                           &median_gpu)) {
        goto fail;
    }

    t2 = now_ms();
    if (!nlm_filter_cpu(&median_cpu, effective_patch_radius, effective_search_radius, effective_nlm_h,
                        use_rician_bias_correction, estimated_sigma, &nlm_cpu)) {
        goto fail;
    }
    t3 = now_ms();

    if (!nlm_filter_gpu(&median_gpu, effective_patch_radius, effective_search_radius, effective_nlm_h,
                        use_rician_bias_correction, estimated_sigma, &nlm_gpu_kernel_ms,
                        &nlm_gpu_total_ms, &nlm_gpu)) {
        goto fail;
    }

    apply_sharpen = params.sharpen_amount > 0.0f;
    if (apply_sharpen) {
        if (!sharpen_unsharp_cpu(&median_cpu, params.sharpen_amount, &median_cpu_sharp) ||
            !sharpen_unsharp_cpu(&median_gpu, params.sharpen_amount, &median_gpu_sharp) ||
            !sharpen_unsharp_cpu(&nlm_cpu, params.sharpen_amount, &nlm_cpu_sharp) ||
            !sharpen_unsharp_cpu(&nlm_gpu, params.sharpen_amount, &nlm_gpu_sharp)) {
            goto fail;
        }
    }

    if (has_ref) {
        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s_clean.pgm", params.output_prefix);
        if (!write_pgm(path, &clean_ref)) {
            goto fail;
        }
    }

    {
        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s_noisy.pgm", params.output_prefix);
        if (!write_pgm(path, &noisy)) {
            goto fail;
        }

        snprintf(path, sizeof(path), "%s_median_cpu.pgm", params.output_prefix);
        if (!write_pgm(path, &median_cpu)) {
            goto fail;
        }

        snprintf(path, sizeof(path), "%s_median_gpu.pgm", params.output_prefix);
        if (!write_pgm(path, &median_gpu)) {
            goto fail;
        }

        snprintf(path, sizeof(path), "%s_nlm_cpu.pgm", params.output_prefix);
        if (!write_pgm(path, &nlm_cpu)) {
            goto fail;
        }

        snprintf(path, sizeof(path), "%s_nlm_gpu.pgm", params.output_prefix);
        if (!write_pgm(path, &nlm_gpu)) {
            goto fail;
        }

        if (apply_sharpen) {
            snprintf(path, sizeof(path), "%s_median_cpu_sharp.pgm", params.output_prefix);
            if (!write_pgm(path, &median_cpu_sharp)) {
                goto fail;
            }

            snprintf(path, sizeof(path), "%s_median_gpu_sharp.pgm", params.output_prefix);
            if (!write_pgm(path, &median_gpu_sharp)) {
                goto fail;
            }

            snprintf(path, sizeof(path), "%s_nlm_cpu_sharp.pgm", params.output_prefix);
            if (!write_pgm(path, &nlm_cpu_sharp)) {
                goto fail;
            }

            snprintf(path, sizeof(path), "%s_nlm_gpu_sharp.pgm", params.output_prefix);
            if (!write_pgm(path, &nlm_gpu_sharp)) {
                goto fail;
            }
        }
    }

    median_t.cpu_ms = t1 - t0;
    median_t.gpu_total_ms = median_gpu_total_ms;
    median_t.gpu_kernel_ms = median_gpu_kernel_ms;

    nlm_t.cpu_ms = t3 - t2;
    nlm_t.gpu_total_ms = nlm_gpu_total_ms;
    nlm_t.gpu_kernel_ms = nlm_gpu_kernel_ms;

    printf("Loaded image: %dx%d\n", noisy.width, noisy.height);
    printf("Estimated sigma: %.3f\n", estimated_sigma);
    printf("Effective params: median-radius=%d, nlm-patch=%d, nlm-search=%d, nlm-h=%.3f, "
           "rician-bias-correct=%d\n",
           effective_median_radius, effective_patch_radius, effective_search_radius, effective_nlm_h,
           use_rician_bias_correction ? 1 : 0);

    if (has_ref) {
        Metrics noisy_m;
        Metrics median_cpu_m;
        Metrics median_gpu_m;
        Metrics nlm_cpu_m;
        Metrics nlm_gpu_m;

        if (!evaluate(&clean_ref, &noisy, params.gpu_metrics, &noisy_m) ||
            !evaluate(&clean_ref, &median_cpu, params.gpu_metrics, &median_cpu_m) ||
            !evaluate(&clean_ref, &median_gpu, params.gpu_metrics, &median_gpu_m) ||
            !evaluate(&clean_ref, &nlm_cpu, params.gpu_metrics, &nlm_cpu_m) ||
            !evaluate(&clean_ref, &nlm_gpu, params.gpu_metrics, &nlm_gpu_m)) {
            goto fail;
        }

        printf("Noisy image quality: PSNR=%.3f dB, SSIM=%.6f\n", noisy_m.psnr, noisy_m.ssim);
        print_report("Median Filter", &median_t, &median_cpu_m, &median_gpu_m);
        print_report("Non-Local Means (after Median)", &nlm_t, &nlm_cpu_m, &nlm_gpu_m);

        if (apply_sharpen) {
            Metrics median_cpu_sharp_m;
            Metrics median_gpu_sharp_m;
            Metrics nlm_cpu_sharp_m;
            Metrics nlm_gpu_sharp_m;

            if (!evaluate(&clean_ref, &median_cpu_sharp, params.gpu_metrics, &median_cpu_sharp_m) ||
                !evaluate(&clean_ref, &median_gpu_sharp, params.gpu_metrics, &median_gpu_sharp_m) ||
                !evaluate(&clean_ref, &nlm_cpu_sharp, params.gpu_metrics, &nlm_cpu_sharp_m) ||
                !evaluate(&clean_ref, &nlm_gpu_sharp, params.gpu_metrics, &nlm_gpu_sharp_m)) {
                goto fail;
            }

            printf("\n=== Sharpening (Unsharp Mask) ===\n");
            printf("Amount:                  %.3f\n", params.sharpen_amount);
            printf("Median CPU sharpened:    PSNR=%.3f dB, SSIM=%.6f\n", median_cpu_sharp_m.psnr,
                   median_cpu_sharp_m.ssim);
            printf("Median GPU sharpened:    PSNR=%.3f dB, SSIM=%.6f\n", median_gpu_sharp_m.psnr,
                   median_gpu_sharp_m.ssim);
            printf("NLM CPU sharpened:       PSNR=%.3f dB, SSIM=%.6f\n", nlm_cpu_sharp_m.psnr,
                   nlm_cpu_sharp_m.ssim);
            printf("NLM GPU sharpened:       PSNR=%.3f dB, SSIM=%.6f\n", nlm_gpu_sharp_m.psnr,
                   nlm_gpu_sharp_m.ssim);
        }
    } else {
        printf("No reference image provided. Reporting timings only.\n");
        print_timing_only_report("Median Filter", &median_t);
        print_timing_only_report("Non-Local Means (after Median)", &nlm_t);
    }

    if (apply_sharpen) {
        printf("\n=== Sharpness (Laplacian Variance) ===\n");
        printf("Median CPU:              %.3f -> %.3f\n",
               compute_sharpness_laplacian_variance(&median_cpu),
               compute_sharpness_laplacian_variance(&median_cpu_sharp));
        printf("Median GPU:              %.3f -> %.3f\n",
               compute_sharpness_laplacian_variance(&median_gpu),
               compute_sharpness_laplacian_variance(&median_gpu_sharp));
        printf("NLM CPU:                 %.3f -> %.3f\n",
               compute_sharpness_laplacian_variance(&nlm_cpu),
               compute_sharpness_laplacian_variance(&nlm_cpu_sharp));
        printf("NLM GPU:                 %.3f -> %.3f\n",
               compute_sharpness_laplacian_variance(&nlm_gpu),
               compute_sharpness_laplacian_variance(&nlm_gpu_sharp));
    }

    printf("\nSaved output images with prefix: %s\n", params.output_prefix);

    image_free(&input_img);
    image_free(&clean_ref);
    image_free(&noisy);
    image_free(&median_cpu);
    image_free(&median_gpu);
    image_free(&nlm_cpu);
    image_free(&nlm_gpu);
    image_free(&median_cpu_sharp);
    image_free(&median_gpu_sharp);
    image_free(&nlm_cpu_sharp);
    image_free(&nlm_gpu_sharp);
    return 0;

fail:
    fprintf(stderr, "Error: %s\n\n", g_error[0] ? g_error : "Unknown failure");
    print_usage(argv[0]);
    image_free(&input_img);
    image_free(&clean_ref);
    image_free(&noisy);
    image_free(&median_cpu);
    image_free(&median_gpu);
    image_free(&nlm_cpu);
    image_free(&nlm_gpu);
    image_free(&median_cpu_sharp);
    image_free(&median_gpu_sharp);
    image_free(&nlm_cpu_sharp);
    image_free(&nlm_gpu_sharp);
    return 1;
}