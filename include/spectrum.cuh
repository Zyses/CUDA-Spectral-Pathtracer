#ifndef SPECTRUM_CUH
#define SPECTRUM_CUH

#include "vec3.cuh"
#include <cuda_runtime.h>

constexpr float LAMBDA_MIN = 380.0f;  
constexpr float LAMBDA_MAX = 780.0f;

struct SpectralSampling {
    int num_samples;
    float lambda_min;
    float lambda_max;
    float delta_lambda;
};

__host__ __device__ inline void wavelength_to_xyz(float lambda, float& x, float& y, float& z) {
    float t1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
    float t2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
    float t3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
    x = 0.362f * expf(-0.5f * t1 * t1) + 1.056f * expf(-0.5f * t2 * t2) - 0.065f * expf(-0.5f * t3 * t3);

    float t4 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
    float t5 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
    y = 0.821f * expf(-0.5f * t4 * t4) + 0.286f * expf(-0.5f * t5 * t5);

    float t6 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
    float t7 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
    z = 1.217f * expf(-0.5f * t6 * t6) + 0.681f * expf(-0.5f * t7 * t7);
}

__host__ __device__ inline float sRGB_gamma_deprecated(float v) {
    return v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

__host__ __device__ inline Color wavelength_to_rgb(float wavelength) {
    float clamped_lambda = fmaxf(380.0f, fminf(wavelength, 780.0f));

    float x_val, y_val, z_val;
    wavelength_to_xyz(clamped_lambda, x_val, y_val, z_val);

    float r =  3.2404542f * x_val - 1.5371385f * y_val - 0.4985314f * z_val;
    float g = -0.9692660f * x_val + 1.8760108f * y_val + 0.0415560f * z_val;
    float b =  0.0556434f * x_val - 0.2040259f * y_val + 1.0572252f * z_val;

    return Color(r, g, b);
}

__host__ __device__ inline float cauchy_dispersion(float wavelength, float A, float B, float C) {
    float lambda_squared = wavelength * wavelength;
    return A + B / lambda_squared + C / (lambda_squared * lambda_squared);
}

__host__ __device__ inline float glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5168f, 4320.0f, 0.0f);
}

__host__ __device__ inline float flint_glass_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.5837f, 10800.0f, 0.0f);
}

__host__ __device__ inline float diamond_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 2.38f, 19470.0f, 0.0f);
}

__host__ __device__ inline float water_ior(float wavelength) {
    return cauchy_dispersion(wavelength, 1.319f, 3370.0f, 0.0f);
}

#endif // SPECTRUM_CUH