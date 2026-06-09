#ifndef SPECTRUM_CUH
#define SPECTRUM_CUH

#include "vec3.cuh"
#include <cuda_runtime.h>

constexpr float LAMBDA_MIN = 380.0f;  
constexpr float LAMBDA_MAX = 780.0f;
constexpr float VISIBLE_RANGE_NM = LAMBDA_MAX - LAMBDA_MIN;
constexpr float CIE_Y_INTEGRAL = 106.856895f;

struct SpectralSampling {
    int num_samples;
    float lambda_min;
    float lambda_max;
    float delta_lambda;
};

enum SpectrumType {
    SPECTRUM_CONSTANT,
    SPECTRUM_GAUSSIAN,
    SPECTRUM_BAND,
    SPECTRUM_BLACKBODY_APPROX
};

struct SpectralProfile {
    SpectrumType type;
    float scale;
    float center_nm;
    float width_nm;
    float secondary_center_nm;
    float secondary_width_nm;
};

__host__ __device__ inline SpectralProfile make_constant_spectrum(float scale) {
    return SpectralProfile{SPECTRUM_CONSTANT, scale, 550.0f, 1.0f, 0.0f, 1.0f};
}

__host__ __device__ inline SpectralProfile make_gaussian_spectrum(float scale, float center_nm, float width_nm) {
    return SpectralProfile{SPECTRUM_GAUSSIAN, scale, center_nm, width_nm, 0.0f, 1.0f};
}

__host__ __device__ inline SpectralProfile make_band_spectrum(float scale, float center_nm, float width_nm) {
    return SpectralProfile{SPECTRUM_BAND, scale, center_nm, width_nm, 0.0f, 1.0f};
}

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

__host__ __device__ inline Color xyz_to_linear_srgb(const Color& xyz) {
    return Color(
         3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
         0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z
    );
}

__host__ __device__ inline float evaluate_spectrum(const SpectralProfile& spectrum, float wavelength) {
    float lambda = fmaxf(LAMBDA_MIN, fminf(wavelength, LAMBDA_MAX));

    switch (spectrum.type) {
        case SPECTRUM_GAUSSIAN: {
            float sigma = fmaxf(spectrum.width_nm, 1.0f);
            float t = (lambda - spectrum.center_nm) / sigma;
            return spectrum.scale * expf(-0.5f * t * t);
        }
        case SPECTRUM_BAND: {
            float half_width = fmaxf(spectrum.width_nm, 1.0f) * 0.5f;
            return fabsf(lambda - spectrum.center_nm) <= half_width ? spectrum.scale : 0.0f;
        }
        case SPECTRUM_BLACKBODY_APPROX: {
            float warm = expf(-0.5f * powf((lambda - 610.0f) / 135.0f, 2.0f));
            float cool = 0.35f * expf(-0.5f * powf((lambda - 450.0f) / 95.0f, 2.0f));
            return spectrum.scale * (warm + cool);
        }
        case SPECTRUM_CONSTANT:
        default:
            return spectrum.scale;
    }
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
