#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "spectrum.cuh"
#include <curand_kernel.h>

struct HitRecord;

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    SPECTRAL
};

struct MaterialData {
    MaterialType type;

    SpectralProfile reflectance;
    float fuzz;
    float ior;
    bool dispersive;
    SpectralProfile emission;
};

__device__ inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ inline bool scatter_lambertian(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);
    attenuation = evaluate_spectrum(mat_data.reflectance, r_in.wavelength);
    return true;
}

__device__ inline bool scatter_metal(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 reflected = reflect(normalize(r_in.direction), rec.normal);
    scattered = Ray(
        rec.p,
        reflected + mat_data.fuzz * random_in_unit_sphere(rand_state),
        r_in.wavelength
    );
    attenuation = evaluate_spectrum(mat_data.reflectance, r_in.wavelength);
    return (dot(scattered.direction, rec.normal) > 0);
}

__device__ inline bool scatter_dielectric(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    attenuation = 1.0f;

    float refraction_ratio;
    float index;

    if (mat_data.dispersive) {
        index = flint_glass_ior(r_in.wavelength);
    } else {
        index = mat_data.ior;
    }

    if (rec.front_face) {
        refraction_ratio = 1.0f / index;
    } else {
        refraction_ratio = index;
    }

    Vec3 unit_direction = normalize(r_in.direction);
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    scattered = Ray(rec.p, direction, r_in.wavelength);
    return true;
}

__device__ inline bool scatter_emissive(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    (void)r_in;
    (void)rec;
    (void)mat_data;
    (void)attenuation;
    (void)scattered;
    (void)rand_state;
    return false;
}

__device__ inline bool scatter_spectral(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

    if (scatter_direction.near_zero()) {
        scatter_direction = rec.normal;
    }

    scattered = Ray(rec.p, scatter_direction, r_in.wavelength);
    attenuation = evaluate_spectrum(mat_data.reflectance, r_in.wavelength);
    return true;
}

__device__ inline bool scatter(
    const Ray& r_in,
    const HitRecord& rec,
    const MaterialData& mat_data,
    float& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    switch (mat_data.type) {
        case LAMBERTIAN:
            return scatter_lambertian(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case METAL:
            return scatter_metal(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case DIELECTRIC:
            return scatter_dielectric(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case EMISSIVE:
            return scatter_emissive(r_in, rec, mat_data, attenuation, scattered, rand_state);
        case SPECTRAL:
            return scatter_spectral(r_in, rec, mat_data, attenuation, scattered, rand_state);
        default:
            return false;
    }
}

__device__ inline float emit(const MaterialData& mat_data, float wavelength, float u, float v, const Point3& p) {
    (void)u;
    (void)v;
    (void)p;

    if (mat_data.type == EMISSIVE) {
        return evaluate_spectrum(mat_data.emission, wavelength);
    }
    return 0.0f;
}

#endif // MATERIAL_CUH
