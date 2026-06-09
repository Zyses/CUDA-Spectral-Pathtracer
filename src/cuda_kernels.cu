#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <stdio.h>

#include "vec3.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "spectrum.cuh"
#include "cuda_utils.cuh"

__global__ void init_rand_state(curandState* rand_state, int width, int height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ float ray_spectral_radiance(
    const Ray& r,
    const HittableData* objects,
    int num_objects,
    const SphereData* spheres,
    const TriangleData* triangles,
    const RectangleXYData* rect_xy,
    const RectangleXZData* rect_xz,
    const RectangleYZData* rect_yz,
    const MaterialData* materials,
    int depth,
    curandState* local_rand_state
) {
    Ray cur_ray = r;
    float cur_attenuation = 1.0f;
    float cur_emitted = 0.0f;

    for (int i = 0; i < depth; i++) {
        HitRecord rec;

        if (hit_objects(
                cur_ray, 0.001f, FLT_MAX, rec,
                objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz
            )) {

            float emitted = emit(materials[rec.material_id], cur_ray.wavelength, rec.u, rec.v, rec.p);
            cur_emitted += cur_attenuation * emitted;

            Ray scattered;
            float attenuation;

            if (scatter(cur_ray, rec, materials[rec.material_id], attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return cur_emitted;
            }
        } else {
            Vec3 unit_direction = normalize(cur_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            float background = (1.0f - t) * 0.02f + t * evaluate_spectrum(make_gaussian_spectrum(0.08f, 465.0f, 120.0f), cur_ray.wavelength);
            return cur_emitted + cur_attenuation * background;
        }

        if (cur_attenuation < 0.001f) {
            float q = fmaxf(cur_attenuation, 0.05f);
            if (curand_uniform(local_rand_state) >= q) {
                return cur_emitted;
            }
            cur_attenuation /= q;
        }
    }

    return cur_emitted;
}

__global__ void render_kernel(
    Color* framebuffer,
    int width,
    int height,
    int samples_per_pixel,
    int max_depth,
    int spectral_samples,
    float lambda_min,
    float lambda_max,
    Camera* camera,
    HittableData* objects,
    int num_objects,
    SphereData* spheres,
    TriangleData* triangles,
    RectangleXYData* rect_xy,
    RectangleXZData* rect_xz,
    RectangleYZData* rect_yz,
    MaterialData* materials,
    curandState* rand_state
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];

    Color pixel_xyz(0, 0, 0);

    float delta_lambda = (lambda_max - lambda_min) / spectral_samples;

    for (int s = 0; s < samples_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width - 1);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height - 1);

        Color sample_xyz(0, 0, 0);
        for (int w = 0; w < spectral_samples; w++) {
            float lambda = lambda_min + (w + curand_uniform(&local_rand_state)) * delta_lambda;

            Ray r = camera->get_ray(u, v, &local_rand_state, lambda);

            float spectral_radiance = ray_spectral_radiance(
                r, objects, num_objects,
                spheres, triangles, rect_xy, rect_xz, rect_yz,
                materials, max_depth, &local_rand_state
            );

            float x, y, z;
            wavelength_to_xyz(lambda, x, y, z);
            sample_xyz += spectral_radiance * Color(x, y, z) * delta_lambda;
        }

        pixel_xyz += sample_xyz;
    }

    rand_state[pixel_index] = local_rand_state;

    framebuffer[pixel_index] = pixel_xyz;
}

extern "C" void launch_init_rand_states(
    curandState* rand_state,
    const ImageProperties& img_props,
    unsigned long seed
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    int width = img_props.width;
    int height = img_props.height;

    void* args[] = {
        &rand_state,
        &width,
        &height,
        &seed
    };

    CHECK_CUDA_ERROR(cudaLaunchKernel(
        reinterpret_cast<const void*>(init_rand_state),
        grid,
        blocks,
        args,
        0,
        nullptr
    ));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

extern "C" void launch_render_kernel(
    Color* framebuffer,
    const ImageProperties& img_props,
    Camera* dev_camera,
    HittableData* dev_objects,
    int num_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    curandState* rand_state
) {
    dim3 blocks(16, 16);
    dim3 grid(
        (img_props.width + blocks.x - 1) / blocks.x,
        (img_props.height + blocks.y - 1) / blocks.y
    );

    int width = img_props.width;
    int height = img_props.height;
    int samples_per_pixel = img_props.samples_per_pixel;
    int max_depth = img_props.max_depth;
    int spectral_samples = img_props.spectral_samples;
    float lambda_min = img_props.lambda_min;
    float lambda_max = img_props.lambda_max;

    void* args[] = {
        &framebuffer,
        &width,
        &height,
        &samples_per_pixel,
        &max_depth,
        &spectral_samples,
        &lambda_min,
        &lambda_max,
        &dev_camera,
        &dev_objects,
        &num_objects,
        &dev_spheres,
        &dev_triangles,
        &dev_rect_xy,
        &dev_rect_xz,
        &dev_rect_yz,
        &dev_materials,
        &rand_state
    };

    CHECK_CUDA_ERROR(cudaLaunchKernel(
        reinterpret_cast<const void*>(render_kernel),
        grid,
        blocks,
        args,
        0,
        nullptr
    ));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
