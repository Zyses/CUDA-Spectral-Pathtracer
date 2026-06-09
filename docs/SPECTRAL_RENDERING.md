# Spectral Rendering Notes

This project traces spectral light paths instead of RGB paths. A ray carries one sampled wavelength in nanometers, and the path throughput is a scalar value for that wavelength.

## Core Types

`Ray` stores the current wavelength:

```cpp
Ray(origin, direction, wavelength_nm)
```

`SpectralProfile` describes analytic reflectance or emission curves:

```cpp
make_constant_spectrum(scale)
make_gaussian_spectrum(scale, center_nm, width_nm)
make_band_spectrum(scale, center_nm, width_nm)
```

The profiles are evaluated with:

```cpp
evaluate_spectrum(profile, wavelength_nm)
```

## Path Throughput

The integrator in `src/cuda_kernels.cu` uses scalar spectral radiance:

```cpp
float spectral_radiance = ray_spectral_radiance(...);
```

At each bounce:

- diffuse and spectral materials multiply throughput by `reflectance(lambda)`
- metals multiply throughput by `reflectance(lambda)` and reflect the ray
- dielectrics transmit or reflect with an IOR evaluated at `lambda`
- emissive materials return `emission(lambda)`

This avoids mixing RGB values inside the path.

## Pixel Accumulation

The framebuffer stores XYZ, not display RGB. For each wavelength sample:

```cpp
wavelength_to_xyz(lambda, x, y, z);
sample_xyz += spectral_radiance * Color(x, y, z) * delta_lambda;
```

After rendering, `image_io.cpp` divides the XYZ integral by the Y integral of a flat visible spectrum. This keeps a constant unit spectrum near display luminance 1 instead of letting the raw nanometer integral overexpose the image.

Then it converts:

```text
normalized XYZ -> linear sRGB -> gamma encoded sRGB
```

The PPM writer clamps negative or out-of-range display values after the color-space conversion.

## Dispersion

Dielectric materials can use a wavelength-dependent index of refraction:

```cpp
index = flint_glass_ior(r_in.wavelength);
```

The IOR functions are based on Cauchy's equation:

```text
n(lambda) = A + B / lambda^2 + C / lambda^4
```

Because every ray carries a single wavelength, refraction direction can differ by wavelength and produce dispersion.

## Object Rotation

`HittableData` contains transform metadata:

```cpp
Point3 rotation_pivot;
Vec3 rotation_axis;
float rotation_degrees;
```

Intersection works by inverse-transforming the ray into the object's local space:

```text
world ray -> inverse object rotation -> primitive hit test
```

If the primitive is hit, the hit point and normal are transformed back:

```text
local hit -> object rotation -> world hit
```

This keeps sphere, triangle, and rectangle intersection code simple while allowing scene helpers to pass an axis and angle.

Example:

```cpp
add_triangular_prism(
    scene,
    Point3(0, 2, -1),
    3.0f,
    3.0f,
    prism_material,
    Vec3(0, 1, 0),
    25.0f
);
```

The angle is in degrees. A zero angle or zero axis disables rotation.

## Current Limitations

- The renderer is spectral in transport, but not a fully calibrated physical renderer.
- Spectral profiles are analytic approximations, not measured data.
- The CIE matching functions are compact analytic approximations.
- The framebuffer type is still named `Color` because it reuses `Vec3`; semantically it stores XYZ until image output.
- Rotations are per primitive entry in `HittableData`. Compound helpers such as prisms assign the same pivot and rotation to all generated triangles.

## Physical Accuracy

The renderer is physically motivated in these areas:

- rays carry a single wavelength instead of RGB triplets
- material reflectance and emission are evaluated at that wavelength
- dielectric dispersion changes the IOR per wavelength
- the final display color is integrated through CIE-like XYZ matching functions
- direct RGB wavelength mixing inside the path has been removed

It is still approximate in these areas:

- light emission values are scene-tuned numbers, not calibrated radiometric units
- the analytic CIE functions are compact approximations, not tabulated official observer data
- reflectance and emission spectra are simple constants, Gaussians, or bands, not measured SPDs
- Lambertian scattering does not include explicit BRDF/PDF energy factors such as `albedo / pi` and cosine-weighted PDF cancellation in a rigorously documented estimator
- there is no Fresnel spectral absorption, Beer-Lambert transmission, polarization, wavelength-dependent medium absorption, or spectral MIS
- output uses simple white normalization, clamp, and gamma conversion instead of a camera exposure and tone-mapping model

So the current state is a real spectral path tracer in the sense that wavelength-dependent transport is represented in the ray/path state. It should not yet be treated as a quantitatively accurate spectral renderer for measurement or prediction.
