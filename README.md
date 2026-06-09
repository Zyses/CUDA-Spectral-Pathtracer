# CUDA Spectral Pathtracer

A CUDA path tracer that transports a single wavelength per ray and converts the accumulated spectral result to display RGB only at the end of the render. The renderer is built as an experimental learning project for spectral light transport, wavelength-dependent materials, and GPU rendering.

## Features

- **Spectral path tracing**: Samples wavelengths between 380nm and 780nm and traces scalar radiance per wavelength
- **CUDA acceleration**: Uses GPU kernels for random-state initialization and rendering
- **Spectral material system**:
  - Lambertian diffuse surfaces
  - Metallic surfaces with configurable roughness
  - Dielectric materials with wavelength-dependent IOR
  - Analytic spectral reflectance profiles
  - Analytic spectral emission profiles
- **Geometry**: Spheres, triangles, axis-aligned rectangles, triangular prisms, and boxes
- **Object rotation**: Objects can define a rotation axis and angle; rays are transformed into object-local space for intersections
- **Camera**: Perspective camera with depth of field
- **Image output**: Timestamped PPM files

## How Spectral Rendering Works

Each camera sample draws multiple wavelengths. A path carries one wavelength and one scalar throughput value. Materials evaluate reflectance, emission, and dielectric IOR at that wavelength.

The renderer accumulates CIE XYZ values per pixel:

```text
XYZ += L(lambda) * CIE_XYZ(lambda) * delta_lambda
```

The final image is white-point normalized, converted from XYZ to linear sRGB, clamped, gamma encoded, and written to PPM. RGB values are not transported along the light path.

See [docs/SPECTRAL_RENDERING.md](docs/SPECTRAL_RENDERING.md) for implementation details.

## Dispersion

Dielectric materials can use Cauchy's equation for wavelength-dependent refraction:

```text
n(lambda) = A + B / lambda^2 + C / lambda^4
```

This lets different wavelengths refract differently through glass-like materials, producing prism and rainbow effects from a white spectral light source.

## Building

### Prerequisites

- CUDA Toolkit 11.0 or newer
- CMake 3.18 or newer
- CUDA-capable GPU, compute capability 7.5 or newer recommended
- MSVC on Windows

### Visual Studio

Open the folder in Visual Studio and build the CMake project with one of the provided presets.

### Command Line

From a Visual Studio Developer Command Prompt:

```bat
cmake --preset x64-release
cmake --build out/build/x64-release --config Release
```

## Usage

Run the compiled executable:

```bat
PathtracerSpectralRealtime.exe
```

The program loads the Rainbow focus showcase scene directly, prints its configuration, and writes a timestamped `.ppm` image.

## Scene Authoring Notes

Materials use analytic spectral profiles:

```cpp
int white = add_lambertian_material(scene, make_constant_spectrum(0.73f));
int light = add_emissive_material(scene, make_constant_spectrum(28.0f));
```

Most object helper functions accept optional rotation parameters:

```cpp
add_triangular_prism(scene, Point3(0, 2, -1), 3, 3, prism_material, Vec3(0, 1, 0), 25.0f);
add_rectangle_xz(scene, -1, 1, -1, 1, 0, white_material, Vec3(1, 0, 0), 15.0f);
```

The rotation is defined by an axis vector and an angle in degrees. Internally the ray is transformed into object-local space before the intersection test, and the hit point and normal are transformed back into world space.

## Future Work

- BVH acceleration
- Tabulated measured SPDs and CIE CMFs
- Textures and UV-driven spectral profiles
- Volumetric scattering
- Better tone mapping
- Denoising
