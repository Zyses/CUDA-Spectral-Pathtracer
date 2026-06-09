#include "scene_setup.h"
#include "cuda_utils.cuh"
#include <cmath>
#include <cfloat>

HittableData make_hittable(
    HittableType type,
    int data_index,
    const Point3& rotation_pivot,
    const Vec3& rotation_axis,
    float rotation_degrees
) {
    HittableData object;
    object.type = type;
    object.data_index = data_index;
    object.rotation_pivot = rotation_pivot;
    object.rotation_axis = rotation_axis;
    object.rotation_degrees = rotation_degrees;
    return object;
}

int add_sphere(
    Scene& scene,
    const Point3& center,
    float radius,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f
) {
    int index = static_cast<int>(scene.spheres.size());
    
    SphereData sphere;
    sphere.center = center;
    sphere.radius = radius;
    sphere.material_id = material_id;
    
    scene.spheres.push_back(sphere);
    
    scene.objects.push_back(make_hittable(SPHERE, index, center, rotation_axis, rotation_degrees));
    
    return index;
}

int add_rectangle_xy(
    Scene& scene,
    float x0,
    float x1,
    float y0,
    float y1,
    float k,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f
) {
    int index = static_cast<int>(scene.rect_xy.size());
    
    RectangleXYData rect;
    rect.x0 = x0;
    rect.x1 = x1;
    rect.y0 = y0;
    rect.y1 = y1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_xy.push_back(rect);
    
    Point3 pivot((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, k);
    scene.objects.push_back(make_hittable(RECTANGLE_XY, index, pivot, rotation_axis, rotation_degrees));
    
    return index;
}

int add_rectangle_xz(
    Scene& scene,
    float x0,
    float x1,
    float z0,
    float z1,
    float k,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f
) {
    int index = static_cast<int>(scene.rect_xz.size());
    
    RectangleXZData rect;
    rect.x0 = x0;
    rect.x1 = x1;
    rect.z0 = z0;
    rect.z1 = z1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_xz.push_back(rect);
    
    Point3 pivot((x0 + x1) * 0.5f, k, (z0 + z1) * 0.5f);
    scene.objects.push_back(make_hittable(RECTANGLE_XZ, index, pivot, rotation_axis, rotation_degrees));
    
    return index;
}

int add_rectangle_yz(
    Scene& scene,
    float y0,
    float y1,
    float z0,
    float z1,
    float k,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f
) {
    int index = static_cast<int>(scene.rect_yz.size());
    
    RectangleYZData rect;
    rect.y0 = y0;
    rect.y1 = y1;
    rect.z0 = z0;
    rect.z1 = z1;
    rect.k = k;
    rect.material_id = material_id;
    
    scene.rect_yz.push_back(rect);
    
    Point3 pivot(k, (y0 + y1) * 0.5f, (z0 + z1) * 0.5f);
    scene.objects.push_back(make_hittable(RECTANGLE_YZ, index, pivot, rotation_axis, rotation_degrees));
    
    return index;
}

int add_triangle(
    Scene& scene,
    const Point3& v0,
    const Point3& v1,
    const Point3& v2,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f,
    const Point3& rotation_pivot = Point3(FLT_MAX, FLT_MAX, FLT_MAX)
) {
    int index = static_cast<int>(scene.triangles.size());
    
    TriangleData triangle;
    triangle.v0 = v0;
    triangle.v1 = v1;
    triangle.v2 = v2;
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    triangle.normal = normalize(cross(edge1, edge2));
    
    triangle.material_id = material_id;
    
    scene.triangles.push_back(triangle);
    
    Point3 pivot = rotation_pivot.x == FLT_MAX ? (v0 + v1 + v2) / 3.0f : rotation_pivot;
    scene.objects.push_back(make_hittable(TRIANGLE, index, pivot, rotation_axis, rotation_degrees));
    
    return index;
}

void add_triangular_prism(
    Scene& scene,
    const Point3& base_center,
    float base_size,
    float height,
    int material_id,
    const Vec3& rotation_axis = Vec3(0, 1, 0),
    float rotation_degrees = 0.0f
) {
    const float h = base_size * std::sqrt(3) / 2.0f;
    const float r = base_size / std::sqrt(3.0f);
    
    Point3 base_vertices[3];
    base_vertices[0] = Point3(base_center.x, base_center.y, base_center.z + r);
    base_vertices[1] = Point3(base_center.x + base_size/2.0f, base_center.y, base_center.z - h/3.0f);
    base_vertices[2] = Point3(base_center.x - base_size/2.0f, base_center.y, base_center.z - h/3.0f);
    
    Point3 top_vertices[3];
    for (int i = 0; i < 3; i++) {
        top_vertices[i] = Point3(base_vertices[i].x, base_vertices[i].y + height, base_vertices[i].z);
    }
    
    Point3 prism_pivot = base_center + Vec3(0.0f, height * 0.5f, 0.0f);
    add_triangle(scene, base_vertices[0], base_vertices[1], base_vertices[2], material_id, rotation_axis, rotation_degrees, prism_pivot);
    
    add_triangle(scene, top_vertices[0], top_vertices[2], top_vertices[1], material_id, rotation_axis, rotation_degrees, prism_pivot);
    
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        add_triangle(scene, base_vertices[i], base_vertices[j], top_vertices[i], material_id, rotation_axis, rotation_degrees, prism_pivot);
        add_triangle(scene, base_vertices[j], top_vertices[j], top_vertices[i], material_id, rotation_axis, rotation_degrees, prism_pivot);
    }
}

void add_box(Scene& scene, const Point3& p0, const Point3& p1, int material_id) {
    add_rectangle_xy(scene, p0.x, p1.x, p0.y, p1.y, p1.z, material_id);
    add_rectangle_xy(scene, p0.x, p1.x, p0.y, p1.y, p0.z, material_id);
    
    add_rectangle_xz(scene, p0.x, p1.x, p0.z, p1.z, p1.y, material_id);
    add_rectangle_xz(scene, p0.x, p1.x, p0.z, p1.z, p0.y, material_id);
    
    add_rectangle_yz(scene, p0.y, p1.y, p0.z, p1.z, p1.x, material_id);
    add_rectangle_yz(scene, p0.y, p1.y, p0.z, p1.z, p0.x, material_id);
}

int add_material(Scene& scene, const MaterialData& material) {
    int index = static_cast<int>(scene.materials.size());
    scene.materials.push_back(material);
    return index;
}

MaterialData make_base_material(MaterialType type) {
    MaterialData material;
    material.type = type;
    material.reflectance = make_constant_spectrum(1.0f);
    material.fuzz = 0.0f;
    material.ior = 1.5f;
    material.dispersive = false;
    material.emission = make_constant_spectrum(0.0f);
    return material;
}

int add_lambertian_material(Scene& scene, const SpectralProfile& reflectance) {
    MaterialData material = make_base_material(LAMBERTIAN);
    material.reflectance = reflectance;
    return add_material(scene, material);
}

int add_metal_material(Scene& scene, const SpectralProfile& reflectance, float fuzz) {
    MaterialData material = make_base_material(METAL);
    material.reflectance = reflectance;
    material.fuzz = fuzz;
    return add_material(scene, material);
}

int add_dielectric_material(Scene& scene, float ior, bool dispersive) {
    MaterialData material = make_base_material(DIELECTRIC);
    material.ior = ior;
    material.dispersive = dispersive;
    return add_material(scene, material);
}

int add_emissive_material(Scene& scene, const SpectralProfile& emission) {
    MaterialData material = make_base_material(EMISSIVE);
    material.emission = emission;
    return add_material(scene, material);
}

int add_spectral_material(Scene& scene, const SpectralProfile& reflectance) {
    MaterialData material = make_base_material(SPECTRAL);
    material.reflectance = reflectance;
    return add_material(scene, material);
}

Scene create_rainbow_scene(const ImageProperties& img_props) {
    Point3 lookfrom(0.5f, 2, 0);
    Point3 lookat(0.25f, 0.5f, 0);
    Vec3 vup(0, 1, 0);
    float dist_to_focus = 1.5f;
    float aperture = 0.0f;
    float aspect_ratio = static_cast<float>(img_props.width) / img_props.height;

    Camera cam(lookfrom, lookat, vup, 60, aspect_ratio, aperture, dist_to_focus);

    Scene scene(cam, img_props);

    int screen_material = add_lambertian_material(scene, make_constant_spectrum(0.82f));
    int dark_material = add_lambertian_material(scene, make_constant_spectrum(0.08f));
    int prism_material = add_dielectric_material(scene, 1.5f, true);
    int light_material = add_emissive_material(scene, make_constant_spectrum(12));

    // Keep the floor and rear wall bright enough to catch the caustic, but make the rest dark.
    add_rectangle_xz(scene, -1, 1, -1, 1, 0, screen_material); // Projection floor
    add_rectangle_xz(scene, -1, 1, -1, 1, 4, dark_material); // Projection roof
    add_rectangle_xy(scene, -1, 1, 0, 4, 1, dark_material);  // Projection wall
    add_rectangle_xy(scene, -1, 1, 0, 4, -1, dark_material);   // Front wall
    add_rectangle_yz(scene, 0, 4, -1, 1, -1, dark_material);   // Left wall
    add_rectangle_yz(scene, 0, 4, -1, 1, 1, dark_material);    // Right wall
    add_rectangle_xz(scene, -1, 1, -1, 1, 4, dark_material);   // Ceiling

    // The prism in the center. Keep the size positive so triangle winding and the dispersion orientation stay readable.
    add_triangular_prism(scene, Point3(0, 0.01f, 0.25f), 1.0f, 1.0f, prism_material, Vec3(0, 1, 0), 70.0f);

    // Smaller source means less angular spread and a clearer projected spectrum.
    add_sphere(scene, Point3(-0.5f, 0.75f, -0.5f), 0.22f, light_material);

    return scene;
}

void allocate_scene_on_gpu(
    const Scene& scene,
    HittableData** dev_objects,
    SphereData** dev_spheres,
    TriangleData** dev_triangles,
    RectangleXYData** dev_rect_xy,
    RectangleXZData** dev_rect_xz,
    RectangleYZData** dev_rect_yz,
    MaterialData** dev_materials,
    Camera** dev_camera
) {
    if (!scene.objects.empty()) {
        *dev_objects = cuda_alloc<HittableData>(scene.objects.size());
        cuda_copy_to_device<HittableData>(scene.objects.data(), *dev_objects, scene.objects.size());
    }
    
    if (!scene.spheres.empty()) {
        *dev_spheres = cuda_alloc<SphereData>(scene.spheres.size());
        cuda_copy_to_device<SphereData>(scene.spheres.data(), *dev_spheres, scene.spheres.size());
    }
    
    if (!scene.triangles.empty()) {
        *dev_triangles = cuda_alloc<TriangleData>(scene.triangles.size());
        cuda_copy_to_device<TriangleData>(scene.triangles.data(), *dev_triangles, scene.triangles.size());
    }
    
    if (!scene.rect_xy.empty()) {
        *dev_rect_xy = cuda_alloc<RectangleXYData>(scene.rect_xy.size());
        cuda_copy_to_device<RectangleXYData>(scene.rect_xy.data(), *dev_rect_xy, scene.rect_xy.size());
    }
    
    if (!scene.rect_xz.empty()) {
        *dev_rect_xz = cuda_alloc<RectangleXZData>(scene.rect_xz.size());
        cuda_copy_to_device<RectangleXZData>(scene.rect_xz.data(), *dev_rect_xz, scene.rect_xz.size());
    }
    
    if (!scene.rect_yz.empty()) {
        *dev_rect_yz = cuda_alloc<RectangleYZData>(scene.rect_yz.size());
        cuda_copy_to_device<RectangleYZData>(scene.rect_yz.data(), *dev_rect_yz, scene.rect_yz.size());
    }
    
    if (!scene.materials.empty()) {
        *dev_materials = cuda_alloc<MaterialData>(scene.materials.size());
        cuda_copy_to_device<MaterialData>(scene.materials.data(), *dev_materials, scene.materials.size());
    }
    
    *dev_camera = cuda_alloc<Camera>(1);
    cuda_copy_to_device<Camera>(&scene.camera, *dev_camera, 1);
}

void free_scene_on_gpu(
    HittableData* dev_objects,
    SphereData* dev_spheres,
    TriangleData* dev_triangles,
    RectangleXYData* dev_rect_xy,
    RectangleXZData* dev_rect_xz,
    RectangleYZData* dev_rect_yz,
    MaterialData* dev_materials,
    Camera* dev_camera
) {
    if (dev_objects) cuda_free(dev_objects);
    if (dev_spheres) cuda_free(dev_spheres);
    if (dev_triangles) cuda_free(dev_triangles);
    if (dev_rect_xy) cuda_free(dev_rect_xy);
    if (dev_rect_xz) cuda_free(dev_rect_xz);
    if (dev_rect_yz) cuda_free(dev_rect_yz);
    if (dev_materials) cuda_free(dev_materials);
    if (dev_camera) cuda_free(dev_camera);
}
