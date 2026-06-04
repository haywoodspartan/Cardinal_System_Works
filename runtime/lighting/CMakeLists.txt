# Cardinal lighting module — two complementary lighting paths share a
# scene + BRDF + tracer core:
#   * raytracer — CPU reference path tracer (RTX-style: primary + direct +
#                 multi-bounce indirect with Cook-Torrance BRDF). Reference
#                 for the eventual RT-RHI shader path; deterministic seeded.
#   * bake      — lightmap atlas + irradiance probe volume integrators
#                 that consume the path tracer to integrate offline
#                 irradiance, plus runtime samplers (bilinear lightmap,
#                 trilinear ambient-cube probe volume).
add_library(cardinal_lighting STATIC
    src/lighting.cpp
    src/raytracer.cpp
    src/bake.cpp
    src/gpu_lighting.cpp
    src/gpu_radiance_cache.cpp
    include/cardinal/lighting/lighting.hpp
    include/cardinal/lighting/raytracer.hpp
    include/cardinal/lighting/bake.hpp
    include/cardinal/lighting/gpu_lighting.hpp
    include/cardinal/lighting/gpu_radiance_cache.hpp)
add_library(cardinal::lighting ALIAS cardinal_lighting)
target_include_directories(cardinal_lighting PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cardinal_lighting
    PUBLIC  cardinal::core cardinal::scene cardinal::render cardinal::compile_flags)
set_target_properties(cardinal_lighting PROPERTIES FOLDER runtime)
