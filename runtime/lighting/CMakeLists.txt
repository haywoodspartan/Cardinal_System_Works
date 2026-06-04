# Cardinal lighting module — two complementary lighting paths share a
# scene + BRDF + tracer core:
#   * raytracer — CPU reference path tracer (RTX-style: primary + direct +
#                 multi-bounce indirect with Cook-Torrance BRDF). Reference
#                 for the eventual RT-RHI shader path; deterministic seeded.
#   * bake      — (follow-up commit) lightmap atlas + irradiance probe
#                 volume integrators that consume this tracer.
add_library(cardinal_lighting STATIC
    src/lighting.cpp
    src/raytracer.cpp
    include/cardinal/lighting/lighting.hpp
    include/cardinal/lighting/raytracer.hpp)
add_library(cardinal::lighting ALIAS cardinal_lighting)
target_include_directories(cardinal_lighting PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cardinal_lighting
    PUBLIC  cardinal::core cardinal::scene cardinal::render cardinal::compile_flags)
set_target_properties(cardinal_lighting PROPERTIES FOLDER runtime)
