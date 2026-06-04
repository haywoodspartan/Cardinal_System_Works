add_library(cardinal_postfx STATIC
    src/postfx.cpp
    include/cardinal/postfx/postfx.hpp)
add_library(cardinal::postfx ALIAS cardinal_postfx)

target_include_directories(cardinal_postfx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# postfx reuses render::Knob for editor UI introspection consistency,
# so it depends on render. CPU reference impls only — GPU shader
# variants land when the RHI postprocess pipeline arrives.
target_link_libraries(cardinal_postfx
    PUBLIC  cardinal::core cardinal::render cardinal::compile_flags)

set_target_properties(cardinal_postfx PROPERTIES FOLDER runtime)
