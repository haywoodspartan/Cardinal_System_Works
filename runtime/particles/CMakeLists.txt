add_library(cardinal_particles STATIC
    src/particles.cpp
    include/cardinal/particles/particles.hpp)
add_library(cardinal::particles ALIAS cardinal_particles)

target_include_directories(cardinal_particles PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_particles
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_particles PROPERTIES FOLDER runtime)

# Opt into the engine-wide precompile (cardinal::core::precompile.hpp).
target_precompile_headers(cardinal_particles PRIVATE
    <cardinal/core/precompile.hpp>)
