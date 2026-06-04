add_library(cardinal_render STATIC
    src/feature_gate.cpp
    src/pipelines.cpp
    src/geo.cpp
    src/algos.cpp
    src/precision.cpp
    src/graph.cpp
    src/gpu_passes.cpp
    src/gpu_raster.cpp
    src/gpu_geometry.cpp
    include/cardinal/render/pipeline.hpp
    include/cardinal/render/tex.hpp
    include/cardinal/render/tess.hpp
    include/cardinal/render/geo.hpp
    include/cardinal/render/algos.hpp
    include/cardinal/render/precision.hpp
    include/cardinal/render/graph.hpp
    include/cardinal/render/gpu_passes.hpp
    include/cardinal/render/gpu_raster.hpp
    include/cardinal/render/gpu_geometry.hpp)
add_library(cardinal::render ALIAS cardinal_render)

target_include_directories(cardinal_render PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_render
    PUBLIC  cardinal::core cardinal::rhi cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_render PROPERTIES FOLDER runtime)
