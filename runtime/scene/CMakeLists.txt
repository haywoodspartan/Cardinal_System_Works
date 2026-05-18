add_library(cardinal_scene STATIC
    src/scene.cpp
    src/renderer.cpp
    src/fly_camera.cpp
    src/terrain.cpp
    src/assets.cpp
    src/light.cpp
    src/scene_vgeom.cpp
    include/cardinal/scene/math.hpp
    include/cardinal/scene/scene.hpp
    include/cardinal/scene/renderer.hpp
    include/cardinal/scene/fly_camera.hpp
    include/cardinal/scene/terrain.hpp
    include/cardinal/scene/assets.hpp
    include/cardinal/scene/light.hpp)
add_library(cardinal::scene ALIAS cardinal_scene)

target_include_directories(cardinal_scene PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_scene
    PUBLIC  cardinal::core cardinal::rhi cardinal::vgeom cardinal::compile_flags)

set_target_properties(cardinal_scene PROPERTIES FOLDER runtime)
