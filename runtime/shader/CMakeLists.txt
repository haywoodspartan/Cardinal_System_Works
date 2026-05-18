add_library(cardinal_shader STATIC
    src/shader.cpp
    include/cardinal/shader/shader.hpp)
add_library(cardinal::shader ALIAS cardinal_shader)

target_include_directories(cardinal_shader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_shader
    PUBLIC  cardinal::core cardinal::rhi cardinal::compile_flags)

set_target_properties(cardinal_shader PROPERTIES FOLDER runtime)
