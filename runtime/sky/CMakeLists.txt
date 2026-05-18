add_library(cardinal_sky STATIC
    src/sky.cpp
    include/cardinal/sky/sky.hpp)
add_library(cardinal::sky ALIAS cardinal_sky)

target_include_directories(cardinal_sky PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_sky
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_sky PROPERTIES FOLDER runtime)
