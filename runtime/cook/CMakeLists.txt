add_library(cardinal_cook STATIC
    src/cook.cpp
    include/cardinal/cook/cook.hpp)
add_library(cardinal::cook ALIAS cardinal_cook)

target_include_directories(cardinal_cook PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# cook needs project (Project layout), shader (compile shaders during cook),
# edit (mesh_ops::make_box for fallback), and asset (codec::encode_*).
target_link_libraries(cardinal_cook
    PUBLIC  cardinal::core cardinal::scene cardinal::project cardinal::shader
            cardinal::edit cardinal::import
            cardinal::compile_flags)

set_target_properties(cardinal_cook PROPERTIES FOLDER runtime)
