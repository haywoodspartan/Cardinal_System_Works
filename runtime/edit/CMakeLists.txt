add_library(cardinal_edit STATIC
    src/undo.cpp
    src/brush.cpp
    src/mesh_ops.cpp
    src/tex_ops.cpp
    src/editor_mode.cpp
    include/cardinal/edit/undo.hpp
    include/cardinal/edit/brush.hpp
    include/cardinal/edit/mesh_ops.hpp
    include/cardinal/edit/tex_ops.hpp
    include/cardinal/edit/editor_mode.hpp)
add_library(cardinal::edit ALIAS cardinal_edit)

target_include_directories(cardinal_edit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Mesh ops needs scene::Vertex / scene::Mat4. Scene already depends on
# core/rhi/render — adding scene as a public dep means anything that
# touches edit headers can resolve scene types.
target_link_libraries(cardinal_edit
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_edit PROPERTIES FOLDER runtime)
