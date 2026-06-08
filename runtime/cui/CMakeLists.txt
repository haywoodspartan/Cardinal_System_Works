# Cardinal UI ("cui" / Cardinal Slate) — a standalone retained-mode UI
# framework. LEAF library: depends only on cardinal::core, knows nothing about
# the engine / RHI / Studio, so both the editor and shipped-game UI can use it.
# A renderer backend (RHI / ImGui / a headless test recorder) consumes the
# graphics-agnostic DrawList; the framework itself needs no GPU.
add_library(cardinal_cui STATIC
    src/widget.cpp
    src/widgets.cpp
    src/context.cpp
    include/cardinal/cui/geometry.hpp
    include/cardinal/cui/draw.hpp
    include/cardinal/cui/theme.hpp
    include/cardinal/cui/widget.hpp
    include/cardinal/cui/widgets.hpp
    include/cardinal/cui/context.hpp)
add_library(cardinal::cui ALIAS cardinal_cui)

target_include_directories(cardinal_cui PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cui
    PUBLIC cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_cui PROPERTIES FOLDER runtime)
