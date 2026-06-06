# cardinal::cmd — unified editor command bus. Headless (no ui/imgui/swapchain
# dependency) so commands are unit-testable without the GUI. Depends on the
# engine subsystems its built-in commands act on (scene math + level
# placement, which transitively bring actor/scene/rhi).
add_library(cardinal_cmd STATIC
    src/command.cpp
    include/cardinal/cmd/command.hpp)
add_library(cardinal::cmd ALIAS cardinal_cmd)

target_include_directories(cardinal_cmd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cmd
    PUBLIC  cardinal::core cardinal::scene cardinal::level cardinal::compile_flags)

set_target_properties(cardinal_cmd PROPERTIES FOLDER runtime)
