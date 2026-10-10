# Cardinal Studio native shell — the editor chrome on Cardinal Slate (cui):
# menu bar + WindowManager desktop + the first real editor panels (Hierarchy,
# Inspector, Console, Log, Stats) bound to a scene::Scene. NO ImGui. GPU-free:
# hosts hand its DrawList to cui_rhi (native) or the ImGui bridge; tests drive
# it headless.
add_library(cardinal_cui_shell STATIC
    src/shell.cpp
    include/cardinal/cui_shell/shell.hpp)
add_library(cardinal::cui_shell ALIAS cardinal_cui_shell)

target_include_directories(cardinal_cui_shell PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cui_shell
    PUBLIC  cardinal::cui cardinal::scene cardinal::input cardinal::console
            cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_cui_shell PROPERTIES FOLDER runtime)
