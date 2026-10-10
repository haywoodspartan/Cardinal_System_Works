# Headless regression suite for the native Studio shell (cui_shell): docking
# layout + viewport rect, hierarchy tree from a scene, inspector two-way sync,
# console execute/complete, log sink, menu dispatch, panel toggles. No GPU.
add_executable(Cardinal_Test_CuiShell main.cpp)
target_link_libraries(Cardinal_Test_CuiShell PRIVATE
    cardinal::cui_shell
    cardinal::console
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_CuiShell PROPERTIES FOLDER tests)

add_test(NAME cui_shell COMMAND Cardinal_Test_CuiShell)
