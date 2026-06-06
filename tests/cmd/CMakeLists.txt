# cardinal::cmd unified command-bus regression suite. Pins: registry
# add/find/has/dispatch/ids (sorted), unknown-id graceful failure, builtin
# registration, and the placement math (screen_to_ground) — the round-trip
# that proves a pick unprojected with the SAME captured view/proj it was
# rendered with lands on the cursor point, while a MISMATCHED projection
# (the original offset bug) misses. Headless, no GUI/Device. Exit 0 = pass.
add_executable(Cardinal_Test_Cmd main.cpp)
target_link_libraries(Cardinal_Test_Cmd PRIVATE
    cardinal::cmd
    cardinal::scene
    cardinal::actor
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Cmd PROPERTIES FOLDER tests)

add_test(NAME cmd COMMAND Cardinal_Test_Cmd)
