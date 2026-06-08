# Deterministic Cardinal UI (cui / Cardinal Slate) regression suite. Exercises
# geometry math, the two-pass box layout, input routing (button click / checkbox
# toggle / slider drag + hover/active state), and draw-command emission — all on
# the CPU, no GPU. Exit 0 = all pass.
add_executable(Cardinal_Test_Cui main.cpp)
target_link_libraries(Cardinal_Test_Cui PRIVATE
    cardinal::cui
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Cui PROPERTIES FOLDER tests)

add_test(NAME cui COMMAND Cardinal_Test_Cui)
