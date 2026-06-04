# Deterministic physics regression suite. Physics is gameplay-critical
# and built directly on the now-locked core math; "deterministic given
# the same input" is an explicit design goal that net replication +
# gameplay rely on. A silent regression (flipped gravity sign, broken
# integrator, tunnelling, non-determinism) breaks the game invisibly.
# World::create() is pure CPU — no rhi::Device — so this is headless,
# fast, and exactly reproducible. cardinal::physics brings core
# transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Physics main.cpp)
target_link_libraries(Cardinal_Test_Physics PRIVATE
    cardinal::physics
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Physics PROPERTIES FOLDER tests)

add_test(NAME physics COMMAND Cardinal_Test_Physics)
