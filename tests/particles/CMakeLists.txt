# Deterministic CPU particle-sim regression suite. Emitter::tick is the
# VFX backbone: a rate-accumulator spawn, semi-implicit-Euler + drag
# integration, age/lifetime cull, and colour-over-life — all driven by
# a seeded splitmix32 RNG. A regression silently breaks every particle
# effect (wrong counts, NaN drift, never-dying particles) or, worse,
# breaks REPLAY determinism. Pure CPU + headless; the System is plain
# std::vector. Exact spawn counts + closed-form integration + two-
# System bit-identical determinism are pinned.
add_executable(Cardinal_Test_Particles main.cpp)
target_link_libraries(Cardinal_Test_Particles PRIVATE
    cardinal::particles
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Particles PROPERTIES FOLDER tests)

add_test(NAME particles COMMAND Cardinal_Test_Particles)
