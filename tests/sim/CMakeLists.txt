# Deterministic simulation-orchestrator regression suite. SimWorld::tick
# is the fixed-timestep heartbeat behind ALL gameplay/replay determinism:
# real_dt clamp, time-scale, the physics accumulator with carry-over and
# a sub-step cap (the spiral-of-death guard), pause / single-step, and
# the strict tick-group order (PreUpdate→PrePhysics→[physics]→
# PostPhysics→Update→LateUpdate→Render, with Render running even when
# paused). A regression silently desyncs every simulation: wrong substep
# count, lost carry remainder, frozen/runaway clock, or out-of-order
# groups. Default parallel_handlers=false ⇒ sequential, no JobSystem
# dependency ⇒ pure + fully deterministic. fixed_dt is chosen as 0.10 so
# the accumulator math is hand-exact. Exit 0 = all pass.
add_executable(Cardinal_Test_Sim main.cpp)
target_link_libraries(Cardinal_Test_Sim PRIVATE
    cardinal::sim
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Sim PROPERTIES FOLDER tests)

add_test(NAME sim COMMAND Cardinal_Test_Sim)
