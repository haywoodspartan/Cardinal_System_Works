# Deterministic memory-pressure / autoscaler regression suite. Three
# stacked pure pieces:
#
#   memory::classify  — used% → Pressure tier, monotone with INCLUSIVE
#                        (>=) tier boundaries. The whole adaptive-quality
#                        system keys off this; an off-by-epsilon or a
#                        flipped comparison silently mis-tiers globally.
#   budget::Broker     — subsystem registry + per-Domain tier tracking;
#                        callbacks fire ONLY on a tier transition (never
#                        steady-state) and ONLY for the matching Domain.
#                        Driven here through debug_force_pressure so the
#                        OS memory::Monitor (non-deterministic) is fully
#                        out of the picture.
#   budget::AutoScaler — the public recipe: register + apply Low at
#                        construction, pick the per-tier value on each
#                        transition, deregister on destruction.
#
# A regression makes the engine scale to the wrong quality tier (stutter
# or VRAM blow-up) with no crash. Pure CPU, headless, deterministic.
# Exit 0 = all pass.
add_executable(Cardinal_Test_AutoScaler main.cpp)
target_link_libraries(Cardinal_Test_AutoScaler PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AutoScaler PROPERTIES FOLDER tests)

add_test(NAME autoscaler COMMAND Cardinal_Test_AutoScaler)
