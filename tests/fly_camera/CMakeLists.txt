# Deterministic fly-camera regression suite. FlyCamera drives the
# editor/game viewport camera from synthetic input; a regression
# silently breaks navigation — inverted look axis, drift, no pitch
# clamp (gimbal flip), broken sprint/scroll speed, or input leaking
# while a panel is unhovered. The integration is pure CPU and fully
# deterministic given FlyInput + dt. Test vectors use axis-aligned
# cameras and yaw/pitch ∈ {0, ±pi/2, pi, pi/4} so every expected value
# is an exact closed form, a round-trip, or a unit-length invariant —
# no <cmath> in the test (the engine does the trig; we assert results).
# Pins: sync_from direction→yaw/pitch, the accept_input + dt<=0 gates,
# WASDQE move + sprint + diagonal normalisation + opposite-cancel,
# RMB look (yaw unbounded, pitch clamped), scroll speed clamp, and the
# "FlyCamera owns orientation after init (no re-sync)" contract.
# Exit 0 = all pass.
add_executable(Cardinal_Test_FlyCamera main.cpp)
target_link_libraries(Cardinal_Test_FlyCamera PRIVATE
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_FlyCamera PROPERTIES FOLDER tests)

add_test(NAME fly_camera COMMAND Cardinal_Test_FlyCamera)
