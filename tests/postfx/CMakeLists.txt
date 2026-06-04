# PostFX chain regression suite. The postfx module is a chain of CPU
# reference impls; the HLSL counterparts will validate byte-for-byte
# against the CPU when they land. This suite pins the chain
# composition (add/remove/move_up/move_down), the ping-pong apply()
# semantics, every stock pass's no-op / disabled / NaN-input behaviour,
# and that the engine's NaN-defensive discipline (fz at every knob
# read) holds: a NaN knob value produces finite pixels. Pure CPU,
# headless, deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_PostFx main.cpp)
target_link_libraries(Cardinal_Test_PostFx PRIVATE
    cardinal::postfx
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_PostFx PROPERTIES FOLDER tests)

add_test(NAME postfx COMMAND Cardinal_Test_PostFx)
