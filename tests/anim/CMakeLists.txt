# Deterministic keyframe-animation regression suite. Curve<T>::sample
# + Track/Clip/Player are the animation backbone driven by the
# sequencer, curve editor and cinematics. A regression in key sorting,
# segment interpolation (the LEFT key's mode governs the segment),
# boundary clamp, or wrap (Clamp/Loop/PingPong) silently corrupts every
# animated property. Pure CPU + fully deterministic; exact small-curve
# values + the Hermite basis are pinned. cardinal::anim pulls scene
# (Vec3) + core transitively.
add_executable(Cardinal_Test_Anim main.cpp)
target_link_libraries(Cardinal_Test_Anim PRIVATE
    cardinal::anim
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Anim PROPERTIES FOLDER tests)

add_test(NAME anim COMMAND Cardinal_Test_Anim)
