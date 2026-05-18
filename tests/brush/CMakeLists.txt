# Deterministic sculpt/paint-brush regression suite. edit::brush is the
# tool layer under every terrain sculpt, vertex-paint and voxel-density
# edit: weight_at() is the shared falloff kernel (None / Linear / Smooth
# smoothstep / Gaussian), stamp_height_grid() rasterizes a brush onto a
# world-space height field (Add/Subtract/Set/Smooth/Erase modes + a
# touched-cell AABB for partial GPU re-upload), and stamp_generic()
# drives the same kernel over arbitrary callback-enumerated samples.
# All of it is pure float math with NO hidden state, so the falloff
# curve values, the dt/strength scaling, every mode's exact delta, the
# AABB clamp/empty rules, and the callback skip-when-zero contract are
# all pinned (rational falloffs exact, Gaussian within an epsilon).
# Exit 0 = all pass.
add_executable(Cardinal_Test_Brush main.cpp)
target_link_libraries(Cardinal_Test_Brush PRIVATE
    cardinal::edit
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Brush PROPERTIES FOLDER tests)

add_test(NAME brush COMMAND Cardinal_Test_Brush)
