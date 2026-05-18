# Deterministic math regression suite — the keystone. Every transform,
# camera, physics step, and net-interpolation lerp bottoms out in
# cardinal::core math (aliased as cardinal::scene). A silent regression
# here corrupts EVERYTHING, including the std140/GPU-upload layout
# invariants (Vec3 = 12 B not 16, Mat4 = 64 B column-major) that RHI
# uniform uploads and cross-module type-punning rely on. Pure +
# deterministic + header-only math, so this is fast and rock-solid.
# cardinal::scene brings core transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Math main.cpp)
target_link_libraries(Cardinal_Test_Math PRIVATE
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Math PROPERTIES FOLDER tests)

add_test(NAME math COMMAND Cardinal_Test_Math)
