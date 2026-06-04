# RTX path-tracer regression suite. Pins the BRDF math (Cook-Torrance:
# Lambertian + GGX + Schlick + Smith), scene intersection (sphere/plane,
# closest-hit + visibility), direct shading vs the host LightSet
# (directional + point + spot + occlusion), recursive bounce throughput
# + Russian-roulette unbiasedness, the Knob editor surface, and the
# NaN-defensive discipline (NaN ray / NaN normal / NaN material → finite
# image). Pure CPU, headless, deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Lighting main.cpp)
target_link_libraries(Cardinal_Test_Lighting PRIVATE
    cardinal::lighting
    cardinal::render
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Lighting PROPERTIES FOLDER tests)

add_test(NAME lighting COMMAND Cardinal_Test_Lighting)
