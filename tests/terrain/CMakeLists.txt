# Deterministic terrain regression suite. The pure terrain surface backs
# ground height (collision/placement) and the on-disk profile format:
#
#   terrain_color_profile  — closed-form height+slope tint ramp; pinned
#                            EXACTLY (clamp, span guard, cliff blend).
#   terrain_height(_profile)— seeded value-noise fbm; values aren't hand-
#                            predictable but the CONTRACT is: determinism
#                            (replay/stream consistency), height_scale +
#                            base_height linearity, disabled-layer ==
#                            absent, octave/frequency clamp equalities,
#                            and global finiteness (no NaN/Inf blowup).
#   TerrainProfileLibrary  — process-wide registry: set/replace/find/
#                            remove/names + text save↔load round-trip +
#                            register_default_terrain_profiles idempotency.
#
# A regression silently shifts every terrain (wrong ground height → fall
# through world / float above it) or corrupts saved profiles. Mesh-gen
# needs rhi::Device and is out of scope. Pure CPU, headless,
# deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Terrain main.cpp)
target_link_libraries(Cardinal_Test_Terrain PRIVATE
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Terrain PROPERTIES FOLDER tests)

add_test(NAME terrain COMMAND Cardinal_Test_Terrain)
