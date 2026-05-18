# Deterministic CPU procedural-texture regression suite. edit::tex_ops is
# the editor's image toolbox: RGBA8 generators (solid / checker /
# gradient / value+fractal noise / voronoi) and in-place ops
# (grayscale / invert / levels / channel-swap / alpha-compose). All of it
# is fixed-point integer + hash math with NO RNG state, so every byte is
# reproducible: this suite pins exact pixels where the math is closed
# (solid, checker cell pattern, gradient endpoints/midpoints, the
# contrast=0 -> flat-127 identity, BT.601 grayscale weights, invert,
# channel reorder, opaque/transparent compose) and pins the determinism /
# clamp / structural invariants for the hash-noise generators (same args
# -> identical buffer, seed-sensitive, the <=0 / 0-count argument clamps).
# Exit 0 = all pass.
add_executable(Cardinal_Test_TexOps main.cpp)
target_link_libraries(Cardinal_Test_TexOps PRIVATE
    cardinal::edit
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_TexOps PROPERTIES FOLDER tests)

add_test(NAME tex_ops COMMAND Cardinal_Test_TexOps)
