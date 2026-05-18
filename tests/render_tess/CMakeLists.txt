# Deterministic tessellation-factor regression suite.
# render::tess is the single canonical hull-shader tess-factor math,
# copied verbatim into shaders/include/cardinal_tessellation.hlsli — the
# CPU LOD picker and the GPU hull shader MUST agree byte-for-byte or
# geometry pops between the two. Every function is a pure inline (header-
# only): the [1,64] hardware clamp, the distance policy (incl. the
# 0.001 distance floor so it can't divide-by-zero/explode), the
# screen-edge policy (incl. the target<1 guard), the curvature blend
# weight (symmetric in |n.n|, clamped), and the Off/Low/Medium/High
# quality scales (incl. the out-of-range fallback) are all exactly
# pinned. Exit 0 = all pass.
add_executable(Cardinal_Test_RenderTess main.cpp)
target_link_libraries(Cardinal_Test_RenderTess PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderTess PROPERTIES FOLDER tests)

add_test(NAME render_tess COMMAND Cardinal_Test_RenderTess)
