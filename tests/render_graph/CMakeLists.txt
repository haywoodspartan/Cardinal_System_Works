# Render-graph regression suite.
#
# The graph framework is the seam between the engine's declarative
# expression of per-frame GPU work and the actual executor. The CpuBackend
# is the primary test surrogate for the eventual RHI compute path: same
# Graph + same passes runs on both. This suite pins:
#   * Resource + pass declaration (Buffer / Texture / Compute / Raster / Copy)
#   * Topological sort correctness (RAW + WAW + WAR ordering)
#   * Cycle detection (compile() returns false, no partial order)
#   * Deterministic execution order (stable id-sorted ready queue)
#   * Barrier counters (RAW / WAW / WAR tallies for editor inspection)
#   * CpuBackend dispatch semantics: bytes written/read tracked,
#     map_buffer_* enforces declared access mode, dispatch() routes
#     through ExecutionContext, push-constant key/value scratch works.
#   * End-to-end "GPU-style" pipeline: cull → transform → tonemap chain
#     producing deterministic byte output across runs (the offline
#     reference the real RHI backend will be validated against).
#   * NaN-defensive: zero-sized buffers, bogus handles, missing record
#     callbacks all behave gracefully.
# Pure CPU, headless, deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_RenderGraph main.cpp)
target_link_libraries(Cardinal_Test_RenderGraph PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderGraph PROPERTIES FOLDER tests)
add_test(NAME render_graph COMMAND Cardinal_Test_RenderGraph)
