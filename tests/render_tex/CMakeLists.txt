# Deterministic texture-math regression suite. render::tex is the
# bookkeeping that drives every RHI texture upload + streaming decision:
# mip-chain counts/dimensions/byte-sizes, block-compressed footprints,
# anisotropy presets, and the VRAM-budget streaming planner. It is
# header-only, constexpr / noexcept-pure, no GPU calls — so the whole
# thing is golden-pinnable. This suite pins the format tables, the
# floor-log2 / mip-count / mip-dim clamps, exact mip-byte + full-chain
# totals (uncompressed AND BC/ASTC, incl. the round-up-to-whole-block
# rule for tiny mips), the aniso/LOD presets, and the streaming
# planner's drop-until-it-fits behaviour incl. the can't-fit and
# zero-texture edge cases. The pure constexpr cases are ALSO static_assert
# compile-time pins, so a drift fails the build itself. Exit 0 = all pass.
add_executable(Cardinal_Test_RenderTex main.cpp)
target_link_libraries(Cardinal_Test_RenderTex PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderTex PROPERTIES FOLDER tests)

add_test(NAME render_tex COMMAND Cardinal_Test_RenderTex)
