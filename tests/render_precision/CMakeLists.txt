# Deterministic FP8 / FP4 / FP16 packed-math regression suite.
# render::precision is the CPU reference for sub-byte float formats; the
# shader twins in cardinal_packed_math.hlsli MUST agree bit-for-bit, and
# the algo registry's "preview" path quantises through this exact code so
# the editor shows the true rounded value. Every conversion is exact +
# deterministic (RNE on the FP32->packed paths). This suite pins the
# metadata tables, the FP16 bit layout + exact round-trips + overflow,
# the full 16-entry FP4 E2M1 / E3M0 decode tables, the values that
# round-trip identically, the saturation behaviour of the reference
# encoder, FP8 NaN/Inf survival, and the pack/unpack lane layout. It
# pins the encoder's ACTUAL deterministic output (incl. where that
# differs from the header's stated value sets — flagged separately).
# Exit 0 = all pass.
add_executable(Cardinal_Test_RenderPrecision main.cpp)
target_link_libraries(Cardinal_Test_RenderPrecision PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderPrecision PROPERTIES FOLDER tests)

add_test(NAME render_precision COMMAND Cardinal_Test_RenderPrecision)
