# Deterministic render-algorithm-registry regression suite.
# render::algo::AlgoRegistry is the catalogue every pipeline queries to
# build its Knob enums + wire shader variants; its cpu_fn reference impls
# are the CPU-side contract HLSL must match byte-for-byte. A regression
# silently mis-wires a shader variant, mis-orders a picker, or drifts a
# tonemap/mip/tess curve away from its GPU twin. The registry + the pure
# math reference impls are fully deterministic; this suite pins the
# curated catalogue (counts, insertion order, is_default lookup which is
# NOT always index 0), register dedupe / out-of-range, the picker
# helpers, and the closed-form cpu_fn outputs. Exit 0 = all pass.
add_executable(Cardinal_Test_RenderAlgos main.cpp)
target_link_libraries(Cardinal_Test_RenderAlgos PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderAlgos PROPERTIES FOLDER tests)

add_test(NAME render_algos COMMAND Cardinal_Test_RenderAlgos)
