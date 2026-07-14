# GPU compute numerical validation — the AEGIS kernels dispatched on a REAL
# device (created windowless), read back, and byte/eps-compared against the
# CpuBackend reference. SKIPS (exit 0, "SKIP" lines) when no capable GPU or
# no async-compute queue exists, so the GPU-less CI stays green; on a dev
# box it validates every authored kernel end-to-end on both backends.
add_executable(Cardinal_Test_GpuCompute main.cpp)
target_link_libraries(Cardinal_Test_GpuCompute PRIVATE
    cardinal::render
    cardinal::rhi
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_GpuCompute PROPERTIES FOLDER tests)

add_test(NAME gpu_compute COMMAND Cardinal_Test_GpuCompute)
