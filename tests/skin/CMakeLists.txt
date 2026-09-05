# Skeletal skinning regression suite. Pins the scene::skin core (influence
# sanitising, skeleton validation, topological global walk, palette =
# identity at bind, LBS kernel on hand-computed rigs) and the render-graph
# SkinPass: CpuBackend record byte-identical to skin_vertices, and — when a
# capable GPU exists — the authored HLSL dispatched through RhiBackend on
# Vulkan + D3D12 vs the CPU reference (eps). SKIPs the GPU half (exit 0)
# without a device so the GPU-less CI stays green.
add_executable(Cardinal_Test_Skin main.cpp)
target_link_libraries(Cardinal_Test_Skin PRIVATE
    cardinal::render
    cardinal::scene
    cardinal::rhi
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Skin PROPERTIES FOLDER tests)

add_test(NAME skin COMMAND Cardinal_Test_Skin)
