# Cardinal UI native RHI backend — draws a cui::DrawList through the RHI
# (Vulkan/D3D12) with no ImGui. The analog of imgui_impl_vulkan, for our own
# framework. Depends on cardinal::cui (the framework) + cardinal::rhi (the GPU).
add_library(cardinal_cui_rhi STATIC
    src/renderer.cpp
    src/font8x8.cpp
    include/cardinal/cui_rhi/renderer.hpp
    include/cardinal/cui_rhi/font8x8.hpp)
add_library(cardinal::cui_rhi ALIAS cardinal_cui_rhi)

target_include_directories(cardinal_cui_rhi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cui_rhi
    PUBLIC cardinal::cui cardinal::rhi cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_cui_rhi PROPERTIES FOLDER runtime)
