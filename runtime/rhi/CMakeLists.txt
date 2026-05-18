set(CARDINAL_RHI_HEADERS
    include/cardinal/rhi/rhi.hpp
)

set(CARDINAL_RHI_SOURCES
    src/rhi.cpp
    src/rhi_vulkan.cpp
    src/rhi_vulkan_alloc.cpp
    src/rhi_vulkan_modern.cpp
    src/rhi_d3d12.cpp
)

add_library(cardinal_rhi STATIC ${CARDINAL_RHI_SOURCES} ${CARDINAL_RHI_HEADERS})
add_library(cardinal::rhi ALIAS cardinal_rhi)

target_include_directories(cardinal_rhi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Vulkan loader + headers via vcpkg manifest mode.
find_package(VulkanHeaders          CONFIG REQUIRED)
find_package(volk                   CONFIG REQUIRED)
find_package(VulkanMemoryAllocator  CONFIG REQUIRED)
# DXC for runtime HLSL -> SPIR-V (Vulkan) / DXIL (D3D12) compilation.
find_package(directx-dxc            CONFIG REQUIRED)

target_link_libraries(cardinal_rhi
    PUBLIC  cardinal::core cardinal::compile_flags cardinal::tp::defs
    PRIVATE Vulkan::Headers volk::volk
            GPUOpen::VulkanMemoryAllocator
            Microsoft::DirectXShaderCompiler
)

if(WIN32)
    # D3D12 backend — links system D3D12 + DXGI; debug info comes from dxguid.
    target_link_libraries(cardinal_rhi PRIVATE d3d12 dxgi dxguid)
endif()

# Optional NVIDIA SDKs — link only what was detected.
if(TARGET cardinal::tp::streamline)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::streamline)
endif()
if(TARGET cardinal::tp::reflex)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::reflex)
endif()
if(TARGET cardinal::tp::nvapi)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::nvapi)
endif()
if(TARGET cardinal::tp::rtxmu)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::rtxmu)
endif()
if(TARGET cardinal::tp::ntc_sdk)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::ntc_sdk)
endif()
if(TARGET cardinal::tp::nrd)
    target_link_libraries(cardinal_rhi PRIVATE cardinal::tp::nrd)
endif()

set_target_properties(cardinal_rhi PROPERTIES FOLDER runtime)
