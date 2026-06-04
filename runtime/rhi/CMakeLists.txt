set(CARDINAL_RHI_HEADERS
    include/cardinal/rhi/rhi.hpp
)

# RHI source layout: per-backend subdirectories so incremental rebuilds
# affect only one backend at a time, and the file tree mirrors the link
# graph. Phase 1 (this layout) keeps each backend as a single large TU;
# phase 2 will split vulkan.cpp / d3d12.cpp into per-class files
# (device / swapchain / buffer / pipeline / texture / accel-structure)
# for finer-grained parallel compile + smarter incremental rebuilds.
set(CARDINAL_RHI_SOURCES
    src/rhi.cpp
    # ---- Vulkan backend (Linux + Windows) -----------------------------
    src/vulkan/vulkan.cpp        # VulkanDevice + Swapchain + Buffer +
                                 # Texture + Pipeline + AccelStructure
    src/vulkan/alloc.cpp         # VMA glue
    src/vulkan/modern.cpp        # NVIDIA SDK shims (DLSS/Reflex/Streamline/RTXMU)
    # ---- D3D12 backend (Windows only) ---------------------------------
    src/d3d12/d3d12.cpp          # D3D12Device + Swapchain + Buffer +
                                 # Texture + Pipeline + AccelStructure
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
