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
    # Phase 2 split (per-class TUs for finer-grained incremental rebuilds).
    # internal.hpp hosts the shared includes + ComPtr + inline helpers +
    # the small leaf classes (Buffer / Texture / Pipeline) + the big-class
    # declarations (Swapchain / Device). Each big class's out-of-line
    # methods live in their own TU below.
    src/d3d12/device.cpp         # D3D12Device methods + create_d3d12_device
                                 # + d3d12_interop functions + non-Windows
                                 # fallback (only TU that defines the symbol
                                 # when CARDINAL_PLATFORM_WINDOWS is 0)
    src/d3d12/swapchain.cpp      # D3D12Swapchain core: init / present /
                                 # resize / per-viewport RTT management
    src/d3d12/reflex.cpp         # D3D12Swapchain Reflex (NvAPI) integration
    src/d3d12/commands.cpp       # D3D12Swapchain command-recording surface:
                                 # bind / draw / push / shadow pass + GPU
                                 # debug-marker regions (PIX / RenderDoc)
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
