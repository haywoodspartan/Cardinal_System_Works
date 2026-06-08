#include "vulkan_internal.hpp"
namespace cardinal::rhi {



bool VulkanDevice::initialize(const DeviceDesc& desc) {
    // Modern SDKs first — Streamline wants to be initialised before any
    // Vulkan resources exist (it doesn't intercept in our manual-hooking
    // mode, but it caches global state at init).
    modern::log_linked_sdks();
    const bool sl_ok      = modern::init_streamline();
    const bool reflex_ok  = modern::init_reflex();
    (void)sl_ok;
    (void)reflex_ok;

    if (volkInitialize() != VK_SUCCESS) {
        cardinal::log::errorf("rhi/vk", "vulkan-1 loader not found — install GPU drivers");
        return false;
    }

    u32 loader_version = volkGetInstanceVersion();
    if (loader_version < VK_API_VERSION_1_3) {
        cardinal::log::errorf("rhi/vk",
            "driver only supports Vulkan %u.%u; need 1.3+",
            VK_VERSION_MAJOR(loader_version), VK_VERSION_MINOR(loader_version));
        return false;
    }

    // Instance.
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "Cardinal";
    app.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app.pEngineName        = "Cardinal Engine";
    app.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
    app.apiVersion         = VK_API_VERSION_1_3;

    cardinal::vector<const char*> layers;
    cardinal::vector<const char*> exts = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#if CARDINAL_PLATFORM_WINDOWS
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif CARDINAL_PLATFORM_LINUX
        VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
    };
    if (desc.enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ic{};
    ic.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic.pApplicationInfo        = &app;
    ic.enabledLayerCount       = static_cast<u32>(layers.size());
    ic.ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data();
    ic.enabledExtensionCount   = static_cast<u32>(exts.size());
    ic.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    if (!vk_check(vkCreateInstance(&ic, nullptr, &instance_), "vkCreateInstance")) {
        return false;
    }
    volkLoadInstance(instance_);

    // Physical device pick (discrete preferred + max VRAM).
    u32 phys_count = 0;
    vkEnumeratePhysicalDevices(instance_, &phys_count, nullptr);
    if (phys_count == 0) {
        cardinal::log::errorf("rhi/vk", "no Vulkan-capable GPUs");
        return false;
    }
    cardinal::vector<VkPhysicalDevice> phys(phys_count);
    vkEnumeratePhysicalDevices(instance_, &phys_count, phys.data());

    u64 best_score = 0;
    for (auto p : phys) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(p, &props);

        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(p, &mem);
        u64 vram = 0;
        for (u32 i = 0; i < mem.memoryHeapCount; ++i) {
            if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                vram += mem.memoryHeaps[i].size;
            }
        }
        u64 score = vram / (1024 * 1024);
        if (desc.prefer_discrete_gpu &&
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1'000'000;
        }
        if (score > best_score) {
            best_score = score;
            physical_  = p;
            cardinal::strncpy(adapter_name_, props.deviceName, sizeof(adapter_name_) - 1);
        }
    }
    if (physical_ == VK_NULL_HANDLE) return false;

    // Pick a graphics queue family. (We don't yet check surface support here —
    // Win32/X11 surfaces always work on the graphics queue in practice; we'll
    // verify on swapchain creation and bail if not.)
    u32 qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qf_count, nullptr);
    cardinal::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qf_count, qfs.data());
    // First graphics family + a DEDICATED compute-only family (COMPUTE without
    // GRAPHICS) — the latter is the async-compute lane that overlaps the
    // graphics queue on real hardware (AEGIS Block 10). A shared graphics+
    // compute family is NOT used for async (it time-slices, not overlaps), so
    // when no dedicated family exists async compute stays disabled.
    for (u32 i = 0; i < qf_count; ++i) {
        const bool gfx = (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool cmp = (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  != 0;
        if (gfx && graphics_queue_family_ == static_cast<u32>(-1))
            graphics_queue_family_ = i;
        if (cmp && !gfx && compute_queue_family_ == static_cast<u32>(-1))
            compute_queue_family_ = i;
    }
    if (graphics_queue_family_ == static_cast<u32>(-1)) {
        cardinal::log::errorf("rhi/vk", "no graphics queue family");
        return false;
    }

    // Identity for the capabilities struct.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_, &props);
        const char* vendor = "Unknown";
        switch (props.vendorID) {
            case 0x10DE: vendor = "NVIDIA Corporation"; break;
            case 0x1002: vendor = "AMD";                break;
            case 0x8086: vendor = "Intel";              break;
            default:                                    break;
        }
        cardinal::strncpy(caps_.vendor_name, vendor, sizeof(caps_.vendor_name) - 1);
        // Best-effort architecture string from device name.
        const char* arch = "Unknown";
        if (props.vendorID == 0x10DE) {
            // RTX 40 = Ada Lovelace, RTX 30 = Ampere, RTX 20 = Turing.
            if (cardinal::strstr(props.deviceName, "RTX 40") || cardinal::strstr(props.deviceName, "RTX 50"))
                arch = "Ada Lovelace+";
            else if (cardinal::strstr(props.deviceName, "RTX 30")) arch = "Ampere";
            else if (cardinal::strstr(props.deviceName, "RTX 20") || cardinal::strstr(props.deviceName, "GTX 16"))
                arch = "Turing";
        } else if (props.vendorID == 0x1002) {
            if (cardinal::strstr(props.deviceName, "RX 7"))      arch = "RDNA 3";
            else if (cardinal::strstr(props.deviceName, "RX 6")) arch = "RDNA 2";
            else if (cardinal::strstr(props.deviceName, "RX 5")) arch = "RDNA";
        }
        cardinal::strncpy(caps_.gpu_arch, arch, sizeof(caps_.gpu_arch) - 1);

        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
        for (u32 i = 0; i < mem.memoryHeapCount; ++i) {
            if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                caps_.vram_bytes += mem.memoryHeaps[i].size;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Query supported features via the pNext chain. We ask for everything we
    // know how to use; whatever the driver reports back is what we'll
    // actually enable below.
    // ------------------------------------------------------------------------
    VkPhysicalDeviceVulkan11Features         q11{};  q11.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features         q12{};  q12.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features         q13{};  q13.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceMeshShaderFeaturesEXT    qmesh{}; qmesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR qas{}; qas.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR    qrtp{}; qrtp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR              qrq{};  qrq.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR  qvrs{}; qvrs.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

    // Chain: f2 -> 11 -> 12 -> 13 -> mesh -> as -> rtp -> rq -> vrs
    qrq.pNext   = &qvrs;
    qrtp.pNext  = &qrq;
    qas.pNext   = &qrtp;
    qmesh.pNext = &qas;
    q13.pNext   = &qmesh;
    q12.pNext   = &q13;
    q11.pNext   = &q12;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &q11;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);

    // ------------------------------------------------------------------------
    // Enumerate device extensions to gate optional ones.
    // ------------------------------------------------------------------------
    u32 ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, nullptr);
    cardinal::vector<VkExtensionProperties> avail_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, avail_exts.data());

    auto has_ext = [&](const char* name) {
        for (const auto& e : avail_exts) {
            if (cardinal::strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    // Required.
    cardinal::vector<const char*> dev_exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    const bool ext_dyn_rendering = has_ext(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    if (ext_dyn_rendering) dev_exts.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

    const bool ext_mesh = has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME) && qmesh.meshShader;
    if (ext_mesh) dev_exts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    const bool ext_def_host = has_ext(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool ext_as       = has_ext(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                              qas.accelerationStructure && ext_def_host;
    const bool ext_rtp      = has_ext(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                              qrtp.rayTracingPipeline && ext_as;
    const bool ext_rq       = has_ext(VK_KHR_RAY_QUERY_EXTENSION_NAME) && qrq.rayQuery && ext_as;

    if (ext_def_host) dev_exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    if (ext_as)       dev_exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    if (ext_rtp)      dev_exts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    if (ext_rq)       dev_exts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);

    // Variable Rate Shading (Block 12). Enable only when the per-pipeline rate
    // feature is actually reported, so vkCreateDevice can't fail on it.
    const bool ext_vrs = has_ext(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) &&
                         qvrs.pipelineFragmentShadingRate;
    if (ext_vrs) dev_exts.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);

    // VK_EXT_memory_budget: enables vkGetPhysicalDeviceMemoryProperties2 with a
    // VkPhysicalDeviceMemoryBudgetPropertiesEXT pNext, which gives us per-heap
    // budget + usage. Universally supported on Vulkan 1.1+ desktop drivers.
    const bool ext_mem_budget = has_ext(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (ext_mem_budget) dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    has_memory_budget_ = ext_mem_budget;

    // VK_NV_low_latency2: prerequisite for NvLL_VK Reflex calls
    // (NvLL_VK_InitLowLatencyDevice, NvLL_VK_LatencySleep,
    // NvLL_VK_SetLatencyMarker). Only available on NVIDIA drivers.
    // Probed unconditionally — falls through silently on AMD / Intel.
    const bool ext_nv_low_latency2 = has_ext("VK_NV_low_latency2");
    if (ext_nv_low_latency2) dev_exts.push_back("VK_NV_low_latency2");
    has_nv_low_latency2_ = ext_nv_low_latency2;

    // ------------------------------------------------------------------------
    // Build the enable chain — only set bits the driver reported as TRUE.
    // ------------------------------------------------------------------------
    VkPhysicalDeviceVulkan11Features e11{};  e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features e12{};  e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features e13{};  e13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceMeshShaderFeaturesEXT emesh{}; emesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR eas{};  eas.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR    ertp{}; ertp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR              erq{};  erq.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR  evrs{}; evrs.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

    // 1.3 core
    e13.dynamicRendering = q13.dynamicRendering;
    e13.synchronization2 = q13.synchronization2;
    e13.maintenance4     = q13.maintenance4;
    e13.subgroupSizeControl = q13.subgroupSizeControl;
    // 1.2 — bindless, BDA, scalar layout, FP16/INT8, descriptor indexing, timeline
    e12.bufferDeviceAddress                      = q12.bufferDeviceAddress;
    e12.descriptorIndexing                       = q12.descriptorIndexing;
    e12.runtimeDescriptorArray                   = q12.runtimeDescriptorArray;
    e12.descriptorBindingPartiallyBound          = q12.descriptorBindingPartiallyBound;
    e12.descriptorBindingVariableDescriptorCount = q12.descriptorBindingVariableDescriptorCount;
    e12.descriptorBindingSampledImageUpdateAfterBind  = q12.descriptorBindingSampledImageUpdateAfterBind;
    e12.descriptorBindingStorageBufferUpdateAfterBind = q12.descriptorBindingStorageBufferUpdateAfterBind;
    e12.shaderSampledImageArrayNonUniformIndexing     = q12.shaderSampledImageArrayNonUniformIndexing;
    e12.scalarBlockLayout                        = q12.scalarBlockLayout;
    e12.shaderFloat16                            = q12.shaderFloat16;
    e12.shaderInt8                               = q12.shaderInt8;
    e12.timelineSemaphore                        = q12.timelineSemaphore;
    e12.storageBuffer8BitAccess                  = q12.storageBuffer8BitAccess;
    e12.uniformAndStorageBuffer8BitAccess        = q12.uniformAndStorageBuffer8BitAccess;
    // 1.1 — wave/subgroup ops
    e11.storageBuffer16BitAccess                 = q11.storageBuffer16BitAccess;
    e11.uniformAndStorageBuffer16BitAccess       = q11.uniformAndStorageBuffer16BitAccess;
    e11.shaderDrawParameters                     = q11.shaderDrawParameters;

    // Mesh shaders
    if (ext_mesh) {
        emesh.meshShader = qmesh.meshShader;
        emesh.taskShader = qmesh.taskShader;
    }
    // Ray tracing
    if (ext_as)  eas.accelerationStructure  = qas.accelerationStructure;
    if (ext_rtp) ertp.rayTracingPipeline    = qrtp.rayTracingPipeline;
    if (ext_rq)  erq.rayQuery               = qrq.rayQuery;
    if (ext_vrs) {
        evrs.pipelineFragmentShadingRate   = VK_TRUE;
        evrs.attachmentFragmentShadingRate = qvrs.attachmentFragmentShadingRate;
    }

    // Build the chain — only attach what we're actually using.
    void* chain_head = &e13;
    e13.pNext = &e12;
    e12.pNext = &e11;
    void** tail_pnext = &e11.pNext;
    if (ext_mesh) { *tail_pnext = &emesh; tail_pnext = &emesh.pNext; }
    if (ext_as)   { *tail_pnext = &eas;   tail_pnext = &eas.pNext;   }
    if (ext_rtp)  { *tail_pnext = &ertp;  tail_pnext = &ertp.pNext;  }
    if (ext_rq)   { *tail_pnext = &erq;   tail_pnext = &erq.pNext;   }
    if (ext_vrs)  { *tail_pnext = &evrs;  tail_pnext = &evrs.pNext;  }

    // ------------------------------------------------------------------------
    // Logical device.
    // ------------------------------------------------------------------------
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qcs[2]{};
    qcs[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcs[0].queueFamilyIndex = graphics_queue_family_;
    qcs[0].queueCount       = 1;
    qcs[0].pQueuePriorities = &prio;
    u32 qc_count = 1;
    // Request the dedicated async-compute queue if a compute-only family exists.
    if (compute_queue_family_ != static_cast<u32>(-1) &&
        compute_queue_family_ != graphics_queue_family_) {
        qcs[1].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcs[1].queueFamilyIndex = compute_queue_family_;
        qcs[1].queueCount       = 1;
        qcs[1].pQueuePriorities = &prio;
        qc_count = 2;
    } else {
        compute_queue_family_ = static_cast<u32>(-1);   // no dedicated lane
    }

    VkDeviceCreateInfo dc{};
    dc.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.pNext                   = chain_head;
    dc.queueCreateInfoCount    = qc_count;
    dc.pQueueCreateInfos       = qcs;
    dc.enabledExtensionCount   = static_cast<u32>(dev_exts.size());
    dc.ppEnabledExtensionNames = dev_exts.data();

    if (!vk_check(vkCreateDevice(physical_, &dc, nullptr, &device_), "vkCreateDevice")) {
        return false;
    }
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
    if (compute_queue_family_ != static_cast<u32>(-1))
        vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    // ------------------------------------------------------------------------
    // Populate the public-facing capability struct.
    // ------------------------------------------------------------------------
    caps_.dynamic_rendering        = q13.dynamicRendering;
    caps_.synchronization2         = q13.synchronization2;
    caps_.buffer_device_address    = q12.bufferDeviceAddress;
    caps_.descriptor_indexing      = q12.descriptorIndexing;
    caps_.runtime_descriptor_array = q12.runtimeDescriptorArray;
    caps_.scalar_block_layout      = q12.scalarBlockLayout;
    caps_.timeline_semaphore       = q12.timelineSemaphore;
    // Async compute needs both the dedicated queue AND timeline semaphores
    // (the Fence + cross-queue handshake are timeline-based).
    caps_.async_compute            = (compute_queue_ != VK_NULL_HANDLE) && q12.timelineSemaphore;
    caps_.max_async_compute_queues = caps_.async_compute ? 1u : 0u;
    caps_.shader_float16           = q12.shaderFloat16;
    caps_.shader_int8              = q12.shaderInt8;
    caps_.shader_int16             = q11.storageBuffer16BitAccess;  // proxy
    caps_.storage_buffer_16bit     = q11.storageBuffer16BitAccess;
    caps_.wave_intrinsics          = true;  // Vulkan 1.1+ has subgroup ops baseline
    caps_.mesh_shader              = ext_mesh && qmesh.meshShader;
    caps_.task_shader              = ext_mesh && qmesh.taskShader;
    caps_.acceleration_structure   = ext_as  && qas.accelerationStructure;
    caps_.ray_tracing_pipeline     = ext_rtp && qrtp.rayTracingPipeline;
    caps_.ray_query                = ext_rq  && qrq.rayQuery;

    // ---- AEGIS 2.0 GPU-feature caps (Blocks 4 / 8 / 12) ----
    caps_.indirect_dispatch        = true;                  // vkCmdDispatchIndirect is core
    caps_.indirect_draw_count      = q12.drawIndirectCount;
    caps_.bindless_resources       = q12.descriptorIndexing && q12.runtimeDescriptorArray;
    caps_.variable_rate_shading    = ext_vrs;                       // ext enabled + pipeline rate
    caps_.variable_rate_image      = ext_vrs && qvrs.attachmentFragmentShadingRate;
    // FP8 math: probe VK_KHR_shader_float8 by NAME (the vendored headers may not
    // define the macro yet, so has_ext takes the literal string). fp4_math has
    // no standard Vulkan ext; tensor_cores + direct_storage_capable are
    // D3D12-native paths here — all left false honestly.
    caps_.fp8_math                 = has_ext("VK_KHR_shader_float8");

    // Vendor-specific paths.
    const bool is_nvidia = cardinal::strcmp(caps_.vendor_name, "NVIDIA Corporation") == 0;
    const bool is_amd    = cardinal::strcmp(caps_.vendor_name, "AMD") == 0;
    const bool is_ada    = cardinal::strstr(caps_.gpu_arch, "Ada") != nullptr;
    caps_.nvidia_dlss_capable     = is_nvidia && caps_.acceleration_structure;
    caps_.nvidia_framegen_capable = caps_.nvidia_dlss_capable && is_ada;
    caps_.amd_fsr3_capable        = is_amd;

    if (!init_vma()) return false;

    // Apply initial settings (clamps to capabilities).
    apply_settings(desc.initial_settings);

    return true;
}

bool VulkanDevice::init_vma() {
    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo aci{};
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.physicalDevice   = physical_;
    aci.device           = device_;
    aci.instance         = instance_;
    aci.pVulkanFunctions = &fns;
    // Required so vmaCreateBuffer honors VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    // which we need for ray tracing inputs and acceleration-structure scratch
    // buffers.
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS || allocator_ == VK_NULL_HANDLE) {
        cardinal::log::errorf("rhi/vk", "vmaCreateAllocator failed");
        return false;
    }

    // Shared sampler for sampled-texture bindings. Clamp-to-edge so a
    // shadow lookup just outside the map reads the edge texel rather
    // than wrapping; linear for cheap 2x2 PCF-ish softening.
    VkSamplerCreateInfo sci{};
    sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter     = VK_FILTER_LINEAR;
    sci.minFilter     = VK_FILTER_LINEAR;
    sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod        = 1.0f;
    if (vkCreateSampler(device_, &sci, nullptr, &default_sampler_) != VK_SUCCESS) {
        cardinal::log::errorf("rhi/vk", "vkCreateSampler (default) failed");
        return false;
    }
    return true;
}

// =============================================================================
// DXC runtime shader compilation
//
// Single source-of-truth: HLSL with SM 6.x. DXC outputs SPIR-V here for
// Vulkan; the same source compiles to DXIL for the (future) D3D12 backend
// by dropping the `-spirv` flag. This is the "runtime selectable" pivot —
// shaders aren't baked at build time, they're text the engine compiles
// when it needs them (enabling hot reload, runtime permutations, etc.).
// =============================================================================
bool VulkanDevice::init_dxc() {
    if (dxc_compiler_ != nullptr) return true;
    HRESULT hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler_));
    if (FAILED(hr) || dxc_compiler_ == nullptr) {
        cardinal::log::errorf("rhi/vk", "DxcCreateInstance(compiler) failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        return false;
    }
    hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils_));
    if (FAILED(hr) || dxc_utils_ == nullptr) {
        cardinal::log::errorf("rhi/vk", "DxcCreateInstance(utils) failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        return false;
    }
    return true;
}

ShaderBlob VulkanDevice::compile_shader(
    ShaderStage stage, const char* hlsl_source, const char* entry_point)
{
    if (!init_dxc()) return {};
    if (hlsl_source == nullptr || entry_point == nullptr) return {};

    LPCWSTR profile = nullptr;
    switch (stage) {
        case ShaderStage::Vertex:   profile = L"vs_6_6"; break;
        case ShaderStage::Fragment: profile = L"ps_6_6"; break;
        case ShaderStage::Compute:  profile = L"cs_6_6"; break;
    }
    if (profile == nullptr) return {};

    wchar_t entry_w[128]{};
    MultiByteToWideChar(CP_UTF8, 0, entry_point, -1, entry_w, 128);

    DxcBuffer src{};
    src.Ptr      = hlsl_source;
    src.Size     = cardinal::strlen(hlsl_source);
    src.Encoding = DXC_CP_UTF8;

    // Args: SM 6.6 baseline so HLSL 2021 + new wave/inline RT ops are valid.
    // 16-bit native types only enabled when both the device supports FP16
    // *and* the runtime setting requests it — that pair is what drives RPM.
    cardinal::vector<LPCWSTR> args = {
        L"-T", profile,
        L"-E", entry_w,
        L"-spirv",                       // SPIR-V output (Vulkan)
        L"-fvk-use-dx-layout",           // HLSL-style cbuffer layout
        L"-fspv-target-env=vulkan1.3",   // emit Vulkan 1.3 SPIR-V
        L"-HV", L"2021",                 // HLSL 2021
        L"-O3",                          // optimize
        // Matrix packing MUST match the D3D12 backend: the renderer
        // pushes the SAME scene::Mat4 bytes to both, and the shared
        // HLSL does mul(pc.mvp, pos). D3D12 compiles with DXC's default
        // (column-major); the old -Zpr (row-major) here transposed
        // every transform on Vulkan only, so placed geometry projected
        // to garbage clip space and never rendered as 3D. Pin
        // column-major explicitly so the two backends can't diverge.
        L"-Zpc",                         // column-major (match D3D12)
    };
    if (caps_.shader_float16 && settings_.prefer_fp16) {
        args.push_back(L"-enable-16bit-types");
    }

    IDxcResult* result = nullptr;
    HRESULT hr = dxc_compiler_->Compile(
        &src, args.data(), static_cast<u32>(args.size()),
        nullptr, IID_PPV_ARGS(&result));
    if (FAILED(hr) || result == nullptr) {
        cardinal::log::errorf("rhi/vk", "DXC Compile call failed (0x%08lX)",
                              static_cast<unsigned long>(hr));
        if (result != nullptr) result->Release();
        return {};
    }

    HRESULT status = E_FAIL;
    result->GetStatus(&status);
    if (FAILED(status)) {
        IDxcBlobUtf8* errs = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
        if (errs != nullptr && errs->GetStringLength() > 0) {
            cardinal::log::errorf("rhi/vk", "HLSL compile error:\n%s",
                                  errs->GetStringPointer());
        } else {
            cardinal::log::errorf("rhi/vk", "HLSL compile failed with no diagnostics");
        }
        if (errs != nullptr)   errs->Release();
        result->Release();
        return {};
    }

    IDxcBlob* obj = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&obj), nullptr);
    if (obj == nullptr || obj->GetBufferSize() == 0) {
        cardinal::log::errorf("rhi/vk", "DXC produced empty object");
        if (obj != nullptr) obj->Release();
        result->Release();
        return {};
    }

    ShaderBlob blob;
    auto* p = static_cast<const u8*>(obj->GetBufferPointer());
    blob.bytes.assign(p, p + obj->GetBufferSize());

    obj->Release();
    result->Release();
    return blob;
}

// Live VRAM telemetry — VK_EXT_memory_budget gives us per-heap budget +
// usage; we sum the device-local heaps (== "VRAM" on a discrete GPU; on
// integrated parts this is the shared heap, which is also the right
// answer). Fallback when the extension isn't present: return budget=total,
// usage=0 so the broker treats us as Low pressure rather than spamming.
Device::VramSnapshot VulkanDevice::query_vram_usage() const noexcept {
    VramSnapshot s{};
    if (physical_ == VK_NULL_HANDLE) return s;

    if (has_memory_budget_) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
        bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

        VkPhysicalDeviceMemoryProperties2 mp2{};
        mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp2.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(physical_, &mp2);

        for (u32 i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
            if ((mp2.memoryProperties.memoryHeaps[i].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                s.budget_bytes        += static_cast<u64>(bp.heapBudget[i]);
                s.current_usage_bytes += static_cast<u64>(bp.heapUsage[i]);
            }
        }
    } else {
        // Best effort: total VRAM as budget, no usage info.
        s.budget_bytes = caps_.vram_bytes;
    }
    return s;
}


cardinal::unique_ptr<Buffer> VulkanDevice::create_buffer(const BufferDesc& desc) {
    auto b = cardinal::make_unique<VulkanBuffer>(*this);
    if (!b->initialize(desc)) return nullptr;
    return b;
}

// Transient single-submit command buffer for one-off GPU work (initial
// image layout transitions, acceleration-structure builds). submit_and_wait
// stalls the queue — only for setup-time, never per-frame.


cardinal::unique_ptr<Texture> VulkanDevice::create_texture(const TextureDesc& desc) {
    auto t = cardinal::make_unique<VulkanTexture>(*this);
    if (!t->initialize(desc)) return nullptr;
    return t;
}


namespace {

VkShaderModule make_shader_module(VkDevice dev, const ShaderBlob& spirv) {
    if (!spirv.ok() || (spirv.bytes.size() % 4) != 0) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.bytes.size();
    ci.pCode    = reinterpret_cast<const u32*>(spirv.bytes.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, nullptr, &mod);
    return mod;
}

}  // namespace

bool VulkanPipeline::initialize(const PipelineDesc& desc) {
    vs_module_ = make_shader_module(dev_.device_, desc.vertex_shader);
    fs_module_ = make_shader_module(dev_.device_, desc.fragment_shader);
    if (vs_module_ == VK_NULL_HANDLE || fs_module_ == VK_NULL_HANDLE) {
        cardinal::log::errorf("rhi/vk", "vkCreateShaderModule failed");
        return false;
    }

    push_constant_size_    = desc.push_constant_size;
    storage_buffer_slots_  = desc.storage_buffer_slots;
    sampled_texture_slots_ = desc.sampled_texture_slots;

    // Descriptor-set 0, visible to all graphics stages:
    //   bindings [0, storage_buffer_slots_)            STORAGE_BUFFER
    //   bindings [storage_buffer_slots_, +sampled_n)   COMBINED_IMAGE_SAMPLER
    // Only built when the pipeline declares at least one of either, so
    // push-constant-only pipelines keep an empty (set-less) layout.
    cardinal::vector<VkDescriptorSetLayoutBinding> dsl_bindings;
    if (storage_buffer_slots_ > 0 || sampled_texture_slots_ > 0) {
        for (u32 s = 0; s < storage_buffer_slots_; ++s) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = s;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT
                              | VK_SHADER_STAGE_FRAGMENT_BIT;
            dsl_bindings.push_back(b);
        }
        for (u32 s = 0; s < sampled_texture_slots_; ++s) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = storage_buffer_slots_ + s;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            dsl_bindings.push_back(b);
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<u32>(dsl_bindings.size());
        dlci.pBindings    = dsl_bindings.data();
        if (!vk_check(
                vkCreateDescriptorSetLayout(dev_.device_, &dlci, nullptr, &dsl_),
                "vkCreateDescriptorSetLayout")) {
            return false;
        }
    }

    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (dsl_ != VK_NULL_HANDLE) {
        lci.setLayoutCount = 1;
        lci.pSetLayouts    = &dsl_;
    }
    VkPushConstantRange pc_range{};
    if (push_constant_size_ > 0) {
        // Visible to both stages so the renderer can put MVP in VS and
        // per-draw scalars in FS without two separate ranges.
        //
        // Vulkan only *guarantees* maxPushConstantsSize >= 128 B; real
        // devices vary (commonly 128 or 256). The scene block is 240 B,
        // so on a 128-byte device vkCreatePipelineLayout would fail
        // opaquely and no scene PSO would come up — the exact twin of
        // the D3D12 root-signature overflow. Fail loudly + cleanly with
        // an actionable message instead (the higher layers already
        // degrade gracefully on a null pipeline). The portable fix —
        // demote large blocks to a per-frame UBO, mirroring the D3D12
        // backend's root-CBV fallback — is the follow-up; the common
        // NVIDIA/AMD desktop limit of 256 B covers the 240 B block.
        VkPhysicalDeviceProperties pdp{};
        vkGetPhysicalDeviceProperties(dev_.vk_physical(), &pdp);
        if (push_constant_size_ > pdp.limits.maxPushConstantsSize) {
            cardinal::log::errorf("rhi/vk",
                "push constant block %u B exceeds device maxPushConstantsSize "
                "%u B — pipeline not created (needs a UBO fallback)",
                push_constant_size_, pdp.limits.maxPushConstantsSize);
            return false;
        }
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
                            | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = push_constant_size_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges    = &pc_range;
    }
    if (!vk_check(
            vkCreatePipelineLayout(dev_.device_, &lci, nullptr, &layout_),
            "vkCreatePipelineLayout")) {
        return false;
    }

    cardinal::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_module_;
    stages[0].pName  = desc.vertex_entry;
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_module_;
    stages[1].pName  = desc.fragment_entry;

    // Vertex input.
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = desc.vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    cardinal::vector<VkVertexInputAttributeDescription> attribs;
    attribs.reserve(desc.vertex_attribs.size());
    for (const auto& a : desc.vertex_attribs) {
        VkVertexInputAttributeDescription ad{};
        ad.location = a.location;
        ad.binding  = 0;
        ad.format   = to_vk_format(a.format);
        ad.offset   = a.offset;
        attribs.push_back(ad);
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (desc.vertex_stride > 0 && !attribs.empty()) {
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &binding;
        vi.vertexAttributeDescriptionCount = static_cast<u32>(attribs.size());
        vi.pVertexAttributeDescriptions    = attribs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = to_vk_topology(desc.topology);

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // color_format == Unknown ⇒ depth-only pipeline (0 colour
    // attachments) — the shadow-map depth pass. Drives both the blend
    // state below and the dynamic-rendering formats further down.
    const bool depth_only = (desc.color_format == Format::Unknown);

    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = depth_only ? 0u : 1u;
    cb.pAttachments    = depth_only ? nullptr : &att;

    cardinal::array<VkDynamicState, 2> dyn_states = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<u32>(dyn_states.size());
    dyn.pDynamicStates    = dyn_states.data();

    VkFormat color_fmt = to_vk_format(desc.color_format);
    // Depth-attachment format declaration. When Unknown, prc reports
    // VK_FORMAT_UNDEFINED and the pipeline expects no depth attachment
    // in the rendering pass. When set (currently D32_SFLOAT for the
    // viewport RTTs), the pipeline must be bound in a context that
    // attaches a matching-format depth image.
    VkFormat depth_fmt = (desc.depth_format == Format::Unknown)
        ? VK_FORMAT_UNDEFINED
        : to_vk_format(desc.depth_format);

    VkPipelineRenderingCreateInfo prc{};
    prc.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prc.colorAttachmentCount    = depth_only ? 0u : 1u;
    prc.pColorAttachmentFormats = depth_only ? nullptr : &color_fmt;
    prc.depthAttachmentFormat   = depth_fmt;

    // Depth-stencil state. Only attach when a depth format is declared;
    // otherwise omit the struct entirely so Vulkan doesn't validate
    // depth state against an absent attachment.
    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dss.depthTestEnable       = (desc.depth_format != Format::Unknown && desc.depth_test) ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable      = (desc.depth_format != Format::Unknown && desc.depth_write) ? VK_TRUE : VK_FALSE;
    // LESS — entries with smaller depth pass. Matches the scene's
    // 1.0 = far / 0.0 = near convention (see CLEAR depth above).
    dss.depthCompareOp        = VK_COMPARE_OP_LESS;
    dss.depthBoundsTestEnable = VK_FALSE;
    dss.stencilTestEnable     = VK_FALSE;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.pNext               = &prc;
    pci.stageCount          = static_cast<u32>(stages.size());
    pci.pStages             = stages.data();
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState   = &ms;
    pci.pColorBlendState    = &cb;
    pci.pDynamicState       = &dyn;
    pci.pDepthStencilState  = (desc.depth_format != Format::Unknown) ? &dss : nullptr;
    pci.layout              = layout_;
    pci.subpass             = 0;

    return vk_check(
        vkCreateGraphicsPipelines(
            dev_.device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_),
        "vkCreateGraphicsPipelines");
}

cardinal::unique_ptr<Pipeline> VulkanDevice::create_pipeline(const PipelineDesc& desc) {
    auto p = cardinal::make_unique<VulkanPipeline>(*this);
    if (!p->initialize(desc)) return nullptr;
    return p;
}

// Compute pipeline (AEGIS graph::RhiBackend). One descriptor set: STORAGE_BUFFER
// bindings [0,storage) read-only + [storage, storage+uav) read-write (Vulkan has
// no distinct UAV type), stage = COMPUTE; push range = COMPUTE. Empty blob =>
// false => create_compute_pipeline returns null (graph telemetry fallback).
bool VulkanPipeline::initialize_compute(const ComputePipelineDesc& desc) {
    is_compute_ = true;
    cs_module_  = make_shader_module(dev_.device_, desc.compute_shader);
    if (cs_module_ == VK_NULL_HANDLE) {
        cardinal::log::errorf("rhi/vk", "compute: empty/invalid shader module");
        return false;
    }
    push_constant_size_   = desc.push_constant_size;
    storage_buffer_slots_ = desc.storage_buffer_slots;
    uav_slots_            = desc.uav_slots;

    const u32 total_ssbo = storage_buffer_slots_ + uav_slots_;
    if (total_ssbo > 0) {
        cardinal::vector<VkDescriptorSetLayoutBinding> b;
        for (u32 s = 0; s < total_ssbo; ++s) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding         = s;                         // [0,storage)=RO, [storage,+uav)=RW
            lb.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb.descriptorCount = 1;
            lb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b.push_back(lb);
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<u32>(b.size());
        dlci.pBindings    = b.data();
        if (!vk_check(vkCreateDescriptorSetLayout(dev_.device_, &dlci, nullptr, &dsl_),
                      "vkCreateDescriptorSetLayout(compute)")) {
            return false;
        }
    }

    VkPushConstantRange pc{};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (dsl_ != VK_NULL_HANDLE) { lci.setLayoutCount = 1; lci.pSetLayouts = &dsl_; }
    if (push_constant_size_ > 0) {
        VkPhysicalDeviceProperties pdp{};
        vkGetPhysicalDeviceProperties(dev_.vk_physical(), &pdp);
        if (push_constant_size_ > pdp.limits.maxPushConstantsSize) {
            cardinal::log::errorf("rhi/vk",
                "compute push block %u B exceeds device maxPushConstantsSize %u",
                push_constant_size_, pdp.limits.maxPushConstantsSize);
            return false;
        }
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = push_constant_size_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges    = &pc;
    }
    if (!vk_check(vkCreatePipelineLayout(dev_.device_, &lci, nullptr, &layout_),
                  "vkCreatePipelineLayout(compute)")) {
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs_module_;
    stage.pName  = desc.compute_entry ? desc.compute_entry : "CSMain";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = layout_;
    return vk_check(vkCreateComputePipelines(dev_.device_, VK_NULL_HANDLE, 1, &cpci,
                                             nullptr, &pipeline_),
                    "vkCreateComputePipelines");
}

cardinal::unique_ptr<Pipeline> VulkanDevice::create_compute_pipeline(const ComputePipelineDesc& desc) {
    if (!desc.compute_shader.ok()) return nullptr;   // empty blob -> graph fallback
    auto p = cardinal::make_unique<VulkanPipeline>(*this);
    if (!p->initialize_compute(desc)) return nullptr;
    return p;
}


namespace {

// One-shot command buffer for AS builds. Submits to graphics queue and
// waits on the queue (sync). Suitable for one-time setup; per-frame builds
// will use a dedicated transfer queue + fence in a future phase.
// Allocate a transient buffer for AS scratch / result storage.
struct ScratchBuffer {
    VmaAllocator     allocator{VK_NULL_HANDLE};
    VkBuffer         buffer{VK_NULL_HANDLE};
    VmaAllocation    alloc{VK_NULL_HANDLE};
    u64              device_address{0};

    ~ScratchBuffer() {
        if (buffer != VK_NULL_HANDLE) vmaDestroyBuffer(allocator, buffer, alloc);
    }

    bool create(VulkanDevice& dev, VkDeviceSize size, VkBufferUsageFlags usage) {
        allocator = dev.vk_allocator();
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size;
        bci.usage       = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateBuffer(allocator, &bci, &aci, &buffer, &alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
        VkBufferDeviceAddressInfo bdai{};
        bdai.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = buffer;
        device_address = vkGetBufferDeviceAddress(dev.vk_device(), &bdai);
        return true;
    }
};

}  // namespace

bool VulkanAccelerationStructure::initialize_blas(const BlasDesc& desc) {
    if (desc.geometries.empty()) {
        cardinal::log::errorf("rhi/vk/blas", "no geometries");
        return false;
    }

    // 1) Build the geometry / range info arrays.
    cardinal::vector<VkAccelerationStructureGeometryKHR>      geoms;
    cardinal::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    cardinal::vector<u32>                                      max_prim_counts;
    geoms.reserve(desc.geometries.size());
    ranges.reserve(desc.geometries.size());
    max_prim_counts.reserve(desc.geometries.size());

    for (const auto& g : desc.geometries) {
        if (g.vertex_buffer == nullptr || g.vertex_count == 0) {
            cardinal::log::errorf("rhi/vk/blas", "geometry missing vertex buffer");
            return false;
        }
        auto* vbuf = static_cast<VulkanBuffer*>(g.vertex_buffer);
        if (vbuf->device_address() == 0) {
            cardinal::log::errorf("rhi/vk/blas",
                "vertex buffer must be created with ShaderDeviceAddress");
            return false;
        }

        VkAccelerationStructureGeometryTrianglesDataKHR tris{};
        tris.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tris.vertexFormat = to_vk_format(g.vertex_format);
        tris.vertexData.deviceAddress = vbuf->device_address() + g.vertex_offset;
        tris.vertexStride             = g.vertex_stride;
        tris.maxVertex                = g.vertex_count - 1;
        tris.indexType                = VK_INDEX_TYPE_NONE_KHR;

        u32 prim_count = g.vertex_count / 3;
        if (g.index_buffer != nullptr && g.index_count > 0) {
            auto* ibuf = static_cast<VulkanBuffer*>(g.index_buffer);
            if (ibuf->device_address() == 0) {
                cardinal::log::errorf("rhi/vk/blas",
                    "index buffer must be ShaderDeviceAddress");
                return false;
            }
            tris.indexType                = g.indices_are_u32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            tris.indexData.deviceAddress  = ibuf->device_address() + g.index_offset;
            prim_count                    = g.index_count / 3;
        }

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tris;
        geom.flags        = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u;
        geoms.push_back(geom);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = prim_count;
        ranges.push_back(range);

        max_prim_counts.push_back(prim_count);
    }

    // 2) Query build sizes.
    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = (desc.prefer_fast_trace
                                ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);
    if (desc.allow_compaction) {
        build_info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = static_cast<u32>(geoms.size());
    build_info.pGeometries   = geoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        dev_.device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info,
        max_prim_counts.data(),
        &sizes);

    // 3) Result + scratch buffers.
    {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = sizes.accelerationStructureSize;
        bci.usage       = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (!vk_check(
                vmaCreateBuffer(dev_.allocator_, &bci, &aci,
                                &result_buffer_, &result_alloc_, nullptr),
                "vmaCreateBuffer (BLAS result)")) {
            return false;
        }
    }

    ScratchBuffer scratch;
    if (!scratch.create(dev_,
                        sizes.buildScratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        cardinal::log::errorf("rhi/vk/blas", "scratch buffer alloc failed");
        return false;
    }

    // 4) Create the AS handle on top of the result buffer.
    {
        VkAccelerationStructureCreateInfoKHR aci{};
        aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        aci.size   = sizes.accelerationStructureSize;
        aci.buffer = result_buffer_;
        if (!vk_check(
                vkCreateAccelerationStructureKHR(dev_.device_, &aci, nullptr, &as_handle_),
                "vkCreateAccelerationStructureKHR")) {
            return false;
        }
    }

    // 5) Build via one-shot command buffer.
    {
        OneShotCmd one(dev_);
        build_info.dstAccelerationStructure  = as_handle_;
        build_info.scratchData.deviceAddress = scratch.device_address;

        cardinal::vector<const VkAccelerationStructureBuildRangeInfoKHR*> range_ptrs;
        range_ptrs.reserve(ranges.size());
        for (const auto& r : ranges) range_ptrs.push_back(&r);

        vkCmdBuildAccelerationStructuresKHR(
            one.cmd, 1, &build_info, range_ptrs.data());

        // Barrier so the AS is readable for subsequent uses.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(one.cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &mb, 0, nullptr, 0, nullptr);

        one.submit_and_wait();
    }

    // 6) Cache the device address.
    {
        VkAccelerationStructureDeviceAddressInfoKHR ai{};
        ai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        ai.accelerationStructure = as_handle_;
        device_address_ = vkGetAccelerationStructureDeviceAddressKHR(dev_.device_, &ai);
    }
    return true;
}

cardinal::unique_ptr<AccelerationStructure>
VulkanDevice::create_blas(const BlasDesc& desc) {
    if (!caps_.acceleration_structure) {
        cardinal::log::errorf("rhi/vk/blas",
            "device does not support VK_KHR_acceleration_structure");
        return nullptr;
    }
    auto blas = cardinal::make_unique<VulkanAccelerationStructure>(*this);
    if (!blas->initialize_blas(desc)) return nullptr;
    return blas;
}

cardinal::unique_ptr<AccelerationStructure>
VulkanDevice::create_tlas(const TlasDesc& /*desc*/) {
    // TODO(phase-5.1): symmetric to BLAS but with VkAccelerationStructureInstanceKHR
    // packed into an instances buffer + VK_GEOMETRY_TYPE_INSTANCES_KHR.
    cardinal::log::errorf("rhi/vk/tlas", "not yet implemented");
    return nullptr;
}

// =============================================================================
// Factory entry point referenced by rhi.cpp.
// =============================================================================
cardinal::unique_ptr<Device> create_vulkan_device(const DeviceDesc& desc) {
    auto dev = cardinal::make_unique<VulkanDevice>();
    if (!dev->initialize(desc)) return nullptr;
    return dev;
}

// =============================================================================
// Vulkan interop — implementations of cardinal/rhi/vulkan_interop.hpp.
// Live here because they need access to the (anonymous-namespace) concrete
// classes. The header forward-declares; we cast based on backend().
// =============================================================================
}  // namespace cardinal::rhi

#include <cardinal/rhi/vulkan_interop.hpp>

namespace cardinal::rhi {

VulkanDeviceHandles vulkan_handles(Device* dev) {
    VulkanDeviceHandles h{};
    if (dev == nullptr || dev->backend() != Backend::Vulkan) return h;
    auto* vd = static_cast<VulkanDevice*>(dev);
    h.instance              = vd->vk_instance();
    h.physical_device       = vd->vk_physical();
    h.device                = vd->vk_device();
    h.graphics_queue        = vd->vk_graphics_queue();
    h.graphics_queue_family = vd->vk_graphics_queue_family();
    return h;
}

VulkanSwapchainHandles vulkan_handles(Swapchain* sw) {
    VulkanSwapchainHandles h{};
    if (sw == nullptr) return h;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    h.swapchain        = vs->vk_swapchain();
    h.color_format     = vs->vk_color_format();
    h.image_count      = vs->vk_image_count();
    h.min_image_count  = vs->vk_min_image_count();
    return h;
}

VkCommandBuffer current_command_buffer(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_current_cmd();
}

u32 vulkan_acquired_image_index(Swapchain* sw) {
    if (sw == nullptr) return 0;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_acquired_image_index();
}

VkImage vulkan_swapchain_image(Swapchain* sw, u32 index) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_image(index);
}

VkImageView vulkan_swapchain_view(Swapchain* sw, u32 index) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_view(index);
}

VkFormat vulkan_viewport_format(Swapchain* sw) {
    if (sw == nullptr) return VK_FORMAT_UNDEFINED;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_format();
}

VkImage vulkan_viewport_image(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_image();
}

VkImage vulkan_viewport_image(Swapchain* sw, u32 id) {
    // Per-id viewport image. Each editor panel binds its own SRV via the
    // matching VkImageView returned by vulkan_viewport_view(sw, id).
    if (sw == nullptr) return VK_NULL_HANDLE;
    return static_cast<VulkanSwapchain*>(sw)->vk_viewport_image(id);
}

VkImageView vulkan_viewport_view(Swapchain* sw, u32 id) {
    // Per-id viewport view. The host renders into viewport[id] via
    // set_active_viewport(id) + scene render, then ImGui samples this view
    // inside the overlay's render pass.
    if (sw == nullptr) return VK_NULL_HANDLE;
    return static_cast<VulkanSwapchain*>(sw)->vk_viewport_view(id);
}

VkImageView vulkan_viewport_view(Swapchain* sw) {
    if (sw == nullptr) return VK_NULL_HANDLE;
    auto* vs = static_cast<VulkanSwapchain*>(sw);
    return vs->vk_viewport_view();
}

}  // namespace cardinal::rhi
