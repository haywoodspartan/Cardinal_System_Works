// =============================================================================
// Cardinal RHI — D3D12Device implementation.
//
// Out-of-line methods of the D3D12Device class plus the create_d3d12_device
// factory + the d3d12_interop functions that downcast Device*/Swapchain*
// pointers to expose the raw ID3D12Device + IDXGISwapChain3 to the editor's
// ImGui-on-D3D12 path. Class declaration lives in internal.hpp; this TU
// just provides the bodies.
//
// Non-Windows fallback (create_d3d12_device returns nullptr) lives in
// this file too — it's the only TU that defines the symbol off-Windows.
// =============================================================================

#include <cardinal/rhi/rhi.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#if !CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {
cardinal::unique_ptr<Device> create_d3d12_device(const DeviceDesc&) {
    cardinal::log::warnf("rhi/d3d12", "D3D12 backend is Windows-only");
    return nullptr;
}
}  // namespace cardinal::rhi

#else  // CARDINAL_PLATFORM_WINDOWS

#include "internal.hpp"

namespace cardinal::rhi {

bool D3D12Device::initialize(const DeviceDesc& desc) {
    if (desc.enable_validation) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            cardinal::log::infof("rhi/d3d12", "D3D12 debug layer enabled");
        }
    }

    if (!create_factory())                          return false;
    if (!select_adapter(desc.prefer_discrete_gpu))  return false;
    if (!create_device())                           return false;
    if (!create_queues())                           return false;
    if (!init_dxc())                                return false;
    populate_capabilities();
    settings_ = desc.initial_settings;

    // NvAPI — global init. Must be paired with NvAPI_Unload at shutdown.
    // Failure is non-fatal: Reflex just stays unavailable on this adapter.
#if CARDINAL_HAS_NVAPI
    {
        const NvAPI_Status s = NvAPI_Initialize();
        nvapi_initialised_ = (s == NVAPI_OK);
        if (nvapi_initialised_) {
            cardinal::log::infof("rhi/d3d12",
                "NvAPI initialised — Reflex available on NVIDIA adapters");
        } else {
            cardinal::log::infof("rhi/d3d12",
                "NvAPI_Initialize returned %d (Reflex disabled)",
                static_cast<int>(s));
        }
    }
#endif

    cardinal::log::infof("rhi/d3d12", "Device initialised (%s)", adapter_name_);
    return true;
}

bool D3D12Device::create_factory() {
    UINT flags = 0;
    D3D_CHECK(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)));

    // Probe DXGI tearing support (Win10+ / variable refresh). Without this
    // flag the swapchain caps Present(0,0) to the display refresh rate via
    // DWM composition, regardless of what the application asks for.
    {
        ComPtr<IDXGIFactory5> f5;
        if (SUCCEEDED(factory_.As(&f5))) {
            BOOL supp = FALSE;
            if (SUCCEEDED(f5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING, &supp, sizeof(supp))))
            {
                tearing_supported_ = (supp == TRUE);
            }
        }
        cardinal::log::infof("rhi/d3d12",
            "DXGI tearing support: %s (uncapped present requires this)",
            tearing_supported_ ? "YES" : "NO");
    }
    return true;
}

bool D3D12Device::select_adapter(bool prefer_discrete) {
    // EnumAdapterByGpuPreference is the modern path — we ask DXGI to sort by
    // performance preference. Fall back to manual enum if it's unavailable.
    DXGI_GPU_PREFERENCE pref = prefer_discrete
        ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
        : DXGI_GPU_PREFERENCE_UNSPECIFIED;

    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adp;
        if (factory_->EnumAdapterByGpuPreference(i, pref, IID_PPV_ARGS(&adp))
                == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 d;
        if (FAILED(adp->GetDesc1(&d))) continue;
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adp.Get(), D3D_FEATURE_LEVEL_11_0,
                                        __uuidof(ID3D12Device), nullptr))) {
            adapter_ = cardinal::move(adp);
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1,
                                adapter_name_, sizeof(adapter_name_) - 1,
                                nullptr, nullptr);
            caps_.vram_bytes = d.DedicatedVideoMemory;
            cardinal::strncpy(caps_.vendor_name,
                d.VendorId == 0x10DE ? "NVIDIA Corporation" :
                d.VendorId == 0x1002 ? "Advanced Micro Devices" :
                d.VendorId == 0x8086 ? "Intel" : "Unknown",
                sizeof(caps_.vendor_name) - 1);
            return true;
        }
    }
    cardinal::log::errorf("rhi/d3d12", "no D3D12-capable adapter found");
    return false;
}

bool D3D12Device::create_device() {
    D3D_CHECK(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                                IID_PPV_ARGS(&device_)));
    return true;
}

bool D3D12Device::create_queues() {
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    q.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    q.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    D3D_CHECK(device_->CreateCommandQueue(&q, IID_PPV_ARGS(&gfx_queue_)));

    // Dedicated async-compute queue (AEGIS Block 10). Every FL11+ device
    // exposes a COMPUTE engine, so this succeeds on conformant drivers — but
    // treat failure as "no async compute" (caps.async_compute stays false)
    // rather than failing device init, so graphics still comes up.
    D3D12_COMMAND_QUEUE_DESC cq{};
    cq.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    cq.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    cq.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(device_->CreateCommandQueue(&cq, IID_PPV_ARGS(&compute_queue_)))) {
        cardinal::log::warnf("rhi/d3d12",
            "async-compute queue creation failed — async compute disabled");
        compute_queue_.Reset();
    }
    return true;
}

bool D3D12Device::init_dxc() {
    D3D_CHECK(DxcCreateInstance(CLSID_DxcUtils,    IID_PPV_ARGS(&dxc_utils_)));
    D3D_CHECK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler_)));
    D3D_CHECK(dxc_utils_->CreateDefaultIncludeHandler(&dxc_includes_));
    return true;
}

// DirectStorage availability probe: the dstorage.dll + dstoragecore.dll are
// deployed next to the exe (NTC SDK). We can't compile against the IDStorage
// API (no dstorage.h vendored), but we CAN honestly report the cap by
// dynamically loading the DLL and resolving its public entry point. The full
// streaming subsystem lights up once dstorage.h is vendored.
static bool probe_direct_storage() {
    HMODULE m = ::LoadLibraryW(L"dstorage.dll");
    if (m == nullptr) return false;
    const bool ok = (::GetProcAddress(m, "DStorageGetFactory") != nullptr);
    ::FreeLibrary(m);
    return ok;
}

void D3D12Device::populate_capabilities() {
    // Probe the bits the engine cares about. D3D12 makes most of what we want
    // mandatory at FL_12_0+ (mesh shaders need _2 options struct), so we just
    // expose what's actually queryable.
    D3D12_FEATURE_DATA_D3D12_OPTIONS  o{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 o6{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
    // Shader model: probed robustly below. A 6.9 request returns E_INVALIDARG on
    // runtimes that predate it — if we ignored the HRESULT, HighestShaderModel
    // would be left at the requested 6.9 (garbage) and mis-light FP8/FP4 caps.
    D3D12_FEATURE_DATA_SHADER_MODEL   sm{D3D_SHADER_MODEL_6_8};

    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,  &o,  sizeof(o));
    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5));
    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &o6, sizeof(o6));
    device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7));
    // Request the highest SM the SDK knows; on failure, step down so the field
    // is always a value the runtime actually returned.
#if defined(D3D_SHADER_MODEL_6_9)
    {
        D3D12_FEATURE_DATA_SHADER_MODEL sm_hi{D3D_SHADER_MODEL_6_9};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm_hi, sizeof(sm_hi))))
            sm = sm_hi;
        else
            device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
    }
#else
    device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
#endif

    caps_.dynamic_rendering        = true;   // D3D12 has always had OMSetRenderTargets
    caps_.synchronization2         = true;
    caps_.buffer_device_address    = true;   // GPU VAs are first-class in D3D12
    caps_.descriptor_indexing      = true;
    caps_.runtime_descriptor_array = true;
    caps_.scalar_block_layout      = true;
    caps_.timeline_semaphore       = true;   // ID3D12Fence is timeline-equivalent
    caps_.async_compute            = (compute_queue_ != nullptr);  // dedicated COMPUTE queue
    caps_.max_async_compute_queues = caps_.async_compute ? 1u : 0u;
    caps_.shader_int16             = true;
    caps_.shader_float16           = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_2);
    caps_.shader_int8              = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_4);
    caps_.storage_buffer_16bit     = true;
    caps_.wave_intrinsics          = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_0);
    caps_.mesh_shader              = (o7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1);
    caps_.task_shader              = caps_.mesh_shader;     // amplification
    caps_.acceleration_structure   = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    caps_.ray_tracing_pipeline     = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    caps_.ray_query                = (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);

    // ---- AEGIS 2.0 GPU-feature caps (Blocks 4 / 8 / 12) ----
    caps_.indirect_dispatch        = true;   // ExecuteIndirect — FL12 baseline
    caps_.indirect_draw_count      = true;   // ExecuteIndirect w/ a count buffer
    caps_.bindless_resources       = (o.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3);
    caps_.variable_rate_shading    = (o6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1);
    caps_.variable_rate_image      = (o6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2);
    caps_.tensor_cores             = (cardinal::strstr(caps_.vendor_name, "NVIDIA") != nullptr)
                                  && (o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0);
    // FP8 (E4M3 / E5M2) math-division tier — SM6.9 cooperative vectors expose
    // native sub-FP16 math. Guarded so older SDKs (no 6.9 enum) compile + stay
    // false. The AEGIS adaptive-geometry tier already consumes this cap.
#if defined(D3D_SHADER_MODEL_6_9)
    caps_.fp8_math = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_9);
    // FP4 (E2M1) is the next sub-byte tier on the same SM6.9 cooperative-vector
    // path; finer per-format HW detection (Blackwell-class) would refine this.
    caps_.fp4_math = caps_.fp8_math;
#endif
    // DirectStorage / RTX IO: dstorage.dll is deployed alongside the exe — probe
    // it dynamically (the full IDStorage streaming API needs dstorage.h vendored).
    caps_.direct_storage_capable = probe_direct_storage();

    caps_.nvidia_dlss_capable      = (cardinal::strstr(caps_.vendor_name, "NVIDIA") != nullptr)
                                  && caps_.ray_tracing_pipeline;
    caps_.nvidia_framegen_capable  = caps_.nvidia_dlss_capable;
    caps_.amd_fsr3_capable         = (cardinal::strstr(caps_.vendor_name, "AMD") != nullptr ||
                                      cardinal::strstr(caps_.vendor_name, "Advanced Micro") != nullptr);

    cardinal::strncpy(caps_.gpu_arch,
        caps_.mesh_shader && caps_.ray_query ? "RTX 30+ class" :
        caps_.acceleration_structure         ? "RT-capable"   :
                                               "raster-only",
        sizeof(caps_.gpu_arch) - 1);
}

cardinal::unique_ptr<Buffer> D3D12Device::create_buffer(const BufferDesc& desc) {
    if (desc.size == 0) return nullptr;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = desc.cpu_writable ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width              = desc.size;
    rd.Height             = 1;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags              = D3D12_RESOURCE_FLAG_NONE;
    // ALLOW_UNORDERED_ACCESS is *illegal* on an UPLOAD-heap resource and is
    // only meaningful for GPU-writable (UAV) buffers. Our cpu_writable
    // storage buffers (per-frame light/material uploads) live on the UPLOAD
    // heap and are bound read-only as root SRVs (StructuredBuffer<T>), so
    // requesting UAV here makes CreateCommittedResource fail with
    // E_INVALIDARG. Only DEFAULT-heap storage buffers may be UAVs.
    if (!desc.cpu_writable &&
        (static_cast<u32>(desc.usage) & static_cast<u32>(BufferUsage::Storage)))
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    const D3D12_RESOURCE_STATES initial =
        desc.cpu_writable ? D3D12_RESOURCE_STATE_GENERIC_READ
                          : D3D12_RESOURCE_STATE_COMMON;

    ComPtr<ID3D12Resource> res;
    HRESULT hr = device_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &rd, initial, nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateCommittedResource (buffer) failed (%s)",
            hr_str(hr));
        return nullptr;
    }
    return cardinal::make_unique<D3D12Buffer>(cardinal::move(res), desc.size, desc.cpu_writable);
}

cardinal::unique_ptr<Texture> D3D12Device::create_texture(const TextureDesc& desc) {
    auto t = cardinal::make_unique<D3D12Texture>();
    if (!t->initialize(device_.Get(), desc)) return nullptr;
    return t;
}

ShaderBlob D3D12Device::compile_shader(ShaderStage stage, const char* hlsl, const char* entry) {
    ShaderBlob out;
    if (hlsl == nullptr || entry == nullptr) return out;

    LPCWSTR target = L"vs_6_5";
    switch (stage) {
        case ShaderStage::Vertex:   target = L"vs_6_5"; break;
        case ShaderStage::Fragment: target = L"ps_6_5"; break;
        case ShaderStage::Compute:  target = L"cs_6_5"; break;
    }
    // Convert entry point char → wide.
    wchar_t entry_w[64]{};
    MultiByteToWideChar(CP_UTF8, 0, entry, -1, entry_w, 63);

    DxcBuffer src{};
    src.Ptr      = hlsl;
    src.Size     = cardinal::strlen(hlsl);
    src.Encoding = DXC_CP_UTF8;

    cardinal::array<LPCWSTR, 8> args = {
        L"-E", entry_w,
        L"-T", target,
        L"-Zi",
        L"-Qstrip_reflect",
        L"-HV", L"2021",
    };

    ComPtr<IDxcResult> result;
    HRESULT hr = dxc_compiler_->Compile(&src, args.data(), static_cast<UINT32>(args.size()),
                                        dxc_includes_.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "DXC Compile call failed (%s)", hr_str(hr));
        return out;
    }
    HRESULT compile_status = S_OK;
    result->GetStatus(&compile_status);
    if (FAILED(compile_status)) {
        ComPtr<IDxcBlobUtf8> errors;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        if (errors && errors->GetStringLength() > 0) {
            cardinal::log::errorf("rhi/d3d12", "DXC error: %s", errors->GetStringPointer());
        } else {
            cardinal::log::errorf("rhi/d3d12", "DXC failed (%s)", hr_str(compile_status));
        }
        return out;
    }
    ComPtr<IDxcBlob> code;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&code), nullptr);
    if (!code || code->GetBufferSize() == 0) {
        cardinal::log::errorf("rhi/d3d12", "DXC returned empty object");
        return out;
    }
    out.bytes.resize(code->GetBufferSize());
    cardinal::memcpy(out.bytes.data(), code->GetBufferPointer(), code->GetBufferSize());
    return out;
}

// Live VRAM telemetry — DXGI 4 path. QueryVideoMemoryInfo is essentially
// free (one user-mode call into dxgi.dll); safe per-frame.
Device::VramSnapshot D3D12Device::query_vram_usage() const noexcept {
    VramSnapshot s{};
    if (!adapter_) return s;
    ComPtr<IDXGIAdapter4> a4;
    if (SUCCEEDED(adapter_.As(&a4))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(a4->QueryVideoMemoryInfo(
                /*NodeIndex=*/0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            s.budget_bytes                    = static_cast<u64>(info.Budget);
            s.current_usage_bytes             = static_cast<u64>(info.CurrentUsage);
            s.available_for_reservation_bytes = static_cast<u64>(info.AvailableForReservation);
            s.current_reservation_bytes       = static_cast<u64>(info.CurrentReservation);
        }
    }
    return s;
}

cardinal::unique_ptr<Pipeline> D3D12Device::create_pipeline(const PipelineDesc& desc) {
    if (!desc.vertex_shader.ok() || !desc.fragment_shader.ok()) {
        cardinal::log::errorf("rhi/d3d12", "create_pipeline: shaders not compiled");
        return nullptr;
    }

    // Root signature: optional 32-bit constants block at slot 0 when the
    // pipeline declares one. push_constant_size is in BYTES; the D3D12
    // root-constants slot counts in DWORDs (4 B each), so we round up
    // for the slot size while passing the byte count up to D3D12 via
    // SetGraphicsRoot32BitConstants below. Visible to all stages so
    // both VS and PS see the block (matches the Vulkan side, which
    // enables both VK_SHADER_STAGE_VERTEX_BIT + FRAGMENT_BIT).
    // Root parameter layout:
    //   [0]            32-bit constants (b0)         — iff push_constant_size>0
    //   [base .. base+N) root SRV (t0, t1, …)        — one per storage slot
    // where base = (push_constant_size>0 ? 1 : 0). Keeping the push
    // block at index 0 preserves set_push_constants' hardcoded root
    // index 0. Root SRVs (not descriptor tables) need no descriptor
    // heap — they bind straight from a GPU virtual address, mirroring
    // the Vulkan storage-buffer descriptor on the other backend.
    // A D3D12 root signature is capped at 64 DWORDs. 32-bit root constants
    // cost one DWORD *per DWORD of payload*, so a large push block (e.g. the
    // 240 B / 60-DWORD scene block) plus root SRVs (2 DWORDs each) + a
    // descriptor table (1 DWORD) blows the budget and CreateRootSignature
    // fails with E_INVALIDARG. Demote big blocks to a root CBV (b0, a flat
    // 2 DWORDs) backed by the per-frame constant ring; keep the cheaper
    // inline root-constants path for genuinely small blocks (<=16 DWORDs,
    // e.g. the shadow pass's single mat4). HLSL `cbuffer Push : register(b0)`
    // is identical either way, so no shader change is needed.
    const bool push_as_cbv = desc.push_constant_size > 64;
    cardinal::vector<D3D12_ROOT_PARAMETER> root_params;
    if (desc.push_constant_size > 0) {
        D3D12_ROOT_PARAMETER rp{};
        if (push_as_cbv) {
            rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rp.Descriptor.ShaderRegister = 0;             // b0
            rp.Descriptor.RegisterSpace  = 0;
        } else {
            rp.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rp.Constants.ShaderRegister = 0;              // b0
            rp.Constants.RegisterSpace  = 0;
            rp.Constants.Num32BitValues = (desc.push_constant_size + 3) / 4;
        }
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    for (u32 s = 0; s < desc.storage_buffer_slots; ++s) {
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp.Descriptor.ShaderRegister = s;                 // t0, t1, …
        rp.Descriptor.RegisterSpace  = 0;
        rp.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    // Sampled textures (shadow map): one descriptor-table root param
    // holding a contiguous SRV range at registers t[storage_slots ..],
    // matching the Vulkan binding convention (textures follow buffers).
    // `range` must outlive D3D12SerializeRootSignature below — it's a
    // synchronous call, so a function-local is fine.
    D3D12_DESCRIPTOR_RANGE srv_range{};
    if (desc.sampled_texture_slots > 0) {
        srv_range.RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srv_range.NumDescriptors     = desc.sampled_texture_slots;
        srv_range.BaseShaderRegister = desc.storage_buffer_slots;   // t after storage
        srv_range.RegisterSpace      = 0;
        srv_range.OffsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp.DescriptorTable.NumDescriptorRanges = 1;
        rp.DescriptorTable.pDescriptorRanges   = &srv_range;
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        root_params.push_back(rp);
    }
    // Static sampler s0 for the sampled textures — clamp + linear, same
    // policy as the Vulkan device default sampler. Static samplers live
    // in the root signature and consume no root-parameter slot.
    D3D12_STATIC_SAMPLER_DESC ss{};
    ss.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    ss.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ss.MaxLOD           = D3D12_FLOAT32_MAX;
    ss.ShaderRegister   = 0;                               // s0
    ss.RegisterSpace    = 0;
    ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    if (!root_params.empty()) {
        rsd.NumParameters = static_cast<UINT>(root_params.size());
        rsd.pParameters   = root_params.data();
    }
    if (desc.sampled_texture_slots > 0) {
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers   = &ss;
    }

    // Fail loudly *here* rather than via an opaque CreateRootSignature
    // E_INVALIDARG if some future slot configuration still overflows the
    // 64-DWORD ceiling. push: CBV=2, else ceil(bytes/4); root SRV=2 each;
    // sampled-texture descriptor table=1.
    {
        const u32 push_dwords = (desc.push_constant_size == 0) ? 0u
                              : (push_as_cbv ? 2u : (desc.push_constant_size + 3u) / 4u);
        const u32 total_dwords = push_dwords
                               + 2u * desc.storage_buffer_slots
                               + (desc.sampled_texture_slots > 0 ? 1u : 0u);
        if (total_dwords > 64u) {
            cardinal::log::errorf("rhi/d3d12",
                "root signature too large: %u DWORDs > 64 "
                "(push=%uB storage=%u tex=%u)",
                total_dwords, desc.push_constant_size,
                desc.storage_buffer_slots, desc.sampled_texture_slots);
            return nullptr;
        }
    }

    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                             &blob, &err);
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "SerializeRootSignature failed (%s%s)",
            hr_str(hr), err ? static_cast<const char*>(err->GetBufferPointer()) : "");
        return nullptr;
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(),
                                      blob->GetBufferSize(), IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateRootSignature failed (%s)", hr_str(hr));
        return nullptr;
    }

    // Input layout from PipelineDesc::vertex_attribs.
    cardinal::vector<D3D12_INPUT_ELEMENT_DESC> ie;
    ie.reserve(desc.vertex_attribs.size());
    static const char* semantic_names[] = { "POSITION", "COLOR", "TEXCOORD", "NORMAL", "TANGENT" };
    for (const auto& a : desc.vertex_attribs) {
        D3D12_INPUT_ELEMENT_DESC el{};
        // Match the HLSL we use: POSITION0 / COLOR0 / TEXCOORD<n> by location.
        const u32 sem_idx = a.location;
        const char* sem   = sem_idx < cardinal::size(semantic_names)
                          ? semantic_names[sem_idx] : "TEXCOORD";
        // Heuristic: location 0 -> POSITION, 1 -> COLOR, 2+ -> TEXCOORD with
        // increasing semantic indices. Matches what our triangle sample uses.
        u32 sem_index = 0;
        if (sem_idx >= 2) { sem = "TEXCOORD"; sem_index = sem_idx - 2; }
        el.SemanticName      = sem;
        el.SemanticIndex     = sem_index;
        el.Format            = to_dxgi(a.format);
        el.InputSlot         = 0;
        el.AlignedByteOffset = a.offset;
        el.InputSlotClass    = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        ie.push_back(el);
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature                = root.Get();
    psoDesc.VS.pShaderBytecode            = desc.vertex_shader.bytes.data();
    psoDesc.VS.BytecodeLength             = desc.vertex_shader.bytes.size();
    psoDesc.PS.pShaderBytecode            = desc.fragment_shader.bytes.data();
    psoDesc.PS.BytecodeLength             = desc.fragment_shader.bytes.size();
    psoDesc.InputLayout                   = { ie.data(), static_cast<UINT>(ie.size()) };
    psoDesc.PrimitiveTopologyType         = to_d3d_topology_type(desc.topology);
    psoDesc.SampleMask                    = UINT_MAX;
    // color_format == Unknown ⇒ depth-only PSO (0 render targets) — the
    // shadow-map depth pass binds OMSetRenderTargets(0,...). The PSO's
    // RT count must match, so honour Unknown rather than forcing 1.
    const bool depth_only = (desc.color_format == Format::Unknown);
    psoDesc.NumRenderTargets              = depth_only ? 0u : 1u;
    if (!depth_only) psoDesc.RTVFormats[0] = to_dxgi(desc.color_format);
    psoDesc.SampleDesc.Count              = 1;

    // Sensible raster defaults — matches the Vulkan path for parity.
    psoDesc.RasterizerState.FillMode      = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode      = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthClipEnable       = TRUE;

    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    // Depth — opt-in per pipeline. When the caller sets depth_format we
    // tell the PSO about the matching DSV format and enable depth test +
    // write. Function = LESS so the nearer fragment wins.
    if (desc.depth_format != Format::Unknown) {
        psoDesc.DepthStencilState.DepthEnable    = desc.depth_test  ? TRUE : FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = desc.depth_write
            ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc      = D3D12_COMPARISON_FUNC_LESS;
        psoDesc.DepthStencilState.StencilEnable  = FALSE;
        psoDesc.DSVFormat = (desc.depth_format == Format::D32_Float)
            ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
    } else {
        psoDesc.DepthStencilState.DepthEnable    = FALSE;
        psoDesc.DepthStencilState.StencilEnable  = FALSE;
    }

    ComPtr<ID3D12PipelineState> pso;
    hr = device_->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        cardinal::log::errorf("rhi/d3d12", "CreateGraphicsPipelineState failed (%s)", hr_str(hr));
        return nullptr;
    }
    return cardinal::make_unique<D3D12Pipeline>(cardinal::move(root), cardinal::move(pso),
                                           desc.topology, desc.vertex_stride,
                                           desc.push_constant_size,
                                           desc.storage_buffer_slots,
                                           desc.sampled_texture_slots,
                                           push_as_cbv);
}

// -----------------------------------------------------------------------------
// Compute pipeline (AEGIS graph::RhiBackend). Mirrors create_pipeline's root
// signature (push block @ b0 + read-only SRVs @ t0.. + read-write UAVs @ u0..)
// minus the input-assembler flag + sampler table, then a compute PSO. Empty
// shader => nullptr (the graph layer's correct telemetry-only fallback until
// real CSMain bytecode is threaded through).
// -----------------------------------------------------------------------------
cardinal::unique_ptr<Pipeline> D3D12Device::create_compute_pipeline(const ComputePipelineDesc& desc) {
    if (!desc.compute_shader.ok()) {
        return nullptr;   // empty blob — RhiBackend falls back to recording-only
    }

    const bool push_as_cbv = desc.push_constant_size > 64;
    cardinal::vector<D3D12_ROOT_PARAMETER> root_params;
    if (desc.push_constant_size > 0) {
        D3D12_ROOT_PARAMETER rp{};
        if (push_as_cbv) {
            rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rp.Descriptor.ShaderRegister = 0;             // b0
        } else {
            rp.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rp.Constants.ShaderRegister = 0;              // b0
            rp.Constants.Num32BitValues = (desc.push_constant_size + 3) / 4;
        }
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    for (u32 s = 0; s < desc.storage_buffer_slots; ++s) {   // read-only SRV t0..
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp.Descriptor.ShaderRegister = s;
        rp.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }
    for (u32 s = 0; s < desc.uav_slots; ++s) {             // read-write UAV u0..
        D3D12_ROOT_PARAMETER rp{};
        rp.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp.Descriptor.ShaderRegister = s;
        rp.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        root_params.push_back(rp);
    }

    // 64-DWORD budget guard (CBV/UAV/SRV root descriptors cost 2 DWORDs each;
    // inline constants cost 1 per DWORD of payload).
    {
        const u32 push_dwords = (desc.push_constant_size == 0) ? 0u
                              : (push_as_cbv ? 2u : (desc.push_constant_size + 3u) / 4u);
        const u32 total_dwords = push_dwords
                               + 2u * desc.storage_buffer_slots
                               + 2u * desc.uav_slots;
        if (total_dwords > 64u) {
            cardinal::log::errorf("rhi/d3d12",
                "compute root signature too large: %u DWORDs > 64 (push=%uB srv=%u uav=%u)",
                total_dwords, desc.push_constant_size,
                desc.storage_buffer_slots, desc.uav_slots);
            return nullptr;
        }
    }

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;   // no IA for compute
    if (!root_params.empty()) {
        rsd.NumParameters = static_cast<UINT>(root_params.size());
        rsd.pParameters   = root_params.data();
    }
    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        cardinal::log::errorf("rhi/d3d12", "compute SerializeRootSignature failed (%s)",
            err ? static_cast<const char*>(err->GetBufferPointer()) : "");
        return nullptr;
    }
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(),
                                            blob->GetBufferSize(), IID_PPV_ARGS(&root)))) {
        cardinal::log::errorf("rhi/d3d12", "compute CreateRootSignature failed");
        return nullptr;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC cd{};
    cd.pRootSignature   = root.Get();
    cd.CS.pShaderBytecode = desc.compute_shader.bytes.data();
    cd.CS.BytecodeLength  = desc.compute_shader.bytes.size();
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device_->CreateComputePipelineState(&cd, IID_PPV_ARGS(&pso)))) {
        cardinal::log::errorf("rhi/d3d12", "CreateComputePipelineState failed");
        return nullptr;
    }
    return cardinal::make_unique<D3D12Pipeline>(cardinal::move(root), cardinal::move(pso),
                                           desc.push_constant_size, push_as_cbv,
                                           desc.storage_buffer_slots, desc.uav_slots);
}

// -----------------------------------------------------------------------------
// Factory entry point
// -----------------------------------------------------------------------------
cardinal::unique_ptr<Device> create_d3d12_device(const DeviceDesc& desc) {
    auto d = cardinal::make_unique<D3D12Device>();
    if (!d->initialize(desc)) return nullptr;
    return d;
}

// -----------------------------------------------------------------------------
// d3d12_interop.hpp implementations — only valid when the underlying object
// is actually D3D12. Backend mismatch returns null/zero.
// -----------------------------------------------------------------------------
D3D12DeviceHandles d3d12_handles(Device* dev) {
    D3D12DeviceHandles h{};
    if (dev == nullptr || dev->backend() != Backend::D3D12) return h;
    auto* d = static_cast<D3D12Device*>(dev);
    h.device    = d->device();
    h.gfx_queue = d->gfx_queue();
    h.factory   = d->factory();
    return h;
}

D3D12SwapchainHandles d3d12_handles(Swapchain* sw) {
    D3D12SwapchainHandles h{};
    if (sw == nullptr) return h;
    auto* s = static_cast<D3D12Swapchain*>(sw);
    h.swapchain          = s->swap_chain();
    h.back_buffer_format = s->back_buffer_format();
    h.back_buffer_count  = s->back_buffer_count();
    return h;
}

ID3D12GraphicsCommandList* d3d12_current_cmd(Swapchain* sw) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->current_cmd();
}
u32 d3d12_acquired_image(Swapchain* sw) {
    if (sw == nullptr) return 0;
    return static_cast<D3D12Swapchain*>(sw)->current_index();
}
ID3D12Resource* d3d12_back_buffer(Swapchain* sw, u32 i) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->back_buffer(i);
}
D3D12_CPU_DESCRIPTOR_HANDLE d3d12_back_buffer_rtv(Swapchain* sw, u32 i) {
    if (sw == nullptr) { D3D12_CPU_DESCRIPTOR_HANDLE h{}; return h; }
    return static_cast<D3D12Swapchain*>(sw)->back_buffer_rtv(i);
}
ID3D12Resource* d3d12_viewport_image(Swapchain* sw) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->viewport_image();
}
ID3D12Resource* d3d12_viewport_image(Swapchain* sw, u32 id) {
    if (sw == nullptr) return nullptr;
    return static_cast<D3D12Swapchain*>(sw)->viewport_image(id);
}
u32 d3d12_viewport_count(Swapchain* sw) {
    if (sw == nullptr) return 0u;
    return static_cast<D3D12Swapchain*>(sw)->d3d12_viewport_count();
}

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
