// =============================================================================
// Cardinal — feature gating against GpuCapabilities.
// =============================================================================
#include <cardinal/render/pipeline.hpp>

#include <cardinal/core/log.hpp>

#include <cstring>

namespace cardinal::render {

FeatureGate::FeatureGate(const rhi::GpuCapabilities& caps) : caps_(caps) {}

FeatureStatus FeatureGate::query(Feature f) const noexcept {
    auto ok   = [](const char* /*why*/) { return FeatureStatus{ true,  "" }; };
    auto fail = [](const char* why)     { return FeatureStatus{ false, why }; };

    switch (f) {
        case Feature::HardwareRT:
            return (caps_.acceleration_structure || caps_.ray_query)
                ? ok("RT supported")
                : fail("Adapter lacks VK_KHR_acceleration_structure / D3D12 raytracing tier");

        case Feature::DLSS_Upscale:
            return caps_.nvidia_dlss_capable
                ? ok("RTX 20-series+ + Tensor cores")
                : fail("NVIDIA RTX 20+ required (Tensor cores)");

        case Feature::DLSS_FrameGen:
            return caps_.nvidia_framegen_capable
                ? ok("RTX 40-series+ Optical Flow Accelerator")
                : fail("NVIDIA RTX 40+ required (Optical Flow Accelerator)");

        case Feature::DLSS_RayReconstruct:
            // Same hardware bar as DLSS Super Resolution, but only useful
            // when ray tracing is actually doing the lighting.
            return (caps_.nvidia_dlss_capable && caps_.acceleration_structure)
                ? ok("RTX + RT denoiser")
                : fail("NVIDIA RTX 20+ AND hardware RT required");

        case Feature::DLAA:
            return caps_.nvidia_dlss_capable
                ? ok("native-resolution AA via NGX")
                : fail("NVIDIA RTX 20+ required");

        case Feature::FSR3_Upscale:
            // FSR3 spatial upscale runs on any GPU with shader model 6 / VK 1.2.
            return caps_.shader_float16
                ? ok("any FP16-capable GPU")
                : fail("FP16 shader support required");

        case Feature::FSR3_FrameGen:
            return caps_.amd_fsr3_capable
                ? ok("AMD RDNA 3+ with native frame generation")
                : fail("AMD RDNA 3+ required for native frame generation");

        case Feature::XeSS_Upscale:
            // Intel-optimised, runs on others as DP4a fallback.
            return caps_.shader_int8
                ? ok("any DP4a-capable GPU (best on Intel Arc XMX)")
                : fail("INT8 / DP4a shader support required");

        case Feature::MeshShaders:
            return caps_.mesh_shader
                ? ok("VK_EXT_mesh_shader / D3D12 mesh-shader tier 1+")
                : fail("Adapter does not expose mesh shaders");

        case Feature::TaskShaders:
            return caps_.task_shader
                ? ok("amplification stage available")
                : fail("Adapter does not expose task / amplification shaders");

        case Feature::VRS:
            // Probed indirectly: descriptor indexing + sync2 are mandated by
            // the same modern feature level. We don't have a dedicated cap
            // bit yet, so use that as a proxy.
            return (caps_.descriptor_indexing && caps_.synchronization2)
                ? ok("variable-rate shading supported")
                : fail("Adapter lacks VK 1.2 / DX12 Ultimate baseline");

        case Feature::BindlessDescriptors:
            return caps_.descriptor_indexing
                ? ok("descriptor indexing + runtime arrays")
                : fail("Adapter lacks descriptor indexing");

        case Feature::FP16Math:
            return caps_.shader_float16
                ? ok("RPM-class shader support")
                : fail("Adapter lacks shader float16");

        case Feature::INT8Math:
            return caps_.shader_int8
                ? ok("INT8 dot-product instructions")
                : fail("Adapter lacks shader int8");

        case Feature::AsyncCompute:
            // Every modern GPU has an async compute queue. We don't have
            // explicit caps yet, so report ok if the device looks modern.
            return caps_.synchronization2
                ? ok("dedicated compute queue available")
                : fail("Adapter does not advertise sync2 — async compute likely unsafe");

        case Feature::ReflexLowLatency:
            return (caps_.nvidia_dlss_capable)
                ? ok("NVIDIA Reflex SDK supported")
                : fail("NVIDIA RTX 20+ required for Reflex");

        case Feature::FP8Math:
            // Hardware FP8 in tensor cores landed with Hopper (H100) / Ada
            // (RTX 40-series) / equivalent AMD CDNA3+. We don't yet expose
            // a dedicated cap bit, so use frame-gen capability as a strong
            // proxy: every frame-gen-capable NVIDIA GPU has FP8 tensor cores.
            // The packed-math library still works on every GPU through
            // emulated unpack/operate/repack — this gate only tells the UI
            // whether NATIVE FP8 ops are likely to compile.
            return caps_.nvidia_framegen_capable
                ? ok("Tensor cores expose native FP8 (E4M3 / E5M2)")
                : fail("Native FP8 needs Hopper / Ada+ class hardware (emulated path still works)");

        case Feature::FP4Math:
            // FP4 tensor ops are Blackwell-class (RTX 50-series, B100/B200)
            // and brand new — DXC support is still evolving. Today this is
            // basically always "no" on shipping hardware; we check the
            // GPU arch label as a coarse heuristic.
            {
                bool blackwell_or_newer =
                    (std::strstr(caps_.gpu_arch, "Blackwell") != nullptr) ||
                    (std::strstr(caps_.gpu_arch, "RTX 50")    != nullptr);
                return blackwell_or_newer
                    ? ok("Blackwell-class FP4 tensor cores")
                    : fail("Native FP4 needs Blackwell-class hardware (emulated path still works)");
            }

        case Feature::Count_: break;
    }
    return fail("unknown feature");
}

const char* FeatureGate::feature_name(Feature f) noexcept {
    switch (f) {
        case Feature::HardwareRT:           return "Hardware Ray Tracing";
        case Feature::DLSS_Upscale:         return "DLSS Super Resolution";
        case Feature::DLSS_FrameGen:        return "DLSS Frame Generation";
        case Feature::DLSS_RayReconstruct:  return "DLSS Ray Reconstruction";
        case Feature::DLAA:                 return "DLAA";
        case Feature::FSR3_Upscale:         return "FSR 3 (upscale)";
        case Feature::FSR3_FrameGen:        return "FSR 3 (frame gen)";
        case Feature::XeSS_Upscale:         return "XeSS";
        case Feature::MeshShaders:          return "Mesh Shaders";
        case Feature::TaskShaders:          return "Task / Amplification Shaders";
        case Feature::VRS:                  return "Variable Rate Shading";
        case Feature::BindlessDescriptors:  return "Bindless Descriptors";
        case Feature::FP16Math:             return "FP16 (Rapid Packed Math)";
        case Feature::INT8Math:             return "INT8 / DP4a";
        case Feature::AsyncCompute:         return "Async Compute";
        case Feature::ReflexLowLatency:     return "NVIDIA Reflex";
        case Feature::FP8Math:              return "FP8 Tensor Math";
        case Feature::FP4Math:              return "FP4 Tensor Math";
        case Feature::Count_:               return "?";
    }
    return "?";
}

const char* FeatureGate::feature_group(Feature f) noexcept {
    switch (f) {
        case Feature::HardwareRT:
        case Feature::DLSS_RayReconstruct:           return "Ray Tracing";
        case Feature::DLSS_Upscale:
        case Feature::DLSS_FrameGen:
        case Feature::DLAA:
        case Feature::FSR3_Upscale:
        case Feature::FSR3_FrameGen:
        case Feature::XeSS_Upscale:                   return "Upscaling / Frame Generation";
        case Feature::MeshShaders:
        case Feature::TaskShaders:                    return "Geometry";
        case Feature::VRS:
        case Feature::BindlessDescriptors:
        case Feature::FP16Math:
        case Feature::INT8Math:
        case Feature::AsyncCompute:                   return "Performance";
        case Feature::ReflexLowLatency:               return "Latency";
        case Feature::FP8Math:
        case Feature::FP4Math:                        return "Packed Math";
        case Feature::Count_:                         return "?";
    }
    return "?";
}

}  // namespace cardinal::render
