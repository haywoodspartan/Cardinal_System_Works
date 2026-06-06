// =============================================================================
// Cardinal RHI — D3D12Swapchain Reflex (NvAPI) implementation.
//
// Split out of swapchain.cpp: the NVIDIA Reflex low-latency integration
// (sleep mode, latency markers, latency read-back) is a self-contained
// concern that only touches dev_.device() + NvAPI. Keeping it in its own TU
// shrinks the swapchain core and isolates the noisy NvAPI dependency.
//
// Compiles to no-ops without CARDINAL_HAS_NVAPI or when NvAPI returned
// non-OK during init. Windows-only.
// =============================================================================

#if defined(_WIN32) && !defined(CARDINAL_NO_D3D12)

#include "internal.hpp"

#if CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {

void D3D12Swapchain::set_reflex_mode(ReflexMode m) {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) { reflex_mode_ = m; return; }
    NV_SET_SLEEP_MODE_PARAMS p{};
    p.version           = NV_SET_SLEEP_MODE_PARAMS_VER;
    p.bLowLatencyMode   = (m != ReflexMode::Off);
    p.bLowLatencyBoost  = (m == ReflexMode::OnPlusBoost);
    p.minimumIntervalUs = (reflex_fps_cap_ > 0u)
                          ? (1'000'000u / reflex_fps_cap_) : 0u;
    p.bUseMarkersToOptimize = TRUE;
    const NvAPI_Status s = NvAPI_D3D_SetSleepMode(
        reinterpret_cast<IUnknown*>(dev_.device()), &p);
    if (s == NVAPI_OK) {
        reflex_mode_ = m;
    } else {
        cardinal::log::warnf("rhi/d3d12",
            "NvAPI_D3D_SetSleepMode(mode=%u) failed (%d)",
            static_cast<u32>(m), static_cast<int>(s));
    }
#else
    reflex_mode_ = m;
#endif
}

void D3D12Swapchain::set_reflex_fps_cap(u32 fps) {
    reflex_fps_cap_ = fps;
    set_reflex_mode(reflex_mode_);   // re-push (encodes minimumIntervalUs)
}

void D3D12Swapchain::reflex_sleep() {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return;
    // It's recommended to call this every frame even when the mode is Off —
    // the driver may still apply control-panel-level FPS caps via this hook.
    NvAPI_D3D_Sleep(reinterpret_cast<IUnknown*>(dev_.device()));
#endif
}

void D3D12Swapchain::reflex_marker(ReflexMarker m, u64 frame_id) {
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return;
    NV_LATENCY_MARKER_PARAMS p{};
    p.version = NV_LATENCY_MARKER_PARAMS_VER;
    p.frameID = frame_id;
    switch (m) {
        case ReflexMarker::SimulationStart:   p.markerType = SIMULATION_START;   break;
        case ReflexMarker::SimulationEnd:     p.markerType = SIMULATION_END;     break;
        case ReflexMarker::RenderSubmitStart: p.markerType = RENDERSUBMIT_START; break;
        case ReflexMarker::RenderSubmitEnd:   p.markerType = RENDERSUBMIT_END;   break;
        case ReflexMarker::PresentStart:      p.markerType = PRESENT_START;      break;
        case ReflexMarker::PresentEnd:        p.markerType = PRESENT_END;        break;
        case ReflexMarker::InputSample:       p.markerType = INPUT_SAMPLE;       break;
        default: return;
    }
    NvAPI_D3D_SetLatencyMarker(
        reinterpret_cast<IUnknown*>(dev_.device()), &p);
#else
    (void)m; (void)frame_id;
#endif
}

Swapchain::ReflexLatency D3D12Swapchain::reflex_latency() const {
    Swapchain::ReflexLatency out{};
#if CARDINAL_HAS_NVAPI
    if (!reflex_supported_) return out;
    NV_LATENCY_RESULT_PARAMS r{};
    r.version = NV_LATENCY_RESULT_PARAMS_VER;
    if (NvAPI_D3D_GetLatency(
            reinterpret_cast<IUnknown*>(dev_.device()), &r) != NVAPI_OK) {
        return out;
    }
    // Average over the most recent 8 valid frames (the buffer holds 64,
    // newest at the end). NvAPI returns timestamps in microseconds; we
    // emit milliseconds for the UI.
    constexpr int kAvg = 8;
    int      n = 0;
    double   tot = 0, sim = 0, render = 0, present = 0, drv = 0, osq = 0, gpu = 0;
    for (int i = 64 - kAvg; i < 64; ++i) {
        const auto& f = r.frameReport[i];
        if (f.frameID == 0) continue;
        if (f.inputSampleTime == 0 || f.gpuRenderEndTime == 0) continue;
        tot     += static_cast<double>(f.gpuRenderEndTime  - f.inputSampleTime);
        sim     += static_cast<double>(f.simEndTime        - f.simStartTime);
        render  += static_cast<double>(f.renderSubmitEndTime - f.renderSubmitStartTime);
        present += static_cast<double>(f.presentEndTime    - f.presentStartTime);
        drv     += static_cast<double>(f.driverEndTime     - f.driverStartTime);
        osq     += static_cast<double>(f.osRenderQueueEndTime - f.osRenderQueueStartTime);
        gpu     += static_cast<double>(f.gpuRenderEndTime  - f.gpuRenderStartTime);
        ++n;
    }
    if (n == 0) return out;
    const double inv = 1.0 / (n * 1000.0);   // µs → ms
    out.valid              = true;
    out.total_latency_ms   = static_cast<float>(tot     * inv);
    out.sim_ms             = static_cast<float>(sim     * inv);
    out.render_ms          = static_cast<float>(render  * inv);
    out.present_ms         = static_cast<float>(present * inv);
    out.driver_ms          = static_cast<float>(drv     * inv);
    out.os_render_queue_ms = static_cast<float>(osq     * inv);
    out.gpu_ms             = static_cast<float>(gpu     * inv);
#endif
    return out;
}

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
#endif  // _WIN32 && !CARDINAL_NO_D3D12
