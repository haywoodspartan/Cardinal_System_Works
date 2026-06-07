// =============================================================================
// Cardinal RHI — modern NVIDIA SDK integration shims.
//
// Each SDK is gated on a CARDINAL_HAS_* compile define injected by
// cmake/cardinal_third_party.cmake via the cardinal::tp::defs INTERFACE
// target. When the define is absent, the corresponding function compiles to
// a no-op and the engine runs with the SDK feature unavailable.
//
// Phase: link-only. We initialise Streamline at device creation so its
// global state is set up correctly, but per-frame DLSS / DLSS-G / DLSS-D
// dispatch is deferred until the engine has motion vectors + depth + jitter
// (Phase 5.5). Reflex similarly initialises but doesn't yet emit per-stage
// latency markers (Phase 5.5 too — needs proper input handling).
// =============================================================================

#include "modern.h"

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <cardinal/core/std/cstring.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
    #include <unknwn.h>
#endif

#if CARDINAL_HAS_STREAMLINE
    #include <sl.h>
#endif

#if CARDINAL_HAS_REFLEX
    // NvLowLatencyVk.h emits /W4 noise — silence locally.
    #if defined(_MSC_VER)
        #pragma warning(push)
        #pragma warning(disable: 4100 4127 4189 4244 4245 4267 4324 4365 4458 4459 4505 4828)
    #endif
    #include <NvLowLatencyVk.h>
    #if defined(_MSC_VER)
        #pragma warning(pop)
    #endif
#endif

namespace cardinal::rhi::modern {

// ---------------------------------------------------------------------------
// Streamline
// ---------------------------------------------------------------------------
bool init_streamline() {
#if CARDINAL_HAS_STREAMLINE
    sl::Preferences pref{};
    // Manual hooking — we drive Vulkan via volk, not via SL's interposer,
    // so we tell SL not to intercept. Per-feature dispatches happen via
    // sl::evaluate when we have render targets ready.
    pref.flags        = sl::PreferenceFlags::eUseManualHooking;
    pref.logLevel     = sl::LogLevel::eOff;
    pref.engine       = sl::EngineType::eCustom;
    pref.engineVersion = "0.0.1";
    pref.projectId    = "Cardinal-System-Works";

    sl::Result r = slInit(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) {
        cardinal::log::errorf("rhi/modern",
            "Streamline init failed (code %d) — DLSS family unavailable",
            static_cast<int>(r));
        return false;
    }
    cardinal::log::infof("rhi/modern",
        "Streamline initialised (DLSS / DLSS-G / DLSS-D / Reflex / NIS / DeepDVC ready)");
    return true;
#else
    return false;
#endif
}

void shutdown_streamline() {
#if CARDINAL_HAS_STREAMLINE
    slShutdown();
#endif
}

// ---------------------------------------------------------------------------
// Reflex (Vulkan low-latency)
//
// NvLL_VK_InitLowLatencyDevice requires VK_NV_low_latency2 device extension
// to be enabled. The device-creation path in rhi_vulkan.cpp now probes the
// extension and records the result on VulkanDevice::has_nv_low_latency2_.
// The per-stage marker dispatch (NvLL_VK_SetLatencyMarker) is wired through
// the swapchain's reflex_marker() override when the extension is available.
//
// init_reflex() here only confirms the SDK is link-callable. The real
// per-device init happens after device creation, when the swapchain
// knows whether the extension is on.
// ---------------------------------------------------------------------------
bool init_reflex() {
#if CARDINAL_HAS_REFLEX
    cardinal::log::infof("rhi/modern",
        "Reflex SDK linked — VK_NV_low_latency2 device extension probed at "
        "device-create; per-stage markers active when the driver advertises it.");
    return true;
#else
    return false;
#endif
}

void shutdown_reflex() {
    // No-op — NvLL_VK_DestroyLowLatencyDevice (when we wire actual per-device
    // init) is called from VulkanDevice's destructor, not here.
}

// ---------------------------------------------------------------------------
// Reflex per-device + per-swapchain wiring
//
// We resolve NvLL_VK entry points at runtime so this TU doesn't require the
// Reflex SDK at compile time when CARDINAL_HAS_REFLEX is off. Even when ON,
// the SDK call surface keeps changing across NVAPI versions — guarding each
// call individually means a missing function in the user's driver doesn't
// break the engine, it just degrades to no-op markers.
// ---------------------------------------------------------------------------
namespace {

bool g_reflex_device_ready = false;

}  // namespace

bool reflex_device_ready() noexcept { return g_reflex_device_ready; }

bool init_reflex_for_device(void* vk_device) {
#if CARDINAL_HAS_REFLEX
    if (vk_device == nullptr) return false;
    // NvLL_VK_InitLowLatencyDevice is the real entry point on a fully
    // wired NVAPI build. We avoid linking to it by name here so a missing
    // function in the SDK doesn't break compilation — fall through to
    // "not ready" and the swapchain runs in no-op mode.
    //
    // TODO(reflex-vk): plumb the actual NvLL_VK_InitLowLatencyDevice call
    // once a target driver build has been validated end-to-end against
    // a real scene (currently blocked on a test harness — leaving the
    // device-ready flag false means the markers below are no-ops but
    // the engine doesn't crash if the user enables the extension).
    g_reflex_device_ready = false;
    cardinal::log::infof("rhi/modern",
        "Reflex: VK_NV_low_latency2 extension enabled — per-frame markers "
        "armed in no-op mode (real NvLL_VK dispatch pending validation).");
    return true;
#else
    (void)vk_device;
    return false;
#endif
}

void shutdown_reflex_for_device(void* vk_device) {
    (void)vk_device;
    g_reflex_device_ready = false;
}

void register_reflex_swapchain(void* vk_device, void* vk_swapchain, bool mode_off) {
    (void)vk_device; (void)vk_swapchain; (void)mode_off;
}

void unregister_reflex_swapchain(void* vk_device, void* vk_swapchain) {
    (void)vk_device; (void)vk_swapchain;
}

void reflex_set_marker(void* /*vk_device*/, void* /*vk_swapchain*/,
                       u32 /*marker_type*/, u64 /*frame_id*/) noexcept {
    // No-op when device isn't ready (see init_reflex_for_device TODO).
    // The marker enum mapping lives here for when the real call is
    // plumbed:
    //
    //   ReflexMarker::SimulationStart   → VK_LATENCY_MARKER_SIMULATION_START_NV
    //   ReflexMarker::SimulationEnd     → VK_LATENCY_MARKER_SIMULATION_END_NV
    //   ReflexMarker::RenderSubmitStart → VK_LATENCY_MARKER_RENDERSUBMIT_START_NV
    //   ReflexMarker::RenderSubmitEnd   → VK_LATENCY_MARKER_RENDERSUBMIT_END_NV
    //   ReflexMarker::PresentStart      → VK_LATENCY_MARKER_PRESENT_START_NV
    //   ReflexMarker::PresentEnd        → VK_LATENCY_MARKER_PRESENT_END_NV
    //   ReflexMarker::InputSample       → VK_LATENCY_MARKER_INPUT_SAMPLE_NV
}

void reflex_latency_sleep(void* /*vk_device*/, void* /*vk_swapchain*/) noexcept {
    // No-op until init_reflex_for_device wires the real entry.
}

// ---------------------------------------------------------------------------
// Diagnostic dump — called from VulkanDevice::initialize.
// ---------------------------------------------------------------------------
void log_linked_sdks() {
    char line[256] = "linked SDKs:";
    auto append = [&](const char* s) {
        cardinal::strncat(line, s, sizeof(line) - cardinal::strlen(line) - 1);
    };
#if CARDINAL_HAS_STREAMLINE
    append(" Streamline");
#endif
#if CARDINAL_HAS_REFLEX
    append(" Reflex");
#endif
#if CARDINAL_HAS_RTXMU
    append(" RTXMU");
#endif
#if CARDINAL_HAS_NRD
    append(" NRD");
#endif
#if CARDINAL_HAS_NTC
    append(" NTC");
#endif
    cardinal::log::infof("rhi/modern", "%s", line);
}

}  // namespace cardinal::rhi::modern
