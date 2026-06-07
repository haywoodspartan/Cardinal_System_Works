// =============================================================================
// Cardinal UI — Vulkan diagnostic helper (Studio decomposition, step 1).
//
// Verbatim move of vk_result_name + check_vk out of studio.cpp. No logic
// change — the same code, in its own translation unit, so the studio.cpp
// god-file shrinks and this self-contained concern is independently
// editable without touching the fragile StudioImpl/Vulkan-submit paths.
// =============================================================================
#include "studio_vk_diag.hpp"

#include <cardinal/core/types.hpp>   // u64 / usize / i64
#include <cardinal/core/std.hpp>     // cardinal::mutex/lock_guard/array/string
#include <cardinal/core/diag/log.hpp>
#include <cardinal/trace/stack.hpp>

namespace cardinal::ui::detail {

namespace {

// Map common VkResult values to their spec name so the log line is actually
// readable. Anything not in the table prints the numeric code as a fallback.
const char* vk_result_name(VkResult r) noexcept {
    switch (r) {
        case VK_SUCCESS:                          return "VK_SUCCESS";
        case VK_NOT_READY:                        return "VK_NOT_READY";
        case VK_TIMEOUT:                          return "VK_TIMEOUT";
        case VK_EVENT_SET:                        return "VK_EVENT_SET";
        case VK_EVENT_RESET:                      return "VK_EVENT_RESET";
        case VK_INCOMPLETE:                       return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:         return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:       return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:      return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:                return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:          return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:          return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:      return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:        return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:        return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:           return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:       return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY:         return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:    return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_SURFACE_LOST_KHR:           return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:   return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR:                   return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:            return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT:      return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
                                                  return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        default:                                  return nullptr;
    }
}

}  // namespace

// ImGui's Vulkan backend invokes this for every Vk* call result. To avoid
// drowning the log when one call fires every frame (a known ImGui-Vulkan
// secondary-window edge case), we:
//   1. Always log the FIRST error with a captured stack trace so the dev can
//      pinpoint the offending call site.
//   2. Throttle subsequent identical errors by code: one repeat log per
//      ~600 occurrences with a count, nothing else.
//   3. Treat VK_SUBOPTIMAL_KHR + VK_ERROR_OUT_OF_DATE_KHR as benign — the
//      backend handles them by flagging a swapchain rebuild and they're
//      noisy on resize.
void check_vk(VkResult r) {
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR) {
        return;
    }
    static cardinal::mutex             s_mu;
    static u64                    s_total{0};
    static cardinal::array<u64, 64>    s_per_code_count{};   // small fixed array indexed by hash
    auto code_slot = [](VkResult v) noexcept -> usize {
        return static_cast<usize>(static_cast<i64>(v) & 63);
    };

    cardinal::lock_guard<cardinal::mutex> g(s_mu);
    ++s_total;
    const usize slot = code_slot(r);
    const u64   prev = s_per_code_count[slot]++;

    const char* name = vk_result_name(r);
    if (prev == 0) {
        // First time we see this code — emit a full diagnostic with stack
        // so the dev can pinpoint the failing call site.
        if (name) {
            cardinal::log::errorf("ui/studio",
                "ImGui Vulkan call returned %s (%d) — first occurrence; capturing stack:",
                name, static_cast<int>(r));
        } else {
            cardinal::log::errorf("ui/studio",
                "ImGui Vulkan call returned (%d) — first occurrence; capturing stack:",
                static_cast<int>(r));
        }
        const auto frames = cardinal::trace::capture(/*skip=*/1, /*max=*/24);
        const cardinal::string s = cardinal::trace::format_full(frames, "    ");
        cardinal::log::errorf("ui/studio", "%s", s.c_str());
    } else if ((prev & 0x3FFu) == 0) {
        // ~Every 1024 hits, emit a one-liner with the running count so we
        // know it's still happening but don't drown the log.
        if (name) {
            cardinal::log::warnf("ui/studio",
                "ImGui Vulkan call still failing %s (%d) — count=%llu (total errors=%llu)",
                name, static_cast<int>(r),
                static_cast<unsigned long long>(prev + 1),
                static_cast<unsigned long long>(s_total));
        } else {
            cardinal::log::warnf("ui/studio",
                "ImGui Vulkan call still failing (%d) — count=%llu",
                static_cast<int>(r),
                static_cast<unsigned long long>(prev + 1));
        }
    }
}

}  // namespace cardinal::ui::detail
