#pragma once

// =============================================================================
// Studio — stats panel (free-function helper).
//
// Pure-read panel: no per-frame state besides the device + swapchain that
// the studio already holds. The host hands us pointers + the live FPS /
// frame counter on each call.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::rhi { class Device; class Swapchain; }

namespace cardinal::ui::panels::stats_panel {

struct Inputs {
    cardinal::rhi::Device*    device{nullptr};
    cardinal::rhi::Swapchain* swapchain{nullptr};
    float                     ema_fps{0.0f};
    cardinal::u64             frame_count{0};
};

void draw(const char* title, bool* p_open, const Inputs& inputs);

}  // namespace cardinal::ui::panels::stats_panel
