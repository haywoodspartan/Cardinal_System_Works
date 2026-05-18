#pragma once

// =============================================================================
// Cardinal startup menu.
//
// Pre-RHI native dialog that lets the user pick which graphics backend to
// boot the engine with. Runs before any GPU work — uses Win32 only on
// Windows so the dialog itself never depends on the backend you haven't
// yet chosen. Returns BackendChoice::Cancel if the user closes the
// window; the application should exit cleanly in that case.
//
// Usage:
//   auto pick = cardinal::launcher::show_startup_menu({"My Game"});
//   if (pick == BackendChoice::Cancel) return 0;
//   rhi::DeviceDesc d{};
//   d.backend = cardinal::launcher::to_rhi_backend(pick);
//   auto device = rhi::create_device(d);
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/rhi/rhi.hpp>

namespace cardinal::launcher {

enum class BackendChoice : u32 {
    Cancel = 0,   // user closed the dialog without picking
    Vulkan,
    D3D12,
    Auto,         // engine picks the best available
};

struct StartupOptions {
    const char* app_title{"Cardinal"};
    const char* subtitle{"Select graphics backend"};
    // If true, suppress the dialog entirely and return Auto when neither
    // backend is available (so the app still tries something — useful for
    // CI/headless smoke tests).
    bool        skip_when_no_options{false};
    // Default selection when the dialog opens. The user can still change.
    BackendChoice default_choice{BackendChoice::Vulkan};
};

// Shows the dialog modally and returns the user's choice. Safe to call
// before any windowing/RHI work. Probes vulkan-1.dll and d3d12.dll +
// D3D12CreateDevice(FL_11_0) to label each option Available / Unavailable.
BackendChoice show_startup_menu(const StartupOptions& opts = {});

// Convenience: BackendChoice → cardinal::rhi::Backend.
inline cardinal::rhi::Backend to_rhi_backend(BackendChoice c) {
    switch (c) {
        case BackendChoice::Vulkan: return cardinal::rhi::Backend::Vulkan;
        case BackendChoice::D3D12:  return cardinal::rhi::Backend::D3D12;
        case BackendChoice::Auto:   return cardinal::rhi::Backend::Auto;
        case BackendChoice::Cancel: return cardinal::rhi::Backend::Auto;
    }
    return cardinal::rhi::Backend::Auto;
}

inline const char* backend_name(BackendChoice c) {
    switch (c) {
        case BackendChoice::Vulkan: return "Vulkan";
        case BackendChoice::D3D12:  return "D3D12";
        case BackendChoice::Auto:   return "Auto";
        case BackendChoice::Cancel: return "Cancel";
    }
    return "?";
}

}  // namespace cardinal::launcher
