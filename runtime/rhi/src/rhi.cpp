#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/platform.hpp>

namespace cardinal::rhi {

// Backend factories implemented in rhi_<backend>.cpp.
std::unique_ptr<Device> create_vulkan_device(const DeviceDesc& desc);
std::unique_ptr<Device> create_d3d12_device (const DeviceDesc& desc);

std::unique_ptr<Device> create_device(const DeviceDesc& desc) {
    Backend chosen = desc.backend;
    if (chosen == Backend::Auto) {
        // Auto prefers Vulkan today because Studio's editor overlay is
        // Vulkan-only. Flip to D3D12-first once cardinal::ui supports both.
        // Falls through to D3D12 if the Vulkan loader is missing.
        if (auto v = create_vulkan_device(desc)) return v;
#if defined(_WIN32)
        return create_d3d12_device(desc);
#else
        return nullptr;
#endif
    }

    switch (chosen) {
        case Backend::Vulkan: return create_vulkan_device(desc);
        case Backend::D3D12:  return create_d3d12_device(desc);
        default:              return nullptr;
    }
}

}  // namespace cardinal::rhi
