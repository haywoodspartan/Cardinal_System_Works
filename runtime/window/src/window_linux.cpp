#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_LINUX

#include <cardinal/window/window.hpp>

// TODO(phase-2.5): wire up X11 (xcb) or Wayland alongside Vulkan WSI.
// For now this is a stub that compiles but cannot create a window.

namespace cardinal::window {

cardinal::unique_ptr<Window> Window::create(const WindowDesc& /*desc*/) {
    return nullptr;
}

// Stub keeps the abstract method satisfied for the Linux build path.
// Concrete implementation lands with the X11/Wayland backend.

}  // namespace cardinal::window

#endif  // CARDINAL_PLATFORM_LINUX
