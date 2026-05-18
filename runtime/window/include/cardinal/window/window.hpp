#pragma once

#include <cardinal/core/types.hpp>

#include <memory>

namespace cardinal::window {

struct WindowDesc {
    const char* title{"Cardinal"};
    u32 width{1280};
    u32 height{720};
    bool resizable{true};
};

// Hook called from the window's message loop BEFORE default handling.
// Returning true tells the window the message has been consumed (skip
// default processing). Allows external systems (Dear ImGui, custom input
// layers) to intercept raw OS messages.
//
//   hwnd / msg / wparam / lparam — raw Win32 args. On Linux they map to
//   the native event payload (TBD when WSI lands).
using RawMessageHook = bool (*)(void* hwnd, u32 msg, u64 wparam, i64 lparam, void* user);

// Native window. The native_handle() return is platform-specific:
//   Windows : HWND
//   Linux   : (TODO — X11/Wayland to be wired in alongside Vulkan WSI)
class Window {
public:
    static std::unique_ptr<Window> create(const WindowDesc& desc);

    virtual ~Window() = default;

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    virtual void  poll_events()         = 0;
    virtual bool  should_close() const  = 0;
    virtual void* native_handle() const = 0;
    virtual u32   width() const         = 0;
    virtual u32   height() const        = 0;

    // Pop the most recent unconsumed resize. Returns true and writes the
    // new client size into out_w/out_h when the user resized the window
    // since the last call; false otherwise. Apps poll this once per frame
    // and forward the size to swapchain->resize() when it returns true.
    virtual bool  consume_resize(u32* out_w, u32* out_h) = 0;

    // Returns true if the window is currently in a "minimised" / zero-area
    // state. Frame loops should skip rendering while this is true so the
    // RHI doesn't try to size a zero-sized swapchain.
    virtual bool  is_minimized() const = 0;

    // ----- Message hook chain ------------------------------------------
    //
    // Multiple subsystems can attach to the OS message stream — Studio
    // (ImGui), the input::Manager (action map), debug overlays, etc. The
    // window holds an ordered chain of (hook, user) pairs and calls them
    // in attach order on every message. The first hook to return true
    // CONSUMES the message (the chain stops and the message is dropped
    // before reaching default handling).
    //
    // Order matters when more than one hook can consume the same message.
    // Convention:
    //   1) Studio (ImGui) — attaches itself in Studio::create. Consumes
    //      messages destined for the editor UI (text input, panel clicks,
    //      modal popups). Returns false for everything else.
    //   2) input::Manager — attaches via the integrator. Translates raw
    //      messages into action / axis state but never consumes (always
    //      returns false), so the OS still gets default processing.
    //
    // Apps that just want a single hook can keep using set_message_hook —
    // it's now a convenience that clears the chain and adds one entry.
    virtual void  add_message_hook(RawMessageHook hook, void* user) = 0;
    virtual void  remove_message_hook(RawMessageHook hook, void* user) = 0;

    // Convenience — clears the chain and installs `hook` as the only entry.
    // Useful for the simple single-consumer case. Pass nullptr as `hook`
    // to clear without re-adding (equivalent to clear_message_hooks()).
    virtual void  set_message_hook(RawMessageHook hook, void* user) = 0;

    // Drop every attached hook. Called during teardown.
    virtual void  clear_message_hooks() = 0;

protected:
    Window() = default;
};

}  // namespace cardinal::window
