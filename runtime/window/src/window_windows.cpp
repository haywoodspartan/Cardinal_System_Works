#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#include <cardinal/window/window.hpp>

#include <Windows.h>

#include <cardinal/core/std/containers.hpp>

namespace cardinal::window {

namespace {

class WindowsWindow final : public Window {
public:
    explicit WindowsWindow(const WindowDesc& desc) {
        register_class_once();

        DWORD style = desc.resizable
                          ? WS_OVERLAPPEDWINDOW
                          : (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX);

        // Adjust for client-area target size.
        RECT r{0, 0, static_cast<LONG>(desc.width), static_cast<LONG>(desc.height)};
        AdjustWindowRect(&r, style, FALSE);

        wchar_t title_w[256]{};
        MultiByteToWideChar(CP_UTF8, 0, desc.title, -1, title_w, 256);

        HINSTANCE hi = GetModuleHandleW(nullptr);
        hwnd_ = CreateWindowExW(
            0, kClassName, title_w, style,
            CW_USEDEFAULT, CW_USEDEFAULT,
            r.right - r.left, r.bottom - r.top,
            nullptr, nullptr, hi, this);

        if (hwnd_ == nullptr) return;

        SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        width_  = desc.width;
        height_ = desc.height;
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
    }

    ~WindowsWindow() override {
        if (hwnd_ != nullptr) {
            DestroyWindow(hwnd_);
        }
    }

    void poll_events() override {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE) != 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    bool  should_close() const override  { return should_close_; }
    void* native_handle() const override { return hwnd_; }
    u32   width() const override         { return width_; }
    u32   height() const override        { return height_; }

    bool  is_minimized() const override  { return minimized_; }

    bool  consume_resize(u32* out_w, u32* out_h) override {
        if (!resize_pending_) return false;
        if (out_w) *out_w = width_;
        if (out_h) *out_h = height_;
        resize_pending_ = false;
        return true;
    }

    void add_message_hook(RawMessageHook hook, void* user) override {
        if (hook == nullptr) return;
        // Don't add the same (hook, user) pair twice — guards against
        // double-attach during reinit cycles.
        for (const auto& h : hooks_) {
            if (h.fn == hook && h.user == user) return;
        }
        hooks_.push_back({hook, user});
    }

    void remove_message_hook(RawMessageHook hook, void* user) override {
        for (auto it = hooks_.begin(); it != hooks_.end(); ++it) {
            if (it->fn == hook && it->user == user) {
                hooks_.erase(it);
                return;
            }
        }
    }

    // Convenience — clear chain + add one. Pass nullptr to just clear.
    void set_message_hook(RawMessageHook hook, void* user) override {
        hooks_.clear();
        if (hook != nullptr) hooks_.push_back({hook, user});
    }

    void clear_message_hooks() override { hooks_.clear(); }

private:
    static constexpr const wchar_t* kClassName = L"CardinalWindow";

    static void register_class_once() {
        static bool registered = []() {
            WNDCLASSEXW wc{};
            wc.cbSize        = sizeof(wc);
            wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
            wc.lpfnWndProc   = &WindowsWindow::wnd_proc;
            wc.hInstance     = GetModuleHandleW(nullptr);
            wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
            wc.hbrBackground = nullptr;  // We'll paint via the renderer; default would flash.
            wc.lpszClassName = kClassName;
            RegisterClassExW(&wc);
            return true;
        }();
        (void)registered;
    }

    static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        auto* self = reinterpret_cast<WindowsWindow*>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA));

        // ----- Lifecycle messages: handled IN-HOUSE, NEVER forwarded -----
        //
        // We deliberately don't forward WM_CLOSE / WM_DESTROY / WM_QUIT /
        // WM_ENDSESSION / WM_SYSCOMMAND(SC_CLOSE) to the ImGui hook for the
        // MAIN window. ImGui_ImplWin32_WndProcHandler reacts to WM_CLOSE
        // by setting the matching viewport's PlatformRequestClose flag —
        // for our hook, that's the MAIN viewport. ImGui::UpdatePlatformWindows
        // then tries to tear the main viewport down mid-frame which trips
        // an IM_ASSERT inside the renderer backend (the main viewport's
        // ViewportData is the special "stub" allocated in Init, not a full
        // CreateWindow-allocated one). Manifests as a CRT abort() dialog.
        //
        // Solution: take full ownership of these. Set should_close_, let
        // the main loop exit cleanly, and the engine destructor handles
        // teardown — including ImGui shutdown — in the right order.
        bool is_lifecycle = false;
        switch (msg) {
            case WM_CLOSE:
            case WM_QUIT:
            case WM_ENDSESSION:
                is_lifecycle = true;
                if (self) self->should_close_ = true;
                return 0;
            case WM_DESTROY:
                is_lifecycle = true;
                if (self) self->should_close_ = true;
                PostQuitMessage(0);
                return 0;
            case WM_SYSCOMMAND:
                // SC_CLOSE is sent when the user clicks the X button (Win32
                // routes it through the system menu). DefWindowProc(SC_CLOSE)
                // synchronously sends WM_CLOSE — by short-circuiting here we
                // avoid the hook seeing either message.
                if ((wp & 0xFFF0) == /*SC_CLOSE*/ 0xF060) {
                    is_lifecycle = true;
                    if (self) self->should_close_ = true;
                    return 0;
                }
                break;
            default:
                break;
        }
        (void)is_lifecycle;

        // Non-lifecycle: walk the hook chain in attach order. First hook
        // to return true CONSUMES the message (Studio's ImGui handler
        // does this for text input / panel clicks). Otherwise every hook
        // sees the event and we fall through to default Win32 handling.
        if (self != nullptr) {
            for (const auto& h : self->hooks_) {
                if (h.fn(hwnd, static_cast<u32>(msg),
                         static_cast<u64>(wp), static_cast<i64>(lp),
                         h.user)) {
                    return 0;
                }
            }
        }

        switch (msg) {
            case WM_SIZE:
                if (self != nullptr) {
                    const u32 nw = LOWORD(lp);
                    const u32 nh = HIWORD(lp);
                    self->minimized_ = (wp == SIZE_MINIMIZED) || (nw == 0) || (nh == 0);
                    if (!self->minimized_ && (nw != self->width_ || nh != self->height_)) {
                        self->width_  = nw;
                        self->height_ = nh;
                        self->resize_pending_ = true;
                    } else if (!self->minimized_) {
                        self->width_  = nw;
                        self->height_ = nh;
                    }
                }
                return 0;
            default:
                break;
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    struct HookEntry { RawMessageHook fn; void* user; };

    HWND                   hwnd_{nullptr};
    bool                   should_close_{false};
    bool                   minimized_{false};
    bool                   resize_pending_{false};
    u32                    width_{0};
    u32                    height_{0};
    cardinal::vector<HookEntry> hooks_;
};

}  // namespace

cardinal::unique_ptr<Window> Window::create(const WindowDesc& desc) {
    return cardinal::make_unique<WindowsWindow>(desc);
}

}  // namespace cardinal::window

#endif  // CARDINAL_PLATFORM_WINDOWS
