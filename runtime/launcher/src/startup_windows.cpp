// =============================================================================
// Cardinal startup menu — Win32 implementation.
//
// Owner-drawn modern tile picker. Renders everything ourselves using GDI+
// (anti-aliased text + rounded rects) so the dialog doesn't look like a
// stock 90's Win32 form. Three "cards" (Vulkan / DirectX 12 / Auto-detect)
// with hover, focus, pressed and disabled visual states. Keyboard nav: Tab
// / arrow keys move focus, Enter activates, Esc cancels.
//
// We probe vulkan-1.dll and d3d12.dll up front so the cards can show
// availability inline. Disabled cards are still focusable but ignore clicks
// and Enter — they exist to inform the user *why* the option is greyed out.
// =============================================================================

#include <cardinal/launcher/startup.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

// WIN32_LEAN_AND_MEAN + NOMINMAX come from cardinal::compile_flags.
#include <Windows.h>
#include <Windowsx.h>
#include <unknwn.h>
#include <dxgi.h>
#include <d3d12.h>
#include <wrl/client.h>

// Vulkan probe — types only, we resolve symbols dynamically via LoadLibraryA.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// GDI+ for anti-aliased text and rounded rects. Header MUST be included
// after Windows.h, and the library has its own min/max which NOMINMAX
// already suppresses, so we're safe.
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace cardinal::launcher {

namespace {

// -----------------------------------------------------------------------------
// Probes
// -----------------------------------------------------------------------------
struct BackendStatus {
    bool        available{false};
    std::string detail;
};

BackendStatus probe_vulkan() {
    BackendStatus s;
    HMODULE m = LoadLibraryA("vulkan-1.dll");
    if (m == nullptr) {
        s.detail = "vulkan-1.dll not found — install GPU driver or Vulkan runtime";
        return s;
    }
    using PFN_vkEnumerateInstanceVersion = VkResult (VKAPI_PTR*)(uint32_t*);
    auto p_get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        ::GetProcAddress(m, "vkGetInstanceProcAddr"));
    if (p_get != nullptr) {
        auto p_ver = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            p_get(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        uint32_t ver = 0;
        if (p_ver != nullptr && p_ver(&ver) == VK_SUCCESS) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "Loader %u.%u.%u — recommended",
                VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver), VK_VERSION_PATCH(ver));
            s.detail = buf;
        } else {
            s.detail = "Loader 1.0 only — engine may run with reduced features";
        }
    } else {
        s.detail = "Loader present (couldn't read version)";
    }
    s.available = true;
    return s;
}

BackendStatus probe_d3d12() {
    BackendStatus s;
    HMODULE m = LoadLibraryA("d3d12.dll");
    if (m == nullptr) {
        s.detail = "d3d12.dll not found — Windows 10 1709+ required";
        return s;
    }
    auto p_create = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
        ::GetProcAddress(m, "D3D12CreateDevice"));
    if (p_create == nullptr) {
        s.detail = "d3d12.dll present but D3D12CreateDevice missing";
        return s;
    }
    HRESULT hr = p_create(nullptr, D3D_FEATURE_LEVEL_11_0,
                          __uuidof(ID3D12Device), nullptr);
    if (FAILED(hr) && hr != S_FALSE) {
        s.detail = "Hardware does not support D3D12 (FL_11_0 required)";
        return s;
    }
    s.available = true;
    s.detail    = "Hardware ready — D3D12 backend (Phase 2.6, baseline)";
    return s;
}

std::string detect_primary_gpu() {
    using namespace Microsoft::WRL;
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) return "(unknown GPU)";

    using PFN_CreateDXGIFactory = HRESULT (WINAPI*)(REFIID, void**);
    auto pCreate = reinterpret_cast<PFN_CreateDXGIFactory>(
        ::GetProcAddress(dxgi, "CreateDXGIFactory1"));
    if (pCreate == nullptr) return "(unknown GPU)";

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(pCreate(__uuidof(IDXGIFactory1), &factory))) return "(unknown GPU)";

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d;
        if (FAILED(adapter->GetDesc1(&d))) continue;
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;     // skip WARP
        char buf[160] = {};
        WideCharToMultiByte(CP_UTF8, 0, d.Description, -1,
                            buf, sizeof(buf) - 1, nullptr, nullptr);
        const auto gb = static_cast<unsigned>(
            d.DedicatedVideoMemory / (1024ull * 1024ull * 1024ull));
        char out[256];
        std::snprintf(out, sizeof(out), "%s (%u GB VRAM)", buf, gb);
        return out;
    }
    return "(no GPU enumerated)";
}

// -----------------------------------------------------------------------------
// Tile model
// -----------------------------------------------------------------------------
struct Tile {
    BackendChoice choice;
    std::string   title;
    std::string   detail;
    std::string   badge;       // "Available", "Coming soon", etc.
    bool          enabled{true};
};

// Visual constants (logical pixels at 96 DPI; scaled at draw-time).
struct Theme {
    Gdiplus::Color bg          {255, 25,  26,  31 };  // window background
    Gdiplus::Color bg_panel    {255, 32,  34,  41 };  // tile background (idle)
    Gdiplus::Color bg_hover    {255, 42,  46,  56 };
    Gdiplus::Color bg_pressed  {255, 18,  20,  26 };
    Gdiplus::Color bg_disabled {255, 28,  29,  34 };
    Gdiplus::Color border      {255, 60,  64,  72 };
    Gdiplus::Color border_focus{255, 0,   120, 215};  // accent (Win11 blue)
    Gdiplus::Color text        {255, 232, 233, 238};
    Gdiplus::Color text_dim    {255, 158, 162, 172};
    Gdiplus::Color text_disable{255, 100, 102, 110};
    Gdiplus::Color accent      {255, 0,   120, 215};
    Gdiplus::Color badge_ok    {255, 56,  176, 0  };
    Gdiplus::Color badge_warn  {255, 224, 168, 0  };
    Gdiplus::Color badge_off   {255, 110, 113, 123};
};

// -----------------------------------------------------------------------------
// Dialog state
// -----------------------------------------------------------------------------
constexpr int kWidth    = 560;
constexpr int kHeight   = 480;
constexpr int kPadding  = 24;
constexpr int kHdrH     = 96;     // height of header block
constexpr int kTileH    = 92;
constexpr int kTileGap  = 12;

struct DialogData {
    BackendChoice choice{BackendChoice::Cancel};
    std::array<Tile, 3> tiles{};

    std::string header;          // "Cardinal — Choose graphics backend"
    std::string subheader;       // "GPU: NVIDIA RTX 4090 (24 GB VRAM)"

    int   hover_index  {-1};
    int   focus_index  {0};
    int   pressed_idx  {-1};
    bool  mouse_tracking{false};

    Theme theme;
    HFONT font_h1{nullptr};      // big title
    HFONT font_h2{nullptr};      // subtitle
    HFONT font_tile_title{nullptr};
    HFONT font_tile_body{nullptr};
    HFONT font_badge{nullptr};

    ULONG_PTR gdiplus_token{0};
    bool      gdiplus_owner{false};
};

DialogData* state_of(HWND h) {
    return reinterpret_cast<DialogData*>(GetWindowLongPtrA(h, GWLP_USERDATA));
}

// Tile rect indexed by tile slot (0..2).
RECT tile_rect(int idx, RECT client) {
    RECT r;
    r.left   = client.left  + kPadding;
    r.right  = client.right - kPadding;
    r.top    = client.top   + kHdrH + idx * (kTileH + kTileGap);
    r.bottom = r.top + kTileH;
    return r;
}

bool point_in_rect(POINT p, RECT r) {
    return p.x >= r.left && p.x < r.right && p.y >= r.top && p.y < r.bottom;
}

int hit_test(POINT p, RECT client) {
    for (int i = 0; i < 3; ++i) if (point_in_rect(p, tile_rect(i, client))) return i;
    return -1;
}

// -----------------------------------------------------------------------------
// Drawing
// -----------------------------------------------------------------------------
void fill_rounded(Gdiplus::Graphics& g, const Gdiplus::RectF& r, int radius,
                  const Gdiplus::Color& fill,
                  const Gdiplus::Color* border = nullptr,
                  float border_width = 1.0f)
{
    using namespace Gdiplus;
    GraphicsPath path;
    const float d = static_cast<float>(radius * 2);
    path.AddArc(r.X,             r.Y,                d, d, 180.0f, 90.0f);
    path.AddArc(r.X + r.Width-d, r.Y,                d, d, 270.0f, 90.0f);
    path.AddArc(r.X + r.Width-d, r.Y + r.Height - d, d, d,   0.0f, 90.0f);
    path.AddArc(r.X,             r.Y + r.Height - d, d, d,  90.0f, 90.0f);
    path.CloseFigure();
    SolidBrush brush(fill);
    g.FillPath(&brush, &path);
    if (border) {
        Pen pen(*border, border_width);
        g.DrawPath(&pen, &path);
    }
}

// Draw a UTF-8 narrow string with GDI+. Converts on the fly.
void draw_text(Gdiplus::Graphics& g, const char* utf8,
               const Gdiplus::FontFamily& family, float pt,
               Gdiplus::FontStyle style, const Gdiplus::Color& col,
               const Gdiplus::RectF& rect, Gdiplus::StringAlignment halign,
               Gdiplus::StringAlignment valign)
{
    using namespace Gdiplus;
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), wlen);
    if (!w.empty() && w.back() == L'\0') w.pop_back();

    Font           font(&family, pt, style, UnitPoint);
    SolidBrush     brush(col);
    StringFormat   fmt;
    fmt.SetAlignment(halign);
    fmt.SetLineAlignment(valign);
    fmt.SetTrimming(StringTrimmingEllipsisCharacter);
    g.DrawString(w.c_str(), -1, &font, rect, &fmt, &brush);
}

void paint_tile(Gdiplus::Graphics& g, const DialogData& s, int idx,
                const Gdiplus::FontFamily& family,
                const Gdiplus::FontFamily& family_bold,
                RECT rc)
{
    using namespace Gdiplus;
    const Tile& t  = s.tiles[idx];
    const bool   hover    = s.hover_index == idx && t.enabled;
    const bool   pressed  = s.pressed_idx == idx && t.enabled;
    const bool   focused  = s.focus_index == idx;

    Color fill = !t.enabled ? s.theme.bg_disabled
               :  pressed   ? s.theme.bg_pressed
               :  hover     ? s.theme.bg_hover
               :              s.theme.bg_panel;

    Color border_col = focused && t.enabled ? s.theme.border_focus : s.theme.border;
    float border_w   = focused && t.enabled ? 2.0f : 1.0f;

    RectF rf(static_cast<float>(rc.left + 1),  static_cast<float>(rc.top + 1),
             static_cast<float>(rc.right - rc.left - 2),
             static_cast<float>(rc.bottom - rc.top - 2));
    fill_rounded(g, rf, 8, fill, &border_col, border_w);

    // Accent strip on the left when enabled (subtle visual hierarchy)
    if (t.enabled) {
        Color stripe = focused ? s.theme.accent : s.theme.border;
        SolidBrush b(stripe);
        g.FillRectangle(&b, rf.X + 8.0f, rf.Y + 14.0f, 4.0f, rf.Height - 28.0f);
    }

    const Color title_col = t.enabled ? s.theme.text     : s.theme.text_disable;
    const Color body_col  = t.enabled ? s.theme.text_dim : s.theme.text_disable;

    // Title (top-left, large)
    RectF title_rc(rf.X + 22.0f, rf.Y + 14.0f, rf.Width - 44.0f - 130.0f, 30.0f);
    draw_text(g, t.title.c_str(), family_bold, 16.0f, FontStyleBold, title_col,
              title_rc, StringAlignmentNear, StringAlignmentCenter);

    // Detail (under title)
    RectF body_rc(rf.X + 22.0f, rf.Y + 44.0f, rf.Width - 44.0f - 24.0f, rf.Height - 56.0f);
    draw_text(g, t.detail.c_str(), family, 9.5f, FontStyleRegular, body_col,
              body_rc, StringAlignmentNear, StringAlignmentNear);

    // Badge (top-right) — coloured pill with status text.
    if (!t.badge.empty()) {
        // Pick badge colour by tile state (Vulkan ok = green, D3D12 disabled
        // = grey, Auto = accent blue).
        Color bg = !t.enabled ? s.theme.badge_off
                 : (idx == 0) ? s.theme.badge_ok
                              : s.theme.accent;
        // Measure the text quickly using DrawString with a measure layout.
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, t.badge.c_str(), -1, nullptr, 0);
        std::wstring w(static_cast<size_t>(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, t.badge.c_str(), -1, w.data(), wlen);
        if (!w.empty() && w.back() == L'\0') w.pop_back();
        Font font(&family_bold, 8.5f, FontStyleBold, UnitPoint);
        RectF measured;
        g.MeasureString(w.c_str(), -1, &font, PointF(0, 0), &measured);

        const float pad_x = 10.0f, pad_y = 4.0f;
        const float bw = measured.Width  + pad_x * 2;
        const float bh = measured.Height + pad_y * 2;
        RectF badge_rc(rf.X + rf.Width - bw - 16.0f, rf.Y + 14.0f, bw, bh);
        fill_rounded(g, badge_rc, 8, bg);

        SolidBrush text_brush(Color(255, 255, 255, 255));
        StringFormat fmt;
        fmt.SetAlignment(StringAlignmentCenter);
        fmt.SetLineAlignment(StringAlignmentCenter);
        g.DrawString(w.c_str(), -1, &font, badge_rc, &fmt, &text_brush);
    }
}

void paint_dialog(HWND hwnd, HDC hdc) {
    using namespace Gdiplus;
    auto* s = state_of(hwnd);
    if (s == nullptr) return;

    RECT rc; GetClientRect(hwnd, &rc);
    const int W = rc.right - rc.left;
    const int H = rc.bottom - rc.top;

    // Double-buffer to a memory DC, then blit. Gives flicker-free repaint.
    HDC      mem    = CreateCompatibleDC(hdc);
    HBITMAP  bmp    = CreateCompatibleBitmap(hdc, W, H);
    HGDIOBJ  oldBmp = SelectObject(mem, bmp);

    Graphics g(mem);
    g.SetSmoothingMode      (SmoothingModeAntiAlias);
    g.SetTextRenderingHint  (TextRenderingHintClearTypeGridFit);
    g.SetCompositingQuality (CompositingQualityHighQuality);

    // Window background.
    SolidBrush bgBrush(s->theme.bg);
    g.FillRectangle(&bgBrush,
        static_cast<INT>(0), static_cast<INT>(0),
        static_cast<INT>(W), static_cast<INT>(H));

    // Header: app title + GPU subtitle. Segoe UI is on every Win10/11 box;
    // fall back to a generic family if it's somehow missing (locked-down VMs).
    // FontFamily is non-copyable / non-assignable, so probe with a temporary
    // and pass the resolved name into the real instances.
    auto resolve_family = [](const wchar_t* primary, const wchar_t* fallback) {
        return FontFamily(primary).IsAvailable() ? primary : fallback;
    };
    const wchar_t* fam_name    = resolve_family(L"Segoe UI",          L"Tahoma");
    const wchar_t* fam_sb_name = resolve_family(L"Segoe UI Semibold", fam_name);
    FontFamily family   (fam_name);
    FontFamily family_sb(fam_sb_name);

    RectF hdr_rc(static_cast<float>(kPadding), 22.0f,
                 static_cast<float>(W - kPadding * 2), 36.0f);
    draw_text(g, s->header.c_str(), family_sb, 18.0f, FontStyleBold,
              s->theme.text, hdr_rc, StringAlignmentNear, StringAlignmentCenter);

    RectF sub_rc(static_cast<float>(kPadding), 56.0f,
                 static_cast<float>(W - kPadding * 2), 24.0f);
    draw_text(g, s->subheader.c_str(), family, 10.0f, FontStyleRegular,
              s->theme.text_dim, sub_rc, StringAlignmentNear, StringAlignmentCenter);

    // Tiles.
    for (int i = 0; i < 3; ++i) {
        paint_tile(g, *s, i, family, family_sb, tile_rect(i, rc));
    }

    // Footer hint.
    RectF foot_rc(static_cast<float>(kPadding),
                  static_cast<float>(H - 32),
                  static_cast<float>(W - kPadding * 2), 20.0f);
    draw_text(g,
        "Tab / arrows to move • Enter to launch • Esc to cancel",
        family, 8.5f, FontStyleRegular, s->theme.text_dim,
        foot_rc, StringAlignmentNear, StringAlignmentCenter);

    BitBlt(hdc, 0, 0, W, H, mem, 0, 0, SRCCOPY);
    SelectObject(mem, oldBmp);
    DeleteObject(bmp);
    DeleteDC(mem);
}

// -----------------------------------------------------------------------------
// Window procedure
// -----------------------------------------------------------------------------
void invalidate_all(HWND h) { InvalidateRect(h, nullptr, FALSE); }

void confirm_choice(HWND hwnd, int idx) {
    auto* s = state_of(hwnd);
    if (s == nullptr || idx < 0 || idx >= 3) return;
    if (!s->tiles[idx].enabled) return;
    s->choice = s->tiles[idx].choice;
    PostMessageA(hwnd, WM_CLOSE, 0, 0);
}

void move_focus(HWND hwnd, int delta) {
    auto* s = state_of(hwnd);
    if (s == nullptr) return;
    // Skip disabled tiles when moving focus.
    int next = s->focus_index;
    for (int step = 0; step < 3; ++step) {
        next = (next + delta + 3) % 3;
        if (s->tiles[next].enabled) break;
    }
    if (next != s->focus_index) {
        s->focus_index = next;
        invalidate_all(hwnd);
    }
}

LRESULT CALLBACK launcher_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lp);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    DialogData* s = state_of(hwnd);

    switch (msg) {
        case WM_CREATE: {
            // Default focus to the first enabled tile.
            for (int i = 0; i < 3; ++i) {
                if (s->tiles[i].enabled) { s->focus_index = i; break; }
            }
            return 0;
        }

        case WM_ERASEBKGND:
            return 1;   // we paint everything in WM_PAINT

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            paint_dialog(hwnd, hdc);
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_MOUSEMOVE: {
            if (!s->mouse_tracking) {
                TRACKMOUSEEVENT tme{ sizeof(tme), TME_LEAVE, hwnd, HOVER_DEFAULT };
                TrackMouseEvent(&tme);
                s->mouse_tracking = true;
            }
            POINT p{ GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
            RECT  rc; GetClientRect(hwnd, &rc);
            int hit = hit_test(p, rc);
            if (hit != -1 && !s->tiles[hit].enabled) hit = -1;
            if (hit != s->hover_index) {
                s->hover_index = hit;
                invalidate_all(hwnd);
                // Hover also becomes the keyboard focus, so Enter behaves intuitively.
                if (hit != -1) { s->focus_index = hit; }
            }
            // Update cursor: hand over enabled tiles, arrow elsewhere.
            SetCursor(LoadCursorA(nullptr,
                MAKEINTRESOURCEA(hit != -1 ? 32649 /*IDC_HAND*/ : 32512 /*IDC_ARROW*/)));
            return 0;
        }

        case WM_MOUSELEAVE: {
            s->mouse_tracking = false;
            if (s->hover_index != -1) { s->hover_index = -1; invalidate_all(hwnd); }
            return 0;
        }

        case WM_LBUTTONDOWN: {
            POINT p{ GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
            RECT  rc; GetClientRect(hwnd, &rc);
            int hit = hit_test(p, rc);
            if (hit != -1 && s->tiles[hit].enabled) {
                s->pressed_idx = hit;
                s->focus_index = hit;
                SetCapture(hwnd);
                invalidate_all(hwnd);
            }
            return 0;
        }

        case WM_LBUTTONUP: {
            ReleaseCapture();
            POINT p{ GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
            RECT  rc; GetClientRect(hwnd, &rc);
            int hit = hit_test(p, rc);
            const int pressed = s->pressed_idx;
            s->pressed_idx = -1;
            invalidate_all(hwnd);
            if (pressed != -1 && hit == pressed && s->tiles[pressed].enabled) {
                confirm_choice(hwnd, pressed);
            }
            return 0;
        }

        case WM_KEYDOWN: {
            switch (wp) {
                case VK_ESCAPE:                  PostMessageA(hwnd, WM_CLOSE, 0, 0); return 0;
                case VK_RETURN: case VK_SPACE:   confirm_choice(hwnd, s->focus_index); return 0;
                case VK_TAB: {
                    const bool back = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
                    move_focus(hwnd, back ? -1 : +1);
                    return 0;
                }
                case VK_UP:   case VK_LEFT:      move_focus(hwnd, -1); return 0;
                case VK_DOWN: case VK_RIGHT:     move_focus(hwnd, +1); return 0;
            }
            break;
        }

        case WM_GETDLGCODE:
            // Consume Tab / arrows / Enter so DefWindowProc doesn't ring the bell.
            return DLGC_WANTALLKEYS | DLGC_WANTARROWS | DLGC_WANTCHARS | DLGC_WANTTAB;

        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;

        case WM_DESTROY:
            if (s->gdiplus_owner) Gdiplus::GdiplusShutdown(s->gdiplus_token);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

}  // namespace

BackendChoice show_startup_menu(const StartupOptions& opts) {
    DialogData s;
    auto vk = probe_vulkan();
    auto dx = probe_d3d12();

    s.tiles[0] = Tile{ BackendChoice::Vulkan, "Vulkan", vk.detail,
                       vk.available ? "AVAILABLE" : "UNAVAILABLE", vk.available };
    s.tiles[1] = Tile{ BackendChoice::D3D12,  "DirectX 12", dx.detail,
                       dx.available ? "AVAILABLE" : "COMING SOON", dx.available };
    s.tiles[2] = Tile{ BackendChoice::Auto,   "Auto-detect",
                       "Let Cardinal pick the best available backend",
                       "RECOMMENDED", true };

    s.header    = std::string(opts.app_title) + " — " + opts.subtitle;
    s.subheader = "GPU: " + detect_primary_gpu();

    if (opts.skip_when_no_options && !vk.available && !dx.available) {
        return BackendChoice::Auto;
    }

    // Initialise GDI+ for the lifetime of this dialog. If the host process
    // already started it (e.g. embedded use) we just no-op shutdown.
    Gdiplus::GdiplusStartupInput gdi_in;
    if (Gdiplus::GdiplusStartup(&s.gdiplus_token, &gdi_in, nullptr) == Gdiplus::Ok) {
        s.gdiplus_owner = true;
    }

    HINSTANCE hi = GetModuleHandleA(nullptr);
    WNDCLASSA wc{};
    wc.lpfnWndProc   = launcher_proc;
    wc.hInstance     = hi;
    wc.lpszClassName = "CardinalLauncherWnd";
    wc.hCursor       = LoadCursorA(nullptr, MAKEINTRESOURCEA(32512 /*IDC_ARROW*/));
    wc.hIcon         = LoadIconA  (nullptr, MAKEINTRESOURCEA(32512 /*IDI_APPLICATION*/));
    wc.hbrBackground = nullptr;     // we paint the bg
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    RegisterClassA(&wc);   // benign if already registered

    // Centre on the work area; account for window chrome via AdjustWindowRect.
    RECT scr{}; SystemParametersInfoA(SPI_GETWORKAREA, 0, &scr, 0);
    RECT win{ 0, 0, kWidth, kHeight };
    AdjustWindowRectEx(&win, WS_POPUP | WS_CAPTION | WS_SYSMENU, FALSE, WS_EX_DLGMODALFRAME);
    const int total_w = win.right - win.left;
    const int total_h = win.bottom - win.top;
    const int x = scr.left + ((scr.right  - scr.left) - total_w) / 2;
    const int y = scr.top  + ((scr.bottom - scr.top)  - total_h) / 2;

    HWND hwnd = CreateWindowExA(WS_EX_DLGMODALFRAME | WS_EX_TOPMOST,
        wc.lpszClassName, opts.app_title,
        WS_POPUP | WS_CAPTION | WS_SYSMENU,
        x, y, total_w, total_h, nullptr, nullptr, hi, &s);
    if (hwnd == nullptr) {
        if (s.gdiplus_owner) Gdiplus::GdiplusShutdown(s.gdiplus_token);
        return BackendChoice::Cancel;
    }

    ShowWindow(hwnd, SW_SHOW);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    MSG msg{};
    while (GetMessageA(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return s.choice;
}

}  // namespace cardinal::launcher

#endif  // CARDINAL_PLATFORM_WINDOWS
