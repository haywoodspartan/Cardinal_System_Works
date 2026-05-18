// =============================================================================
// Cardinal — Win32 input backend.
//
// Translates raw WM_KEYDOWN / WM_LBUTTONDOWN / WM_MOUSEMOVE / WM_MOUSEWHEEL
// into Manager::push_*() calls. The integrator wires us in by installing a
// raw message hook on cardinal::window::Window:
//
//   manager.attach_to_window(window.get());
//   window.set_message_hook(cardinal::input::win32_message_hook, manager.get());
// =============================================================================
#include <cardinal/input/input.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <Windows.h>
#include <windowsx.h>     // GET_X_LPARAM / GET_Y_LPARAM

namespace cardinal::input {

namespace {

KeyCode vk_to_keycode(WPARAM vk) noexcept {
    if (vk >= 'A' && vk <= 'Z') return static_cast<KeyCode>(static_cast<u32>(KeyCode::A) + (vk - 'A'));
    if (vk >= '0' && vk <= '9') return static_cast<KeyCode>(static_cast<u32>(KeyCode::K0) + (vk - '0'));
    switch (vk) {
        case VK_SPACE:    return KeyCode::Space;
        case VK_TAB:      return KeyCode::Tab;
        case VK_RETURN:   return KeyCode::Enter;
        case VK_ESCAPE:   return KeyCode::Escape;
        case VK_BACK:     return KeyCode::Backspace;
        case VK_LSHIFT:   return KeyCode::LeftShift;  case VK_RSHIFT:   return KeyCode::RightShift;
        case VK_LCONTROL: return KeyCode::LeftCtrl;   case VK_RCONTROL: return KeyCode::RightCtrl;
        case VK_LMENU:    return KeyCode::LeftAlt;    case VK_RMENU:    return KeyCode::RightAlt;
        case VK_SHIFT:    return KeyCode::LeftShift;  // generic — we collapse to Left
        case VK_CONTROL:  return KeyCode::LeftCtrl;
        case VK_MENU:     return KeyCode::LeftAlt;
        case VK_UP:       return KeyCode::Up;
        case VK_DOWN:     return KeyCode::Down;
        case VK_LEFT:     return KeyCode::Left;
        case VK_RIGHT:    return KeyCode::Right;
        case VK_F1:  return KeyCode::F1;  case VK_F2:  return KeyCode::F2;
        case VK_F3:  return KeyCode::F3;  case VK_F4:  return KeyCode::F4;
        case VK_F5:  return KeyCode::F5;  case VK_F6:  return KeyCode::F6;
        case VK_F7:  return KeyCode::F7;  case VK_F8:  return KeyCode::F8;
        case VK_F9:  return KeyCode::F9;  case VK_F10: return KeyCode::F10;
        case VK_F11: return KeyCode::F11; case VK_F12: return KeyCode::F12;
        case VK_NUMPAD0: return KeyCode::NumPad0; case VK_NUMPAD1: return KeyCode::NumPad1;
        case VK_NUMPAD2: return KeyCode::NumPad2; case VK_NUMPAD3: return KeyCode::NumPad3;
        case VK_NUMPAD4: return KeyCode::NumPad4; case VK_NUMPAD5: return KeyCode::NumPad5;
        case VK_NUMPAD6: return KeyCode::NumPad6; case VK_NUMPAD7: return KeyCode::NumPad7;
        case VK_NUMPAD8: return KeyCode::NumPad8; case VK_NUMPAD9: return KeyCode::NumPad9;
        case VK_OEM_MINUS:  return KeyCode::Minus;
        case VK_OEM_PLUS:   return KeyCode::Equals;
        case VK_OEM_4:      return KeyCode::LBracket;
        case VK_OEM_6:      return KeyCode::RBracket;
        case VK_OEM_1:      return KeyCode::Semicolon;
        case VK_OEM_7:      return KeyCode::Quote;
        case VK_OEM_COMMA:  return KeyCode::Comma;
        case VK_OEM_PERIOD: return KeyCode::Period;
        case VK_OEM_2:      return KeyCode::Slash;
        case VK_OEM_5:      return KeyCode::Backslash;
        default: return KeyCode::Unknown;
    }
}

}  // namespace

// Public entry — install via window->set_message_hook(win32_message_hook, manager_ptr).
extern "C" bool win32_message_hook(void* /*hwnd*/, u32 msg, u64 wparam, i64 lparam, void* user) {
    auto* mgr = static_cast<Manager*>(user);
    if (mgr == nullptr) return false;
    switch (msg) {
        case WM_KEYDOWN: case WM_SYSKEYDOWN: {
            const KeyCode k = vk_to_keycode(static_cast<WPARAM>(wparam));
            if (k != KeyCode::Unknown) mgr->push_key(k, true);
            return false;   // let ImGui see it too
        }
        case WM_KEYUP: case WM_SYSKEYUP: {
            const KeyCode k = vk_to_keycode(static_cast<WPARAM>(wparam));
            if (k != KeyCode::Unknown) mgr->push_key(k, false);
            return false;
        }
        case WM_LBUTTONDOWN: mgr->push_mouse_button(MouseButton::Left,   true);  return false;
        case WM_LBUTTONUP:   mgr->push_mouse_button(MouseButton::Left,   false); return false;
        case WM_RBUTTONDOWN: mgr->push_mouse_button(MouseButton::Right,  true);  return false;
        case WM_RBUTTONUP:   mgr->push_mouse_button(MouseButton::Right,  false); return false;
        case WM_MBUTTONDOWN: mgr->push_mouse_button(MouseButton::Middle, true);  return false;
        case WM_MBUTTONUP:   mgr->push_mouse_button(MouseButton::Middle, false); return false;
        case WM_XBUTTONDOWN: case WM_XBUTTONUP: {
            const u32 which = GET_XBUTTON_WPARAM(static_cast<WPARAM>(wparam));
            const MouseButton b = (which == XBUTTON1) ? MouseButton::X1 : MouseButton::X2;
            mgr->push_mouse_button(b, msg == WM_XBUTTONDOWN);
            return false;
        }
        case WM_MOUSEMOVE: {
            const int x = static_cast<int>(GET_X_LPARAM(static_cast<LPARAM>(lparam)));
            const int y = static_cast<int>(GET_Y_LPARAM(static_cast<LPARAM>(lparam)));
            mgr->push_mouse_move(x, y);
            return false;
        }
        case WM_MOUSEWHEEL: {
            const short z = static_cast<short>(HIWORD(static_cast<DWORD>(wparam)));
            mgr->push_mouse_wheel(z / WHEEL_DELTA);
            return false;
        }
    }
    return false;
}

}  // namespace cardinal::input

#endif  // CARDINAL_PLATFORM_WINDOWS
