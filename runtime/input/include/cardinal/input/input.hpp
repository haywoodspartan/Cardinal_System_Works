#pragma once

// =============================================================================
// Cardinal — Input system.
//
// One process-wide Manager owns:
//   - Raw keyboard / mouse / gamepad state (this frame + last frame)
//   - An ActionMap of high-level "Jump" / "Move" / "Fire" bindings
//   - The OS message hook that feeds it
//
// Three layers, smallest at the bottom:
//   1. Keys / buttons / axes  — physical inputs (KeyCode::W, MouseButton::Left)
//   2. Action bindings        — gameplay names ("Move.X" → A/D + LeftStick.X)
//   3. Per-frame query        — Manager::down("Jump"), Manager::axis("Move.X")
//
// State transitions exposed: just_pressed (rising edge), just_released
// (falling edge), held, value (for axes). Mouse delta is per-frame.
//
// The Win32 message hook lives in input_windows.cpp; on Linux we'll wire
// X11 events when WSI lands. Both feed the same Manager via push_*() so
// gameplay code is platform-agnostic.
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string/shared_ptr
#include <cardinal/core/std/containers.hpp>   // array/unordered_map/vector

namespace cardinal::input {

// ---------------------------------------------------------------------------
// Physical inputs.
//
// KeyCode covers ~100 keys — we use the SDL/Vulkan-friendly subset; full
// keyboard layout coverage is the integrator's job (extend the enum as
// you need country-specific keys).
// ---------------------------------------------------------------------------
enum class KeyCode : u32 {
    Unknown = 0,
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    K0, K1, K2, K3, K4, K5, K6, K7, K8, K9,
    Space, Tab, Enter, Escape, Backspace,
    LeftShift, RightShift, LeftCtrl, RightCtrl, LeftAlt, RightAlt,
    Up, Down, Left, Right,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    NumPad0, NumPad1, NumPad2, NumPad3, NumPad4,
    NumPad5, NumPad6, NumPad7, NumPad8, NumPad9,
    Minus, Equals, LBracket, RBracket, Semicolon, Quote, Comma, Period, Slash, Backslash,
    Count
};

const char* key_code_name(KeyCode k) noexcept;
KeyCode     key_code_from_string(const char* s) noexcept;

enum class MouseButton : u32 { Left = 0, Right = 1, Middle = 2, X1 = 3, X2 = 4, Count };

enum class GamepadButton : u32 {
    A, B, X, Y, LB, RB, Back, Start, LStick, RStick,
    DUp, DDown, DLeft, DRight, Count
};
enum class GamepadAxis : u32 {
    LeftStickX, LeftStickY, RightStickX, RightStickY,
    LeftTrigger, RightTrigger, Count
};

// ---------------------------------------------------------------------------
// Per-frame button state.
//
//   .down       : currently pressed
//   .pressed    : down this frame, NOT down last frame (rising edge)
//   .released   : up this frame, was down last frame (falling edge)
//   .held_time  : seconds since first press (0 when not down)
// ---------------------------------------------------------------------------
struct ButtonState {
    bool down{false};
    bool pressed{false};
    bool released{false};
    float held_time{0.0f};
};

// ---------------------------------------------------------------------------
// Mouse state, all in pixels relative to the window's client area.
// ---------------------------------------------------------------------------
struct MouseState {
    int  x{0}, y{0};
    int  dx{0}, dy{0};       // delta since last frame
    int  wheel{0};           // accumulated this frame
    cardinal::array<ButtonState, static_cast<usize>(MouseButton::Count)> buttons{};
};

// ---------------------------------------------------------------------------
// Action / Axis bindings. An action's value is the OR of every binding
// being pressed; an axis is the sum of all positive contributions minus
// the sum of all negative contributions, clamped to [-1, +1].
// ---------------------------------------------------------------------------
enum class BindKind : u32 { Key, MouseButton, GamepadButton, GamepadAxis };
struct Binding {
    BindKind kind{BindKind::Key};
    u32      code{0};            // KeyCode / MouseButton / GamepadButton / GamepadAxis as u32
    float    scale{1.0f};        // for axis: +1 / -1 / arbitrary
};

struct Action {
    cardinal::string          name;
    cardinal::vector<Binding> bindings;
};

struct AxisBinding {
    cardinal::string          name;
    cardinal::vector<Binding> contributions;
};

// ---------------------------------------------------------------------------
// Manager — process-wide. Use Manager::create() to construct, attach the
// OS hook via attach_to_window(), then call begin_frame() once per
// frame BEFORE the first action query so .pressed / .released are valid.
// ---------------------------------------------------------------------------
class Manager {
public:
    static cardinal::shared_ptr<Manager> create();
    ~Manager();
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;

    // ----- OS hookup -------------------------------------------------
    // Install a window message hook so OS events feed our state. Pass
    // the cardinal::window::Window* (we only treat it as void* here so
    // input doesn't link to window).
    void attach_to_window(void* window_ptr) noexcept;
    void detach_from_window() noexcept;

    // ----- Per-frame -------------------------------------------------
    // Call once per frame BEFORE you query anything. Advances state from
    // the OS-pushed deltas, recomputes .pressed/.released edges, decays
    // mouse delta + wheel.
    void begin_frame(float dt) noexcept;

    // ----- Direct queries -------------------------------------------
    const ButtonState& key  (KeyCode k)     const noexcept;
    const ButtonState& mouse(MouseButton b) const noexcept;
    const MouseState&  mouse_state()        const noexcept;

    bool  key_down  (KeyCode k)     const noexcept { return key(k).down; }
    bool  key_pressed (KeyCode k)   const noexcept { return key(k).pressed; }
    bool  key_released(KeyCode k)   const noexcept { return key(k).released; }
    bool  mouse_down  (MouseButton b) const noexcept { return mouse(b).down; }
    bool  mouse_pressed (MouseButton b) const noexcept { return mouse(b).pressed; }

    // ----- Action map -----------------------------------------------
    void  bind_action(const cardinal::string& name, Binding b);
    void  bind_axis  (const cardinal::string& name, Binding b);
    void  clear_action(const cardinal::string& name);
    void  clear_axis  (const cardinal::string& name);
    void  reset_bindings() noexcept;

    bool  action_down    (const cardinal::string& name) const;
    bool  action_pressed (const cardinal::string& name) const;
    bool  action_released(const cardinal::string& name) const;
    float axis(const cardinal::string& name) const;

    // ----- Gameplay gate (Play-In-Editor pattern) ------------------
    // When the studio is hosting an editor session, gameplay code shouldn't
    // see WASD / Jump / Fire. We mirror UE5's PIE behaviour: action_*() and
    // axis() return false / 0 unless gameplay is active.
    //
    // Direct key / mouse queries (key_down(), mouse_*(), mouse_state())
    // are NOT gated — Studio hotkeys (F5/F6/F8/ESC) live above this layer
    // and need to fire whether or not the game is running.
    //
    // Defaults to true so headless tests / non-Studio hosts get the prior
    // "always on" behaviour without opt-in.
    void  set_gameplay_active(bool active) noexcept;
    bool  is_gameplay_active() const noexcept;

    // Inspection — used by the Studio Input panel.
    cardinal::vector<cardinal::string>           action_names() const;
    cardinal::vector<cardinal::string>           axis_names()   const;
    const cardinal::vector<Binding>&        action_bindings(const cardinal::string& name) const;
    const cardinal::vector<Binding>&        axis_bindings  (const cardinal::string& name) const;

    // ----- Producer side (OS-hook) ---------------------------------
    // Called by the platform message pump. Idempotent — last-event-wins.
    void push_key(KeyCode k, bool down) noexcept;
    void push_mouse_button(MouseButton b, bool down) noexcept;
    void push_mouse_move(int x, int y) noexcept;
    void push_mouse_wheel(int delta) noexcept;

    // ----- Statistics for the Studio panel -------------------------
    struct Stats {
        u64 events_processed{0};
        u64 keys_down_now    {0};
        u64 mouse_buttons_down{0};
    };
    Stats stats() const noexcept;

private:
    Manager();
    struct Impl;
    Impl* impl_;
};

}  // namespace cardinal::input
