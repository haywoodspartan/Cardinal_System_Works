#pragma once
// =============================================================================
// Cardinal UI — the window system (Cardinal Slate's own, no ImGui).
//
//   Window        a framed, titled panel: drag the title bar to move, drag the
//                 bottom-right grip / right / bottom edge to resize, close and
//                 collapse buttons, a min size, and ONE content child that is
//                 clipped to the content area (DrawList clip stack).
//   WindowManager the desktop: owns windows in z-order (last child = front),
//                 brings the pressed window to the front, keeps floating
//                 windows on-screen and snaps them to its edges, and DOCKS
//                 windows into Left / Right / Top / Bottom strips (splitting a
//                 strip evenly among its windows). client_rect() is whatever
//                 the strips leave free — the host's 3D viewport.
//
// Everything is plain retained widgets: a Window works inside any tree, the
// manager is just a container, and all of it is testable without a GPU.
// Hosts drive it like any Ui: layout -> update_input -> paint; the manager
// re-sorts z-order from the Ui's active widget at paint time, so no extra
// per-frame call is needed.
// =============================================================================

#include "widget.hpp"
#include "context.hpp"

namespace cardinal::cui {

enum class DockSide : u8 { None = 0, Left, Right, Top, Bottom };

class WindowManager;

class Window : public Widget {
public:
    // `frame` is the floating rect (manager space: position + size).
    Window(cardinal::string title, Rect frame);

    // Content — a single logical child (usually a Stack / ScrollPanel).
    Widget* set_content(cardinal::unique_ptr<Widget> content);
    Widget* content() const noexcept { return content_; }

    const cardinal::string& title() const noexcept { return title_; }
    void set_title(cardinal::string t) { title_ = cardinal::move(t); }

    // Floating frame (ignored while docked; restored on undock).
    const Rect& frame() const noexcept { return frame_; }
    void set_frame(const Rect& r) noexcept { frame_ = r; }
    void set_min_size(Vec2 s) noexcept { min_size_ = s; }
    Vec2 min_size() const noexcept { return min_size_; }

    // Chrome toggles.
    void set_closable(bool v) noexcept    { closable_ = v; }
    void set_collapsible(bool v) noexcept { collapsible_ = v; }
    void set_resizable(bool v) noexcept   { resizable_ = v; }
    void set_movable(bool v) noexcept     { movable_ = v; }
    bool closable() const noexcept    { return closable_; }
    bool collapsible() const noexcept { return collapsible_; }
    bool resizable() const noexcept   { return resizable_; }
    bool movable() const noexcept     { return movable_; }

    // Collapsed = title bar only; the content is hidden (not painted, not hit).
    bool collapsed() const noexcept { return collapsed_; }
    void set_collapsed(bool c);

    // Close = hide (set_visible(false)) + callback. The window stays owned by
    // its parent so a host can re-show it (set_visible(true) + bring_to_front).
    void close();
    void set_on_close(cardinal::function<void()> f)    { on_close_ = cardinal::move(f); }
    // Fired on the press that starts any interaction with the chrome.
    void set_on_activate(cardinal::function<void()> f) { on_activate_ = cardinal::move(f); }

    DockSide dock_side() const noexcept { return dock_side_; }
    bool     docked() const noexcept    { return dock_side_ != DockSide::None; }

    // Chrome geometry (valid after arrange). Content rect is the frame minus
    // the title bar and border; collapsed windows have an empty content rect.
    Rect title_rect()    const noexcept;
    Rect content_rect()  const noexcept;
    Rect close_rect()    const noexcept;
    Rect collapse_rect() const noexcept;
    Rect grip_rect()     const noexcept;
    float title_height() const noexcept { return title_h_; }

    // True when `w` is this window or lives in its subtree.
    bool contains_widget(const Widget* w) const noexcept;

    // --- Widget ---------------------------------------------------------
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_drag(Vec2 mouse) override;
    void on_click() override;

private:
    friend class WindowManager;
    enum class Drag : u8 { None, Press, Move, ResizeRight, ResizeBottom, ResizeCorner };
    Drag classify_(Vec2 p) const noexcept;
    void apply_content_visibility_();

    cardinal::string title_;
    Rect  frame_;
    Vec2  min_size_{120.0f, 60.0f};
    float title_h_{24.0f};
    float font_size_{14.0f};
    float border_{1.0f};
    float grip_{12.0f};
    float edge_{6.0f};
    bool  closable_{true};
    bool  collapsible_{true};
    bool  resizable_{true};
    bool  movable_{true};
    bool  collapsed_{false};
    DockSide dock_side_{DockSide::None};
    u32   dock_order_{0};           // strip slot order (stable across z-order changes)
    Rect  floating_frame_{};        // saved while docked
    Widget* content_{nullptr};
    WindowManager* manager_{nullptr};
    // Drag state (reset when the Ui releases us — see paint()).
    bool  drag_valid_{false};
    Drag  drag_{Drag::None};
    Vec2  press_{};
    Rect  start_frame_{};
    cardinal::function<void()> on_close_;
    cardinal::function<void()> on_activate_;
};

class WindowManager : public Widget {
public:
    WindowManager() = default;

    // Adds a floating window on top. Returns the raw pointer (manager owns it).
    Window* add_window(cardinal::unique_ptr<Window> w);
    // Detach a window (returns ownership; null if not ours). Clear the Ui's
    // focus / popup first if they point into it.
    cardinal::unique_ptr<Window> remove_window(Window* w);

    cardinal::vector<Window*> windows() const;       // back-to-front
    Window* front() const noexcept;                  // topmost floating/docked window
    Window* active() const noexcept { return active_; }
    void    bring_to_front(Window* w);

    // Docking. Thickness = strip width (Left/Right) or height (Top/Bottom).
    void  dock(Window* w, DockSide side);
    void  undock(Window* w);
    void  set_dock_thickness(DockSide side, float px);
    float dock_thickness(DockSide side) const noexcept;
    Rect  dock_rect(DockSide side) const noexcept;   // strip rect (after arrange)
    // The area no docked strip covers — the host's viewport.
    Rect  client_rect() const noexcept { return client_; }

    // Floating windows snap to the manager's edges within this distance.
    void  set_snap_distance(float px) noexcept { snap_ = px; }
    float snap_distance() const noexcept { return snap_; }

    // --- Widget ---------------------------------------------------------
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override;

private:
    friend class Window;
    void  on_window_pressed_(Window* w);
    void  sync_z_order_(const Ui& ui);
    Rect  clamp_floating_(Rect f) const noexcept;

    float thickness_[5] { 0.0f, 260.0f, 260.0f, 120.0f, 180.0f };   // indexed by DockSide
    Rect  strips_[5]{};
    Rect  client_{};
    float snap_{10.0f};
    Window* active_{nullptr};
    const Widget* last_ui_active_{nullptr};
    u32   dock_seq_{0};
};

}  // namespace cardinal::cui
