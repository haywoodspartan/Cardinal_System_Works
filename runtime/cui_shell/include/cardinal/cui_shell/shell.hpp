#pragma once
// =============================================================================
// Cardinal Studio native shell — the editor chrome on Cardinal Slate (cui),
// with no ImGui anywhere.
//
//   Shell = menu bar + cui::WindowManager desktop + the first real editor
//   panels, all bound to a scene::Scene:
//     Hierarchy  (Left dock)    TreeView of entities (parent_id grouping)
//     Inspector  (Right dock)   name, position / rotation / scale, visible,
//                               tint — two-way synced NumberFields
//     Console    (Bottom dock)  cardinal::console commands + autocomplete
//     Log        (Bottom dock)  live cardinal::log sink
//     Stats      (floating)     three host-fed lines (fps, backend, ...)
//   viewport_rect() is the manager's client rect — the host renders the 3D
//   view there. The shell is GPU-free: hosts hand its DrawList to any
//   renderer (cui_rhi native, or the ImGui bridge), and tests drive it
//   headless with synthetic InputState frames.
//
// Frame contract (same as a bare cui::Ui):
//     shell.layout({w, h});
//     shell.update(make_input_state(*input));   // input + panel sync
//     shell.paint(draw_list);
// =============================================================================

#include <cardinal/cui/context.hpp>
#include <cardinal/cui/widgets.hpp>
#include <cardinal/cui/bind.hpp>
#include <cardinal/cui/window.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/input/input.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/thread.hpp>   // cardinal::mutex / lock_guard

namespace cardinal::cui_shell {

struct ShellDesc {
    float left_dock{280.0f};     // Hierarchy strip width
    float right_dock{320.0f};    // Inspector strip width
    float bottom_dock{220.0f};   // Console + Log strip height
    bool  stats_window{true};
};

// Translate the engine's per-frame input into a cui snapshot. Characters are
// synthesised from key press edges + shift (the window layer has no WM_CHAR
// feed yet), so typing works for the ASCII set the console needs.
cui::InputState make_input_state(const input::Manager& in);

class Shell {
public:
    enum class Panel : u8 { Hierarchy = 0, Inspector, Console, Log, Stats, Count };

    explicit Shell(scene::Scene& scene, const ShellDesc& desc = {});
    ~Shell();
    Shell(const Shell&)            = delete;
    Shell& operator=(const Shell&) = delete;

    // ---- Frame ---------------------------------------------------------
    void layout(cui::Vec2 size);
    void update(const cui::InputState& in);
    void paint(cui::DrawList& dl);

    cui::Ui&            ui() noexcept        { return ui_; }
    cui::WindowManager& windows() noexcept   { return *wm_; }
    cui::MenuBar&       menu_bar() noexcept  { return *menu_; }

    // Where the host draws the 3D view (nothing docked covers it).
    cui::Rect viewport_rect() const noexcept { return wm_->client_rect(); }
    // True when the cursor is over the viewport and no widget claims it —
    // hosts gate camera navigation on this.
    bool mouse_over_viewport(cui::Vec2 p) const noexcept;
    // True while a text field owns the keyboard (hosts suppress hotkeys).
    bool wants_keyboard() const noexcept { return ui_.focused() != nullptr; }

    // ---- Menus ---------------------------------------------------------
    int  add_menu(cardinal::string title);
    void add_menu_item(int menu, cardinal::string label, cardinal::function<void()> fn);
    // "View" menu with a toggle per panel.
    int  add_view_menu();

    // ---- Panels --------------------------------------------------------
    cui::Window* window(Panel p) const noexcept;
    void show_panel(Panel p, bool show);
    void toggle_panel(Panel p);

    // ---- Hierarchy + selection ------------------------------------------
    void refresh_hierarchy();                 // forced rebuild (auto on change)
    u32  selected_entity() const noexcept { return selected_; }
    void select_entity(u32 id);
    cui::TreeView& hierarchy() noexcept { return *tree_; }

    // ---- Inspector -----------------------------------------------------
    // axis 0..2. Rotation fields are in DEGREES (entity stores radians).
    cui::NumberField<float>* position_field(int axis) noexcept;
    cui::NumberField<float>* rotation_field(int axis) noexcept;
    cui::NumberField<float>* scale_field(int axis) noexcept;

    // ---- Console + log -------------------------------------------------
    void console_execute(const cardinal::string& line);
    void console_print(cardinal::string line);
    void console_accept_suggestion();         // Tab: take the first suggestion
    cui::TextField& console_input() noexcept { return *con_in_; }
    const cardinal::vector<cardinal::string>& console_lines() const noexcept { return con_lines_; }
    const cardinal::vector<cardinal::string>& log_lines() const noexcept { return log_lines_; }

    // ---- Stats ---------------------------------------------------------
    void set_stats(cardinal::string line0, cardinal::string line1, cardinal::string line2);

private:
    struct LogTap final : cardinal::log::Sink {
        cardinal::mutex                     mtx;
        cardinal::vector<cardinal::string>  pending;
        void on_message(cardinal::log::Level lvl, const char* category, const char* msg) override;
    };
    struct Bound {                 // one two-way bound float
        float local{0.0f};
        float snap{0.0f};
        cui::NumberField<float>* field{nullptr};
    };

    void build_();
    void build_hierarchy_window_(cui::Window& w);
    void build_inspector_window_(cui::Window& w);
    void build_console_window_(cui::Window& w);
    void build_log_window_(cui::Window& w);
    void build_stats_window_(cui::Window& w);
    void sync_popup_();
    void sync_hierarchy_();
    void sync_inspector_();
    void drain_log_();
    void refresh_console_view_();
    u64  hierarchy_signature_() const noexcept;

    scene::Scene&       scene_;
    ShellDesc           desc_;
    cui::Ui             ui_;
    cui::MenuBar*       menu_{nullptr};
    cui::WindowManager* wm_{nullptr};
    cui::Window*        panels_[static_cast<usize>(Panel::Count)]{};
    cardinal::vector<cardinal::vector<cardinal::function<void()>>> menu_items_;
    int                 view_menu_{-1};

    // Hierarchy.
    cui::TreeView*            tree_{nullptr};
    cardinal::vector<u32>     handle_to_id_;
    u64                       hier_sig_{0};
    u32                       selected_{0};
    bool                      selecting_{false};

    // Inspector.
    cui::Label*    insp_name_{nullptr};
    cui::Widget*   insp_body_{nullptr};
    cui::Label*    insp_empty_{nullptr};
    Bound          pos_[3], rot_[3], scl_[3], tint_[3];
    bool           vis_local_{true}, vis_snap_{true};
    cui::Checkbox* vis_box_{nullptr};
    u32            insp_entity_{0};

    // Console.
    cui::ScrollPanel*                  con_scroll_{nullptr};
    cui::ListView*                     con_list_{nullptr};
    cui::ListView*                     con_sug_{nullptr};
    cui::TextField*                    con_in_{nullptr};
    cardinal::vector<cardinal::string> con_lines_;

    // Log.
    LogTap                             log_tap_;
    cui::ScrollPanel*                  log_scroll_{nullptr};
    cui::ListView*                     log_list_{nullptr};
    cardinal::vector<cardinal::string> log_lines_;

    // Stats.
    cui::Label* stats_[3]{};
};

}  // namespace cardinal::cui_shell
