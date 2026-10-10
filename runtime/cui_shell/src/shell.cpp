// =============================================================================
// Cardinal Studio native shell — implementation. See shell.hpp.
// =============================================================================
#include <cardinal/cui_shell/shell.hpp>
#include <cardinal/console/console.hpp>
#include <cardinal/core/std/cmath.hpp>

#include <cstdio>   // std::snprintf

namespace cardinal::cui_shell {

namespace {

constexpr float kRadToDeg = 57.29577951308232f;
constexpr float kDegToRad = 0.017453292519943295f;
constexpr usize kMaxConsoleLines = 400;
constexpr usize kMaxLogLines     = 500;

cardinal::string fmt_id(u32 id) {
    char b[16];
    std::snprintf(b, sizeof(b), "#%u", id);
    return cardinal::string(b);
}

}  // namespace

// -----------------------------------------------------------------------------
// Input translation
// -----------------------------------------------------------------------------
cui::InputState make_input_state(const input::Manager& in) {
    using K = input::KeyCode;
    cui::InputState s;
    const auto& ms = in.mouse_state();
    s.mouse      = { static_cast<float>(ms.x), static_cast<float>(ms.y) };
    s.mouse_down = in.mouse(input::MouseButton::Left).down;
    s.scroll     = static_cast<float>(ms.wheel);   // notches; + = wheel up

    const bool shift = in.key_down(K::LeftShift) || in.key_down(K::RightShift);
    const bool ctrl  = in.key_down(K::LeftCtrl)  || in.key_down(K::RightCtrl);
    if (!ctrl) {
        for (u32 i = 0; i < 26; ++i) {
            const K k = static_cast<K>(static_cast<u32>(K::A) + i);
            if (in.key_pressed(k)) s.text += static_cast<char>((shift ? 'A' : 'a') + i);
        }
        static constexpr char kDigitShift[10] = { ')', '!', '@', '#', '$', '%', '^', '&', '*', '(' };
        for (u32 i = 0; i < 10; ++i) {
            const K k = static_cast<K>(static_cast<u32>(K::K0) + i);
            if (in.key_pressed(k)) s.text += shift ? kDigitShift[i] : static_cast<char>('0' + i);
        }
        for (u32 i = 0; i < 10; ++i) {
            const K k = static_cast<K>(static_cast<u32>(K::NumPad0) + i);
            if (in.key_pressed(k)) s.text += static_cast<char>('0' + i);
        }
        struct Sym { K k; char lo; char hi; };
        static constexpr Sym kSyms[] = {
            { K::Space, ' ', ' ' },   { K::Minus, '-', '_' },     { K::Equals, '=', '+' },
            { K::LBracket, '[', '{' }, { K::RBracket, ']', '}' }, { K::Semicolon, ';', ':' },
            { K::Quote, '\'', '"' },   { K::Comma, ',', '<' },    { K::Period, '.', '>' },
            { K::Slash, '/', '?' },    { K::Backslash, '\\', '|' },
        };
        for (const Sym& sym : kSyms)
            if (in.key_pressed(sym.k)) s.text += shift ? sym.hi : sym.lo;
    }
    s.key_backspace = in.key_pressed(K::Backspace);
    s.key_left      = in.key_pressed(K::Left);
    s.key_right     = in.key_pressed(K::Right);
    s.key_enter     = in.key_pressed(K::Enter);
    s.key_escape    = in.key_pressed(K::Escape);
    return s;
}

// -----------------------------------------------------------------------------
// Log sink
// -----------------------------------------------------------------------------
void Shell::LogTap::on_message(cardinal::log::Level lvl, const char* category, const char* msg) {
    cardinal::string line = "[";
    line += cardinal::log::level_name(lvl);
    line += "] [";
    line += category ? category : "";
    line += "] ";
    line += msg ? msg : "";
    cardinal::lock_guard<cardinal::mutex> lg(mtx);
    pending.push_back(cardinal::move(line));
    if (pending.size() > kMaxLogLines) pending.erase(pending.begin());
}

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
Shell::Shell(scene::Scene& scene, const ShellDesc& desc)
    : scene_(scene), desc_(desc) {
    build_();
    cardinal::log::add_sink(&log_tap_);
}

Shell::~Shell() {
    cardinal::log::remove_sink(&log_tap_);
    ui_.clear_focus();
    ui_.set_popup(nullptr);
}

void Shell::build_() {
    using namespace cui;
    auto root = cardinal::make_unique<Stack>(Axis::Vertical, 0.0f);
    menu_ = static_cast<MenuBar*>(root->add(cardinal::make_unique<MenuBar>()));
    menu_->set_on_item([this](int m, int i) {
        if (m < 0 || static_cast<usize>(m) >= menu_items_.size()) return;
        auto& items = menu_items_[static_cast<usize>(m)];
        if (i < 0 || static_cast<usize>(i) >= items.size()) return;
        if (items[static_cast<usize>(i)]) items[static_cast<usize>(i)]();
    });

    auto wm = cardinal::make_unique<WindowManager>();
    wm_ = wm.get();
    wm_->set_dock_thickness(DockSide::Left,   desc_.left_dock);
    wm_->set_dock_thickness(DockSide::Right,  desc_.right_dock);
    wm_->set_dock_thickness(DockSide::Bottom, desc_.bottom_dock);

    auto make = [&](Panel p, const char* title, const char* id, Rect frame, DockSide side) {
        auto w = cardinal::make_unique<Window>(title, frame);
        w->set_id(id);
        Window* raw = wm_->add_window(cardinal::move(w));
        if (side != DockSide::None) wm_->dock(raw, side);
        panels_[static_cast<usize>(p)] = raw;
        return raw;
    };
    Window* hier = make(Panel::Hierarchy, "Hierarchy", "shell.hierarchy",
                        { {0.0f, 0.0f}, {desc_.left_dock, 400.0f} }, DockSide::Left);
    Window* insp = make(Panel::Inspector, "Inspector", "shell.inspector",
                        { {0.0f, 0.0f}, {desc_.right_dock, 400.0f} }, DockSide::Right);
    Window* con  = make(Panel::Console, "Console", "shell.console",
                        { {0.0f, 0.0f}, {600.0f, desc_.bottom_dock} }, DockSide::Bottom);
    Window* lg   = make(Panel::Log, "Log", "shell.log",
                        { {0.0f, 0.0f}, {600.0f, desc_.bottom_dock} }, DockSide::Bottom);
    Window* st   = make(Panel::Stats, "Stats", "shell.stats",
                        { {desc_.left_dock + 16.0f, 40.0f}, {280.0f, 96.0f} }, DockSide::None);
    build_hierarchy_window_(*hier);
    build_inspector_window_(*insp);
    build_console_window_(*con);
    build_log_window_(*lg);
    build_stats_window_(*st);
    st->set_resizable(false);
    if (!desc_.stats_window) st->set_visible(false);

    root->add_flex(cardinal::move(wm), 1.0f);
    ui_.set_root(cardinal::move(root));
    refresh_hierarchy();
}

void Shell::build_hierarchy_window_(cui::Window& w) {
    using namespace cui;
    auto scroll = cardinal::make_unique<ScrollPanel>(0.0f);
    tree_ = static_cast<TreeView*>(scroll->add(cardinal::make_unique<TreeView>()));
    tree_->set_on_select([this](int h) {
        if (selecting_) return;
        if (h < 0 || static_cast<usize>(h) >= handle_to_id_.size()) { selected_ = 0; return; }
        selected_ = handle_to_id_[static_cast<usize>(h)];
    });
    w.set_content(cardinal::move(scroll));
}

void Shell::build_inspector_window_(cui::Window& w) {
    using namespace cui;
    auto panel = cardinal::make_unique<cui::Panel>(ui_.theme().panel_bg, Edges::all(8.0f));
    auto stack = cardinal::make_unique<Stack>(Axis::Vertical, 6.0f);
    insp_empty_ = static_cast<Label*>(stack->add(cardinal::make_unique<Label>("No selection")));

    auto body = cardinal::make_unique<Stack>(Axis::Vertical, 6.0f);
    insp_name_ = static_cast<Label*>(body->add(cardinal::make_unique<Label>("-")));
    auto row3 = [&](const char* caption, Bound* b, float lo, float hi) {
        body->add(cardinal::make_unique<Label>(caption));
        auto row = cardinal::make_unique<Stack>(Axis::Horizontal, 4.0f);
        for (int i = 0; i < 3; ++i) {
            auto f = cardinal::make_unique<NumberField<float>>(&b[i].local, lo, hi, false, 80.0f);
            b[i].field = static_cast<NumberField<float>*>(row->add(cardinal::move(f)));
        }
        body->add(cardinal::move(row));
    };
    row3("Position",       pos_, -1.0e6f, 1.0e6f);
    row3("Rotation (deg)", rot_, -3.6e4f, 3.6e4f);
    row3("Scale",          scl_, -1.0e4f, 1.0e4f);
    vis_box_ = static_cast<Checkbox*>(body->add(cardinal::make_unique<Checkbox>("Visible", &vis_local_)));
    body->add(cardinal::make_unique<Label>("Tint (r, g, b)"));
    for (int i = 0; i < 3; ++i)
        body->add(cardinal::make_unique<Slider>(&tint_[i].local, 0.0f, 1.0f));
    insp_body_ = stack->add(cardinal::move(body));
    insp_body_->set_visible(false);

    panel->add(cardinal::move(stack));
    w.set_content(cardinal::move(panel));
}

void Shell::build_console_window_(cui::Window& w) {
    using namespace cui;
    auto panel  = cardinal::make_unique<cui::Panel>(ui_.theme().panel_bg, Edges::all(6.0f));
    auto stack  = cardinal::make_unique<Stack>(Axis::Vertical, 4.0f);
    auto scroll = cardinal::make_unique<ScrollPanel>(0.0f);
    con_list_   = static_cast<ListView*>(scroll->add(cardinal::make_unique<ListView>()));
    con_scroll_ = static_cast<ScrollPanel*>(stack->add_flex(cardinal::move(scroll), 1.0f));
    con_sug_    = static_cast<ListView*>(stack->add(cardinal::make_unique<ListView>()));
    con_sug_->set_visible(false);
    auto field  = cardinal::make_unique<TextField>(cardinal::string{}, 200.0f);
    field->set_placeholder("command (Tab completes, Enter runs)...");
    con_in_     = static_cast<TextField*>(stack->add(cardinal::move(field)));
    panel->add(cardinal::move(stack));
    w.set_content(cardinal::move(panel));

    con_in_->set_on_change([this](const cardinal::string& t) {
        if (t.empty()) { con_sug_->clear_items(); con_sug_->set_visible(false); return; }
        auto s = console::Registry::instance().complete(t, 8);
        const bool have = !s.empty();
        con_sug_->set_items(cardinal::move(s));
        con_sug_->set_visible(have);
    });
    con_sug_->set_on_select([this](int i) {
        if (i < 0 || static_cast<usize>(i) >= con_sug_->items().size()) return;
        con_in_->set_text(con_sug_->items()[static_cast<usize>(i)]);
        con_in_->set_caret(con_in_->text().size());
    });
    con_in_->set_on_submit([this](const cardinal::string& line) {
        if (line.empty()) return;
        console_execute(line);
        con_in_->set_text(cardinal::string{});
        con_in_->set_caret(0);
        con_sug_->clear_items();
        con_sug_->set_visible(false);
    });
    con_lines_.push_back("Cardinal Studio (native) console - type a command, Tab completes");
    refresh_console_view_();
}

void Shell::build_log_window_(cui::Window& w) {
    using namespace cui;
    auto panel  = cardinal::make_unique<cui::Panel>(ui_.theme().panel_bg, Edges::all(6.0f));
    auto scroll = cardinal::make_unique<ScrollPanel>(0.0f);
    log_list_   = static_cast<ListView*>(scroll->add(cardinal::make_unique<ListView>()));
    log_scroll_ = static_cast<ScrollPanel*>(panel->add(cardinal::move(scroll)));
    w.set_content(cardinal::move(panel));
}

void Shell::build_stats_window_(cui::Window& w) {
    using namespace cui;
    auto panel = cardinal::make_unique<cui::Panel>(ui_.theme().panel_bg, Edges::all(8.0f));
    auto stack = cardinal::make_unique<Stack>(Axis::Vertical, 4.0f);
    for (auto& s : stats_)
        s = static_cast<Label*>(stack->add(cardinal::make_unique<Label>("-")));
    panel->add(cardinal::move(stack));
    w.set_content(cardinal::move(panel));
}

// -----------------------------------------------------------------------------
// Frame
// -----------------------------------------------------------------------------
void Shell::layout(cui::Vec2 size) { ui_.layout(size); }

void Shell::sync_popup_() {
    ui_.set_popup(menu_->open_menu() >= 0 ? static_cast<cui::Widget*>(menu_) : nullptr);
}

void Shell::update(const cui::InputState& in) {
    sync_popup_();
    ui_.update_input(in);
    sync_popup_();
    sync_hierarchy_();
    sync_inspector_();
    drain_log_();
}

void Shell::paint(cui::DrawList& dl) { ui_.paint(dl); }

bool Shell::mouse_over_viewport(cui::Vec2 p) const noexcept {
    if (!wm_->client_rect().contains(p)) return false;
    if (menu_->open_menu() >= 0) return false;      // dropdown owns the cursor
    if (ui_.active() != nullptr) return false;      // mid-drag on a widget
    cui::Widget* root = ui_.root();
    return root == nullptr || root->hit_test(p) == nullptr;   // nothing claims p
}

// -----------------------------------------------------------------------------
// Menus
// -----------------------------------------------------------------------------
int Shell::add_menu(cardinal::string title) {
    const int m = menu_->add_menu(cardinal::move(title));
    while (menu_items_.size() <= static_cast<usize>(m)) menu_items_.emplace_back();
    return m;
}

void Shell::add_menu_item(int menu, cardinal::string label, cardinal::function<void()> fn) {
    if (menu < 0) return;
    while (menu_items_.size() <= static_cast<usize>(menu)) menu_items_.emplace_back();
    menu_->add_item(menu, cardinal::move(label));
    menu_items_[static_cast<usize>(menu)].push_back(cardinal::move(fn));
}

int Shell::add_view_menu() {
    if (view_menu_ >= 0) return view_menu_;
    view_menu_ = add_menu("View");
    static constexpr const char* kNames[] = { "Hierarchy", "Inspector", "Console", "Log", "Stats" };
    for (usize i = 0; i < static_cast<usize>(Panel::Count); ++i) {
        const Panel p = static_cast<Panel>(i);
        add_menu_item(view_menu_, kNames[i], [this, p] { toggle_panel(p); });
    }
    return view_menu_;
}

// -----------------------------------------------------------------------------
// Panels
// -----------------------------------------------------------------------------
cui::Window* Shell::window(Panel p) const noexcept {
    const usize i = static_cast<usize>(p);
    return i < static_cast<usize>(Panel::Count) ? panels_[i] : nullptr;
}

void Shell::show_panel(Panel p, bool show) {
    cui::Window* w = window(p);
    if (w == nullptr) return;
    if (!show) {
        // Drop dangling Ui pointers into the hidden subtree first.
        if (w->contains_widget(ui_.focused())) ui_.clear_focus();
        w->set_visible(false);
        return;
    }
    w->set_visible(true);
    w->set_collapsed(false);
    wm_->bring_to_front(w);
}

void Shell::toggle_panel(Panel p) {
    cui::Window* w = window(p);
    if (w != nullptr) show_panel(p, !w->visible());
}

// -----------------------------------------------------------------------------
// Hierarchy
// -----------------------------------------------------------------------------
u64 Shell::hierarchy_signature_() const noexcept {
    u64 h = 1469598103934665603ull;
    auto mix = [&h](u64 v) { h ^= v; h *= 1099511628211ull; };
    mix(scene_.entities().size());
    for (const auto& e : scene_.entities()) {
        mix(e.id);
        mix(static_cast<u64>(e.parent_id) << 32);
        for (char c : e.name) mix(static_cast<u8>(c));
        mix(0xABCDu);
    }
    return h;
}

void Shell::refresh_hierarchy() {
    tree_->clear();
    handle_to_id_.clear();
    const auto& ents = scene_.entities();
    // Depth-first from the roots. parent_id == 0 (or unknown) = root.
    auto exists = [&](u32 id) {
        if (id == 0) return false;
        for (const auto& e : ents) if (e.id == id) return true;
        return false;
    };
    cardinal::vector<bool> placed(ents.size(), false);
    auto add_children = [&](auto& self, u32 parent_id, int parent_handle, int depth) -> void {
        if (depth > 64) return;
        for (usize i = 0; i < ents.size(); ++i) {
            if (placed[i]) continue;
            const auto& e = ents[i];
            const bool is_root = (e.parent_id == 0 || !exists(e.parent_id));
            if (parent_id == 0 ? !is_root : e.parent_id != parent_id) continue;
            placed[i] = true;
            const int h = tree_->add_node(parent_handle, e.name + "  " + fmt_id(e.id), true);
            if (static_cast<usize>(h) >= handle_to_id_.size()) handle_to_id_.resize(static_cast<usize>(h) + 1, 0u);
            handle_to_id_[static_cast<usize>(h)] = e.id;
            self(self, e.id, h, depth + 1);
        }
    };
    add_children(add_children, 0u, -1, 0);
    // Anything left (cycles) goes top-level.
    for (usize i = 0; i < ents.size(); ++i) {
        if (placed[i]) continue;
        const int h = tree_->add_node(-1, ents[i].name + "  " + fmt_id(ents[i].id), true);
        if (static_cast<usize>(h) >= handle_to_id_.size()) handle_to_id_.resize(static_cast<usize>(h) + 1, 0u);
        handle_to_id_[static_cast<usize>(h)] = ents[i].id;
    }
    hier_sig_ = hierarchy_signature_();
    // Re-apply selection (handles were rebuilt).
    if (selected_ != 0 && scene_.find_by_id(selected_) == nullptr) selected_ = 0;
    select_entity(selected_);
}

void Shell::sync_hierarchy_() {
    if (hierarchy_signature_() != hier_sig_) refresh_hierarchy();
}

void Shell::select_entity(u32 id) {
    selected_ = (id != 0 && scene_.find_by_id(id) != nullptr) ? id : 0;
    selecting_ = true;
    int handle = -1;
    for (usize h = 0; h < handle_to_id_.size(); ++h)
        if (handle_to_id_[h] == selected_ && selected_ != 0) { handle = static_cast<int>(h); break; }
    tree_->set_selected(handle);
    selecting_ = false;
}

// -----------------------------------------------------------------------------
// Inspector
// -----------------------------------------------------------------------------
cui::NumberField<float>* Shell::position_field(int axis) noexcept {
    return (axis >= 0 && axis < 3) ? pos_[axis].field : nullptr;
}
cui::NumberField<float>* Shell::rotation_field(int axis) noexcept {
    return (axis >= 0 && axis < 3) ? rot_[axis].field : nullptr;
}
cui::NumberField<float>* Shell::scale_field(int axis) noexcept {
    return (axis >= 0 && axis < 3) ? scl_[axis].field : nullptr;
}

void Shell::sync_inspector_() {
    scene::Entity* e = (selected_ != 0) ? scene_.find_by_id(selected_) : nullptr;
    const bool have = e != nullptr;
    insp_empty_->set_visible(!have);
    insp_body_->set_visible(have);
    if (!have) {
        if (insp_entity_ != 0) {
            for (auto* b : { pos_, rot_, scl_, tint_ })
                for (int i = 0; i < 3; ++i)
                    if (b[i].field != nullptr && ui_.is_focused(b[i].field)) ui_.clear_focus();
        }
        insp_entity_ = 0;
        return;
    }

    // Selection changed: adopt the entity wholesale (no push of stale locals).
    const bool fresh = (insp_entity_ != e->id);
    insp_entity_ = e->id;
    insp_name_->set_text(e->name + "  " + fmt_id(e->id));

    // Two-way sync per value: a local that moved away from its snapshot is a
    // user edit -> push; otherwise an entity value that moved is external ->
    // pull (skipped while that field is being typed into).
    auto sync = [&](Bound& b, float target_now, auto&& apply) {
        const bool focused = (b.field != nullptr) && ui_.is_focused(b.field);
        if (!fresh && b.local != b.snap) {
            apply(b.local);
            b.snap = b.local;
        } else if (fresh || (!focused && target_now != b.snap)) {
            b.local = target_now;
            b.snap  = target_now;
            if (b.field != nullptr) b.field->refresh();
        }
    };
    float* t = &e->transform.translation.x;
    float* r = &e->transform.rotation_euler.x;
    float* s = &e->transform.scale.x;
    float* c = &e->tint.x;
    for (int i = 0; i < 3; ++i) {
        sync(pos_[i], t[i], [&](float v) { t[i] = v; });
        sync(rot_[i], r[i] * kRadToDeg, [&](float v) { r[i] = v * kDegToRad; });
        sync(scl_[i], s[i], [&](float v) { s[i] = v; });
        sync(tint_[i], c[i], [&](float v) { c[i] = v; });
    }
    if (fresh) {
        vis_local_ = vis_snap_ = e->visible;
    } else if (vis_local_ != vis_snap_) {
        e->visible = vis_local_; vis_snap_ = vis_local_;
    } else if (e->visible != vis_snap_) {
        vis_local_ = vis_snap_ = e->visible;
    }
}

// -----------------------------------------------------------------------------
// Console + log
// -----------------------------------------------------------------------------
void Shell::refresh_console_view_() {
    cardinal::vector<cardinal::string> items = con_lines_;
    con_list_->set_items(cardinal::move(items));
    con_scroll_->scroll_to_end();
}

void Shell::console_print(cardinal::string line) {
    con_lines_.push_back(cardinal::move(line));
    while (con_lines_.size() > kMaxConsoleLines) con_lines_.erase(con_lines_.begin());
    refresh_console_view_();
}

void Shell::console_execute(const cardinal::string& line) {
    if (line.empty()) return;
    con_lines_.push_back("> " + line);
    cardinal::vector<cardinal::string> out_lines;
    console::Output out([&out_lines](const cardinal::string& s) {
        // Split multi-line output into rows.
        cardinal::string cur;
        for (char ch : s) {
            if (ch == '\n') { if (!cur.empty()) out_lines.push_back(cur); cur.clear(); }
            else cur += ch;
        }
        if (!cur.empty()) out_lines.push_back(cur);
    });
    const bool ok = console::Registry::instance().execute(line, out);
    if (!ok && out_lines.empty()) out_lines.push_back("unknown command: " + line + "  (Tab completes)");
    for (auto& l : out_lines) con_lines_.push_back(cardinal::move(l));
    while (con_lines_.size() > kMaxConsoleLines) con_lines_.erase(con_lines_.begin());
    refresh_console_view_();
}

void Shell::console_accept_suggestion() {
    if (!con_sug_->visible() || con_sug_->items().empty()) return;
    con_in_->set_text(con_sug_->items()[0]);
    con_in_->set_caret(con_in_->text().size());
    con_sug_->clear_items();
    con_sug_->set_visible(false);
}

void Shell::drain_log_() {
    cardinal::vector<cardinal::string> got;
    {
        cardinal::lock_guard<cardinal::mutex> lg(log_tap_.mtx);
        got.swap(log_tap_.pending);
    }
    if (got.empty()) return;
    for (auto& l : got) log_lines_.push_back(cardinal::move(l));
    while (log_lines_.size() > kMaxLogLines) log_lines_.erase(log_lines_.begin());
    cardinal::vector<cardinal::string> items = log_lines_;
    log_list_->set_items(cardinal::move(items));
    log_scroll_->scroll_to_end();
}

// -----------------------------------------------------------------------------
// Stats
// -----------------------------------------------------------------------------
void Shell::set_stats(cardinal::string a, cardinal::string b, cardinal::string c) {
    stats_[0]->set_text(cardinal::move(a));
    stats_[1]->set_text(cardinal::move(b));
    stats_[2]->set_text(cardinal::move(c));
}

}  // namespace cardinal::cui_shell
