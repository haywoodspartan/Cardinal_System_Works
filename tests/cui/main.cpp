// =============================================================================
// Cardinal UI (cui / Cardinal Slate) — deterministic regression suite.
//
// The framework is fully CPU-testable: build a widget tree, run the two-pass
// layout, assert computed rects; feed synthetic input, assert interaction
// (button click callback, checkbox toggle, slider value, hover/active state);
// paint into a DrawList and assert emitted commands. No GPU, no clock, no
// window — exit 0 = all pass.
// =============================================================================

#include <cardinal/cui/context.hpp>
#include <cardinal/cui/widgets.hpp>
#include <cardinal/cui/bind.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace ui = cardinal::cui;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("cuitest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}

// ---- geometry: Rect helpers + Constraints clamping -------------------------
void test_geometry() {
    ui::Rect r{ {10.0f, 20.0f}, {100.0f, 40.0f} };
    CHECK(ap(r.right(), 110.0f));
    CHECK(ap(r.bottom(), 60.0f));
    CHECK(ap(r.center().x, 60.0f));
    CHECK(r.contains({ 50.0f, 30.0f }));
    CHECK(!r.contains({ 5.0f, 30.0f }));

    const ui::Rect in = r.inset(5, 5, 5, 5);
    CHECK(ap(in.pos.x, 15.0f) && ap(in.width(), 90.0f));

    ui::Constraints c = ui::Constraints::loose(200.0f, 100.0f);
    CHECK(ap(c.clamp_w(300.0f), 200.0f));   // clamped to max
    CHECK(ap(c.clamp_h(50.0f), 50.0f));     // within range
    const ui::Constraints d = c.deflate(ui::Edges::all(10.0f));
    CHECK(ap(d.max_w, 180.0f) && ap(d.max_h, 80.0f));
}

// ---- layout: a vertical stack lays children out top-to-bottom --------------
void test_layout_vertical_stack() {
    ui::Ui u;
    auto* root = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(u.theme().panel_bg, ui::Edges::all(8.0f))));
    auto stack = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 4.0f);
    auto* lbl = static_cast<ui::Label*>(stack->add(cardinal::make_unique<ui::Label>("Hello")));
    auto* btn = static_cast<ui::Button*>(stack->add(cardinal::make_unique<ui::Button>("Click")));
    root->add(cardinal::move(stack));

    u.layout({ 400.0f, 300.0f });

    // Root fills the screen.
    CHECK(ap(root->rect().width(), 400.0f) && ap(root->rect().height(), 300.0f));
    // Children sit inside the panel padding (x >= 8).
    CHECK(lbl->rect().pos.x >= 8.0f - 1e-3f);
    CHECK(btn->rect().pos.x >= 8.0f - 1e-3f);
    // Vertical order: the button is below the label.
    CHECK(btn->rect().top() > lbl->rect().top());
    // The label has a non-zero measured size.
    CHECK(lbl->rect().width() > 0.0f && lbl->rect().height() > 0.0f);
}

// ---- interaction: a press+release on a button fires its callback -----------
void test_button_click() {
    ui::Ui u;
    int clicks = 0;
    auto* btn = static_cast<ui::Button*>(
        u.set_root(cardinal::make_unique<ui::Button>("Go", [&clicks]() { ++clicks; })));
    u.layout({ 200.0f, 80.0f });

    const ui::Vec2 c = btn->rect().center();

    u.update_input({ c, false, 0.0f });    // seed prev (hover)
    CHECK(u.is_hovered(btn));
    u.update_input({ c, true,  0.0f });    // press
    CHECK(u.is_active(btn));
    CHECK(clicks == 0);                     // not yet — click fires on release
    u.update_input({ c, false, 0.0f });    // release on the same widget
    CHECK(clicks == 1);
    CHECK(!u.is_active(btn));

    // Press on the button, release OFF it → no click.
    u.update_input({ c, true,  0.0f });
    u.update_input({ { -50.0f, -50.0f }, false, 0.0f });
    CHECK(clicks == 1);                     // unchanged
}

// ---- interaction: checkbox toggles its bound bool on click -----------------
void test_checkbox_toggle() {
    ui::Ui u;
    bool flag = false;
    auto* cb = static_cast<ui::Checkbox*>(
        u.set_root(cardinal::make_unique<ui::Checkbox>("Enabled", &flag)));
    u.layout({ 200.0f, 40.0f });
    const ui::Vec2 c = cb->rect().center();

    u.update_input({ c, false, 0.0f });
    u.update_input({ c, true,  0.0f });
    u.update_input({ c, false, 0.0f });
    CHECK(flag == true);
    u.update_input({ c, true,  0.0f });
    u.update_input({ c, false, 0.0f });
    CHECK(flag == false);
}

// ---- interaction: slider maps cursor x to value ----------------------------
void test_slider_drag() {
    ui::Ui u;
    float value = 0.0f;
    auto* sl = static_cast<ui::Slider*>(
        u.set_root(cardinal::make_unique<ui::Slider>(&value, 0.0f, 100.0f)));
    u.layout({ 200.0f, 30.0f });

    const ui::Rect r = sl->rect();
    // Press at the track center → ~50.
    u.update_input({ { r.left(), r.center().y }, false, 0.0f });   // seed
    u.update_input({ r.center(), true, 0.0f });                    // press+drag mid
    CHECK(ap(value, 50.0f, 1.0f));
    // Drag to the far right → ~100 (clamped).
    u.update_input({ { r.right() + 50.0f, r.center().y }, true, 0.0f });
    CHECK(ap(value, 100.0f, 1e-2f));
    u.update_input({ { r.right() + 50.0f, r.center().y }, false, 0.0f });   // release
}

// ---- paint: emits a backdrop + the label text ------------------------------
void test_paint_emits_commands() {
    ui::Ui u;
    auto* root = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(u.theme().panel_bg)));
    root->add(cardinal::make_unique<ui::Label>("Inspector"));
    u.layout({ 300.0f, 200.0f });

    ui::DrawList dl;
    u.paint(dl);
    CHECK(dl.size() > 0);
    CHECK(dl.count(ui::DrawKind::Text) == 1);          // the label
    CHECK(dl.count(ui::DrawKind::RectFilled) >= 2);    // backdrop + panel fill

    // The text command carries the label string.
    bool found = false;
    for (const auto& cmd : dl.cmds())
        if (cmd.kind == ui::DrawKind::Text && cmd.text == "Inspector") found = true;
    CHECK(found);
}

// A fixed-size leaf that paints a red rect — for exact layout + clip assertions.
struct FixedBox : ui::Widget {
    ui::Vec2 sz;
    explicit FixedBox(ui::Vec2 s) : sz(s) {}
    ui::Vec2 measure(const ui::Constraints& c) override { return { c.clamp_w(sz.x), c.clamp_h(sz.y) }; }
    void     paint(ui::PaintContext& ctx) override {
        ctx.dl->rect_filled(rect_, ctx.apply(ui::Color{255, 0, 0, 255}));
    }
};

// ---- Canvas: 9-point anchor placement (PA ComputePos model) ---------------
void test_canvas_anchors() {
    ui::Ui u;
    auto* canvas = static_cast<ui::Canvas*>(u.set_root(cardinal::make_unique<ui::Canvas>()));
    auto* tl = static_cast<FixedBox*>(canvas->add_anchored(
        cardinal::make_unique<FixedBox>(ui::Vec2{20, 10}), ui::Anchor::TopLeft));
    auto* ce = static_cast<FixedBox*>(canvas->add_anchored(
        cardinal::make_unique<FixedBox>(ui::Vec2{20, 10}), ui::Anchor::Center));
    auto* br = static_cast<FixedBox*>(canvas->add_anchored(
        cardinal::make_unique<FixedBox>(ui::Vec2{20, 10}), ui::Anchor::BottomRight, ui::Vec2{5, 3}));

    u.layout({ 200, 100 });
    CHECK(canvas->rect().width() == 200.0f && canvas->rect().height() == 100.0f);
    // Top-left at origin.
    CHECK(ap(tl->rect().pos.x, 0.0f) && ap(tl->rect().pos.y, 0.0f));
    // Center: (200-20)/2, (100-10)/2.
    CHECK(ap(ce->rect().pos.x, 90.0f) && ap(ce->rect().pos.y, 45.0f));
    // Bottom-right, inset by offset: 200-20-5, 100-10-3.
    CHECK(ap(br->rect().pos.x, 175.0f) && ap(br->rect().pos.y, 87.0f));
}

// ---- find by id (PA FindControl) ------------------------------------------
void test_find_by_id() {
    ui::Ui u;
    auto* root = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(u.theme().panel_bg)));
    auto stack = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical);
    auto* btn = static_cast<ui::Button*>(stack->add(cardinal::make_unique<ui::Button>("Go")));
    btn->set_id("go_button");
    root->add(cardinal::move(stack));
    CHECK(u.root()->find("go_button") == btn);
    CHECK(u.root()->find("missing") == nullptr);
}

// ---- enable / ignore gate hit-testing (PA DISABLE / IGNORE) ---------------
void test_state_hit_test() {
    ui::Ui u;
    auto* btn = static_cast<ui::Button*>(u.set_root(cardinal::make_unique<ui::Button>("Go")));
    u.layout({ 200, 80 });
    const ui::Vec2 c = btn->rect().center();
    CHECK(u.root()->hit_test(c) == btn);          // normal -> hit
    btn->set_enabled(false);
    CHECK(u.root()->hit_test(c) == nullptr);      // disabled -> no hit
    btn->set_enabled(true);
    btn->set_ignore_input(true);
    CHECK(u.root()->hit_test(c) == nullptr);      // ignore-input -> no hit
}

// ---- alpha cascade (PA SetAlphaChild) -------------------------------------
void test_alpha_cascade() {
    ui::Ui u;
    auto* panel = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(ui::Color{100, 100, 100, 200})));
    panel->set_alpha(0.5f);                       // fade the whole subtree
    panel->add(cardinal::make_unique<ui::Label>("X"));
    u.layout({ 100, 100 });

    ui::DrawList dl;
    u.paint(dl);
    // The label text (theme alpha 255) must be dimmed to ~127 by the 0.5 cascade.
    bool dim_text = false;
    for (const auto& cmd : dl.cmds())
        if (cmd.kind == ui::DrawKind::Text && cmd.color.a >= 120 && cmd.color.a <= 132) dim_text = true;
    CHECK(dim_text);
}

// ---- ScrollPanel: wheel scroll + child clipping ---------------------------
void test_scroll_clip() {
    ui::Ui u;
    auto* sp = static_cast<ui::ScrollPanel*>(
        u.set_root(cardinal::make_unique<ui::ScrollPanel>(0.0f)));
    FixedBox* first = nullptr;
    for (int i = 0; i < 5; ++i) {
        auto* b = static_cast<FixedBox*>(
            sp->add(cardinal::make_unique<FixedBox>(ui::Vec2{100, 20})));
        if (i == 0) first = b;
    }
    u.layout({ 100, 50 });                     // viewport 50 tall; content = 5*20 = 100
    CHECK(ap(first->rect().pos.y, 0.0f));      // first child starts at the top

    // Wheel down -> content scrolls up.
    u.update_input({ { 50, 25 }, false, -2.0f });
    u.layout({ 100, 50 });                     // re-layout applies the new scroll
    CHECK(sp->scroll() > 0.0f);
    CHECK(first->rect().pos.y < 0.0f);         // first child pushed above the top

    // Painted child rects carry the panel's clip (height 50).
    ui::DrawList dl;
    u.paint(dl);
    bool clipped = false;
    for (const auto& c : dl.cmds())
        if (c.kind == ui::DrawKind::RectFilled && c.color.r == 255 && c.color.g == 0
            && ap(c.clip.size.y, 50.0f)) clipped = true;
    CHECK(clipped);
}

// ---- TextField: focus, typing, caret editing, submit, escape ---------------
void test_textfield_focus_typing() {
    ui::Ui u;
    auto root = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 4.0f);
    auto* tf  = static_cast<ui::TextField*>(
        root->add(cardinal::make_unique<ui::TextField>("ab")));
    cardinal::string submitted;
    tf->set_on_submit([&](const cardinal::string& s) { submitted = s; });
    u.set_root(cardinal::move(root));
    u.layout({ 400.0f, 300.0f });

    const ui::Vec2 c = tf->rect().center();
    ui::InputState in{};
    in.mouse = c;
    u.update_input(in);                            // seed prev (hover)
    in.mouse_down = true;  u.update_input(in);     // press -> gains focus
    CHECK(u.is_focused(tf));
    in.mouse_down = false; u.update_input(in);     // release -> click (caret=end)
    CHECK(tf->caret() == 2u);

    auto key = [&](auto&& fill) {
        ui::InputState k{};
        k.mouse = c;
        fill(k);
        u.update_input(k);
    };
    key([](ui::InputState& k) { k.text = "X"; });
    CHECK(tf->text() == cardinal::string("abX") && tf->caret() == 3u);
    key([](ui::InputState& k) { k.key_left = true; });
    CHECK(tf->caret() == 2u);
    key([](ui::InputState& k) { k.key_backspace = true; });
    CHECK(tf->text() == cardinal::string("aX") && tf->caret() == 1u);
    key([](ui::InputState& k) { k.key_home = true; });
    CHECK(tf->caret() == 0u);
    key([](ui::InputState& k) { k.key_delete = true; });
    CHECK(tf->text() == cardinal::string("X"));
    key([](ui::InputState& k) { k.key_end = true; });
    CHECK(tf->caret() == 1u);
    key([](ui::InputState& k) { k.key_enter = true; });
    CHECK(submitted == cardinal::string("X"));

    // Escape clears focus (consumed by the Ui); further keys are ignored.
    key([](ui::InputState& k) { k.key_escape = true; });
    CHECK(!u.is_focused(tf));
    key([](ui::InputState& k) { k.text = "z"; });
    CHECK(tf->text() == cardinal::string("X"));

    // Refocus, then press empty space -> focus clears.
    in.mouse = c;
    in.mouse_down = true;  u.update_input(in);
    CHECK(u.is_focused(tf));
    in.mouse_down = false; u.update_input(in);
    in.mouse = { -50.0f, -50.0f };
    in.mouse_down = true;  u.update_input(in);
    CHECK(!u.is_focused(tf));
    in.mouse_down = false; u.update_input(in);
}

// ---- ListView: press-select + drag-through + callback ----------------------
void test_listview_select() {
    ui::Ui u;
    auto root = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 0.0f);
    auto* lv  = static_cast<ui::ListView*>(root->add(cardinal::make_unique<ui::ListView>()));
    lv->add_item("alpha");
    lv->add_item("beta");
    lv->add_item("gamma");
    int picked = -1;
    lv->set_on_select([&](int i) { picked = i; });
    u.set_root(cardinal::move(root));
    u.layout({ 300.0f, 200.0f });
    CHECK(lv->selected() == -1);

    const float row_h = lv->rect().height() / 3.0f;
    auto press_at = [&](float ry) {
        ui::InputState in{};
        in.mouse = { lv->rect().left() + 20.0f, lv->rect().top() + ry };
        u.update_input(in);
        in.mouse_down = true;  u.update_input(in);   // press -> selects
        in.mouse_down = false; u.update_input(in);   // release
    };
    press_at(row_h * 1.5f);                          // middle of row 1
    CHECK(lv->selected() == 1 && picked == 1);
    press_at(row_h * 2.5f);                          // row 2
    CHECK(lv->selected() == 2 && picked == 2);
}

// ---- TreeView: expand/collapse via arrow + label select --------------------
void test_treeview_expand_select() {
    ui::Ui u;
    auto root = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 0.0f);
    auto* tv  = static_cast<ui::TreeView*>(root->add(cardinal::make_unique<ui::TreeView>()));
    const int n_root = tv->add_node(-1, "World");
    const int n_a    = tv->add_node(n_root, "EntityA");
    const int n_b    = tv->add_node(n_root, "EntityB");
    (void)n_b;
    int picked = -1;
    tv->set_on_select([&](int h) { picked = h; });
    u.set_root(cardinal::move(root));
    u.layout({ 300.0f, 200.0f });

    // 3 visible rows (root + 2 children), 22px each.
    CHECK(ap(tv->rect().height() / 3.0f, 22.0f, 0.5f));

    auto click_at = [&](float rx, float ry) {
        ui::InputState in{};
        in.mouse = { tv->rect().left() + rx, tv->rect().top() + ry };
        u.update_input(in);
        in.mouse_down = true;  u.update_input(in);
        in.mouse_down = false; u.update_input(in);   // release -> on_click acts
    };
    // Click EntityA's LABEL (row 1, depth 1 -> label starts at 4+16+14=34).
    click_at(60.0f, 22.0f * 1.5f);
    CHECK(tv->selected() == n_a && picked == n_a);
    // Click the root's ARROW (row 0, x within [4, 18)) -> collapse.
    click_at(8.0f, 11.0f);
    CHECK(!tv->expanded(n_root));
    u.layout({ 300.0f, 200.0f });                    // re-measure: 1 visible row
    CHECK(ap(tv->rect().height(), 22.0f, 0.5f));
    tv->set_expanded(n_root, true);
    u.layout({ 300.0f, 200.0f });
    CHECK(ap(tv->rect().height(), 66.0f, 0.5f));
}

// ---- Combo: open/select/close + popup priority + focus-loss close ----------
void test_combo_dropdown() {
    ui::Ui u;
    auto* panel = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(u.theme().panel_bg, ui::Edges::all(8.0f))));
    auto stack = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 4.0f);
    auto* cb = static_cast<ui::Combo*>(stack->add(cardinal::make_unique<ui::Combo>()));
    cb->set_items({ "Alpha", "Beta", "Gamma" });
    int selects = 0, last = -1;
    cb->set_on_select([&](int i) { ++selects; last = i; });
    int btn_clicks = 0;
    auto* btn = static_cast<ui::Button*>(stack->add(
        cardinal::make_unique<ui::Button>("Go", [&]() { ++btn_clicks; })));
    panel->add(cardinal::move(stack));
    u.layout({ 300.0f, 300.0f });

    // Closed: intrinsic header height (line_height(14)+8 = 24), not stretched.
    const float header_h = ui::line_height(14.0f) + 8.0f;
    CHECK(ap(cb->rect().height(), header_h));
    CHECK(!cb->open());

    ui::DrawList closed_dl;
    u.paint(closed_dl);
    const auto closed_texts = closed_dl.count(ui::DrawKind::Text);

    auto click_at = [&](ui::Vec2 p) {
        ui::InputState in{};
        in.mouse = p;              u.update_input(in);
        in.mouse_down = true;      u.update_input(in);
        in.mouse_down = false;     u.update_input(in);
    };

    // Header point: stable regardless of dropdown extension (the rect stays
    // extended until the next layout after a close, so center() would drift
    // into the dropdown region — always aim at the header band).
    const ui::Vec2 header_pt{ cb->rect().left() + 10.0f,
                              cb->rect().top() + header_h * 0.5f };

    // Click the header -> opens + takes focus; rect extends over the dropdown.
    click_at(header_pt);
    CHECK(cb->open());
    CHECK(u.is_focused(cb));
    u.layout({ 300.0f, 300.0f });
    CHECK(ap(cb->rect().height(), header_h + 3.0f * 22.0f));

    // Open paints the dropdown rows (more text commands than closed).
    ui::DrawList open_dl;
    u.paint(open_dl);
    CHECK(open_dl.count(ui::DrawKind::Text) > closed_texts);

    // Host popup contract: while open the combo wins input over later siblings.
    u.set_popup(cb);
    // Click row 1 ("Beta"): y = header bottom + 1.5 rows.
    click_at({ cb->rect().left() + 10.0f,
               cb->rect().top() + header_h + 1.5f * 22.0f });
    CHECK(cb->selected() == 1 && last == 1 && selects == 1);
    CHECK(!cb->open());
    u.set_popup(nullptr);

    // Reopen; a click on the BUTTON's center (inside the dropdown overlap)
    // must go to the popup, not the button.
    click_at(header_pt);
    u.layout({ 300.0f, 300.0f });
    u.set_popup(cb);
    click_at(btn->rect().center());
    CHECK(btn_clicks == 0);
    CHECK(!cb->open());
    u.set_popup(nullptr);

    // WITHOUT the popup contract, the later sibling wins the overlap (known
    // tree-order limitation) — and the focus steal still closes the combo.
    click_at(header_pt);
    u.layout({ 300.0f, 300.0f });
    click_at(btn->rect().center());
    CHECK(btn_clicks == 1);
    CHECK(!cb->open());

    // Escape closes via Ui-consumed focus clear.
    click_at(header_pt);
    CHECK(cb->open());
    ui::InputState esc{};
    esc.mouse = cb->rect().center();
    esc.key_escape = true;
    u.update_input(esc);
    CHECK(!cb->open() && !u.is_focused(cb));
}

// ---- MenuBar: open/switch/fire-item/stray-close/focus-close ----------------
void test_menubar() {
    ui::Ui u;
    auto* panel = static_cast<ui::Panel*>(
        u.set_root(cardinal::make_unique<ui::Panel>(u.theme().panel_bg, ui::Edges::all(0.0f))));
    auto* mb = static_cast<ui::MenuBar*>(panel->add(cardinal::make_unique<ui::MenuBar>()));
    const int m_file = mb->add_menu("File");
    mb->add_item(m_file, "Open");
    mb->add_item(m_file, "Save");
    const int m_edit = mb->add_menu("Edit");
    mb->add_item(m_edit, "Undo");
    int fired = 0, last_menu = -1, last_item = -1;
    mb->set_on_item([&](int m, int i) { ++fired; last_menu = m; last_item = i; });
    u.layout({ 400.0f, 300.0f });
    CHECK(ap(mb->rect().height(), 24.0f));   // closed bar

    const float tw_file = ui::text_advance("File", 14.0f) + 20.0f;
    const float tw_edit = ui::text_advance("Edit", 14.0f) + 20.0f;

    auto click_at = [&](ui::Vec2 p) {
        ui::InputState in{};
        in.mouse = p;              u.update_input(in);
        in.mouse_down = true;      u.update_input(in);
        in.mouse_down = false;     u.update_input(in);
    };

    // Open "File".
    click_at({ mb->rect().left() + tw_file * 0.5f, mb->rect().top() + 12.0f });
    CHECK(mb->open_menu() == m_file);
    u.layout({ 400.0f, 300.0f });
    CHECK(ap(mb->rect().height(), 24.0f + 2.0f * 22.0f));   // bar + 2 items
    u.set_popup(mb);

    // Click "Save" (row 1 under the File title).
    click_at({ mb->rect().left() + 5.0f,
               mb->rect().top() + 24.0f + 1.5f * 22.0f });
    CHECK(fired == 1 && last_menu == m_file && last_item == 1);
    CHECK(mb->open_menu() == -1);
    u.set_popup(nullptr);

    // Open "File", then click "Edit" -> switches menus.
    click_at({ mb->rect().left() + tw_file * 0.5f, mb->rect().top() + 12.0f });
    CHECK(mb->open_menu() == m_file);
    click_at({ mb->rect().left() + tw_file + tw_edit * 0.5f, mb->rect().top() + 12.0f });
    CHECK(mb->open_menu() == m_edit);

    // Stray click below the bar but outside the popup just closes.
    u.layout({ 400.0f, 300.0f });
    click_at({ mb->rect().right() - 10.0f, mb->rect().top() + 30.0f });
    CHECK(mb->open_menu() == -1);
    CHECK(fired == 1);                       // nothing fired

    // Reopen, then press outside the bar entirely -> focus loss closes.
    click_at({ mb->rect().left() + tw_file * 0.5f, mb->rect().top() + 12.0f });
    CHECK(mb->open_menu() == m_file);
    click_at({ 200.0f, 250.0f });
    CHECK(mb->open_menu() == -1);
}

// ---- Tabs: page switching, visibility, content hit-testing -----------------
void test_tabs_pages() {
    ui::Ui u;
    auto tabs_owned = cardinal::make_unique<ui::Tabs>();
    auto* tabs = tabs_owned.get();
    auto* lbl = tabs->add_tab("One", cardinal::make_unique<ui::Label>("PageA"));
    int clicks = 0;
    auto* btn = tabs->add_tab("Two",
        cardinal::make_unique<ui::Button>("B", [&]() { ++clicks; }));
    int changes = 0, last_tab = -1;
    tabs->set_on_change([&](int i) { ++changes; last_tab = i; });
    u.set_root(cardinal::move(tabs_owned));
    u.layout({ 400.0f, 300.0f });

    CHECK(tabs->selected() == 0);
    CHECK(lbl->visible() && !btn->visible());

    // Only the visible page paints: 2 tab labels + "PageA".
    ui::DrawList dl;
    u.paint(dl);
    CHECK(dl.count(ui::DrawKind::Text) == 3);

    const float tw_one = ui::text_advance("One", 14.0f) + 20.0f;
    const float tw_two = ui::text_advance("Two", 14.0f) + 20.0f;

    auto click_at = [&](ui::Vec2 p) {
        ui::InputState in{};
        in.mouse = p;              u.update_input(in);
        in.mouse_down = true;      u.update_input(in);
        in.mouse_down = false;     u.update_input(in);
    };

    // Click the "Two" header -> switch page.
    click_at({ tw_one + tw_two * 0.5f, 13.0f });
    CHECK(tabs->selected() == 1 && changes == 1 && last_tab == 1);
    CHECK(!lbl->visible() && btn->visible());
    // The new page is arranged into the content region (below the 26px strip).
    CHECK(btn->rect().top() >= 26.0f - 1e-3f);
    CHECK(ap(btn->rect().height(), 300.0f - 26.0f));

    // The shown page is hit-testable: click the button.
    click_at(btn->rect().center());
    CHECK(clicks == 1);

    // Clicking the already-selected header fires no change.
    click_at({ tw_one + tw_two * 0.5f, 13.0f });
    CHECK(changes == 1);

    // Programmatic switch swaps visibility without firing on_change.
    tabs->set_selected(0);
    CHECK(lbl->visible() && !btn->visible() && changes == 1);
}

// ---- Stack flex-grow: leftover main-axis space split by weight -------------
void test_stack_flex() {
    ui::Ui u;
    auto stack = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 0.0f);
    auto* top  = stack->add(cardinal::make_unique<ui::Label>("top"));
    auto* fill = stack->add_flex(cardinal::make_unique<ui::ScrollPanel>(), 1.0f);
    auto* bot  = stack->add(cardinal::make_unique<ui::Label>("bottom"));
    u.set_root(cardinal::move(stack));
    u.layout({ 200.0f, 300.0f });

    const float lh = ui::line_height(14.0f);
    CHECK(ap(top->rect().height(), lh));
    CHECK(ap(bot->rect().height(), lh));
    CHECK(ap(fill->rect().height(), 300.0f - 2.0f * lh));   // absorbs leftover
    CHECK(ap(fill->rect().top(), lh));
    CHECK(ap(bot->rect().top(), 300.0f - lh));               // pinned to bottom

    // Two flex children split leftover by weight (1 : 3).
    ui::Ui u2;
    auto s2 = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 0.0f);
    auto* lbl = s2->add(cardinal::make_unique<ui::Label>("x"));
    auto* a   = s2->add_flex(cardinal::make_unique<ui::ScrollPanel>(), 1.0f);
    auto* b   = s2->add_flex(cardinal::make_unique<ui::ScrollPanel>(), 3.0f);
    u2.set_root(cardinal::move(s2));
    u2.layout({ 100.0f, 100.0f + lh });
    (void)lbl;
    CHECK(ap(a->rect().height(), 25.0f));
    CHECK(ap(b->rect().height(), 75.0f));
}

// ---- ScrollPanel::scroll_to_end + TextField::set_caret ---------------------
void test_scroll_end_and_caret() {
    ui::Ui u;
    auto sp = cardinal::make_unique<ui::ScrollPanel>(0.0f);
    for (int i = 0; i < 20; ++i)
        sp->add(cardinal::make_unique<ui::Label>("row"));
    auto* p = static_cast<ui::ScrollPanel*>(u.set_root(cardinal::move(sp)));
    u.layout({ 100.0f, 100.0f });
    const float lh = ui::line_height(14.0f);
    p->scroll_to_end();
    u.layout({ 100.0f, 100.0f });                    // arrange clamps
    CHECK(ap(p->scroll(), 20.0f * lh - 100.0f));

    ui::TextField tf("hello");
    tf.set_caret(2);   CHECK(tf.caret() == 2u);
    tf.set_caret(999); CHECK(tf.caret() == 5u);      // clamped to size
}

// ---- NumberField<T>: parse / clamp / canonicalise / refresh ----------------
void test_number_field() {
    ui::Ui u;
    int bound = 7;
    auto stack = cardinal::make_unique<ui::Stack>(ui::Axis::Vertical, 4.0f);
    auto* nf = static_cast<ui::NumberField<int>*>(stack->add(
        cardinal::make_unique<ui::NumberField<int>>(&bound, 0, 100)));
    u.set_root(cardinal::move(stack));
    u.layout({ 300.0f, 100.0f });
    CHECK(nf->text() == cardinal::string("7"));      // ctor refresh

    const ui::Vec2 c = nf->rect().center();
    ui::InputState in{};
    in.mouse = c;              u.update_input(in);
    in.mouse_down = true;      u.update_input(in);   // focus
    in.mouse_down = false;     u.update_input(in);
    CHECK(u.is_focused(nf));

    auto key = [&](auto&& fill) {
        ui::InputState k{}; k.mouse = c; fill(k); u.update_input(k);
    };
    key([](ui::InputState& k) { k.text = "2"; });    // "7" -> "72"
    CHECK(bound == 7);                               // not live: no adopt yet
    key([](ui::InputState& k) { k.key_enter = true; });
    CHECK(bound == 72);
    CHECK(nf->text() == cardinal::string("72"));

    // Clamp on commit: append digits to exceed max -> 100.
    key([](ui::InputState& k) { k.text = "99"; });   // "7299"
    key([](ui::InputState& k) { k.key_enter = true; });
    CHECK(bound == 100);
    CHECK(nf->text() == cardinal::string("100"));    // canonicalised

    // External mutation + refresh.
    bound = 42;
    nf->refresh();
    CHECK(nf->text() == cardinal::string("42"));

    // Float field, live mode: value tracks every keystroke.
    ui::Ui uf;
    float fbound = 0.5f;
    auto* ff = static_cast<ui::NumberField<float>*>(
        uf.set_root(cardinal::make_unique<ui::NumberField<float>>(
            &fbound, 0.0f, 10.0f, /*live=*/true)));
    uf.layout({ 300.0f, 60.0f });
    const ui::Vec2 fc = ff->rect().center();
    ui::InputState fin{};
    fin.mouse = fc;            uf.update_input(fin);
    fin.mouse_down = true;     uf.update_input(fin);
    fin.mouse_down = false;    uf.update_input(fin);
    ui::InputState fk{}; fk.mouse = fc; fk.text = "2";   // "0.5" -> "0.52"
    uf.update_input(fk);
    CHECK(ap(fbound, 0.52f, 1e-4f));
}

// ---- EnumCombo<E>: dropdown selection writes the bound enum ----------------
void test_enum_combo() {
    enum class Mode : int { A = 0, B, C };
    ui::Ui u;
    Mode bound = Mode::B;
    auto* ec = static_cast<ui::EnumCombo<Mode>*>(
        u.set_root(cardinal::make_unique<ui::EnumCombo<Mode>>(
            &bound, cardinal::vector<cardinal::string>{ "A", "B", "C" })));
    u.layout({ 300.0f, 200.0f });
    CHECK(ec->selected() == 1);                      // ctor refresh from value

    auto click_at = [&](ui::Vec2 p) {
        ui::InputState in{};
        in.mouse = p;              u.update_input(in);
        in.mouse_down = true;      u.update_input(in);
        in.mouse_down = false;     u.update_input(in);
    };
    const float header_h = ui::line_height(14.0f) + 8.0f;
    const ui::Vec2 hp{ ec->rect().left() + 10.0f, ec->rect().top() + header_h * 0.5f };
    click_at(hp);                                    // open
    CHECK(ec->open());
    u.layout({ 300.0f, 200.0f });
    click_at({ ec->rect().left() + 10.0f,
               ec->rect().top() + header_h + 2.5f * 22.0f });   // row 2 = "C"
    CHECK(!ec->open());
    CHECK(bound == Mode::C);
    bound = Mode::A;
    ec->refresh();
    CHECK(ec->selected() == 0);
}

}  // namespace

int main() {
    test_geometry();
    test_layout_vertical_stack();
    test_button_click();
    test_checkbox_toggle();
    test_slider_drag();
    test_paint_emits_commands();
    test_canvas_anchors();
    test_find_by_id();
    test_state_hit_test();
    test_alpha_cascade();
    test_scroll_clip();
    test_textfield_focus_typing();
    test_listview_select();
    test_treeview_expand_select();
    test_combo_dropdown();
    test_menubar();
    test_tabs_pages();
    test_stack_flex();
    test_scroll_end_and_caret();
    test_number_field();
    test_enum_combo();

    if (g_fail == 0) {
        cardinal::log::infof("cuitest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cuitest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
