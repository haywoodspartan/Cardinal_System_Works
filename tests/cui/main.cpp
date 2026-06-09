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

    if (g_fail == 0) {
        cardinal::log::infof("cuitest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cuitest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
