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

// A fixed-size leaf for exact layout assertions.
struct FixedBox : ui::Widget {
    ui::Vec2 sz;
    explicit FixedBox(ui::Vec2 s) : sz(s) {}
    ui::Vec2 measure(const ui::Constraints& c) override { return { c.clamp_w(sz.x), c.clamp_h(sz.y) }; }
    void     paint(ui::PaintContext&) override {}
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

    if (g_fail == 0) {
        cardinal::log::infof("cuitest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cuitest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
