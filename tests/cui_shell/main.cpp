// =============================================================================
// Cardinal — native Studio shell (cui_shell) regression suite. Headless: the
// shell is driven with synthetic InputState frames against a CPU-only
// scene::Scene (entities with no meshes). Exit 0 = all pass.
// =============================================================================

#include <cardinal/cui_shell/shell.hpp>
#include <cardinal/console/console.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cstdlib.hpp>   // cardinal::atof

namespace {

namespace ui = cardinal::cui;
namespace sh = cardinal::cui_shell;
namespace sc = cardinal::scene;

int g_checks = 0, g_fail = 0;
void check_impl(bool ok, const char* e, int l) {
    ++g_checks;
    if (!ok) { ++g_fail; cardinal::log::errorf("shelltest", "FAIL  L%d  %s", l, e); }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)
bool ap(float a, float b, float e = 1e-3f) { float d = a > b ? a - b : b - a; return d <= e; }

constexpr float W = 1280.0f, H = 720.0f;

void frame(sh::Shell& s, ui::Vec2 mouse, bool down, const char* text = "",
           bool enter = false, bool backspace = false) {
    s.layout({W, H});
    ui::InputState in;
    in.mouse = mouse; in.mouse_down = down;
    in.text = text; in.key_enter = enter; in.key_backspace = backspace;
    s.update(in);
    ui::DrawList dl;
    s.paint(dl);
}
void click(sh::Shell& s, ui::Vec2 p) {
    frame(s, p, false);
    frame(s, p, true);
    frame(s, p, false);
}

void test_layout_and_viewport() {
    sc::Scene scene;
    sh::Shell shell(scene);
    frame(shell, {0.0f, 0.0f}, false);
    auto* hier = shell.window(sh::Shell::Panel::Hierarchy);
    auto* insp = shell.window(sh::Shell::Panel::Inspector);
    auto* con  = shell.window(sh::Shell::Panel::Console);
    auto* lg   = shell.window(sh::Shell::Panel::Log);
    CHECK(hier && hier->docked() && hier->dock_side() == ui::DockSide::Left);
    CHECK(insp && insp->docked() && insp->dock_side() == ui::DockSide::Right);
    CHECK(con && con->dock_side() == ui::DockSide::Bottom);
    CHECK(lg  && lg->dock_side()  == ui::DockSide::Bottom);
    // Menu bar (24 px) above the desktop; strips carve the viewport.
    const ui::Rect vp = shell.viewport_rect();
    CHECK(ap(vp.left(), 280.0f));
    CHECK(ap(vp.top(), 24.0f));
    CHECK(ap(vp.width(), W - 280.0f - 320.0f));
    CHECK(ap(vp.height(), H - 24.0f - 220.0f));
    // Console + Log split the bottom strip side by side.
    CHECK(ap(con->rect().width(), W * 0.5f) && ap(lg->rect().width(), W * 0.5f));
    CHECK(ap(lg->rect().left(), W * 0.5f));
    CHECK(shell.mouse_over_viewport({640.0f, 300.0f}));
    CHECK(!shell.mouse_over_viewport({100.0f, 300.0f}));
    CHECK(!shell.mouse_over_viewport({640.0f, 650.0f}));
    // Hovering a floating window over the viewport is not "over the viewport".
    auto* st = shell.window(sh::Shell::Panel::Stats);
    const ui::Vec2 sc = st->title_rect().center();
    frame(shell, sc, false);
    CHECK(!shell.mouse_over_viewport(sc));
    // Panel toggles.
    shell.toggle_panel(sh::Shell::Panel::Stats);
    CHECK(!st->visible());
    shell.show_panel(sh::Shell::Panel::Stats, true);
    CHECK(st->visible());
    shell.show_panel(sh::Shell::Panel::Log, false);
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(ap(con->rect().width(), W));      // Console takes the whole strip
}

void test_hierarchy_and_selection() {
    sc::Scene scene;
    // add_entity returns a reference into a vector: take ids right away.
    const cardinal::u32 ground_id = scene.add_entity("Ground").id;
    const cardinal::u32 box_id    = scene.add_entity("Box").id;
    auto& child = scene.add_entity("Child");
    child.parent_id = box_id;
    sh::Shell shell(scene);
    frame(shell, {0.0f, 0.0f}, false);
    auto& tree = shell.hierarchy();
    CHECK(tree.label(0).find("Ground") == 0u);
    CHECK(tree.label(1).find("Box") == 0u);
    CHECK(tree.label(2).find("Child") == 0u);
    CHECK(tree.selected() == -1 && shell.selected_entity() == 0);

    shell.select_entity(box_id);
    CHECK(shell.selected_entity() == box_id);
    CHECK(tree.selected() == 1);

    // Clicking a row selects through the tree (row 0 = Ground).
    const ui::Rect tr = tree.rect();
    click(shell, { tr.left() + 60.0f, tr.top() + 10.0f });
    CHECK(shell.selected_entity() == ground_id);

    // Scene change (new entity) -> the tree rebuilds and keeps the selection.
    scene.add_entity("Late");
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(tree.label(3).find("Late") == 0u);
    CHECK(shell.selected_entity() == ground_id && tree.selected() == 0);

    // Removing the selected entity clears the selection.
    scene.remove_entity(ground_id);
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(shell.selected_entity() == 0 && tree.selected() == -1);
}

void test_inspector_sync() {
    sc::Scene scene;
    auto& box = scene.add_entity("Box");
    const cardinal::u32 id = box.id;
    sh::Shell shell(scene);
    frame(shell, {0.0f, 0.0f}, false);
    auto* px = shell.position_field(0);
    CHECK(px != nullptr);

    shell.select_entity(id);
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(px->text() == "0");

    // External edit (gizmo / script) shows up in the field.
    scene.find_by_id(id)->transform.translation.x = 2.5f;
    scene.find_by_id(id)->transform.rotation_euler.y = 1.5707963f;
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(px->text() == "2.5");
    CHECK(ap(static_cast<float>(cardinal::atof(shell.rotation_field(1)->text().c_str())), 90.0f, 0.01f));

    // Typing into the field + Enter pushes to the entity immediately.
    const ui::Vec2 fc = px->rect().center();
    click(shell, fc);
    CHECK(shell.wants_keyboard());
    frame(shell, fc, false, "", false, true);
    frame(shell, fc, false, "", false, true);
    frame(shell, fc, false, "", false, true);
    frame(shell, fc, false, "7");
    frame(shell, fc, false, "", true);
    CHECK(ap(scene.find_by_id(id)->transform.translation.x, 7.0f));
    // While focused, an external change does not clobber the edit text.
    scene.find_by_id(id)->transform.translation.x = 9.0f;
    frame(shell, fc, false);
    CHECK(px->text() == "7");
    // Blur (click the viewport) -> pull the external value.
    click(shell, {640.0f, 300.0f});
    frame(shell, {640.0f, 300.0f}, false);
    CHECK(px->text() == "9");

    // Rotation is edited in degrees, stored in radians.
    auto* ry = shell.rotation_field(1);
    click(shell, ry->rect().center());
    for (int i = 0; i < 12; ++i) frame(shell, ry->rect().center(), false, "", false, true);
    frame(shell, ry->rect().center(), false, "45");
    frame(shell, ry->rect().center(), false, "", true);
    CHECK(ap(scene.find_by_id(id)->transform.rotation_euler.y, 0.7853982f, 1e-4f));

    // Selecting another entity adopts it without pushing stale locals.
    const cardinal::u32 oid = scene.add_entity("Other").id;
    scene.find_by_id(oid)->transform.translation.x = -3.0f;
    click(shell, {640.0f, 300.0f});
    shell.select_entity(oid);
    frame(shell, {640.0f, 300.0f}, false);
    CHECK(px->text() == "-3");
    CHECK(ap(scene.find_by_id(oid)->transform.translation.x, -3.0f));
    CHECK(ap(scene.find_by_id(id)->transform.translation.x, 9.0f));
}

void test_console_and_log() {
    sc::Scene scene;
    sh::Shell shell(scene);
    frame(shell, {0.0f, 0.0f}, false);

    int echoed = 0;
    CARDINAL_CCOMMAND("shelltest.echo", "echo args",
        [&echoed](const cardinal::vector<cardinal::string>& argv, cardinal::console::Output& out) {
            ++echoed;
            cardinal::string s;
            for (cardinal::usize i = 1; i < argv.size(); ++i) { if (i > 1) s += ' '; s += argv[i]; }
            out.line(s + "\nsecond");
        });

    shell.console_execute("shelltest.echo hi there");
    CHECK(echoed == 1);
    const auto& lines = shell.console_lines();
    CHECK(lines.size() >= 3u);
    CHECK(lines[lines.size() - 3] == "> shelltest.echo hi there");
    CHECK(lines[lines.size() - 2] == "hi there");
    CHECK(lines[lines.size() - 1] == "second");
    shell.console_execute("definitely.not.a.command");
    CHECK(shell.console_lines().back().find("unknown command") == 0u);

    // Typing a prefix shows suggestions; Tab (accept) completes the input.
    auto& in = shell.console_input();
    click(shell, in.rect().center());
    CHECK(shell.wants_keyboard());
    frame(shell, in.rect().center(), false, "shelltest.e");
    CHECK(in.text() == "shelltest.e");
    shell.console_accept_suggestion();
    CHECK(in.text() == "shelltest.echo");
    // Enter runs it through the field's submit path and clears the input.
    frame(shell, in.rect().center(), false, " x");
    frame(shell, in.rect().center(), false, "", true);
    CHECK(echoed == 2);
    CHECK(in.text().empty());
    CHECK(shell.console_lines().back() == "second");

    // Log sink: engine log lines land in the Log panel on the next update.
    cardinal::log::infof("shelltest", "hello %d", 42);
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(!shell.log_lines().empty());
    CHECK(shell.log_lines().back().find("hello 42") != cardinal::string::npos);
    CHECK(shell.log_lines().back().find("[shelltest]") != cardinal::string::npos);

    cardinal::console::Registry::instance().unregister("shelltest.echo");
}

void test_menu_dispatch() {
    sc::Scene scene;
    sh::Shell shell(scene);
    int quit = 0;
    const int m_file = shell.add_menu("File");
    shell.add_menu_item(m_file, "New", [] {});
    shell.add_menu_item(m_file, "Quit", [&quit] { ++quit; });
    const int m_view = shell.add_view_menu();
    CHECK(m_view == m_file + 1);
    frame(shell, {0.0f, 0.0f}, false);

    auto& mb = shell.menu_bar();
    const float tw_file = ui::text_advance("File", 14.0f) + 20.0f;
    click(shell, { mb.rect().left() + tw_file * 0.5f, mb.rect().top() + 12.0f });
    CHECK(mb.open_menu() == m_file);
    // While a menu is open the viewport does not take navigation input.
    CHECK(!shell.mouse_over_viewport({640.0f, 300.0f}));
    frame(shell, {0.0f, 0.0f}, false);   // re-layout with the dropdown open
    click(shell, { mb.rect().left() + 5.0f, mb.rect().top() + 24.0f + 1.5f * 22.0f });   // "Quit" (row 1)
    CHECK(quit == 1);
    CHECK(mb.open_menu() == -1);

    // View > Hierarchy toggles the panel.
    const float tw_view = ui::text_advance("View", 14.0f) + 20.0f;
    click(shell, { mb.rect().left() + tw_file + tw_view * 0.5f, mb.rect().top() + 12.0f });
    CHECK(mb.open_menu() == m_view);
    frame(shell, {0.0f, 0.0f}, false);
    click(shell, { mb.rect().left() + tw_file + 5.0f, mb.rect().top() + 24.0f + 0.5f * 22.0f });
    CHECK(!shell.window(sh::Shell::Panel::Hierarchy)->visible());
    frame(shell, {0.0f, 0.0f}, false);
    CHECK(ap(shell.viewport_rect().left(), 0.0f));
}

}  // namespace

int main() {
    cardinal::log::set_min_level(cardinal::log::Level::Info);
    test_layout_and_viewport();
    test_hierarchy_and_selection();
    test_inspector_sync();
    test_console_and_log();
    test_menu_dispatch();
    if (g_fail == 0) {
        cardinal::log::infof("shelltest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("shelltest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
