// =============================================================================
// Cardinal Studio (native) — the editor shell on Cardinal Slate. NO ImGui.
//
// Host responsibilities (everything else is cardinal::cui_shell):
//   - native window + OS input + RHI device/swapchain
//   - a scene::Scene with a few primitives + scene::ForwardRenderer drawing it
//     into the back buffer each frame (the shell's viewport_rect() is the area
//     no panel covers; the UI is composited on top by cui_rhi)
//   - orbit camera on the viewport (LMB/RMB drag = orbit, wheel = dolly,
//     F = frame the selection) — only when the shell says the cursor is over
//     the viewport and no widget owns it
//   - console commands (scene.spawn / scene.remove / scene.list / view.mode /
//     app.quit) registered with cardinal::console
//   - Esc quits when no text field has focus.
// Backend: "d3d12" / "vulkan" as argv[1] (or CARDINAL_RHI), else Auto.
// =============================================================================

#include <cardinal/rhi/rhi.hpp>
#include <cardinal/window/window.hpp>
#include <cardinal/input/input.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/renderer.hpp>
#include <cardinal/console/console.hpp>
#include <cardinal/cui_shell/shell.hpp>
#include <cardinal/cui_rhi/renderer.hpp>
#include <cardinal/core/diag/log.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>   // std::getenv

using cardinal::u32;
namespace cui = cardinal::cui;
namespace scn = cardinal::scene;
namespace sh  = cardinal::cui_shell;

namespace {

struct OrbitCamera {
    scn::Vec3 target{0.0f, 0.5f, 0.0f};
    float yaw{0.6f}, pitch{0.35f}, distance{7.0f};
    void apply(scn::Camera& cam) const {
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cy = std::cos(yaw),   sy = std::sin(yaw);
        cam.target   = target;
        cam.position = { target.x + distance * cp * sy,
                         target.y + distance * sp,
                         target.z + distance * cp * cy };
        cam.up = { 0.0f, 1.0f, 0.0f };
    }
};

cardinal::string f2s(float v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f", static_cast<double>(v));
    return cardinal::string(b);
}

}  // namespace

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);

    cardinal::rhi::Backend backend = cardinal::rhi::Backend::Auto;
    const char* want = (argc > 1) ? argv[1] : nullptr;
    if (want == nullptr) want = std::getenv("CARDINAL_RHI");
    if (want != nullptr) {
        if (want[0] == 'd' || want[0] == 'D') backend = cardinal::rhi::Backend::D3D12;
        else if (want[0] == 'v' || want[0] == 'V') backend = cardinal::rhi::Backend::Vulkan;
    }

    cardinal::window::WindowDesc wd{};
    wd.title     = "Cardinal Studio (native) - Cardinal Slate, no ImGui";
    wd.width     = 1600;
    wd.height    = 900;
    wd.resizable = true;
    auto window = cardinal::window::Window::create(wd);
    if (window == nullptr) { std::fprintf(stderr, "window creation failed\n"); return 1; }

    auto input = cardinal::input::Manager::create();
    input->attach_to_window(window.get());

    cardinal::rhi::DeviceDesc dd{};
    dd.backend = backend;
    auto device = cardinal::rhi::create_device(dd);
    if (device == nullptr) { std::fprintf(stderr, "device creation failed\n"); return 1; }
    const char* backend_name =
        device->backend() == cardinal::rhi::Backend::Vulkan ? "Vulkan" : "D3D12";
    std::printf("RHI device: %s [%s]\n", device->adapter_name(), backend_name);

    auto swapchain = device->create_swapchain(
        window->native_handle(), window->width(), window->height());
    if (swapchain == nullptr) { std::fprintf(stderr, "swapchain creation failed\n"); return 1; }

    auto ui_renderer = cardinal::cui_rhi::Renderer::create(*device, swapchain->color_format());
    if (ui_renderer == nullptr) { std::fprintf(stderr, "cui renderer creation failed\n"); return 1; }

    // ---- Scene ------------------------------------------------------------
    scn::Scene scene;
    auto mesh_box    = scn::Mesh::make_box(*device, 1.0f);
    auto mesh_plane  = scn::Mesh::make_plane(*device, 12.0f, 8);
    auto mesh_sphere = scn::Mesh::make_sphere(*device, 0.6f, 24);
    {
        auto& ground = scene.add_entity("Ground");
        ground.mesh = mesh_plane;
        ground.tint = { 0.35f, 0.40f, 0.36f };
        auto& box = scene.add_entity("Box");
        box.mesh = mesh_box;
        box.transform.translation = { -1.5f, 0.5f, 0.0f };
        box.tint = { 0.85f, 0.45f, 0.30f };
        auto& ball = scene.add_entity("Sphere");
        ball.mesh = mesh_sphere;
        ball.transform.translation = { 1.5f, 0.6f, 0.0f };
        ball.tint = { 0.35f, 0.55f, 0.90f };
        auto& top = scene.add_entity("Box.Top");
        top.mesh = mesh_box;
        top.transform.translation = { -1.5f, 1.5f, 0.0f };
        top.transform.scale = { 0.6f, 0.6f, 0.6f };
        top.tint = { 0.95f, 0.85f, 0.35f };
        top.parent_id = box.id;
    }
    auto renderer = scn::ForwardRenderer::create(*device, *swapchain);
    if (renderer == nullptr) { std::fprintf(stderr, "forward renderer creation failed\n"); return 1; }
    renderer->set_show_axes(true);
    scn::ViewMode view_mode = scn::ViewMode::Solid;
    OrbitCamera orbit;

    // ---- Shell ------------------------------------------------------------
    sh::Shell shell(scene);
    bool quit = false;
    const int m_file = shell.add_menu("File");
    shell.add_menu_item(m_file, "Spawn box",    [&] { auto& e = scene.add_entity("Box"); e.mesh = mesh_box; e.transform.translation = {0.0f, 0.5f, 2.0f}; });
    shell.add_menu_item(m_file, "Spawn sphere", [&] { auto& e = scene.add_entity("Sphere"); e.mesh = mesh_sphere; e.transform.translation = {0.0f, 0.6f, -2.0f}; });
    shell.add_menu_item(m_file, "Quit",         [&] { quit = true; });
    shell.add_view_menu();
    const int m_help = shell.add_menu("Help");
    shell.add_menu_item(m_help, "About", [&] {
        shell.console_print("Cardinal Studio (native): the editor shell on Cardinal Slate. No ImGui.");
        shell.show_panel(sh::Shell::Panel::Console, true);
    });

    // ---- Console commands ---------------------------------------------------
    namespace cv = cardinal::console;
    CARDINAL_CCOMMAND("scene.spawn", "scene.spawn box|sphere [name]  - add a primitive at the origin",
        [&](const cardinal::vector<cardinal::string>& argv, cv::Output& out) {
            const cardinal::string kind = argv.size() > 1 ? argv[1] : "box";
            const cardinal::string name = argv.size() > 2 ? argv[2] : kind;
            auto& e = scene.add_entity(name);
            e.mesh = (kind == "sphere") ? mesh_sphere : mesh_box;
            e.transform.translation = { 0.0f, 0.5f, 0.0f };
            shell.select_entity(e.id);
            out("spawned %s #%u", name.c_str(), e.id);
        });
    CARDINAL_CCOMMAND("scene.remove", "scene.remove <id>  - remove an entity",
        [&](const cardinal::vector<cardinal::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: scene.remove <id>"); return; }
            const u32 id = static_cast<u32>(std::strtoul(argv[1].c_str(), nullptr, 10));
            out(scene.remove_entity(id) ? "removed #%u" : "no entity #%u", id);
        });
    CARDINAL_CCOMMAND("scene.list", "scene.list  - list entities",
        [&](const cardinal::vector<cardinal::string>&, cv::Output& out) {
            for (const auto& e : scene.entities())
                out("#%u  %s  (%.2f, %.2f, %.2f)%s", e.id, e.name.c_str(),
                    static_cast<double>(e.transform.translation.x),
                    static_cast<double>(e.transform.translation.y),
                    static_cast<double>(e.transform.translation.z),
                    e.visible ? "" : "  [hidden]");
        });
    CARDINAL_CCOMMAND("view.mode", "view.mode solid|wire|normals",
        [&](const cardinal::vector<cardinal::string>& argv, cv::Output& out) {
            const cardinal::string m = argv.size() > 1 ? argv[1] : "solid";
            if (m == "wire")         view_mode = scn::ViewMode::Wireframe;
            else if (m == "normals") view_mode = scn::ViewMode::Normals;
            else                     view_mode = scn::ViewMode::Solid;
            out("view mode: %s", m.c_str());
        });
    CARDINAL_CCOMMAND("app.quit", "app.quit  - exit", [&](const cardinal::vector<cardinal::string>&, cv::Output&) { quit = true; });

    cardinal::log::infof("studio_native", "Cardinal Studio (native) online — %s, %s", device->adapter_name(), backend_name);
    std::puts("Cardinal Studio (native) window up - Cardinal Slate shell, no ImGui. Esc to quit.");

    auto prev = std::chrono::steady_clock::now();
    float fps_smooth = 0.0f;
    u32 frame_no = 0;
    bool orbiting = false;
    while (!window->should_close() && !quit) {
        window->poll_events();

        const auto  now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        if (dt > 0.1f) dt = 0.1f;
        input->begin_frame(dt);
        if (dt > 0.0f) fps_smooth = fps_smooth * 0.9f + (1.0f / dt) * 0.1f;

        u32 nw = 0, nh = 0;
        if (window->consume_resize(&nw, &nh)) swapchain->resize(nw, nh);
        if (window->is_minimized()) continue;

        const float w = static_cast<float>(swapchain->width());
        const float h = static_cast<float>(swapchain->height());

        // ---- UI ------------------------------------------------------------
        shell.layout({ w, h });
        cui::InputState in = sh::make_input_state(*input);
        // Wheel over the viewport dollies the camera, not a scroll panel.
        const bool over_vp = shell.mouse_over_viewport(in.mouse);
        const float wheel = in.scroll;
        if (over_vp) in.scroll = 0.0f;
        shell.update(in);
        if (input->key_pressed(cardinal::input::KeyCode::Tab) && shell.wants_keyboard())
            shell.console_accept_suggestion();
        if (input->key_pressed(cardinal::input::KeyCode::Escape) && !shell.wants_keyboard()) break;

        // ---- Camera ------------------------------------------------------
        const auto& ms = input->mouse_state();
        const bool orbit_btn = input->mouse(cardinal::input::MouseButton::Right).down ||
                               input->mouse(cardinal::input::MouseButton::Left).down;
        if (orbit_btn && (orbiting || over_vp)) {
            orbiting = true;
            orbit.yaw   -= static_cast<float>(ms.dx) * 0.006f;
            orbit.pitch += static_cast<float>(ms.dy) * 0.006f;
            if (orbit.pitch >  1.45f) orbit.pitch =  1.45f;
            if (orbit.pitch < -1.45f) orbit.pitch = -1.45f;
        } else {
            orbiting = false;
        }
        if (over_vp && wheel != 0.0f) {
            orbit.distance *= std::pow(0.9f, wheel);
            if (orbit.distance < 0.5f)   orbit.distance = 0.5f;
            if (orbit.distance > 200.0f) orbit.distance = 200.0f;
        }
        if (input->key_pressed(cardinal::input::KeyCode::F) && !shell.wants_keyboard()) {
            if (auto* e = scene.find_by_id(shell.selected_entity())) orbit.target = e->transform.translation;
        }
        orbit.apply(scene.camera());

        // ---- Stats ---------------------------------------------------------
        if ((frame_no++ % 10) == 0) {
            const cui::Rect vp = shell.viewport_rect();
            shell.set_stats(cardinal::string("fps ") + f2s(fps_smooth) + "   " + backend_name,
                            "entities " + cardinal::to_string(scene.entities().size()) +
                                "   selected #" + cardinal::to_string(shell.selected_entity()),
                            "viewport " + cardinal::to_string(static_cast<int>(vp.width())) + " x " +
                                cardinal::to_string(static_cast<int>(vp.height())));
        }

        cui::DrawList dl;
        shell.paint(dl);

        // ---- Render: scene into the back buffer, UI composited on top -------
        swapchain->begin_frame(0.05f, 0.05f, 0.06f, 1.0f);
        renderer->render(scene, view_mode, w / (h > 1.0f ? h : 1.0f));
        ui_renderer->render(dl, *swapchain, w, h);
        swapchain->end_frame();
    }

    for (const char* n : { "scene.spawn", "scene.remove", "scene.list", "view.mode", "app.quit" })
        cv::Registry::instance().unregister(n);
    return 0;
}
