// =============================================================================
// Cardinal Slate — native windowed UI sample (NO ImGui).
//
// Proves Cardinal's OWN UI framework as a real native, GPU-rendered windowed
// app — the equivalent of an ImGui app, but driven entirely by cardinal::cui
// (the framework) + cardinal::cui_rhi (the native RHI renderer). It opens a
// native window, creates the RHI device + swapchain, builds a Slate widget
// tree, and each frame: maps OS mouse input into the framework, lays out,
// paints to a draw list, and renders that draw list through the RHI (Vulkan/
// D3D12). Esc to quit.
// =============================================================================

#include <cardinal/rhi/rhi.hpp>
#include <cardinal/window/window.hpp>
#include <cardinal/input/input.hpp>
#include <cardinal/cui/context.hpp>
#include <cardinal/cui/widgets.hpp>
#include <cardinal/cui_rhi/renderer.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>   // std::getenv

using cardinal::u32;
namespace cui = cardinal::cui;

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);

    // Backend select: "d3d12" / "vulkan" arg (or CARDINAL_RHI env), else Auto.
    cardinal::rhi::Backend backend = cardinal::rhi::Backend::Auto;
    const char* want = (argc > 1) ? argv[1] : nullptr;
    if (want == nullptr) want = std::getenv("CARDINAL_RHI");
    if (want != nullptr) {
        if (want[0] == 'd' || want[0] == 'D') backend = cardinal::rhi::Backend::D3D12;
        else if (want[0] == 'v' || want[0] == 'V') backend = cardinal::rhi::Backend::Vulkan;
    }

    cardinal::window::WindowDesc wd{};
    wd.title     = "Cardinal Slate - native UI (no ImGui)";
    wd.width     = 900;
    wd.height    = 600;
    wd.resizable = true;
    auto window = cardinal::window::Window::create(wd);
    if (window == nullptr) { std::fprintf(stderr, "window creation failed\n"); return 1; }

    auto input = cardinal::input::Manager::create();
    input->attach_to_window(window.get());

    cardinal::rhi::DeviceDesc dd{};
    dd.backend = backend;
    auto device = cardinal::rhi::create_device(dd);
    if (device == nullptr) { std::fprintf(stderr, "device creation failed\n"); return 1; }
    std::printf("RHI device: %s [%s]\n", device->adapter_name(),
                device->backend() == cardinal::rhi::Backend::Vulkan ? "Vulkan" : "D3D12");

    auto swapchain = device->create_swapchain(
        window->native_handle(), window->width(), window->height());
    if (swapchain == nullptr) { std::fprintf(stderr, "swapchain creation failed\n"); return 1; }

    auto renderer = cardinal::cui_rhi::Renderer::create(*device, swapchain->color_format());
    if (renderer == nullptr) { std::fprintf(stderr, "cui renderer creation failed\n"); return 1; }

    // ---- Build the Cardinal Slate widget tree ----------------------------
    // Bound state lives here so it outlives the tree.
    cui::Ui      ui;
    bool         enabled = true;
    float        slider  = 0.5f;
    int          clicks  = 0;
    cui::Label*  clicks_lbl = nullptr;
    {
        using namespace cui;
        auto panel = cardinal::make_unique<Panel>(ui.theme().panel_bg, Edges::all(16.0f));
        auto stack = cardinal::make_unique<Stack>(Axis::Vertical, 10.0f);
        stack->add(cardinal::make_unique<Label>("CARDINAL SLATE - NATIVE UI (NO IMGUI)"));
        stack->add(cardinal::make_unique<Button>("CLICK ME", [&clicks]() { ++clicks; }));
        clicks_lbl = static_cast<Label*>(
            stack->add(cardinal::make_unique<Label>("CLICKS: 0")));
        stack->add(cardinal::make_unique<Checkbox>("ENABLED", &enabled));
        stack->add(cardinal::make_unique<Slider>(&slider, 0.0f, 1.0f));
        panel->add(cardinal::move(stack));
        ui.set_root(cardinal::move(panel));
    }

    std::puts("Cardinal Slate native window up - our own UI framework, GPU-rendered "
              "through the RHI. Esc to quit.");

    auto prev = std::chrono::steady_clock::now();
    while (!window->should_close()) {
        window->poll_events();

        const auto  now = std::chrono::steady_clock::now();
        const float dt  = std::chrono::duration<float>(now - prev).count();
        prev = now;
        input->begin_frame(dt);

        if (input->key_pressed(cardinal::input::KeyCode::Escape)) break;

        u32 nw = 0, nh = 0;
        if (window->consume_resize(&nw, &nh)) swapchain->resize(nw, nh);
        if (window->is_minimized()) continue;

        const float w = static_cast<float>(swapchain->width());
        const float h = static_cast<float>(swapchain->height());

        // Map OS mouse into the framework's input snapshot.
        const auto& m = input->mouse_state();
        cui::InputState in;
        in.mouse      = { static_cast<float>(m.x), static_cast<float>(m.y) };
        in.mouse_down = input->mouse(cardinal::input::MouseButton::Left).down;

        ui.layout({ w, h });
        ui.update_input(in);
        if (clicks_lbl) clicks_lbl->set_text("CLICKS: " + cardinal::to_string(clicks));

        cui::DrawList dl;
        ui.paint(dl);

        swapchain->begin_frame(0.06f, 0.06f, 0.07f, 1.0f);
        renderer->render(dl, *swapchain, w, h);
        swapchain->end_frame();
    }

    return 0;
}
