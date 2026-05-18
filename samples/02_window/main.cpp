#include <cardinal/rhi/rhi.hpp>
#include <cardinal/window/window.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>

using cardinal::u32;

// Vertex layout that matches the HLSL VSIn below.
struct Vertex {
    float pos[2];
    float color[3];
};

// Source-of-truth shader, compiled at runtime. Two entry points; the same
// VSOut is the interpolated payload from VS to PS.
constexpr const char* triangle_hlsl = R"(
struct VSIn {
    float2 pos   : POSITION0;
    float3 color : COLOR0;
};

struct VSOut {
    float4 pos   : SV_POSITION;
    float3 color : COLOR0;
};

VSOut VSMain(VSIn i) {
    VSOut o;
    o.pos   = float4(i.pos, 0.0, 1.0);
    o.color = i.color;
    return o;
}

float4 PSMain(VSOut i) : SV_TARGET {
    return float4(i.color, 1.0);
}
)";

int main() {
    std::setbuf(stdout, nullptr);

    cardinal::window::WindowDesc wdesc{};
    wdesc.title     = "Cardinal — vertex-buffer triangle";
    wdesc.width     = 1280;
    wdesc.height    = 720;
    wdesc.resizable = true;

    auto window = cardinal::window::Window::create(wdesc);
    if (window == nullptr) { std::fprintf(stderr, "Window creation failed\n"); return 1; }

    cardinal::rhi::DeviceDesc ddesc{};
    auto device = cardinal::rhi::create_device(ddesc);
    if (device == nullptr) { std::fprintf(stderr, "Device creation failed\n"); return 1; }
    std::printf("RHI device: %s [%s]\n", device->adapter_name(),
                device->backend() == cardinal::rhi::Backend::Vulkan ? "Vulkan" : "D3D12");

    auto swapchain = device->create_swapchain(
        window->native_handle(), window->width(), window->height());
    if (swapchain == nullptr) { std::fprintf(stderr, "Swapchain creation failed\n"); return 1; }
    std::printf("Swapchain %ux%u\n", swapchain->width(), swapchain->height());

    // Phase 5.x viewport — render the scene at half-res into an off-screen
    // image, then end_frame() blit-scales it to the swapchain. Foundation
    // for DLSS (low-res in, high-res out) and the future editor viewport
    // panel (which will ImGui::Image() this same image).
    swapchain->set_viewport_size(swapchain->width()  / 2,
                                 swapchain->height() / 2);
    std::printf("Viewport %ux%u (rendering at 50%% scale)\n",
                swapchain->viewport_width(), swapchain->viewport_height());

    // ---- Geometry ----------------------------------------------------------
    const Vertex verts[] = {
        {{ 0.0f,  0.5f}, {1.0f, 0.2f, 0.2f}},
        {{ 0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
        {{-0.5f, -0.5f}, {0.2f, 0.4f, 1.0f}},
    };

    cardinal::rhi::BufferDesc bd{};
    bd.size         = sizeof(verts);
    bd.usage        = static_cast<u32>(cardinal::rhi::BufferUsage::Vertex)
                    | static_cast<u32>(cardinal::rhi::BufferUsage::AccelerationStructureBuild)
                    | static_cast<u32>(cardinal::rhi::BufferUsage::ShaderDeviceAddress);
    bd.cpu_writable = true;
    auto vbuf = device->create_buffer(bd);
    if (vbuf == nullptr) { std::fprintf(stderr, "Vertex buffer creation failed\n"); return 1; }
    vbuf->upload(verts, sizeof(verts));

    // ---- Phase 5: build a BLAS from the triangle, if RT is supported ------
    const auto& caps = device->capabilities();
    std::printf("RT support — accel_struct=%d ray_pipeline=%d ray_query=%d\n",
                caps.acceleration_structure, caps.ray_tracing_pipeline, caps.ray_query);
    std::unique_ptr<cardinal::rhi::AccelerationStructure> blas;
    if (caps.acceleration_structure) {
        cardinal::rhi::BlasDesc bld{};
        cardinal::rhi::BlasGeometryDesc geom{};
        geom.vertex_buffer = vbuf.get();
        geom.vertex_format = cardinal::rhi::Format::R32G32_Float;  // matches `pos`
        geom.vertex_stride = sizeof(Vertex);
        geom.vertex_count  = 3;
        geom.opaque        = true;
        bld.geometries.push_back(geom);
        bld.prefer_fast_trace = true;
        blas = device->create_blas(bld);
        if (blas != nullptr) {
            std::printf("BLAS built — device address: 0x%016llx\n",
                        static_cast<unsigned long long>(blas->device_address()));
        } else {
            std::fprintf(stderr, "BLAS build returned null\n");
        }
    }

    // ---- Shaders + pipeline ------------------------------------------------
    auto vs_blob = device->compile_shader(cardinal::rhi::ShaderStage::Vertex,
                                          triangle_hlsl, "VSMain");
    auto fs_blob = device->compile_shader(cardinal::rhi::ShaderStage::Fragment,
                                          triangle_hlsl, "PSMain");
    if (!vs_blob.ok() || !fs_blob.ok()) {
        std::fprintf(stderr, "Shader compile failed\n");
        return 1;
    }

    cardinal::rhi::PipelineDesc pd{};
    pd.vertex_shader   = std::move(vs_blob);
    pd.fragment_shader = std::move(fs_blob);
    pd.vertex_stride   = sizeof(Vertex);
    pd.vertex_attribs  = {
        { 0, cardinal::rhi::Format::R32G32_Float,    offsetof(Vertex, pos)   },
        { 1, cardinal::rhi::Format::R32G32B32_Float, offsetof(Vertex, color) },
    };
    pd.color_format    = swapchain->color_format();
    pd.topology        = cardinal::rhi::PrimitiveTopology::TriangleList;

    auto pipeline = device->create_pipeline(pd);
    if (pipeline == nullptr) { std::fprintf(stderr, "Pipeline creation failed\n"); return 1; }

    std::puts("Sample driving geometry through public RHI — vertex buffer + pipeline + draw.");

    // ---- Render loop -------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    while (!window->should_close()) {
        window->poll_events();

        const float t = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t0).count();
        const float r = 0.05f + 0.05f * std::sin(t * 1.0f);
        const float g = 0.05f + 0.05f * std::sin(t * 1.3f + 2.0f);
        const float b = 0.10f + 0.05f * std::sin(t * 1.7f + 4.0f);

        swapchain->begin_frame(r, g, b, 1.0f);
        swapchain->bind_pipeline(pipeline.get());
        swapchain->bind_vertex_buffer(vbuf.get(), 0);
        swapchain->draw(3, 1, 0, 0);
        swapchain->end_frame();
    }

    return 0;
}
