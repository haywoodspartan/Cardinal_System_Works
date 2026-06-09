#pragma once

// =============================================================================
// Cardinal UI — native RHI renderer for Cardinal Slate (cui).
//
// This is the GPU backend that draws a cui::DrawList directly through the RHI
// (Vulkan / D3D12), with NO ImGui involved — the analog of imgui_impl_vulkan,
// but for our own framework. It lets Cardinal Slate run as a real native,
// GPU-rendered windowed UI (see samples/10_slate_native).
//
// Everything renders through ONE pipeline of colored 2D quads (position +
// packed RGBA). Rects, strokes and lines become quads directly; text is drawn
// as tiny quads from an embedded 8x8 bitmap font (font8x8.hpp). This avoids
// needing a font-atlas texture (the public RHI has no texture-upload path yet)
// and keeps the whole UI to a single vertex buffer + draw.
//
// Usage (between Swapchain::begin_frame and end_frame):
//     renderer.render(draw_list, swapchain, screen_w, screen_h);
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/cui/draw.hpp>
#include <cardinal/rhi/rhi.hpp>

namespace cardinal::cui_rhi {

// One UI vertex: pixel-space position + packed RGBA (R8G8B8A8_UNORM).
struct Vertex {
    float x, y;
    u32   rgba;
};

class Renderer {
public:
    // Build the colored-quad pipeline against the swapchain's color format.
    // Returns null on shader-compile / pipeline-creation failure.
    static cardinal::unique_ptr<Renderer> create(cardinal::rhi::Device& device,
                                                 cardinal::rhi::Format color_format);

    // Convert the draw list to quads, upload, and record the draw into the
    // swapchain's current frame. Call between begin_frame() and end_frame().
    void render(const cardinal::cui::DrawList& dl,
                cardinal::rhi::Swapchain& sc,
                float screen_w, float screen_h);

    cardinal::usize last_vertex_count() const noexcept { return verts_.size(); }

    // Convert a draw list into the colored-quad vertex stream (the geometry the
    // renderer uploads). Pure + device-free, so it's unit-testable headlessly.
    static void build_geometry(const cardinal::cui::DrawList& dl,
                               cardinal::vector<Vertex>& out);

private:
    Renderer() = default;
    void ensure_capacity(u32 slot, cardinal::usize vertex_count);

    static constexpr u32 kFrames = 2;   // ring-buffer the VB across frames-in-flight

    cardinal::rhi::Device*                        device_{nullptr};
    cardinal::unique_ptr<cardinal::rhi::Pipeline> pipeline_;
    cardinal::unique_ptr<cardinal::rhi::Buffer>   vbuf_[kFrames];
    cardinal::usize                               vbuf_capacity_[kFrames]{0, 0};   // in vertices
    u32                                           frame_{0};
    cardinal::vector<Vertex>                      verts_;
};

// Pack a cui::Color into the R8G8B8A8_UNORM layout the shader reads.
inline u32 pack_color(cardinal::cui::Color c) noexcept {
    return (static_cast<u32>(c.r))        |
           (static_cast<u32>(c.g) <<  8)  |
           (static_cast<u32>(c.b) << 16)  |
           (static_cast<u32>(c.a) << 24);
}

}  // namespace cardinal::cui_rhi
