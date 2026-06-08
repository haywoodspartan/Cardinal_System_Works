// =============================================================================
// Cardinal UI — native RHI renderer for Cardinal Slate (see renderer.hpp).
// =============================================================================
#include <cardinal/cui_rhi/renderer.hpp>
#include <cardinal/cui_rhi/font8x8.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/cmath.hpp>   // cardinal::sqrt

namespace cardinal::cui_rhi {

namespace rhi = cardinal::rhi;
namespace cui = cardinal::cui;

namespace {

// One pipeline, one shader: pixel-space position -> NDC, flat vertex color.
// (Color arrives as a normalized float4 from the R8G8B8A8_UNORM attribute.)
constexpr const char* kShader = R"HLSL(
// Screen size for pixel-space -> NDC, delivered as a push/root constant.
// Cross-backend: DXC requires [[vk::push_constant]] on a STRUCT global for
// SPIR-V (Vulkan), while D3D12 binds set_push_constants to root 32-bit
// constants at b0 (an HLSL `cbuffer ... : register(b0)`). DXC predefines
// __spirv__ when generating SPIR-V, so branch on it.
#ifdef __spirv__
struct PushData { float2 uScreen; };
[[vk::push_constant]] PushData g_push;
#define U_SCREEN g_push.uScreen
#else
cbuffer Push : register(b0) { float2 g_uScreen; };
#define U_SCREEN g_uScreen
#endif

struct VSIn  { float2 pos : POSITION0; float4 col : COLOR0; };
struct VSOut { float4 pos : SV_POSITION; float4 col : COLOR0; };

VSOut VSMain(VSIn i) {
    VSOut o;
    float2 s = U_SCREEN;
    o.pos = float4((i.pos.x / s.x) * 2.0 - 1.0,
                   1.0 - (i.pos.y / s.y) * 2.0,   // top-left origin
                   0.0, 1.0);
    o.col = i.col;
    return o;
}

float4 PSMain(VSOut i) : SV_TARGET { return i.col; }
)HLSL";

}  // namespace

cardinal::unique_ptr<Renderer> Renderer::create(rhi::Device& device, rhi::Format color_format) {
    rhi::ShaderBlob vs = device.compile_shader(rhi::ShaderStage::Vertex,   kShader, "VSMain");
    rhi::ShaderBlob fs = device.compile_shader(rhi::ShaderStage::Fragment, kShader, "PSMain");
    if (!vs.ok() || !fs.ok()) {
        cardinal::log::errorf("cui_rhi", "UI shader compile failed");
        return nullptr;
    }

    rhi::PipelineDesc pd;
    pd.vertex_shader   = cardinal::move(vs);
    pd.fragment_shader = cardinal::move(fs);
    pd.vertex_entry    = "VSMain";
    pd.fragment_entry  = "PSMain";
    pd.vertex_attribs  = {
        { 0, rhi::Format::R32G32_Float,   0 },   // pos
        { 1, rhi::Format::R8G8B8A8_UNORM, 8 },   // packed color
    };
    pd.vertex_stride      = sizeof(Vertex);   // 12
    pd.topology           = rhi::PrimitiveTopology::TriangleList;
    pd.color_format       = color_format;
    pd.depth_format       = rhi::Format::Unknown;
    pd.depth_test         = false;
    pd.depth_write        = false;
    pd.push_constant_size  = sizeof(float) * 2;   // float2 screen size

    auto pipe = device.create_pipeline(pd);
    if (!pipe) {
        cardinal::log::errorf("cui_rhi", "UI pipeline creation failed");
        return nullptr;
    }

    auto r = cardinal::unique_ptr<Renderer>(new Renderer());
    r->device_   = &device;
    r->pipeline_ = cardinal::move(pipe);
    return r;
}

void Renderer::ensure_capacity(u32 slot, cardinal::usize vcount) {
    if (vbuf_[slot] && vcount <= vbuf_capacity_[slot]) return;
    cardinal::usize cap = vbuf_capacity_[slot] ? vbuf_capacity_[slot] : 4096;
    while (cap < vcount) cap *= 2;
    rhi::BufferDesc bd;
    bd.size         = cap * sizeof(Vertex);
    bd.usage        = static_cast<u32>(rhi::BufferUsage::Vertex);
    bd.cpu_writable = true;
    vbuf_[slot]          = device_->create_buffer(bd);
    vbuf_capacity_[slot] = vbuf_[slot] ? cap : 0;
}

void Renderer::quad(float x, float y, float w, float h, u32 rgba) {
    const Vertex a{ x,     y,     rgba };
    const Vertex b{ x + w, y,     rgba };
    const Vertex c{ x + w, y + h, rgba };
    const Vertex d{ x,     y + h, rgba };
    verts_.push_back(a); verts_.push_back(b); verts_.push_back(c);
    verts_.push_back(a); verts_.push_back(c); verts_.push_back(d);
}

void Renderer::stroke(float x, float y, float w, float h, float t, u32 rgba) {
    if (t < 1.0f) t = 1.0f;
    quad(x,         y,         w, t,         rgba);   // top
    quad(x,         y + h - t, w, t,         rgba);   // bottom
    quad(x,         y,         t, h,         rgba);   // left
    quad(x + w - t, y,         t, h,         rgba);   // right
}

void Renderer::line(float x0, float y0, float x1, float y1, float t, u32 rgba) {
    if (t < 1.0f) t = 1.0f;
    float dx = x1 - x0, dy = y1 - y0;
    const float len = cardinal::sqrt(dx * dx + dy * dy);
    if (len <= 0.0001f) return;
    dx /= len; dy /= len;                       // unit direction
    const float px = -dy * (t * 0.5f);          // perpendicular * half-thickness
    const float py =  dx * (t * 0.5f);
    const Vertex a{ x0 + px, y0 + py, rgba };
    const Vertex b{ x1 + px, y1 + py, rgba };
    const Vertex c{ x1 - px, y1 - py, rgba };
    const Vertex d{ x0 - px, y0 - py, rgba };
    verts_.push_back(a); verts_.push_back(b); verts_.push_back(c);
    verts_.push_back(a); verts_.push_back(c); verts_.push_back(d);
}

void Renderer::text(float x, float y, const cardinal::string& s, float font_size, u32 rgba) {
    if (font_size <= 0.0f) font_size = 14.0f;
    const float px      = font_size / 8.0f;     // square cell pixel
    const float advance = px * 6.0f;            // 6 of 8 columns (matches text_advance 0.75)
    float pen = x;
    for (char ch : s) {
        const u8* g = glyph8x8(ch);
        for (int row = 0; row < 8; ++row) {
            const u8 bits = g[row];
            if (bits == 0) continue;
            for (int col = 0; col < 8; ++col) {
                if (bits & (1u << col)) {
                    // +0.5 so adjacent pixels overlap slightly (no seams)
                    quad(pen + col * px, y + row * px, px + 0.5f, px + 0.5f, rgba);
                }
            }
        }
        pen += advance;
    }
}

void Renderer::render(const cui::DrawList& dl, rhi::Swapchain& sc,
                      float screen_w, float screen_h) {
    verts_.clear();
    for (const auto& cmd : dl.cmds()) {
        const u32 col = pack_color(cmd.color);
        switch (cmd.kind) {
        case cui::DrawKind::RectFilled:
            quad(cmd.rect.pos.x, cmd.rect.pos.y, cmd.rect.size.x, cmd.rect.size.y, col);
            break;
        case cui::DrawKind::RectStroke:
            stroke(cmd.rect.pos.x, cmd.rect.pos.y, cmd.rect.size.x, cmd.rect.size.y,
                   cmd.thickness, col);
            break;
        case cui::DrawKind::Line:
            line(cmd.rect.pos.x, cmd.rect.pos.y, cmd.p1.x, cmd.p1.y, cmd.thickness, col);
            break;
        case cui::DrawKind::Text:
            text(cmd.rect.pos.x, cmd.rect.pos.y, cmd.text, cmd.font_size, col);
            break;
        }
    }
    if (verts_.empty()) return;

    const u32 slot = frame_;
    ensure_capacity(slot, verts_.size());
    if (!vbuf_[slot]) return;
    vbuf_[slot]->upload(verts_.data(), verts_.size() * sizeof(Vertex), 0);

    sc.bind_pipeline(pipeline_.get());
    sc.bind_vertex_buffer(vbuf_[slot].get(), 0);
    const float screen[2] = { screen_w, screen_h };
    sc.set_push_constants(0, screen, sizeof(screen));
    sc.draw(static_cast<u32>(verts_.size()), 1, 0, 0);

    frame_ = (frame_ + 1) % kFrames;
}

}  // namespace cardinal::cui_rhi
