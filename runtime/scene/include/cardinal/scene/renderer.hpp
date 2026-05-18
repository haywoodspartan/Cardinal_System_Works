#pragma once

// =============================================================================
// Cardinal — forward 3D renderer (Phase 5 baseline).
//
// Walks a Scene and submits a draw per visible entity through the RHI.
// Per-frame view/projection are baked into a CPU-updated transform that's
// applied to vertex data on the CPU (no push constants yet). This is
// pragmatic scaffolding so the editor has *something* to render in its
// viewports while the proper GPU-side constants land in the next iteration.
//
// One ForwardRenderer per swapchain. Reuses the same pipeline across
// frames; the active ViewMode picks which fragment shader variant to bind.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/light.hpp>
#include <cardinal/scene/scene.hpp>

namespace cardinal::rhi { class Device; class Swapchain; class Pipeline; class Buffer; }

namespace cardinal::scene {

// Per-frame counters published after render(). Reset on every render()
// call. Useful for the editor stats overlay and for verifying the
// frustum cull is doing useful work (drawn < total when the camera
// excludes part of the scene).
struct FrameStats {
    u32 entities_total{0};      // every entity walked in Phase 1
    u32 entities_drawn{0};      // passed visibility cull + had geometry
    u32 entities_culled{0};     // failed frustum cull (visible but off-screen)
    u32 work_items{0};          // EntityWork count after entity-splitting
    u32 vertices_emitted{0};    // total verts in cpu_verts_ this frame
    u32 draw_calls{0};          // GPU draw submissions this frame (== work_items today)
    u32 pc_pushes{0};           // push-constant writes (skipped when consecutive draws share MVP)

    // vgeom aggregates — summed across every vgeom-selected mesh this
    // frame. master_tris is what the source meshes WOULD draw without
    // vgeom; drawn_tris is what the per-frame cluster cut actually
    // emits. cluster counts complement the panel's per-mesh table.
    u32 vgeom_master_tris{0};
    u32 vgeom_drawn_tris{0};
    u32 vgeom_cut_clusters{0};
    u32 vgeom_culled_clusters{0};

    // Per-phase wall-clock cost in microseconds. Measured with
    // chrono::high_resolution_clock around each phase. Phase 2 is the
    // wall time of the parallel transform — when JobSystem fans out
    // across N workers this is the longest worker's runtime, not the
    // sum (so Phase 2 can be ~1/N of the work-item total).
    f32 cull_us{0.0f};         // frustum_cull_aabbs pre-pass
    f32 phase1_us{0.0f};       // visibility + sizing + EntityWork build
    f32 phase2_us{0.0f};       // parallel CPU vertex transform
    f32 sort_us{0.0f};         // painter's-algorithm depth sort (fallback path)
    f32 submit_us{0.0f};       // GPU buffer upload + bind + draw
};

class ForwardRenderer {
public:
    static cardinal::unique_ptr<ForwardRenderer> create(rhi::Device& dev,
                                                   rhi::Swapchain& sw);
    virtual ~ForwardRenderer() = default;

    // Renders the scene into whatever the swapchain is currently targeting
    // (the off-screen viewport RTT if one is active, else the back buffer).
    // Must be called between sw->begin_frame() and sw->end_frame().
    //
    // `aspect` is the destination aspect ratio (e.g. viewport_w / viewport_h).
    virtual void render(Scene& scene, ViewMode mode, float aspect) = 0;

    // Optional lighting + ambient for the Solid view mode. When unset (or
    // empty), falls back to the cheap built-in directional light and tint.
    // Pass nullptr to clear.
    virtual void set_light_set(const LightSet* lights) = 0;

    // Stats from the most recent render() call. Zeroed before any
    // render. Hosts query this for editor overlays / profiling panels.
    virtual FrameStats frame_stats() const = 0;

    // Editor gizmo toggles. When on, the renderer appends a wireframe
    // pass after the main draws — entity world-AABBs (green = visible,
    // red = culled), the camera frustum (white), world-origin axes, and
    // per-light cones/spheres/arrows. All default off so non-editor
    // hosts pay nothing.
    virtual void set_show_aabbs(bool on)   = 0;
    virtual void set_show_frustum(bool on) = 0;
    virtual void set_show_axes(bool on)    = 0;   // RGB axes at world origin
    virtual void set_show_lights(bool on)  = 0;   // cones / spheres / dirs for the bound LightSet

protected:
    ForwardRenderer() = default;
};

}  // namespace cardinal::scene
