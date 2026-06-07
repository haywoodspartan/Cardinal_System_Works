#pragma once

// =============================================================================
// Cardinal — post-processing chain.
//
// A `Chain` is an ordered sequence of `Pass`es applied after the main scene
// render. Each Pass reads an RGBA32F framebuffer + writes an RGBA32F
// framebuffer; the Chain ping-pongs between two scratch buffers so the
// final result lands in the caller's output.
//
// Pass design mirrors render::Pipeline's Knob-driven introspection — the
// editor renders a generic settings panel without knowing about specific
// passes, plugins register new passes the same way they register algos.
//
// Today's stock passes (all CPU reference impls; GPU shader variants land
// when the RHI post-process pipeline arrives):
//
//   Bloom                  — threshold + box-blur + add
//   FXAA                   — luminance-edge detection + directional blur
//   Vignette               — radial darkening
//   ChromaticAberration    — per-channel UV shift radially
//   FilmGrain              — hash-based per-pixel noise
//
// All passes honour the Knob NaN-safety discipline established elsewhere
// in the engine: every public float knob value is sanitised at the use
// site (NaN → safe default) so a corrupt Knob slot can't poison the
// framebuffer.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/budget/memory.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/render/pipeline.hpp>   // for render::Knob

namespace cardinal::postfx {

// Reuse the render module's Knob so the editor panel renders postfx
// settings with the same widget code it uses for pipeline knobs.
using Knob = cardinal::render::Knob;

// ---------------------------------------------------------------------------
// Pass — one effect in the chain.
// ---------------------------------------------------------------------------
class Pass {
public:
    virtual ~Pass() = default;
    Pass(const Pass&)            = delete;
    Pass& operator=(const Pass&) = delete;

    // Stable identifier ("bloom", "fxaa", …). Used to persist chain order
    // + pick up the matching GPU shader variant when shaders land.
    virtual const char* id()          const noexcept = 0;
    virtual const char* label()       const noexcept = 0;
    virtual const char* description() const noexcept = 0;

    // Mutable so the editor can write directly. Persistent across hot
    // reloads (chain-side serialisation is a follow-up).
    virtual cardinal::vector<Knob>& knobs() noexcept = 0;

    // Run the pass over `in_rgba` → `out_rgba`. Both buffers are
    // width*height*4 floats, row-major, top-left origin. `in_rgba` may
    // equal `out_rgba` (in-place); passes that need separate read/write
    // buffers allocate scratch internally.
    //
    // Implementations MUST tolerate non-finite knob values (sanitise at
    // use site) and non-finite input pixels (clamp / propagate-as-zero).
    virtual void apply_cpu(const float* in_rgba,
                           float*       out_rgba,
                           u32          width,
                           u32          height) noexcept = 0;

    // Whether the chain runs this pass. Toggle from the editor.
    bool enabled{true};

protected:
    Pass() = default;
};

// ---------------------------------------------------------------------------
// Chain — owns a sequence of passes, applies them in order.
// ---------------------------------------------------------------------------
class Chain {
public:
    static cardinal::shared_ptr<Chain> create();
    virtual ~Chain() = default;
    Chain(const Chain&)            = delete;
    Chain& operator=(const Chain&) = delete;

    // ----- Chain composition --------------------------------------------
    virtual void  add  (cardinal::unique_ptr<Pass> p) = 0;
    virtual bool  remove(usize i) = 0;                  // false if i OOB
    virtual void  clear() = 0;
    virtual bool  move_up  (usize i) = 0;               // false if i==0 / OOB
    virtual bool  move_down(usize i) = 0;               // false if i==back / OOB

    virtual const cardinal::vector<cardinal::unique_ptr<Pass>>&
                                            passes() const noexcept = 0;
    virtual usize                           size()   const noexcept = 0;

    // Pass lookup by id (first match). Returns nullptr if missing.
    virtual Pass* find(const char* id) noexcept = 0;

    // ----- Apply --------------------------------------------------------
    // Run the active chain. The Chain owns one scratch buffer reused
    // across frames; this resizes if (w,h) changes. On an empty / fully-
    // disabled chain, copies in_rgba → out_rgba (cheap memcpy when
    // distinct, no-op when same pointer).
    virtual void apply_cpu(const float* in_rgba,
                           float*       out_rgba,
                           u32          width,
                           u32          height) = 0;

    // Per-frame counters published after apply_cpu(). Useful for the
    // editor stats overlay and profiling.
    struct Stats {
        u32 passes_run             {0};
        u32 passes_skipped_disabled{0};
        f32 total_us               {0.0f};
    };
    virtual Stats stats() const noexcept = 0;

protected:
    Chain() = default;
};

// ---------------------------------------------------------------------------
// Stock pass factories. Each returns a freshly-constructed Pass with the
// default knob set — callers can re-tune knobs() before adding to a Chain.
// ---------------------------------------------------------------------------
cardinal::unique_ptr<Pass> make_bloom_pass();
cardinal::unique_ptr<Pass> make_fxaa_pass();
cardinal::unique_ptr<Pass> make_vignette_pass();
cardinal::unique_ptr<Pass> make_chromatic_aberration_pass();
cardinal::unique_ptr<Pass> make_film_grain_pass();

}  // namespace cardinal::postfx
