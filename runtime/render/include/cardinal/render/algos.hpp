#pragma once

// =============================================================================
// Cardinal — runtime-selectable math algorithms for GPU render functions.
//
// The render system is full of "you could pick from N algorithms" decisions:
//
//   * Tonemap       — Linear / Reinhard / Filmic / ACES (approx) / ACES (full) /
//                     AgX / Lottes
//   * Hash / RNG    — Wang / PCG / Hash13 / Interleaved-gradient noise
//   * Mip filter    — Box / Tent / Kaiser / Lanczos2 / Catmull-Rom
//   * Cluster cull  — None / Frustum / Frustum+Cone / Frustum+Cone+HiZ (stub)
//   * Tess factor   — Distance / Edge-screen / Phong (PN-tri)
//   * Sampling      — Halton / Sobol / Bayer / Hammersley / Blue noise
//
// AlgoRegistry centralises that catalogue. Each Algo has:
//
//   - id       : stable string used to wire shader variants + persist UI
//   - label    : human-friendly name shown in the picker
//   - tooltip  : one-liner explaining the trade-off
//   - cpu_fn   : optional CPU reference impl so the editor can preview /
//                unit-test without firing a GPU pass
//   - hlsl     : optional shader-side identifier (function name in the
//                cardinal_*.hlsli library) — pipelines specialise their
//                shader using this to generate the right variant
//
// Engine code (and plugins) register new algorithms with `register_algo()`.
// Pipelines build their enum knobs by querying `AlgoRegistry::category()`
// — no enum proliferation, no hard-coded per-category combo lists.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/budget/memory.hpp>
#include <cardinal/core/std/containers.hpp>

namespace cardinal::render::algo {

// Categories are identified by short ids that double as both the stable
// persistence key and the prefix on shader variant macros.
enum class CategoryId : u32 {
    Tonemap = 0,           // outputs linear → display-encoded
    HashRng,               // pseudo-random / hash functions
    MipFilter,             // downsample kernel for mip generation
    ClusterCull,           // visibility test for meshlets
    TessFactor,            // hull-shader factor policy
    Sampling,              // sampling pattern for AA / GI
    NormalEncode,          // normal-vector compression scheme
    PrecisionMath,         // FP32 / FP16 / FP8-E4M3 / FP8-E5M2 / FP4-E2M1 / FP4-E3M0
    Count_
};

const char* category_name(CategoryId id) noexcept;
const char* category_description(CategoryId id) noexcept;

// Polymorphic value passed to a CPU reference impl. The typed slot a
// CPU function reads depends on the category — every algorithm in the
// same category honours the same slot contract:
//
//   Tonemap         : in.color3 = HDR colour     → out.color3 = display
//   HashRng         : in.seed   = u32 seed       → out.unit3 = (x,y,z) in [0,1]
//   MipFilter       : in.samples4 = 2x2 footprint → out.color3 = filtered
//   ClusterCull     : in.cluster + in.frustum + in.camera → out.flag (0/1)
//   TessFactor      : in.distance / in.edge_pixels → out.factor
//   NormalEncode    : in.unit3 = normal           → out.color3 (xy + .z=0)
//   Sampling        : in.index  = sample index   → out.color3 = (u,v,0)
struct AlgoIn {
    float       distance{0.0f};
    float       edge_pixels{0.0f};
    u32         seed{0};
    u32         index{0};
    float       color3[3]{};
    float       unit3[3]{};
    float       samples4[12]{};   // 4 RGB samples
    // Cluster cull also wants pointer-to-bounds + frustum + camera, so
    // for that category the registry uses a typed function pointer instead
    // (see ClusterCullFn below).
};
struct AlgoOut {
    float       color3[3]{};
    float       unit3[3]{};
    float       factor{0.0f};
    int         flag{0};
};

// Most algorithms fit a single CPU functor signature. Cluster-cull uses
// a different shape — declared separately to avoid forcing the rest into
// a more awkward variant.
using AlgoFn = cardinal::function<void(const AlgoIn&, AlgoOut&)>;

struct Algo {
    CategoryId  category{CategoryId::Tonemap};
    cardinal::string id;            // "aces_approx", "wang_hash", ...
    cardinal::string label;         // "ACES (approx)"
    cardinal::string tooltip;
    cardinal::string hlsl_function; // empty if no GPU-side counterpart
    AlgoFn      cpu_fn;        // empty if no CPU-side reference impl
    bool        is_default{false};
    bool        is_user{false};// true when registered after engine init
};

// ---------------------------------------------------------------------------
// AlgoRegistry — singleton-but-explicit. instance() lazily constructs and
// populates with the engine's curated set.
// ---------------------------------------------------------------------------
class AlgoRegistry {
public:
    static AlgoRegistry& instance();

    // Register a new algorithm. Plugins / users call this; engine init
    // calls it for each curated entry. Duplicate ids in the same category
    // log a warning and reject the new one (caller wins → first-loaded).
    bool register_algo(Algo a);

    // Read-only enumeration of every algorithm in a category, in
    // insertion order (the default impl is always first).
    const cardinal::vector<Algo>& list(CategoryId c) const;

    // Lookup by id (within a category). Returns nullptr when missing.
    const Algo* find(CategoryId c, const char* id) const;

    // Pick the default algorithm for a category (first registered).
    const Algo* default_for(CategoryId c) const;

private:
    AlgoRegistry();
    AlgoRegistry(const AlgoRegistry&)            = delete;
    AlgoRegistry& operator=(const AlgoRegistry&) = delete;

    struct Impl;
    cardinal::unique_ptr<Impl> p_;
};

// ---------------------------------------------------------------------------
// Helpers — turn a category into a Knob-friendly enum_labels list. Pipelines
// call these to populate their Knob lists at construction time.
// ---------------------------------------------------------------------------
cardinal::vector<cardinal::string> labels_for(CategoryId c);
int                      default_index_for(CategoryId c);
const char*              hlsl_for_choice(CategoryId c, int idx);
const char*              id_for_choice  (CategoryId c, int idx);

}  // namespace cardinal::render::algo
