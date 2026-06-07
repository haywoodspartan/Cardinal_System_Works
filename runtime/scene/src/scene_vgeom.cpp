// =============================================================================
// Cardinal — scene-side virtualized geometry attachment.
//
// vgeom is a lower-layer module that operates on raw vertex blobs. Mesh
// is a scene concept. The integration glue (per-Mesh registry + cook-
// from-Mesh helper) lives here so the layering goes
//   core ← vgeom ← scene
// instead of cycling.
// =============================================================================
#include <cardinal/scene/scene.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>

#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/thread.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::scene {

namespace {

struct Attachment {
    cardinal::shared_ptr<cardinal::vgeom::Hierarchy> hierarchy;
    bool                                        enabled{true};
    cardinal::vgeom::FrameStats                 last_stats{};
};

// Side table keyed by Mesh*. The Mesh outlives its entry; eviction is
// implicit when a Mesh is destroyed (lookups simply miss). Stale
// entries persist after Mesh death until explicitly cleared, but
// they're tiny — and migrating this to a real Mesh field is a one-
// line follow-up once the API stabilises.
cardinal::mutex                                       g_mtx;
cardinal::unordered_map<const Mesh*, Attachment>      g_attach;

}  // namespace

bool vgeom_attach(Mesh& mesh) {
    if (mesh.cpu_vertices() == nullptr || mesh.vertex_count() == 0) {
        cardinal::log::warnf("scene/vgeom",
            "vgeom_attach: mesh '%s' has no CPU shadow",
            mesh.name().c_str());
        return false;
    }

    // Static asserts at the boundary: vgeom::Vertex must be layout-
    // compatible with scene::Vertex so the reinterpret_cast below is
    // legal. If these ever fire, the cook will silently misinterpret
    // verts (catastrophic).
    static_assert(sizeof(cardinal::vgeom::Vertex) == sizeof(Vertex),
        "vgeom::Vertex / scene::Vertex size mismatch");
    static_assert(alignof(cardinal::vgeom::Vertex) == alignof(Vertex),
        "vgeom::Vertex / scene::Vertex alignment mismatch");

    cardinal::vgeom::CookDesc d{};
    d.vertices     = reinterpret_cast<const cardinal::vgeom::Vertex*>(mesh.cpu_vertices());
    d.vertex_count = mesh.vertex_count();
    d.name         = mesh.name().c_str();

    auto h = cardinal::vgeom::cook(d);
    if (h == nullptr) return false;

    Attachment a{};
    a.hierarchy = cardinal::move(h);
    a.enabled   = true;

    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    g_attach[&mesh] = cardinal::move(a);
    return true;
}

bool vgeom_attached(const Mesh& mesh) noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    return g_attach.find(&mesh) != g_attach.end();
}

bool vgeom_enabled(const Mesh& mesh) noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    auto it = g_attach.find(&mesh);
    return it != g_attach.end() && it->second.enabled;
}

void vgeom_set_enabled(Mesh& mesh, bool on) noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    auto it = g_attach.find(&mesh);
    if (it != g_attach.end()) it->second.enabled = on;
}

cardinal::shared_ptr<cardinal::vgeom::Hierarchy> vgeom_hierarchy_of(const Mesh& mesh) {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    auto it = g_attach.find(&mesh);
    return it == g_attach.end() ? nullptr : it->second.hierarchy;
}

cardinal::vgeom::FrameStats vgeom_last_stats(const Mesh& mesh) noexcept {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    auto it = g_attach.find(&mesh);
    return it == g_attach.end() ? cardinal::vgeom::FrameStats{} : it->second.last_stats;
}

void vgeom_publish_stats(Mesh& mesh, const cardinal::vgeom::FrameStats& s) {
    cardinal::lock_guard<cardinal::mutex> lg(g_mtx);
    auto it = g_attach.find(&mesh);
    if (it != g_attach.end()) it->second.last_stats = s;
}

}  // namespace cardinal::scene
