// =============================================================================
// Cardinal — deterministic virtualized-geometry (vgeom) regression suite.
//
// vgeom is the Nanite-style runtime path: cook() turns a source mesh into
// a LOD cluster Hierarchy; select() walks it per-frame and emits the cut.
// Both are pure CPU + fully deterministic, so this suite pins the
// structural contract a cook/select refactor must not silently break:
//
//   * cook input guards (null / 0 / non-triangle-list → nullptr);
//   * Hierarchy well-formedness — clusters[0] is the root, master tri
//     count, the radius*0.5 first-cut error proxy, vertex-slice bounds,
//     and reciprocal parent/child link integrity;
//   * cook determinism (no RNG — identical mesh ⇒ identical hierarchy);
//   * select null-safety, id validity, stats accounting
//     (cut == ids.size(), drawn == Σ cluster tris, master propagated),
//     determinism, and the tolerance monotonicity (stricter ⇒ ≥ cut).
//
// Zero deps (same harness as the other suites). Exit 0 = all pass.
// =============================================================================

#include <cardinal/vgeom/vgeom.hpp>
#include <cardinal/vgeom/cluster.hpp>
#include <cardinal/core/math.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace vg = cardinal::vgeom;
using cardinal::u32;
using cardinal::usize;
using M4  = cardinal::core::Mat4;
using V3  = cardinal::core::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("vgeomtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Deterministic N×N-quad grid as a triangle list (no index buffer, per
// CookDesc). Gentle z bump so bounds/normals aren't degenerate.
cardinal::vector<vg::Vertex> make_grid(int n) {
    cardinal::vector<vg::Vertex> v;
    v.reserve(static_cast<usize>(n) * n * 6);
    auto P = [&](int x, int y) {
        const float fx = static_cast<float>(x);
        const float fy = static_cast<float>(y);
        const float fz = 0.25f * static_cast<float>((x * 7 + y * 13) % 5);
        vg::Vertex vert;
        vert.position = { fx, fy, fz };
        vert.normal   = { 0.0f, 0.0f, 1.0f };
        vert.color    = { fx / n, fy / n, 0.5f };
        return vert;
    };
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            v.push_back(P(x,   y  )); v.push_back(P(x+1, y  )); v.push_back(P(x+1, y+1));
            v.push_back(P(x,   y  )); v.push_back(P(x+1, y+1)); v.push_back(P(x,   y+1));
        }
    return v;
}

bool v3eq(const vg::Vec3& a, const vg::Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// ---- cook: input guards -------------------------------------------
void test_cook_guards() {
    vg::CookDesc d{};
    CHECK(vg::cook(d) == nullptr);                       // null verts, 0 count

    auto grid = make_grid(8);
    d.vertices = grid.data();
    d.vertex_count = 0;
    CHECK(vg::cook(d) == nullptr);                       // 0 count

    d.vertex_count = 7;                                  // not a multiple of 3
    CHECK(vg::cook(d) == nullptr);

    d.vertices = nullptr;
    d.vertex_count = static_cast<u32>(grid.size());
    CHECK(vg::cook(d) == nullptr);                       // null ptr
}

// ---- cook: hierarchy well-formedness ------------------------------
void test_cook_hierarchy() {
    auto grid = make_grid(16);                           // 512 tris, 1536 verts
    vg::CookDesc d{};
    d.vertices     = grid.data();
    d.vertex_count = static_cast<u32>(grid.size());
    d.name         = "grid16";
    auto h = vg::cook(d);
    CHECK(h != nullptr);
    if (!h) return;

    CHECK(h->master_tri_count == d.vertex_count / 3u);
    CHECK(!h->clusters.empty());
    CHECK(h->total_cluster_count == static_cast<u32>(h->clusters.size()));
    CHECK(h->level_count >= 1u);
    CHECK(h->leaf_count >= 1u);

    // clusters[0] is the root.
    const vg::Cluster& root = h->clusters[0];
    CHECK(root.parent_id == vg::Cluster::kInvalid);
    CHECK(v3eq(h->root_center, root.center));
    CHECK(h->root_radius == root.radius);
    if (h->level_count > 1u) CHECK(root.child_count > 0u);

    bool any_leaf = false;
    bool err_proxy_ok = true, slices_ok = true, links_ok = true;
    for (usize i = 0; i < h->clusters.size(); ++i) {
        const vg::Cluster& c = h->clusters[i];

        // First-cut error proxy: geometric_error == radius * 0.5 for
        // EVERY cluster (NOT 0 at leaves — see cook.cpp).
        if (c.geometric_error != c.radius * 0.5f) err_proxy_ok = false;

        // Vertex slice in-bounds, non-empty, triangle-list multiple of 3.
        if (static_cast<usize>(c.vertex_offset) + c.vertex_count
              > h->vertices.size()) slices_ok = false;
        if (c.vertex_count == 0u || (c.vertex_count % 3u) != 0u)
            slices_ok = false;

        if (c.child_count == 0u) {
            any_leaf = true;
        } else {
            // Children contiguous, in-bounds, and reciprocally linked.
            if (c.first_child_id == vg::Cluster::kInvalid) links_ok = false;
            else {
                const usize end = static_cast<usize>(c.first_child_id)
                                + c.child_count;
                if (end > h->clusters.size()) links_ok = false;
                else for (u32 k = 0; k < c.child_count; ++k)
                    if (h->clusters[c.first_child_id + k].parent_id
                          != static_cast<u32>(i)) links_ok = false;
            }
        }
    }
    CHECK(err_proxy_ok);
    CHECK(slices_ok);
    CHECK(links_ok);
    CHECK(any_leaf);

    // Determinism: no RNG, index-ordered BFS — same mesh ⇒ same build.
    auto h2 = vg::cook(d);
    CHECK(h2 != nullptr);
    if (h2) {
        CHECK(h2->clusters.size() == h->clusters.size());
        CHECK(h2->master_tri_count == h->master_tri_count);
        CHECK(h2->level_count == h->level_count);
        CHECK(h2->leaf_count == h->leaf_count);
        bool same = (h2->clusters.size() == h->clusters.size());
        for (usize i = 0; same && i < h->clusters.size(); ++i) {
            const auto& a = h->clusters[i];
            const auto& b = h2->clusters[i];
            if (!v3eq(a.center, b.center) || a.radius != b.radius ||
                a.parent_id != b.parent_id ||
                a.first_child_id != b.first_child_id ||
                a.child_count != b.child_count ||
                a.vertex_count != b.vertex_count) same = false;
        }
        CHECK(same);
    }
}

// ---- select: null-safety + per-frame cut contract -----------------
void test_select() {
    // Null hierarchy → empty, no crash.
    {
        vg::SelectInput in{};
        vg::SelectOutput out;
        out.cluster_ids.push_back(123u);                 // must be cleared
        vg::select(in, out);
        CHECK(out.cluster_ids.empty());
    }

    auto grid = make_grid(16);
    vg::CookDesc d{};
    d.vertices     = grid.data();
    d.vertex_count = static_cast<u32>(grid.size());
    auto h = vg::cook(d);
    CHECK(h != nullptr);
    if (!h) return;

    const V3 c = h->root_center;
    const float r = h->root_radius > 0.0f ? h->root_radius : 1.0f;

    vg::SelectInput in{};
    in.hierarchy = h.get();
    in.model     = M4::identity();
    in.view      = M4::look_at({ c.x, c.y, c.z + r * 4.0f }, c, { 0, 1, 0 });
    in.proj      = M4::perspective(1.0f, 16.0f / 9.0f, 0.05f, 10000.0f);
    in.viewport_pixel_height = 1080.0f;

    auto valid_consistent = [&](const vg::SelectOutput& o) -> bool {
        if (o.stats.master_tri_count != h->master_tri_count) return false;
        if (o.stats.cut_cluster_count != o.cluster_ids.size()) return false;
        u32 drawn = 0;
        for (u32 id : o.cluster_ids) {
            if (id >= h->clusters.size()) return false;
            drawn += h->clusters[id].vertex_count / 3u;
        }
        return drawn == o.stats.drawn_tri_count;
    };

    // Permissive: a huge error tolerance ⇒ select never descends ⇒ the
    // cut is small (root-ward). Strict + same camera ⇒ descends further.
    vg::SelectOutput coarse;
    in.pixel_error_tolerance = 1.0e9f;
    vg::select(in, coarse);
    CHECK(!coarse.cluster_ids.empty());                  // camera looks at it
    CHECK(valid_consistent(coarse));

    vg::SelectOutput fine;
    in.pixel_error_tolerance = 1.0e-6f;
    vg::select(in, fine);
    CHECK(valid_consistent(fine));
    CHECK(fine.cluster_ids.size() >= coarse.cluster_ids.size());  // monotone
    CHECK(fine.cluster_ids.size() <= h->clusters.size());

    // Determinism: identical input ⇒ identical cut.
    vg::SelectOutput fine2;
    vg::select(in, fine2);
    bool same = (fine2.cluster_ids.size() == fine.cluster_ids.size());
    for (usize i = 0; same && i < fine.cluster_ids.size(); ++i)
        if (fine2.cluster_ids[i] != fine.cluster_ids[i]) same = false;
    CHECK(same);
    CHECK(fine2.stats.drawn_tri_count == fine.stats.drawn_tri_count);
}

}  // namespace

int main() {
    test_cook_guards();
    test_cook_hierarchy();
    test_select();

    if (g_fail == 0) {
        cardinal::log::infof("vgeomtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("vgeomtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
