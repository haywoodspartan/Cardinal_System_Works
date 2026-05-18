// =============================================================================
// Cardinal — scene + primitives.
// =============================================================================
#include <cardinal/scene/scene.hpp>

#include <cardinal/core/async.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/core/simd_math.hpp>
#include <cardinal/rhi/rhi.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/cmath.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/limits.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::scene {

const char* view_mode_name(ViewMode m) {
    switch (m) {
        case ViewMode::Solid:      return "Solid";
        case ViewMode::Wireframe:  return "Wireframe";
        case ViewMode::Polygons:   return "Polygons";
        case ViewMode::Heightmap:  return "Heightmap";
        case ViewMode::Normals:    return "Normals";
        case ViewMode::RTXPreview: return "RTX Preview";
    }
    return "?";
}

// ---- Mesh -------------------------------------------------------------------
cardinal::shared_ptr<Mesh> Mesh::from_vertices(rhi::Device& dev, const Vertex* verts, u32 count) {
    if (count == 0 || verts == nullptr) return nullptr;
    auto m = cardinal::shared_ptr<Mesh>(new Mesh());
    rhi::BufferDesc bd{};
    bd.size         = sizeof(Vertex) * count;
    bd.usage        = static_cast<u32>(rhi::BufferUsage::Vertex);
    bd.cpu_writable = true;             // ForwardRenderer rewrites verts/frame
    auto buf = dev.create_buffer(bd);
    if (buf == nullptr) {
        cardinal::log::errorf("scene", "Mesh: vertex buffer alloc failed");
        return nullptr;
    }
    buf->upload(verts, sizeof(Vertex) * count);
    m->vbuf_   = cardinal::move(buf);
    m->vcount_ = count;
    m->cpu_.assign(verts, verts + count);
    return m;
}

// Box centred on origin, ±size/2 on each axis. 36 vertices (no indexing).
cardinal::shared_ptr<Mesh> Mesh::make_box(rhi::Device& dev, float size) {
    const float h = size * 0.5f;
    const Vec3 col{ 0.85f, 0.86f, 0.92f };
    // 6 faces × 2 tris × 3 verts. Per-face shared normal.
    Vertex v[36] = {
        // +X
        {{ h,-h,-h},{1,0,0},col},{{ h, h,-h},{1,0,0},col},{{ h, h, h},{1,0,0},col},
        {{ h,-h,-h},{1,0,0},col},{{ h, h, h},{1,0,0},col},{{ h,-h, h},{1,0,0},col},
        // -X
        {{-h,-h, h},{-1,0,0},col},{{-h, h, h},{-1,0,0},col},{{-h, h,-h},{-1,0,0},col},
        {{-h,-h, h},{-1,0,0},col},{{-h, h,-h},{-1,0,0},col},{{-h,-h,-h},{-1,0,0},col},
        // +Y
        {{-h, h,-h},{0,1,0},col},{{-h, h, h},{0,1,0},col},{{ h, h, h},{0,1,0},col},
        {{-h, h,-h},{0,1,0},col},{{ h, h, h},{0,1,0},col},{{ h, h,-h},{0,1,0},col},
        // -Y
        {{-h,-h, h},{0,-1,0},col},{{-h,-h,-h},{0,-1,0},col},{{ h,-h,-h},{0,-1,0},col},
        {{-h,-h, h},{0,-1,0},col},{{ h,-h,-h},{0,-1,0},col},{{ h,-h, h},{0,-1,0},col},
        // +Z
        {{-h,-h, h},{0,0,1},col},{{ h,-h, h},{0,0,1},col},{{ h, h, h},{0,0,1},col},
        {{-h,-h, h},{0,0,1},col},{{ h, h, h},{0,0,1},col},{{-h, h, h},{0,0,1},col},
        // -Z
        {{ h,-h,-h},{0,0,-1},col},{{-h,-h,-h},{0,0,-1},col},{{-h, h,-h},{0,0,-1},col},
        {{ h,-h,-h},{0,0,-1},col},{{-h, h,-h},{0,0,-1},col},{{ h, h,-h},{0,0,-1},col},
    };
    auto m = from_vertices(dev, v, 36);
    if (m) m->set_name("Box");
    return m;
}

// Plane on Y=0, facing +Y. subdivisions controls grid density (>=1).
cardinal::shared_ptr<Mesh> Mesh::make_plane(rhi::Device& dev, float size, u32 subdivisions) {
    if (subdivisions < 1) subdivisions = 1;
    const u32 N = subdivisions;
    const float step = size / static_cast<float>(N);
    const Vec3  col { 0.42f, 0.45f, 0.50f };
    const Vec3  nrm { 0.0f, 1.0f, 0.0f };

    cardinal::vector<Vertex> verts;
    verts.reserve(N * N * 6);
    for (u32 j = 0; j < N; ++j) {
        for (u32 i = 0; i < N; ++i) {
            const float x0 = -size * 0.5f + step * static_cast<float>(i);
            const float z0 = -size * 0.5f + step * static_cast<float>(j);
            const float x1 = x0 + step;
            const float z1 = z0 + step;
            verts.push_back({{x0, 0, z0}, nrm, col});
            verts.push_back({{x1, 0, z0}, nrm, col});
            verts.push_back({{x1, 0, z1}, nrm, col});
            verts.push_back({{x0, 0, z0}, nrm, col});
            verts.push_back({{x1, 0, z1}, nrm, col});
            verts.push_back({{x0, 0, z1}, nrm, col});
        }
    }
    auto m = from_vertices(dev, verts.data(), static_cast<u32>(verts.size()));
    if (m) m->set_name("Plane");
    return m;
}

// UV sphere — N segments around, N/2 stacks. No indexing.
cardinal::shared_ptr<Mesh> Mesh::make_sphere(rhi::Device& dev, float radius, u32 segments) {
    if (segments < 4) segments = 4;
    const u32 stacks = segments / 2;
    const Vec3 col{ 0.85f, 0.55f, 0.50f };
    cardinal::vector<Vertex> verts;
    verts.reserve(segments * stacks * 6);
    for (u32 j = 0; j < stacks; ++j) {
        const float v0 = static_cast<float>(j)     / static_cast<float>(stacks);
        const float v1 = static_cast<float>(j + 1) / static_cast<float>(stacks);
        const float p0 = v0 * kPi;
        const float p1 = v1 * kPi;
        for (u32 i = 0; i < segments; ++i) {
            const float u0 = static_cast<float>(i)     / static_cast<float>(segments);
            const float u1 = static_cast<float>(i + 1) / static_cast<float>(segments);
            const float t0 = u0 * 2.0f * kPi;
            const float t1 = u1 * 2.0f * kPi;
            auto pt = [&](float t, float p) {
                Vec3 n{ cardinal::sin(p) * cardinal::cos(t), cardinal::cos(p), cardinal::sin(p) * cardinal::sin(t) };
                Vec3 pos = n * radius;
                return Vertex{ pos, n, col };
            };
            Vertex a = pt(t0, p0), b = pt(t1, p0), c = pt(t1, p1), d = pt(t0, p1);
            verts.push_back(a); verts.push_back(b); verts.push_back(c);
            verts.push_back(a); verts.push_back(c); verts.push_back(d);
        }
    }
    auto m = from_vertices(dev, verts.data(), static_cast<u32>(verts.size()));
    if (m) m->set_name("Sphere");
    return m;
}

void Mesh::bounding_sphere(Vec3& out_center, float& out_radius) const noexcept {
    if (!bs_cached_) {
        Vec3  c{}; float r2 = 0.0f;
        if (vcount_ > 0) {
            for (u32 i = 0; i < vcount_; ++i) c += cpu_[i].position;
            c = c * (1.0f / static_cast<float>(vcount_));
            for (u32 i = 0; i < vcount_; ++i) {
                const Vec3 d = cpu_[i].position - c;
                const float dd = dot(d, d);
                if (dd > r2) r2 = dd;
            }
        }
        bs_center_ = c;
        bs_radius_ = cardinal::sqrt(r2);
        bs_cached_ = true;
    }
    out_center = bs_center_;
    out_radius = bs_radius_;
}

void Mesh::bounding_aabb(Vec3& out_min, Vec3& out_max) const noexcept {
    if (!ab_cached_) {
        if (vcount_ == 0) {
            ab_min_ = Vec3{0,0,0};
            ab_max_ = Vec3{0,0,0};
        } else {
            Vec3 mn = cpu_[0].position;
            Vec3 mx = mn;
            for (u32 i = 1; i < vcount_; ++i) {
                const Vec3& p = cpu_[i].position;
                if (p.x < mn.x) mn.x = p.x; else if (p.x > mx.x) mx.x = p.x;
                if (p.y < mn.y) mn.y = p.y; else if (p.y > mx.y) mx.y = p.y;
                if (p.z < mn.z) mn.z = p.z; else if (p.z > mx.z) mx.z = p.z;
            }
            ab_min_ = mn;
            ab_max_ = mx;
        }
        ab_cached_ = true;
    }
    out_min = ab_min_;
    out_max = ab_max_;
}

// ---- Scene ------------------------------------------------------------------
Entity& Scene::add_entity(cardinal::string name) {
    Entity e;
    e.id   = next_id_++;
    e.name = cardinal::move(name);
    entities_.push_back(cardinal::move(e));
    return entities_.back();
}

bool Scene::remove_entity(u32 id) {
    for (auto it = entities_.begin(); it != entities_.end(); ++it) {
        if (it->id == id) {
            // Reparent any children to root so they don't dangle pointing at
            // a now-deleted parent. Cheap O(N) sweep — fine for editor scenes.
            for (auto& child : entities_) {
                if (child.parent_id == id) child.parent_id = 0u;
            }
            entities_.erase(it);
            return true;
        }
    }
    return false;
}

Entity* Scene::find_by_id(u32 id) {
    for (auto& e : entities_) if (e.id == id) return &e;
    return nullptr;
}

// ----- Hierarchy --------------------------------------------------------------

cardinal::vector<u32> Scene::children_of(u32 parent_id) const {
    cardinal::vector<u32> out;
    out.reserve(8);
    for (const auto& e : entities_) {
        if (e.parent_id == parent_id) out.push_back(e.id);
    }
    return out;
}

bool Scene::would_create_cycle(u32 entity_id, u32 new_parent_id) const {
    if (entity_id == 0u) return false;
    if (new_parent_id == 0u) return false;       // root reparent never cycles
    if (new_parent_id == entity_id) return true; // self-parent

    // Walk up from new_parent_id towards root; if we hit entity_id we'd
    // close a cycle. Bounded by the number of entities to handle corrupt
    // input (e.g. an existing cycle from a buggy plugin).
    u32 walker = new_parent_id;
    const usize cap = entities_.size() + 1u;
    for (usize i = 0; i < cap && walker != 0u; ++i) {
        if (walker == entity_id) return true;
        // Find walker's parent.
        u32 next = 0u;
        for (const auto& e : entities_) {
            if (e.id == walker) { next = e.parent_id; break; }
        }
        walker = next;
    }
    return false;
}

bool Scene::set_parent(u32 entity_id, u32 new_parent_id) {
    if (entity_id == 0u) return false;
    if (would_create_cycle(entity_id, new_parent_id)) return false;
    Entity* e = find_by_id(entity_id);
    if (e == nullptr) return false;
    // Validate parent exists (or is root).
    if (new_parent_id != 0u && find_by_id(new_parent_id) == nullptr) return false;
    e->parent_id = new_parent_id;
    return true;
}

// Recompute every entity's (chunk_x, chunk_y, chunk_z) against a uniform
// 3D grid. Mirrors world::WorldGrid::chunk_of without dragging the world
// module's header in here. Returns count whose membership changed.
usize Scene::assign_chunks(float chunk_size_units) noexcept {
    if (chunk_size_units <= 1e-3f) return 0;
    const float inv = 1.0f / chunk_size_units;

    // Above the threshold, fan out to the worker pool — assign_chunks runs
    // on every frame and scales linearly with entity count, so for cinematic
    // scenes (10k+ entities) it's worth one parallel_for. Below the
    // threshold the fork/join overhead dominates; stay serial.
    constexpr usize kParallelThreshold = 1024;

    if (entities_.size() < kParallelThreshold ||
        cardinal::async::pool() == nullptr)
    {
        usize changed = 0;
        for (auto& e : entities_) {
            const float wx = e.transform.translation.x;
            const float wy = e.transform.translation.y;
            const float wz = e.transform.translation.z;
            if (!cardinal::isfinite(wx) || !cardinal::isfinite(wy) || !cardinal::isfinite(wz)) continue;
            const i32 cx = static_cast<i32>(cardinal::floor(wx * inv));
            const i32 cy = static_cast<i32>(cardinal::floor(wy * inv));
            const i32 cz = static_cast<i32>(cardinal::floor(wz * inv));
            if (e.chunk_x != cx || e.chunk_y != cy || e.chunk_z != cz) {
                e.chunk_x = cx; e.chunk_y = cy; e.chunk_z = cz;
                ++changed;
            }
        }
        return changed;
    }

    cardinal::atomic<usize> changed_atomic{0};
    cardinal::async::parallel_for(
        0u, static_cast<u32>(entities_.size()),
        [this, inv, &changed_atomic](u32 i) {
            auto& e = entities_[i];
            const float wx = e.transform.translation.x;
            const float wy = e.transform.translation.y;
            const float wz = e.transform.translation.z;
            if (!cardinal::isfinite(wx) || !cardinal::isfinite(wy) || !cardinal::isfinite(wz)) return;
            const i32 cx = static_cast<i32>(cardinal::floor(wx * inv));
            const i32 cy = static_cast<i32>(cardinal::floor(wy * inv));
            const i32 cz = static_cast<i32>(cardinal::floor(wz * inv));
            if (e.chunk_x != cx || e.chunk_y != cy || e.chunk_z != cz) {
                e.chunk_x = cx; e.chunk_y = cy; e.chunk_z = cz;
                changed_atomic.fetch_add(1, cardinal::memory_order_relaxed);
            }
        });
    return changed_atomic.load(cardinal::memory_order_relaxed);
}

// ---- Picking ----------------------------------------------------------------
void entity_world_sphere(const Entity& e, Vec3& out_center, float& out_radius) {
    Vec3  lc{}; float lr = 0.0f;
    if (e.mesh) e.mesh->bounding_sphere(lc, lr);
    // Apply uniform scale + translation. Non-uniform scale would need an
    // OBB or per-axis bound; for the editor's primitives this is enough.
    const float s = cardinal::max({ cardinal::abs(e.transform.scale.x),
                               cardinal::abs(e.transform.scale.y),
                               cardinal::abs(e.transform.scale.z) });
    out_center = e.transform.translation
               + Vec3{ lc.x * e.transform.scale.x,
                       lc.y * e.transform.scale.y,
                       lc.z * e.transform.scale.z };
    out_radius = lr * s;
}

void entity_world_aabb(const Entity& e, Vec3& out_min, Vec3& out_max) {
    Vec3 lmn{}, lmx{};
    if (e.mesh) e.mesh->bounding_aabb(lmn, lmx);
    // Use the entity's full transform matrix and the Arvo incremental
    // technique — same algorithm cardinal::core::simd::transform_aabb_array
    // uses, just one entity at a time. Handles arbitrary rotation +
    // non-uniform scale without overstating the bound the way the
    // sphere variant has to.
    const Mat4 M = e.transform.matrix();
    auto get = [&](int col, int row) { return M.m[col][row]; };
    Vec3 mn{ get(3,0), get(3,1), get(3,2) };
    Vec3 mx{ get(3,0), get(3,1), get(3,2) };
    for (int ai = 0; ai < 3; ++ai) {
        const float lo_i = (&lmn.x)[ai];
        const float hi_i = (&lmx.x)[ai];
        for (int ao = 0; ao < 3; ++ao) {
            const float coeff = get(ai, ao);
            const float a = coeff * lo_i;
            const float b = coeff * hi_i;
            (&mn.x)[ao] += cardinal::min(a, b);
            (&mx.x)[ao] += cardinal::max(a, b);
        }
    }
    out_min = mn;
    out_max = mx;
}

u32 pick_entity(const Scene& scene, const Ray& ray, float* out_t) {
    // SIMD-batched ray-vs-AABB picking. Walk visible entities once,
    // build SoA world-AABB scratch + parallel id table, then a single
    // simd::ray_aabb_intersect_array call computes per-entity entry-t
    // (INF for misses). Final scalar reduction picks the smallest non-
    // INF t and returns its entity id.
    //
    // AABB vs sphere: tighter on non-spherical / rotated geometry, so
    // fewer "I clicked next to it but it picked the wrong one" cases.
    // The renderer uses the same primitive for cull; eventually
    // picking could share that scratch — kept independent for now so
    // picking works whether or not the renderer has rendered this
    // frame (mouse hover, headless tests, etc.).
    static thread_local cardinal::vector<u32> ids;
    static thread_local cardinal::vector<f32> mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
    static thread_local cardinal::vector<f32> t_out;

    ids.clear();
    mn_x.clear(); mn_y.clear(); mn_z.clear();
    mx_x.clear(); mx_y.clear(); mx_z.clear();

    for (const auto& e : scene.entities()) {
        if (!e.visible || !e.mesh) continue;
        Vec3 mn{}, mx{};
        entity_world_aabb(e, mn, mx);
        // Skip degenerate AABBs (empty mesh / zero-thickness slab) —
        // the slab math doesn't have a well-defined t for them.
        if (mn.x >= mx.x || mn.y >= mx.y || mn.z >= mx.z) continue;
        ids.push_back(e.id);
        mn_x.push_back(mn.x); mn_y.push_back(mn.y); mn_z.push_back(mn.z);
        mx_x.push_back(mx.x); mx_y.push_back(mx.y); mx_z.push_back(mx.z);
    }

    const usize n = ids.size();
    if (n == 0) return 0;

    t_out.resize(n);

    // Pre-compute inv_dir. Ray::direction is documented as normalised
    // (unproject_ndc_ray normalises before returning); for axis-parallel
    // rays the IEEE 1/0 = +INF carries through the slab math correctly.
    const f32 inv_dx = 1.0f / ray.direction.x;
    const f32 inv_dy = 1.0f / ray.direction.y;
    const f32 inv_dz = 1.0f / ray.direction.z;

    cardinal::core::simd::ray_aabb_intersect_array(
        t_out.data(),
        ray.origin.x, ray.origin.y, ray.origin.z,
        inv_dx, inv_dy, inv_dz,
        mn_x.data(), mn_y.data(), mn_z.data(),
        mx_x.data(), mx_y.data(), mx_z.data(),
        n);

    // Single-pass argmin over t_out: SIMD-batched reduce. INF-valued
    // misses lose to any real hit, so a finite best_t means "we hit
    // something". On AVX-512 this runs 16 lanes / iter with native
    // mask_blend_epi32 for the index update.
    f32 best_t  = 0.0f;
    u32 best_i  = 0;
    cardinal::core::simd::min_index_f32(&best_t, &best_i, t_out.data(), n);

    if (best_t == cardinal::numeric_limits<f32>::infinity()) return 0;
    if (out_t) *out_t = best_t;
    return ids[best_i];
}

}  // namespace cardinal::scene
