#pragma once

// =============================================================================
// Cardinal — scene graph (Phase 5 baseline).
//
// Flat entity list with per-entity Transform + optional Mesh handle. Real
// hierarchy / parenting / ECS lands in Phase 5.2; the editor doesn't need
// either to start producing useful imagery.
//
// View modes describe how an editor viewport should display the scene:
//   Solid      — lambert-shaded triangles (default)
//   Wireframe  — line topology
//   Polygons   — flat-shaded with random per-triangle colour (debug)
//   Heightmap  — visualise vertex Y as a colour ramp (terrain authoring)
//   Normals    — visualise per-vertex normals
//   RTXPreview — placeholder for the eventual ray-traced pass
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::rhi { class Device; class Buffer; }

namespace cardinal::scene {

enum class ViewMode : u32 {
    Solid = 0,
    Wireframe,
    Polygons,
    Heightmap,
    Normals,
    RTXPreview,
};

const char* view_mode_name(ViewMode);

// Vertex format used by the forward renderer's primitive meshes.
struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec3 color;
};

// Forward decl — vgeom is a lower-layer module; including its header
// creates a layering cycle (scene -> vgeom -> scene). Use the free
// functions below for vgeom integration.
}  // namespace cardinal::scene

namespace cardinal::vgeom {
struct Hierarchy;
struct FrameStats;
}  // namespace cardinal::vgeom

namespace cardinal::scene {

// Mesh = an immutable vertex+index buffer pair owned by the GPU device.
class Mesh {
public:
    static cardinal::shared_ptr<Mesh> from_vertices(rhi::Device& dev,
                                               const Vertex* verts, u32 count);

    // Built-in primitives.
    static cardinal::shared_ptr<Mesh> make_box(rhi::Device&,   float size = 1.0f);
    static cardinal::shared_ptr<Mesh> make_plane(rhi::Device&, float size = 10.0f, u32 subdivisions = 8);
    static cardinal::shared_ptr<Mesh> make_sphere(rhi::Device&,float radius = 1.0f, u32 segments = 24);

    rhi::Buffer*  vertex_buffer() const noexcept { return vbuf_.get(); }
    u32           vertex_count() const noexcept  { return vcount_; }
    // CPU shadow of the mesh — kept by from_vertices() so the forward
    // renderer can do its host-side transform pass without re-reading from
    // GPU memory. May be empty for meshes uploaded out-of-band.
    const Vertex* cpu_vertices() const noexcept  { return cpu_.data(); }

    // Local-space bounding sphere — centroid + farthest-vertex radius.
    // Computed once on first call from the CPU shadow, then cached.
    void bounding_sphere(Vec3& out_center, float& out_radius) const noexcept;

    // Local-space bounding AABB — tight {min,max} over all CPU vertices.
    // Cached on first call. Significantly tighter than the bounding
    // sphere (no dead radius around long-thin meshes), at the cost of
    // 2 Vec3 of state instead of Vec3+f32. Used by the renderer's
    // SIMD frustum-vs-AABB cull.
    void bounding_aabb(Vec3& out_min, Vec3& out_max) const noexcept;

    const cardinal::string& name() const noexcept     { return name_; }
    void set_name(cardinal::string n) { name_ = cardinal::move(n); }

private:
    Mesh() = default;
    cardinal::unique_ptr<rhi::Buffer> vbuf_;
    u32                          vcount_{0};
    cardinal::vector<Vertex>          cpu_;
    cardinal::string                  name_;
    mutable Vec3                 bs_center_{};
    mutable float                bs_radius_{0.0f};
    mutable bool                 bs_cached_{false};
    mutable Vec3                 ab_min_{};
    mutable Vec3                 ab_max_{};
    mutable bool                 ab_cached_{false};
};

struct Transform {
    Vec3 translation{0, 0, 0};
    Vec3 rotation_euler{0, 0, 0};   // radians
    Vec3 scale{1, 1, 1};

    Mat4 matrix() const {
        return Mat4::translation(translation)
             * Mat4::rotation_xyz(rotation_euler)
             * Mat4::scaling(scale);
    }
};

// Per-entity surface parameters consumed by the GPU lighting PS.
//
// `tint` (on Entity) is still the CPU-baked vertex-colour multiplier —
// it stays where it is so the existing colour path is undisturbed.
// Material carries the things the fragment shader needs at shade time
// that have no good per-vertex home: specular response today, PBR
// (roughness / metalness / emissive) as that lands.
//
// These ride a StructuredBuffer<GpuMaterial> at RHI storage slot 1
// (lights own slot 0) — the same primitive the push-block→SSBO arc
// added — so adding fields here never re-pressures the push block.
//
// Defaults intentionally equal the constants the scene PS currently
// hardcodes (SPEC_INTENSITY 0.4 / SPEC_POWER 24) so wiring the buffer
// in is a no-op visually until a host actually varies them.
struct Material {
    float specular_intensity{0.4f};   // 0 = matte, →1 strong highlight
    float specular_power{24.0f};      // tight-plastic … →chrome at high
    // Reserved for the PBR pass — kept in the struct now so the GPU
    // mirror + buffer stride stabilise before hosts depend on them.
    float roughness{0.5f};            // not yet read by the PS
    float metalness{0.0f};            // not yet read by the PS
};

class Scene;

class Entity {
public:
    u32                   id{0};
    cardinal::string           name;
    Transform             transform;
    cardinal::shared_ptr<Mesh> mesh;
    Vec3                  tint{1, 1, 1};
    Material              material{};
    bool                  visible{true};

    // Parent-child hierarchy. parent_id == 0 means "this entity is a root"
    // (root entities have no parent). Children inherit nothing transform-wise
    // from the parent today — the field is purely for editor grouping
    // (Outliner / Hierarchy tree, fold-by-parent, drag-to-reparent). True
    // transform inheritance (world = parent_world * local) is a follow-up
    // when the renderer learns to walk the hierarchy in topological order.
    u32                   parent_id{0};

    // World-grid chunk membership cache (3D). Three i32 = full ±2.1×10⁹
    // range per axis, far past anything practical. Refreshed by
    // Scene::assign_chunks each frame; callers can also set directly when
    // an entity teleports to bypass the auto-update on that frame.
    // (0, 0, 0) is just the origin chunk, NOT a sentinel — the field is
    // always meaningful after assign_chunks runs.
    i32                   chunk_x{0};
    i32                   chunk_y{0};
    i32                   chunk_z{0};
};

class Camera {
public:
    Vec3  position{0.0f, 1.5f, 4.0f};
    Vec3  target  {0.0f, 0.5f, 0.0f};
    Vec3  up      {0.0f, 1.0f, 0.0f};
    float fov_y_rad{60.0f * kDegToRad};
    float z_near{0.05f};
    float z_far{500.0f};

    Mat4 view() const { return Mat4::look_at(position, target, up); }
    Mat4 proj(float aspect) const { return Mat4::perspective(fov_y_rad, aspect, z_near, z_far); }
};

class Scene {
public:
    Entity& add_entity(cardinal::string name);
    bool    remove_entity(u32 id);
    Entity* find_by_id(u32 id);

    cardinal::vector<Entity>& entities() noexcept { return entities_; }
    const cardinal::vector<Entity>& entities() const noexcept { return entities_; }

    // ----- Hierarchy helpers ----------------------------------------------
    //
    // Each Entity has a parent_id field (0 = root). The Outliner / Hierarchy
    // panel uses these helpers to fold the flat entity vector into a tree.
    //
    // children_of(parent_id) is O(N) — linear scan over entities_ — but that
    // matches typical scene sizes for an editor (hundreds, not millions).
    // For mature scenes consider caching a parent→children adjacency list.
    //
    // set_parent enforces no-cycles: assigning P as the parent of an entity
    // that's already (transitively) a parent of P is rejected (returns false).
    // The Studio's drag-to-reparent UI uses the return value to bail out
    // gracefully rather than corrupt the tree.
    cardinal::vector<u32> children_of(u32 parent_id) const;
    bool             set_parent(u32 entity_id, u32 new_parent_id);
    bool             would_create_cycle(u32 entity_id, u32 new_parent_id) const;
    Camera& camera() noexcept             { return camera_; }
    const Camera& camera() const noexcept { return camera_; }

    // ----- World-grid streaming integration (3D) -----
    //
    // Optional "visible chunks" set. When non-empty, the renderer treats
    // entities whose (chunk_x, chunk_y, chunk_z) tuple is NOT in this set
    // as invisible. When empty (the default), all entities render normally.
    //
    // ChunkKey / ChunkSet are aliases of the canonical core types so the
    // world streamer + scene speak the exact same struct (no conversions
    // when the host iterates `WorldStreamer::active()` and pushes into
    // the scene). Equality + hash logic lives in cardinal::core::ChunkCoord.
    using ChunkKey = cardinal::core::ChunkCoord;
    using ChunkSet = cardinal::core::ChunkSet;

    void clear_visible_chunks() noexcept                  { visible_chunks_.clear(); }
    void mark_chunk_visible(i32 x, i32 y, i32 z)          { visible_chunks_.insert({x, y, z}); }
    bool has_visible_chunk_filter() const noexcept        { return !visible_chunks_.empty(); }
    bool is_chunk_visible(i32 x, i32 y, i32 z) const noexcept {
        return visible_chunks_.count({x, y, z}) > 0;
    }
    usize visible_chunk_count() const noexcept            { return visible_chunks_.size(); }
    const ChunkSet& visible_chunks() const noexcept       { return visible_chunks_; }

    // Combined visibility query — entity-flag AND optional chunk filter.
    bool is_entity_visible(const Entity& e) const noexcept {
        if (!e.visible) return false;
        if (visible_chunks_.empty()) return true;
        return visible_chunks_.count({e.chunk_x, e.chunk_y, e.chunk_z}) > 0;
    }

    // Lighting (single directional source for now — Phase 5.3 grows this).
    Vec3  sun_direction{-0.4f, -1.0f, -0.3f};
    Vec3  sun_color    { 1.0f,  0.95f, 0.85f };
    Vec3  ambient_color{ 0.06f, 0.07f, 0.09f };

    // Bulk-recompute every entity's (chunk_x, chunk_y, chunk_z) membership
    // using a uniform XYZ chunk size in world units. Returns the count
    // whose chunk membership actually changed (cheap to call every frame
    // — usually most entities are stationary).
    usize assign_chunks(f32 chunk_size_units) noexcept;

private:
    cardinal::vector<Entity>     entities_;
    Camera                  camera_;
    u32                     next_id_{1};
    ChunkSet                visible_chunks_{};
};

// ---------------------------------------------------------------------------
// Picking — returns the entity id of the closest visible mesh hit by `ray`,
// or 0 when nothing is hit. Tests against each entity's world-space
// bounding sphere (mesh local sphere * uniform-scale + entity translation).
// out_t (optional) is the parametric distance along the ray to the hit.
// ---------------------------------------------------------------------------
u32 pick_entity(const Scene& scene, const Ray& ray, float* out_t = nullptr);

// World-space bounding sphere for an entity — convenience wrapper.
void entity_world_sphere(const Entity& e, Vec3& out_center, float& out_radius);

// World-space bounding AABB for an entity — combines the mesh's local
// AABB with the entity transform via the Arvo incremental method
// (same algorithm cardinal::core::simd::transform_aabb_array uses,
// applied to one entity at a time). Tighter than the sphere variant
// for non-spherical or rotated geometry; consumed by the renderer's
// SIMD frustum-vs-AABB cull pass.
void entity_world_aabb(const Entity& e, Vec3& out_min, Vec3& out_max);

// ---------------------------------------------------------------------------
// Virtualized geometry (vgeom) — per-Mesh attachment.
//
// Cooks `mesh`'s CPU vertices into a vgeom hierarchy and registers it
// against this Mesh*. The renderer then transparently picks a per-frame
// LOD cut for any entity whose mesh has an attached + enabled hierarchy
// — drastically cutting effective polygon count while preserving
// visible LOD. Toggle per-mesh with vgeom_set_enabled.
//
// Returns true on success. Idempotent — re-attaching reuses the
// existing hierarchy.
//
// (Lives here in scene because vgeom is a lower layer and can't see
// scene::Mesh; this is the natural integration point.)
// ---------------------------------------------------------------------------
bool vgeom_attach (Mesh& mesh);
bool vgeom_attached(const Mesh& mesh) noexcept;
bool vgeom_enabled (const Mesh& mesh) noexcept;
void vgeom_set_enabled(Mesh& mesh, bool on) noexcept;

// Hierarchy lookup for the renderer. Returns null when not attached.
cardinal::shared_ptr<cardinal::vgeom::Hierarchy> vgeom_hierarchy_of(const Mesh& mesh);

// Per-mesh stats published by the renderer each frame (drawn vs master
// triangle count, cull statistics). Studio panel reads these.
cardinal::vgeom::FrameStats vgeom_last_stats(const Mesh& mesh) noexcept;
void                        vgeom_publish_stats(Mesh& mesh,
                                                const cardinal::vgeom::FrameStats& s);

}  // namespace cardinal::scene
