// =============================================================================
// Cardinal — asset catalog + default-asset registration.
// =============================================================================
#include <cardinal/scene/assets.hpp>

#include <cardinal/core/log.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/scene/terrain.hpp>

#include <algorithm>
#include <unordered_set>

namespace cardinal::scene {

// ---------------------------------------------------------------------------
// AssetCatalog::Impl
// ---------------------------------------------------------------------------
struct AssetCatalog::Impl {
    std::vector<AssetDesc> entries;
};

AssetCatalog::AssetCatalog() : p_(std::make_unique<Impl>()) {}

AssetCatalog& AssetCatalog::instance() {
    static AssetCatalog g;
    return g;
}

bool AssetCatalog::register_asset(AssetDesc d) {
    if (d.id.empty() || d.factory == nullptr) {
        cardinal::log::warnf("scene/assets",
            "register_asset rejected — empty id or null factory");
        return false;
    }
    for (const auto& e : p_->entries) {
        if (e.id == d.id) {
            cardinal::log::warnf("scene/assets",
                "register_asset duplicate id '%s' — keeping original",
                d.id.c_str());
            return false;
        }
    }
    p_->entries.push_back(std::move(d));
    return true;
}

const std::vector<AssetDesc>& AssetCatalog::all() const noexcept {
    return p_->entries;
}

const AssetDesc* AssetCatalog::find(const char* id) const noexcept {
    if (id == nullptr) return nullptr;
    for (const auto& e : p_->entries) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

std::vector<std::string> AssetCatalog::categories() const {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (const auto& e : p_->entries) {
        if (seen.insert(e.category).second) out.push_back(e.category);
    }
    return out;
}

AssetSpawnResult AssetCatalog::spawn(const char* id,
                                     const AssetSpawnContext& ctx) const
{
    const auto* d = find(id);
    if (d == nullptr) {
        cardinal::log::warnf("scene/assets", "spawn: unknown asset id '%s'",
                             id ? id : "(null)");
        return {};
    }
    if (d->factory == nullptr || ctx.scene == nullptr) return {};
    return d->factory(ctx);
}

// ---------------------------------------------------------------------------
// register_default_assets — engine-shipped catalogue.
//
// Each entry captures shared_ptr<Mesh> handles created at registration
// time so spawning a primitive is just a buffer-handle copy + entity
// emplace. Terrain factories build a fresh mesh per spawn (chunked
// terrain composes them via spawn_terrain_grid instead of going through
// the catalog directly, but a single-chunk asset is handy for testing).
// ---------------------------------------------------------------------------
void register_default_assets(rhi::Device& device) {
    auto& cat = AssetCatalog::instance();

    // Primitives — meshes built once, shared across every spawn.
    auto cube         = Mesh::make_box   (device, 1.0f);
    auto sphere_lo    = Mesh::make_sphere(device, 0.7f, 16);
    auto sphere_hi    = Mesh::make_sphere(device, 0.7f, 32);
    auto plane_small  = Mesh::make_plane (device, 2.0f, 2);
    auto plane_med    = Mesh::make_plane (device, 8.0f, 8);

    auto reg_primitive = [&](const char* id, const char* label,
                             const char* tip, std::shared_ptr<Mesh> mesh,
                             Vec3 tint, float lift_y)
    {
        AssetDesc d{};
        d.id        = id;
        d.label     = label;
        d.category  = "Primitives";
        d.tooltip   = tip;
        d.kind      = AssetKind::Primitive;
        d.factory   = [mesh, tint, lift_y, label](const AssetSpawnContext& ctx) -> AssetSpawnResult {
            if (ctx.scene == nullptr) return {};
            auto& e = ctx.scene->add_entity(label);
            e.mesh = mesh;
            e.transform.translation = Vec3{ ctx.position.x,
                                            ctx.position.y + lift_y,
                                            ctx.position.z };
            e.tint = tint;
            return AssetSpawnResult{ { e.id }, e.id };
        };
        cat.register_asset(std::move(d));
    };

    reg_primitive("primitive.cube",       "Cube",            "Unit cube primitive",
                  cube,         { 0.95f, 0.55f, 0.40f }, 0.5f);
    reg_primitive("primitive.sphere",     "Sphere (low)",    "16-segment UV sphere",
                  sphere_lo,    { 0.45f, 0.70f, 0.95f }, 0.7f);
    reg_primitive("primitive.sphere.hi",  "Sphere (high)",   "32-segment UV sphere",
                  sphere_hi,    { 0.55f, 0.80f, 0.55f }, 0.7f);
    reg_primitive("primitive.plane.s",    "Plane (small)",   "2x2 unit plane",
                  plane_small,  { 0.70f, 0.70f, 0.70f }, 0.0f);
    reg_primitive("primitive.plane.m",    "Plane (medium)",  "8x8 unit plane",
                  plane_med,    { 0.55f, 0.58f, 0.62f }, 0.0f);

    // Composite — tree (trunk + leaves) is a 2-entity spawn so the gizmo
    // can grab either part and rotate/move independently.
    {
        AssetDesc d{};
        d.id       = "composite.tree.basic";
        d.label    = "Tree (basic)";
        d.category = "Foliage";
        d.tooltip  = "Stylised trunk + leaves spawned as two entities";
        d.kind     = AssetKind::Composite;
        d.factory  = [cube, sphere_hi](const AssetSpawnContext& ctx) -> AssetSpawnResult {
            if (ctx.scene == nullptr) return {};
            auto& trunk = ctx.scene->add_entity("Trunk");
            trunk.mesh = cube;
            trunk.transform.translation = { ctx.position.x, ctx.position.y + 0.5f, ctx.position.z };
            trunk.transform.scale       = { 0.2f, 1.0f, 0.2f };
            trunk.tint = { 0.42f, 0.27f, 0.16f };
            auto& leaves = ctx.scene->add_entity("Leaves");
            leaves.mesh = sphere_hi;
            leaves.transform.translation = { ctx.position.x, ctx.position.y + 1.5f, ctx.position.z };
            leaves.transform.scale       = { 1.2f, 1.2f, 1.2f };
            leaves.tint = { 0.30f, 0.55f, 0.25f };
            return AssetSpawnResult{ { trunk.id, leaves.id }, leaves.id };
        };
        cat.register_asset(std::move(d));
    }

    // Terrain — single-chunk asset so the user can drop a hill anywhere.
    // For massive terrains use spawn_terrain_grid via the Create menu modal.
    {
        AssetDesc d{};
        d.id       = "terrain.chunk.basic";
        d.label    = "Terrain Chunk";
        d.category = "Terrain";
        d.tooltip  = "Single 32u² heightmap chunk (default fbm noise)";
        d.kind     = AssetKind::Terrain;
        d.factory  = [](const AssetSpawnContext& ctx) -> AssetSpawnResult {
            if (ctx.scene == nullptr || ctx.device == nullptr) return {};
            TerrainParams p{};
            auto mesh = generate_terrain_chunk(*ctx.device, p, 0, 0);
            if (mesh == nullptr) return {};
            auto& e = ctx.scene->add_entity("Terrain");
            e.mesh = mesh;
            e.transform.translation = ctx.position;
            e.tint = { 1.0f, 1.0f, 1.0f };   // mesh already has slope colours
            return AssetSpawnResult{ { e.id }, e.id };
        };
        cat.register_asset(std::move(d));
    }

    cardinal::log::infof("scene/assets",
        "Asset catalog online — %zu entries across %zu categories",
        cat.all().size(), cat.categories().size());
}

}  // namespace cardinal::scene
