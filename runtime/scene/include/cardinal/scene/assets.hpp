#pragma once

// =============================================================================
// Cardinal — Asset catalog.
//
// Single registry of every "thing the editor knows how to drop into a scene".
// An asset is identified by a stable string id ("primitive.cube",
// "foliage.tree.basic", "terrain.chunk", ...) and exposes:
//
//   - a label + category for the Asset Palette UI
//   - a tooltip describing what the asset does
//   - a factory function that turns a (Scene, position) call into one or
//     more entities, returning their ids
//
// The catalog is intentionally tiny and dependency-free — it holds no GPU
// resources itself. Factories close over shared_ptr<Mesh> handles created
// at registration time, so spawning a primitive is a pointer copy + entity
// emplace; spawning a terrain chunk is a fresh mesh build.
//
// Callers register additional assets at any time (engine startup, plugin
// load, hot-reload) via `register_asset`. Default registration of the
// engine's built-in primitives + tree composite happens in
// `register_default_assets`, called from the sample / launcher boot code.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::rhi { class Device; }

namespace cardinal::scene {

class Scene;

enum class AssetKind : u32 {
    Primitive = 0,    // single mesh, single entity
    Composite,        // multi-entity (e.g. trunk + leaves)
    Terrain,          // generates a fresh mesh per spawn
};

struct AssetSpawnContext {
    Scene*       scene{nullptr};
    rhi::Device* device{nullptr};
    Vec3         position{0, 0, 0};
};

struct AssetSpawnResult {
    cardinal::vector<u32> entity_ids;     // every entity this spawn created
    u32              primary_id{0};  // the one the editor should select
};

using AssetFactory = cardinal::function<AssetSpawnResult(const AssetSpawnContext&)>;

struct AssetDesc {
    cardinal::string  id;          // stable identifier — never localised
    cardinal::string  label;       // user-visible name
    cardinal::string  category;    // "Primitives", "Foliage", "Terrain", "Custom"
    cardinal::string  tooltip;
    AssetKind    kind{AssetKind::Primitive};
    AssetFactory factory;
};

// ---------------------------------------------------------------------------
// AssetCatalog — singleton registry. Iteration order matches insertion.
// ---------------------------------------------------------------------------
class AssetCatalog {
public:
    static AssetCatalog& instance();

    bool register_asset(AssetDesc d);
    const cardinal::vector<AssetDesc>& all() const noexcept;
    const AssetDesc* find(const char* id) const noexcept;

    // Categories in insertion order (deduplicated).
    cardinal::vector<cardinal::string> categories() const;

    // Convenience: spawn by id with a context. Returns an empty result when
    // the id is unknown.
    AssetSpawnResult spawn(const char* id, const AssetSpawnContext& ctx) const;

private:
    AssetCatalog();
    AssetCatalog(const AssetCatalog&)            = delete;
    AssetCatalog& operator=(const AssetCatalog&) = delete;

    struct Impl;
    cardinal::unique_ptr<Impl> p_;
};

// Registers the engine-shipped defaults: cube, sphere (lo/hi), plane,
// tree composite, and a single terrain-chunk asset (with default params).
// Safe to call multiple times — duplicates are dropped on id collision.
void register_default_assets(rhi::Device& device);

}  // namespace cardinal::scene
