#pragma once

// =============================================================================
// Cardinal — Asset Database (the asset pipeline's queryable backbone).
//
// The import/cook/pack/asset modules are the asset *tools* (ingest -> bake ->
// archive -> load). This module is the missing *index* that turns them into a
// pipeline: a queryable registry of every cooked asset + its dependency graph —
// Cardinal's analog of UE's Asset Registry.
//
// Built from a cook pass: every cooked asset becomes an AssetRecord with a
// stable id, type, source + content hash, and its dependencies (e.g. a material
// records the texture keys it references, discovered via the runtime Registry).
// Serialized to <cooked>/assets.db so the editor + runtime can browse by type,
// resolve refs, and find what depends on a given asset (for re-cook / validate).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/cook/cook.hpp>     // cook::AssetType, cook::CookResult

namespace cardinal::asset { class Registry; }

namespace cardinal::assetdb {

// One indexed asset.
struct AssetRecord {
    cardinal::string                   key;          // runtime key (= source relpath)
    u64                                id{0};        // stable id (hash of key)
    cook::AssetType                    type{cook::AssetType::Unknown};
    cardinal::string                   source;       // source relpath it was cooked from
    u64                                source_hash{0};
    cardinal::vector<cardinal::string> deps;         // keys this asset references
};

// Stable 64-bit id for a key (FNV-1a). Stable across everything but a rename
// (rename-survival via persisted .meta GUIDs is a future step).
u64 asset_id(const cardinal::string& key) noexcept;

// Queryable index of cooked assets + their dependency graph.
class AssetDatabase {
public:
    void            add(AssetRecord rec);
    void            clear() noexcept;
    cardinal::usize size() const noexcept { return records_.size(); }

    // Lookups.
    const AssetRecord* find(const cardinal::string& key) const noexcept;
    const AssetRecord* find_by_id(u64 id) const noexcept;
    cook::AssetType    type_of(const cardinal::string& key) const noexcept;

    // Queries.
    cardinal::vector<cardinal::string>   all_keys() const;            // sorted
    cardinal::vector<const AssetRecord*> by_type(cook::AssetType t) const;
    cardinal::vector<cardinal::string>   dependencies(const cardinal::string& key) const;
    cardinal::vector<cardinal::string>   dependents(const cardinal::string& key) const;
    // Dep keys referenced by some asset but not present in the db (broken refs).
    cardinal::vector<cardinal::string>   missing_dependencies() const;

    const cardinal::vector<AssetRecord>& records() const noexcept { return records_; }

    // Text serialization ("# Cardinal assetdb v1").
    cardinal::string serialize() const;
    static bool      deserialize(const cardinal::string& text, AssetDatabase& out);

private:
    cardinal::vector<AssetRecord>                            records_;
    cardinal::unordered_map<cardinal::string, cardinal::usize> by_key_;   // key -> index
};

// Build the database from a cook pass. Indexes every Cooked/Skipped result;
// for materials, extracts texture-key dependencies via `registry` (which must be
// mounted on the cooked output, or have the materials registered in-memory).
AssetDatabase build_database(const cardinal::vector<cook::CookResult>& results,
                             cardinal::asset::Registry& registry);

// Persist / load the database to a file (default site: <cooked>/assets.db).
bool save_database(const AssetDatabase& db, const cardinal::string& path);
bool load_database(const cardinal::string& path, AssetDatabase& out);

}  // namespace cardinal::assetdb
