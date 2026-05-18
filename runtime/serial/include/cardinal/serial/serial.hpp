#pragma once

// =============================================================================
// Cardinal — Serialization (save / load).
//
// Text-format save files. One file = one "world snapshot": a list of actor
// entries, each with name + class_name + transform + reflected GameActor
// properties. Round-trips cleanly through cardinal::game::ClassRegistry —
// load knows how to spawn the right class and apply its property values.
//
// Format (deliberately human-readable so saves are diffable + git-able):
//
//   # Cardinal save v1
//   actor "Spinner" {
//     class      = "SpinnerActor"
//     position   = (0.000, 1.500, 0.000)
//     rotation   = (0.000, 0.000, 0.000)
//     scale      = (1.000, 1.000, 1.000)
//     prop float rpm = 30.000
//     prop vec3 axis = (0.000, 1.000, 0.000)
//     prop bool reverse = false
//   }
//
// Top-level fields beyond actors will land here too (sky state, sequencer,
// camera) — for now we focus on actors which are the primary game state.
//
// Known gap (tracked): this module operates on cardinal::actor::World
// (gameplay) + Sky, but Engine exposes cardinal::scene::Scene (rendering
// primitives) and doesn't surface World/Sky accessors. The StudioMin RHI
// panel has a "Scene info" read-only section as a placeholder until either
// (a) Engine::world() / sky() get added and panel Save/Load buttons hook
// the existing save_world / save_sky here, OR (b) a separate
// scene::serialize / deserialize lands that walks scene::Entity[] (mesh
// refs would need a re-create story — likely a CookDesc-keyed cache).
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::unique_ptr/string
#include <cardinal/core/containers.hpp>   // unordered_map/vector

namespace cardinal::actor { class World; }
namespace cardinal::game  { class Game;  }
namespace cardinal::scene { class Mesh; class Scene; }
namespace cardinal::sky   { class Sky;   }

namespace cardinal::serial {

struct SaveStats {
    u32 actors_written{0};
    u32 properties_written{0};
    u64 bytes_written{0};
};

struct LoadStats {
    u32 actors_spawned{0};
    u32 properties_applied{0};
    u32 actors_skipped{0};      // class not registered
    u32 errors{0};
    cardinal::vector<cardinal::string> warnings;
};

// ---------------------------------------------------------------------------
// World save / load.
//
// save_world walks every alive actor, emits a block per actor with its
// class + transform + reflected properties. Non-GameActor actors are
// emitted with class="" so they round-trip as anonymous transform-only
// entries.
//
// load_world DOES NOT clear the existing world by default — callers can
// pre-clear via World::sweep() + destroy() if they want a "load over
// blank slate" workflow. The `replace_existing` param does that for you.
// ---------------------------------------------------------------------------
SaveStats save_world(const cardinal::actor::World& world,
                     const cardinal::string& path,
                     cardinal::string* error_out = nullptr);

LoadStats load_world(cardinal::game::Game& game,
                     const cardinal::string& path,
                     bool replace_existing = false,
                     cardinal::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Sky save / load — small format extension that stores the current hour,
// time scale, frozen flag, and every phase key.
// ---------------------------------------------------------------------------
bool save_sky(const cardinal::sky::Sky& sky, const cardinal::string& path,
              cardinal::string* error_out = nullptr);
bool load_sky(cardinal::sky::Sky&       sky, const cardinal::string& path,
              cardinal::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Scene save / load — round-trips scene::Entity[] (rendering primitives).
//
// Mesh refs are saved by Mesh::name() string. On load, the caller passes
// a MeshRegistry (mesh-name → shared_ptr) so the loader can re-attach
// the right mesh to each entity. Entities whose mesh has an empty name
// are skipped on save (they can't be recovered); entities whose mesh
// name isn't in the registry are skipped on load (logged in stats).
//
// This is a partial first-cut: a full asset-pipeline iteration would
// persist mesh data itself (CookDesc → on-disk + load-time cook). For
// now, the contract is "host owns its mesh registry; serial just
// records identities".
// ---------------------------------------------------------------------------
struct SceneSaveStats {
    u32 entities_written{0};
    u32 entities_skipped_meshless{0};   // entity.mesh == nullptr
    u32 entities_skipped_unnamed{0};    // mesh name was empty
    u64 bytes_written{0};
};
struct SceneLoadStats {
    u32 entities_spawned{0};
    u32 entities_skipped_unknown_mesh{0};
    u32 errors{0};
    cardinal::vector<cardinal::string> warnings;
};

using MeshRegistry =
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<cardinal::scene::Mesh>>;

SceneSaveStats save_scene(const cardinal::scene::Scene& scene,
                          const cardinal::string& path,
                          cardinal::string* error_out = nullptr);

SceneLoadStats load_scene(cardinal::scene::Scene& scene,
                          const MeshRegistry& mesh_registry,
                          const cardinal::string& path,
                          bool replace_existing = false,
                          cardinal::string* error_out = nullptr);

// ---------------------------------------------------------------------------
// Helpers — usable directly when an integrator wants a custom blob format.
// emit_*: append to `out` (one line per call). parse_*: read from `in`,
// return whether the line matched the expected schema.
// ---------------------------------------------------------------------------
namespace text {
    void emit_kv (cardinal::string& out, const char* key, const char* value);
    void emit_kf (cardinal::string& out, const char* key, float value);
    void emit_kv3(cardinal::string& out, const char* key,
                  float x, float y, float z);
    bool parse_kv (const cardinal::string& line, cardinal::string* key, cardinal::string* value);
}

}  // namespace cardinal::serial
