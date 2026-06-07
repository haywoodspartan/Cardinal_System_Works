#pragma once

// =============================================================================
// Cardinal — Mass Entities (data-oriented ECS).
//
// Sibling to cardinal::actor: where the actor system is OO + per-entity
// pointer-chase friendly (good for hundreds of authored gameplay objects),
// Mass is data-oriented + cache friendly (good for thousands of agents:
// crowds, particles, swarms, projectiles).
//
// Architecture:
//   - Each EntityId is a 32-bit slot index into archetype-specific
//     contiguous component arrays
//   - Components are POD structs registered via register_component<T>()
//   - Archetypes are computed from a component-bitmask; entities sharing
//     a bitmask live in one Chunk
//   - Systems iterate (component slice tuple) over each Chunk in parallel
//   - Add/remove component => move entity between archetypes (O(component bytes))
//
// This is intentionally simpler than EnTT — single-archetype iteration,
// no signal/observer machinery, no sparse sets. The win is "thousands of
// boids ticking in 1ms", not "engine team productivity".
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::mass {

inline constexpr u32 kMaxComponents = 32;

using EntityId       = u32;
inline constexpr EntityId kInvalidEntity = 0xFFFFFFFFu;
using ComponentMask  = u32;     // 1 bit per registered component (0..31)

// ---------------------------------------------------------------------------
// World — owns all entities + component storage.
// ---------------------------------------------------------------------------
struct ComponentDesc {
    cardinal::string  name;
    u32          size{0};       // bytes per component
    u32          align{1};
    u32          bit_index{0};  // 0..31
};

class World {
public:
    static cardinal::shared_ptr<World> create();
    ~World();

    // ----- Component registration -----------------------------------
    template <class T>
    u32 register_component(const cardinal::string& name) {
        // NOTE: pass the FULL 64-bit hash. add<T>/get<T> look it up via
        // bit_for_type_(typeid(T).hash_code()) un-truncated; an earlier
        // static_cast<u32> here keyed type_to_bit with a truncated hash,
        // so on any platform whose hash_code() has high bits set (MSVC)
        // every typed add<T>/get<T> silently missed and returned
        // false/nullptr. type_hash is u64 end-to-end.
        return register_component_internal_(
            name, static_cast<u32>(sizeof(T)), static_cast<u32>(alignof(T)),
            typeid(T).hash_code());
    }
    u32 component_count() const noexcept;
    const ComponentDesc* describe_component(u32 bit) const noexcept;

    // ----- Entity lifecycle -----------------------------------------
    EntityId create_entity();
    bool     destroy_entity(EntityId e);
    bool     alive(EntityId e) const noexcept;
    usize    entity_count() const noexcept;

    // Add a component — copies bytes from `data` into the entity's slot.
    // Returns false on bad EntityId or unknown component bit.
    bool add_component   (EntityId e, u32 bit, const void* data, usize size);
    bool remove_component(EntityId e, u32 bit);
    bool has_component   (EntityId e, u32 bit) const noexcept;

    // Typed wrappers — type T must have been registered first via
    // register_component<T>().
    template <class T>
    bool add(EntityId e, const T& value) {
        const u32 bit = bit_for_type_(typeid(T).hash_code());
        return add_component(e, bit, &value, sizeof(T));
    }
    template <class T>
    T* get(EntityId e) {
        const u32 bit = bit_for_type_(typeid(T).hash_code());
        return static_cast<T*>(get_component_raw_(e, bit));
    }
    template <class T>
    const T* get(EntityId e) const {
        const u32 bit = bit_for_type_(typeid(T).hash_code());
        return static_cast<const T*>(const_cast<World*>(this)->get_component_raw_(e, bit));
    }

    // ----- Iteration ------------------------------------------------
    // for_each(mask, fn) — calls fn(EntityId) for every entity whose mask
    // includes ALL bits in `required`.
    void for_each(ComponentMask required,
                  const cardinal::function<void(EntityId)>& fn) const;

    // Typed iteration helper. The lambda gets pointers to component arrays
    // for one chunk at a time + the chunk's entity-id slice.
    template <class FnChunk>
    void for_each_chunk(ComponentMask required, FnChunk&& fn) {
        for_each_chunk_internal_(required,
            [fn = cardinal::move(fn)](u32 entity_count, const EntityId* ids,
                                  const cardinal::vector<u8*>& comp_ptrs) {
                fn(entity_count, ids, comp_ptrs);
            });
    }

    // Parallel chunk iteration. Each chunk runs as its own JobSystem
    // task — chunks are independent (separate component arrays) so no
    // synchronisation is needed inside the callback. Falls back to
    // sequential when no JobSystem is bound.
    //
    // Use this when the per-entity work is non-trivial (>= a few
    // hundred ns per entity × 1024 entities = ~hundreds of µs per
    // chunk, well above the dispatch overhead). For trivial work
    // (single component read/write) plain for_each_chunk is faster
    // because the dispatch round-trip dominates.
    //
    // Important: the callback runs on a worker thread. Don't touch
    // World::add/remove_component or destroy_entity from inside it —
    // those mutate archetype storage and are not thread-safe.
    void for_each_chunk_parallel(
        ComponentMask required,
        const cardinal::function<void(u32, const EntityId*,
                                 const cardinal::vector<u8*>&)>& fn);

    // ----- Stats ----------------------------------------------------
    struct Stats {
        u32 entities       {0};
        u32 archetype_count{0};
        u32 chunk_count    {0};
        u32 component_count{0};
        u64 entities_created_total{0};
        u64 entities_destroyed_total{0};
    };
    Stats stats() const noexcept;

private:
    World() = default;

    u32 register_component_internal_(const cardinal::string& name, u32 size,
                                     u32 align, u64 type_hash);
    u32 bit_for_type_(u64 type_hash) const noexcept;
    void* get_component_raw_(EntityId e, u32 bit);

    void for_each_chunk_internal_(
        ComponentMask required,
        const cardinal::function<void(u32, const EntityId*,
                                 const cardinal::vector<u8*>&)>& fn);

    struct Impl;
    Impl* impl_{nullptr};
};

// ---------------------------------------------------------------------------
// Built-in components — common ones we ship for the boids / crowd demos.
// Users register additional ones at boot time.
// ---------------------------------------------------------------------------
struct CTransform {
    cardinal::scene::Vec3 position{0,0,0};
    cardinal::scene::Vec3 forward {0,0,-1};
};
struct CVelocity {
    cardinal::scene::Vec3 v{0,0,0};
    f32                   speed{1.0f};
};
struct CHealth {
    f32 hp{1.0f};
    f32 max_hp{1.0f};
};
struct CFaction {
    u32 id{0};
};

}  // namespace cardinal::mass
