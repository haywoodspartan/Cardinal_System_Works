#include <cardinal/mass/mass.hpp>

#include <cardinal/core/async.hpp>
#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/atomic.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/cstring.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::mass {

namespace {
constexpr u32 kEntitiesPerChunk = 1024;
}

struct Chunk {
    ComponentMask                          archetype_mask{0};
    u32                                    count{0};
    cardinal::vector<EntityId>                  entity_ids;          // size <= kEntitiesPerChunk
    // Per-bit storage; only allocated for bits in archetype_mask. Each is
    // a flat array of `count * stride` bytes.
    cardinal::array<cardinal::vector<u8>, kMaxComponents> data;
};

struct Archetype {
    ComponentMask          mask{0};
    cardinal::vector<u32>       chunk_indices;          // into world.chunks_
};

struct EntitySlot {
    u32  archetype_index{0xFFFFFFFFu};   // -1 = dead
    u32  chunk_index    {0xFFFFFFFFu};
    u32  row            {0xFFFFFFFFu};   // index inside chunk
    u32  generation     {0};
};

struct World::Impl {
    cardinal::vector<ComponentDesc>             components;
    cardinal::unordered_map<u64, u32>           type_to_bit;     // typeid().hash_code() -> bit

    cardinal::vector<Chunk>                     chunks;
    cardinal::vector<Archetype>                 archetypes;
    cardinal::unordered_map<ComponentMask, u32> mask_to_archetype;

    cardinal::vector<EntitySlot>                slots;
    cardinal::vector<EntityId>                  free_entity_ids;
    // Start at 0 so the first push_back makes slots[0] valid. Previously
    // started at 1 which left slots[1] OOR on the very first create_entity.
    EntityId                               next_entity_id{0};

    u64                                    entities_created_total{0};
    u64                                    entities_destroyed_total{0};

    u32 ensure_archetype(ComponentMask mask) {
        auto it = mask_to_archetype.find(mask);
        if (it != mask_to_archetype.end()) return it->second;
        Archetype a{};
        a.mask = mask;
        archetypes.push_back(cardinal::move(a));
        const u32 idx = static_cast<u32>(archetypes.size() - 1);
        mask_to_archetype[mask] = idx;
        return idx;
    }

    u32 ensure_chunk_with_room(u32 archetype_idx) {
        Archetype& a = archetypes[archetype_idx];
        for (u32 ci : a.chunk_indices) {
            if (chunks[ci].count < kEntitiesPerChunk) return ci;
        }
        Chunk c{};
        c.archetype_mask = a.mask;
        c.entity_ids.reserve(kEntitiesPerChunk);
        // Allocate per-component storage for bits in mask.
        for (u32 b = 0; b < kMaxComponents; ++b) {
            if (a.mask & (1u << b)) {
                c.data[b].reserve(static_cast<usize>(components[b].size) * kEntitiesPerChunk);
            }
        }
        chunks.push_back(cardinal::move(c));
        const u32 ci = static_cast<u32>(chunks.size() - 1);
        a.chunk_indices.push_back(ci);
        return ci;
    }

    void* component_ptr(u32 chunk_idx, u32 row, u32 bit) {
        Chunk& c = chunks[chunk_idx];
        const u32 stride = components[bit].size;
        return c.data[bit].data() + static_cast<usize>(row) * stride;
    }

    // Move entity from (cur_arch, cur_chunk, cur_row) to a new archetype
    // adding/removing component `bit`. Optionally `set_data` populates the
    // newly-added component.
    //
    // CRITICAL: ensure_chunk_with_room() may push_back into `chunks` and
    // invalidate any in-scope Chunk& we're holding. We therefore work with
    // INDICES (src_chunk_idx / dst_chunk_idx) and re-acquire references
    // through chunks[idx] after every operation that could realloc.
    // EntitySlot& is also re-acquired after destruct- / archetype-mutation.
    void move_entity(EntityId e, ComponentMask new_mask,
                     u32 changed_bit, bool adding,
                     const void* set_data, usize set_size)
    {
        // Snapshot the source position by VALUE before any reallocation.
        const u32 src_chunk_idx = slots[e].chunk_index;
        const u32 src_row       = slots[e].row;

        // ensure_archetype grows `archetypes` (we don't hold an Archetype&).
        const u32 dst_arch  = ensure_archetype(new_mask);
        // ensure_chunk_with_room MAY push_back into `chunks`, invalidating
        // any prior Chunk& we held. Take the index, not the reference.
        const u32 dst_chunk_idx = ensure_chunk_with_room(dst_arch);
        // Re-acquire references AFTER the potential realloc. Use them only
        // for short stretches; never hold across another resize.
        const u32 dst_row = chunks[dst_chunk_idx].count;

        // Grow each present component vector in dst by one slot (zero-init).
        // dst data vectors are heap-allocated themselves so they can grow
        // without invalidating the outer `chunks` vector.
        for (u32 b = 0; b < kMaxComponents; ++b) {
            if ((new_mask & (1u << b)) == 0) continue;
            const u32 stride = components[b].size;
            chunks[dst_chunk_idx].data[b].resize(
                chunks[dst_chunk_idx].data[b].size() + stride, 0);
        }
        chunks[dst_chunk_idx].entity_ids.push_back(e);

        // Copy shared components from src into dst. Indexing chunks[] each
        // step keeps us safe even if a future change adds an inner realloc.
        const ComponentMask src_mask = chunks[src_chunk_idx].archetype_mask;
        const ComponentMask common   = (src_mask & new_mask);
        for (u32 b = 0; b < kMaxComponents; ++b) {
            if ((common & (1u << b)) == 0) continue;
            const u32 stride = components[b].size;
            cardinal::memcpy(chunks[dst_chunk_idx].data[b].data() +
                            static_cast<usize>(dst_row) * stride,
                        chunks[src_chunk_idx].data[b].data() +
                            static_cast<usize>(src_row) * stride,
                        stride);
        }
        if (adding && set_data != nullptr) {
            const u32 stride = components[changed_bit].size;
            const usize copy_n = cardinal::min(static_cast<usize>(stride), set_size);
            cardinal::memcpy(chunks[dst_chunk_idx].data[changed_bit].data() +
                            static_cast<usize>(dst_row) * stride,
                        set_data, copy_n);
        }

        // Swap-remove src_row from src chunk.
        const u32 src_last = chunks[src_chunk_idx].count - 1;
        if (src_row != src_last) {
            for (u32 b = 0; b < kMaxComponents; ++b) {
                if ((src_mask & (1u << b)) == 0) continue;
                const u32 stride = components[b].size;
                cardinal::memcpy(chunks[src_chunk_idx].data[b].data() +
                                static_cast<usize>(src_row) * stride,
                            chunks[src_chunk_idx].data[b].data() +
                                static_cast<usize>(src_last) * stride,
                            stride);
            }
            const EntityId moved_id = chunks[src_chunk_idx].entity_ids[src_last];
            chunks[src_chunk_idx].entity_ids[src_row] = moved_id;
            slots[moved_id].row = src_row;
        }
        for (u32 b = 0; b < kMaxComponents; ++b) {
            if ((src_mask & (1u << b)) == 0) continue;
            const u32 stride = components[b].size;
            chunks[src_chunk_idx].data[b].resize(
                chunks[src_chunk_idx].data[b].size() - stride);
        }
        chunks[src_chunk_idx].entity_ids.pop_back();
        --chunks[src_chunk_idx].count;

        ++chunks[dst_chunk_idx].count;
        slots[e].archetype_index = dst_arch;
        slots[e].chunk_index     = dst_chunk_idx;
        slots[e].row             = dst_row;
    }
};

cardinal::shared_ptr<World> World::create() {
    auto w = cardinal::shared_ptr<World>(new World());
    w->impl_ = new Impl{};
    // Reserve archetype 0 = "no components" (the default for fresh entities).
    w->impl_->ensure_archetype(0u);
    cardinal::log::infof("mass", "Mass world online");
    return w;
}

World::~World() { delete impl_; }

u32 World::register_component_internal_(const cardinal::string& name, u32 size,
                                        u32 align, u64 type_hash)
{
    if (impl_->components.size() >= kMaxComponents) {
        cardinal::log::errorf("mass", "kMaxComponents (%u) exceeded", kMaxComponents);
        return 0xFFFFFFFFu;
    }
    auto it = impl_->type_to_bit.find(type_hash);
    if (it != impl_->type_to_bit.end()) return it->second;
    ComponentDesc d{};
    d.name      = name;
    d.size      = size;
    d.align     = align;
    d.bit_index = static_cast<u32>(impl_->components.size());
    impl_->components.push_back(cardinal::move(d));
    impl_->type_to_bit[type_hash] = d.bit_index;
    return d.bit_index;
}

u32 World::component_count() const noexcept {
    return static_cast<u32>(impl_->components.size());
}
const ComponentDesc* World::describe_component(u32 bit) const noexcept {
    if (bit >= impl_->components.size()) return nullptr;
    return &impl_->components[bit];
}

u32 World::bit_for_type_(u64 type_hash) const noexcept {
    auto it = impl_->type_to_bit.find(type_hash);
    return it == impl_->type_to_bit.end() ? 0xFFFFFFFFu : it->second;
}

EntityId World::create_entity() {
    EntityId e;
    if (!impl_->free_entity_ids.empty()) {
        e = impl_->free_entity_ids.back();
        impl_->free_entity_ids.pop_back();
    } else {
        e = impl_->next_entity_id++;
        // Resize-to-fit is the safest invariant: slots[e] is always valid
        // after this branch regardless of where next_entity_id starts.
        if (e >= impl_->slots.size()) impl_->slots.resize(e + 1);
    }
    EntitySlot& s = impl_->slots[e];
    const u32 arch = impl_->ensure_archetype(0u);
    const u32 chunk = impl_->ensure_chunk_with_room(arch);
    Chunk& c = impl_->chunks[chunk];
    s.archetype_index = arch;
    s.chunk_index     = chunk;
    s.row             = c.count;
    s.generation     += 1;
    c.entity_ids.push_back(e);
    ++c.count;
    ++impl_->entities_created_total;
    return e;
}

bool World::destroy_entity(EntityId e) {
    if (e >= impl_->slots.size()) return false;
    EntitySlot& s = impl_->slots[e];
    if (s.archetype_index == 0xFFFFFFFFu) return false;
    Chunk& c = impl_->chunks[s.chunk_index];
    const u32 last = c.count - 1;
    if (s.row != last) {
        for (u32 b = 0; b < kMaxComponents; ++b) {
            if ((c.archetype_mask & (1u << b)) == 0) continue;
            const u32 stride = impl_->components[b].size;
            cardinal::memcpy(c.data[b].data() + static_cast<usize>(s.row) * stride,
                        c.data[b].data() + static_cast<usize>(last) * stride,
                        stride);
        }
        EntityId moved = c.entity_ids[last];
        c.entity_ids[s.row] = moved;
        impl_->slots[moved].row = s.row;
    }
    for (u32 b = 0; b < kMaxComponents; ++b) {
        if ((c.archetype_mask & (1u << b)) == 0) continue;
        c.data[b].resize(c.data[b].size() - impl_->components[b].size);
    }
    c.entity_ids.pop_back();
    --c.count;
    s.archetype_index = 0xFFFFFFFFu;
    impl_->free_entity_ids.push_back(e);
    ++impl_->entities_destroyed_total;
    return true;
}

bool World::alive(EntityId e) const noexcept {
    if (e >= impl_->slots.size()) return false;
    return impl_->slots[e].archetype_index != 0xFFFFFFFFu;
}
usize World::entity_count() const noexcept {
    usize n = 0;
    for (const auto& c : impl_->chunks) n += c.count;
    return n;
}

bool World::add_component(EntityId e, u32 bit, const void* data, usize size) {
    if (!alive(e) || bit >= impl_->components.size()) return false;
    EntitySlot& s = impl_->slots[e];
    const Chunk& c = impl_->chunks[s.chunk_index];
    const ComponentMask new_mask = c.archetype_mask | (1u << bit);
    if (new_mask == c.archetype_mask) {
        // Already present — overwrite in place.
        if (data) {
            void* p = impl_->component_ptr(s.chunk_index, s.row, bit);
            cardinal::memcpy(p, data, cardinal::min(static_cast<usize>(impl_->components[bit].size), size));
        }
        return true;
    }
    impl_->move_entity(e, new_mask, bit, /*adding=*/true, data, size);
    return true;
}

bool World::remove_component(EntityId e, u32 bit) {
    if (!alive(e) || bit >= impl_->components.size()) return false;
    EntitySlot& s = impl_->slots[e];
    const Chunk& c = impl_->chunks[s.chunk_index];
    const ComponentMask new_mask = c.archetype_mask & ~(1u << bit);
    if (new_mask == c.archetype_mask) return false;
    impl_->move_entity(e, new_mask, bit, /*adding=*/false, nullptr, 0);
    return true;
}

bool World::has_component(EntityId e, u32 bit) const noexcept {
    if (!alive(e) || bit >= impl_->components.size()) return false;
    return (impl_->chunks[impl_->slots[e].chunk_index].archetype_mask & (1u << bit)) != 0;
}

void* World::get_component_raw_(EntityId e, u32 bit) {
    if (!alive(e) || bit >= impl_->components.size()) return nullptr;
    const EntitySlot& s = impl_->slots[e];
    Chunk& c = impl_->chunks[s.chunk_index];
    if ((c.archetype_mask & (1u << bit)) == 0) return nullptr;
    const u32 stride = impl_->components[bit].size;
    return c.data[bit].data() + static_cast<usize>(s.row) * stride;
}

void World::for_each(ComponentMask required,
                     const cardinal::function<void(EntityId)>& fn) const
{
    if (!fn) return;
    for (const auto& c : impl_->chunks) {
        if ((c.archetype_mask & required) != required) continue;
        for (u32 r = 0; r < c.count; ++r) fn(c.entity_ids[r]);
    }
}

void World::for_each_chunk_internal_(
    ComponentMask required,
    const cardinal::function<void(u32, const EntityId*,
                             const cardinal::vector<u8*>&)>& fn)
{
    if (!fn) return;
    cardinal::vector<u8*> ptrs(kMaxComponents, nullptr);
    for (auto& c : impl_->chunks) {
        if ((c.archetype_mask & required) != required) continue;
        for (u32 b = 0; b < kMaxComponents; ++b) {
            ptrs[b] = (required & (1u << b)) ? c.data[b].data() : nullptr;
        }
        fn(c.count, c.entity_ids.data(), ptrs);
    }
}

void World::for_each_chunk_parallel(
    ComponentMask required,
    const cardinal::function<void(u32, const EntityId*,
                             const cardinal::vector<u8*>&)>& fn)
{
    if (!fn) return;

    // Filter matching chunks first so each worker job knows its target
    // index in O(1). The chunk vector itself isn't mutated during
    // iteration (callers must respect the "no add/remove during
    // iteration" rule), so it's safe to capture by reference.
    cardinal::vector<Chunk*> matches;
    matches.reserve(impl_->chunks.size());
    for (auto& c : impl_->chunks) {
        if ((c.archetype_mask & required) == required) matches.push_back(&c);
    }
    if (matches.empty()) return;

    // grain=0 lets parallel_for pick ~4 chunks per worker. With small
    // match counts (<32) it falls through the inline threshold and
    // runs sequentially — no dispatch overhead.
    cardinal::async::parallel_for(
        0u, static_cast<u32>(matches.size()),
        [&](u32 i) {
            // Build the ptrs vector LOCALLY per worker — don't share a
            // captured one across threads.
            cardinal::vector<u8*> ptrs(kMaxComponents, nullptr);
            Chunk& c = *matches[i];
            for (u32 b = 0; b < kMaxComponents; ++b) {
                ptrs[b] = (required & (1u << b)) ? c.data[b].data() : nullptr;
            }
            fn(c.count, c.entity_ids.data(), ptrs);
        });
}

World::Stats World::stats() const noexcept {
    Stats s{};
    s.entities        = static_cast<u32>(entity_count());
    s.archetype_count = static_cast<u32>(impl_->archetypes.size());
    s.chunk_count     = static_cast<u32>(impl_->chunks.size());
    s.component_count = static_cast<u32>(impl_->components.size());
    s.entities_created_total   = impl_->entities_created_total;
    s.entities_destroyed_total = impl_->entities_destroyed_total;
    return s;
}

}  // namespace cardinal::mass
