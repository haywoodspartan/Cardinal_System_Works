// =============================================================================
// Cardinal — deterministic Mass-ECS regression suite.
//
// cardinal::mass is the data-oriented sibling of cardinal::actor: 32-bit
// entity slots, POD components addressed by a registration-order bit,
// archetype chunks (1024 entities/chunk), and a swap-remove storage
// model. The whole surface is pure CPU + fully deterministic, so this
// suite pins the contract that gameplay (crowds / swarms / projectiles)
// relies on and that a storage refactor must not silently break:
//
//   * register_component — bit == registration order, idempotent per
//     type, describe_component name/size/bounds;
//   * entity lifecycle — ids from 0, free-list LIFO reuse, alive/count,
//     destroy double-free + OOR rejection, cumulative stats;
//   * add/remove/has/get — archetype move keeps SHARED components, in-
//     place overwrite when already present, dead/bad-bit guards;
//   * for_each — (archetype & required) == required mask filter;
//   * for_each_chunk — typed per-chunk slices, only required bits mapped;
//   * the 1024-entity chunk split + Stats.
//
// Zero deps (same harness as the other suites). Exit 0 = all pass.
// =============================================================================

#include <cardinal/mass/mass.hpp>
#include <cardinal/core/log.hpp>

namespace {

namespace ms = cardinal::mass;
using Vec3 = cardinal::scene::Vec3;
using cardinal::u32;
using cardinal::u8;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("masstest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool v3eq(const Vec3& a, double x, double y, double z) {
    return a.x == x && a.y == y && a.z == z;     // exact: all test values fit f32
}

// ---- empty world + component registration -------------------------
void test_register() {
    auto w = ms::World::create();
    CHECK(w != nullptr);
    CHECK(w->component_count() == 0u);
    CHECK(w->entity_count() == 0u);
    const auto s0 = w->stats();
    CHECK(s0.entities == 0u && s0.component_count == 0u);
    CHECK(s0.archetype_count == 1u);             // archetype 0 reserved at create
    CHECK(s0.chunk_count == 0u);                 // no chunk until first entity
    CHECK(s0.entities_created_total == 0u &&
          s0.entities_destroyed_total == 0u);

    const u32 tb = w->register_component<ms::CTransform>("Transform");
    const u32 vb = w->register_component<ms::CVelocity>("Velocity");
    const u32 hb = w->register_component<ms::CHealth>("Health");
    CHECK(tb == 0u && vb == 1u && hb == 2u);     // bit == registration order
    CHECK(w->component_count() == 3u);

    const auto* d = w->describe_component(tb);
    CHECK(d != nullptr);
    if (d) {
        CHECK(d->name == "Transform");
        CHECK(d->size == static_cast<u32>(sizeof(ms::CTransform)));
        CHECK(d->bit_index == 0u);
    }
    CHECK(w->describe_component(99u) == nullptr);  // OOR bit

    // Idempotent: re-registering the same type returns the same bit and
    // does NOT add a component.
    CHECK(w->register_component<ms::CTransform>("ignored") == tb);
    CHECK(w->component_count() == 3u);
}

// ---- entity lifecycle + free-list reuse ---------------------------
void test_lifecycle() {
    auto w = ms::World::create();
    const ms::EntityId e0 = w->create_entity();
    const ms::EntityId e1 = w->create_entity();
    CHECK(e0 == 0u && e1 == 1u);                 // ids from 0 upward
    CHECK(w->alive(e0) && w->alive(e1));
    CHECK(w->entity_count() == 2u);
    CHECK(!w->alive(0xFFFFFFFFu));               // OOR id

    CHECK(w->destroy_entity(e0));
    CHECK(!w->alive(e0));
    CHECK(w->entity_count() == 1u);
    CHECK(!w->destroy_entity(e0));               // already dead
    CHECK(!w->destroy_entity(0xFFFFFFFFu));      // OOR

    // Free-list is LIFO — the next create reuses the freed slot id.
    const ms::EntityId e2 = w->create_entity();
    CHECK(e2 == e0);
    CHECK(w->alive(e2) && w->entity_count() == 2u);

    const auto s = w->stats();
    CHECK(s.entities_created_total == 3u);       // e0, e1, e2
    CHECK(s.entities_destroyed_total == 1u);
    CHECK(s.entities == 2u);
}

// ---- add/has/get/remove + archetype move keeps shared comps -------
void test_components() {
    auto w = ms::World::create();
    const u32 tb = w->register_component<ms::CTransform>("T");
    const u32 vb = w->register_component<ms::CVelocity>("V");

    const ms::EntityId e = w->create_entity();
    CHECK(!w->has_component(e, tb));
    CHECK(w->get<ms::CTransform>(e) == nullptr); // absent → null

    // Bind every get<>() to a local and guard the deref, so a future
    // regression FAILs cleanly instead of crashing on a null deref.
    ms::CTransform t; t.position = {1, 2, 3};
    CHECK(w->add<ms::CTransform>(e, t));
    CHECK(w->has_component(e, tb));
    {
        const auto* gt = w->get<ms::CTransform>(e);
        CHECK(gt != nullptr && v3eq(gt->position, 1, 2, 3));
    }

    ms::CVelocity vel; vel.v = {4, 5, 6}; vel.speed = 7.0f;
    CHECK(w->add<ms::CVelocity>(e, vel));         // archetype {T} → {T,V}
    {
        const auto* gv = w->get<ms::CVelocity>(e);
        CHECK(gv != nullptr && v3eq(gv->v, 4, 5, 6) && gv->speed == 7.0f);
        // The shared component must survive the archetype move:
        const auto* gt = w->get<ms::CTransform>(e);
        CHECK(gt != nullptr && v3eq(gt->position, 1, 2, 3));
    }

    // Re-add an already-present component → in-place overwrite, no move.
    ms::CTransform t2; t2.position = {9, 9, 9};
    CHECK(w->add<ms::CTransform>(e, t2));
    CHECK(w->has_component(e, vb));               // V still present
    {
        const auto* gt = w->get<ms::CTransform>(e);
        CHECK(gt != nullptr && v3eq(gt->position, 9, 9, 9));
    }

    // remove → archetype {T,V} → {T}; V gone, T intact.
    CHECK(w->remove_component(e, vb));
    CHECK(!w->has_component(e, vb));
    CHECK(w->has_component(e, tb));
    {
        const auto* gt = w->get<ms::CTransform>(e);
        CHECK(gt != nullptr && v3eq(gt->position, 9, 9, 9));
    }
    CHECK(w->get<ms::CVelocity>(e) == nullptr);
    CHECK(!w->remove_component(e, vb));           // not present → false

    // Guards: bad bit, then dead entity.
    CHECK(!w->add_component(e, 99u, &t, sizeof(t)));
    CHECK(w->destroy_entity(e));
    CHECK(!w->add_component(e, tb, &t, sizeof(t)));
    CHECK(!w->has_component(e, tb));
}

// ---- for_each mask filtering --------------------------------------
void test_for_each() {
    auto w = ms::World::create();
    const u32 tb = w->register_component<ms::CTransform>("T");
    const u32 vb = w->register_component<ms::CVelocity>("V");
    const ms::EntityId A = w->create_entity();   // {T}
    const ms::EntityId B = w->create_entity();   // {T,V}
    const ms::EntityId C = w->create_entity();   // {T,V}
    const ms::EntityId D = w->create_entity();   // {}
    ms::CTransform t; ms::CVelocity v;
    w->add<ms::CTransform>(A, t);
    w->add<ms::CTransform>(B, t); w->add<ms::CVelocity>(B, v);
    w->add<ms::CTransform>(C, t); w->add<ms::CVelocity>(C, v);
    (void)D;

    auto count = [&](ms::ComponentMask m) {
        u32 k = 0; w->for_each(m, [&](ms::EntityId) { ++k; }); return k;
    };
    CHECK(count(0u) == 4u);                       // required 0 ⊆ every archetype
    CHECK(count(1u << tb) == 3u);                 // A, B, C
    CHECK(count((1u << tb) | (1u << vb)) == 2u);  // B, C
    CHECK(count(1u << vb) == 2u);                 // B, C

    cardinal::vector<ms::EntityId> got;
    w->for_each((1u << tb) | (1u << vb),
                [&](ms::EntityId e) { got.push_back(e); });
    CHECK(got.size() == 2u);
    bool hasB = false, hasC = false;
    for (auto e : got) { hasB |= (e == B); hasC |= (e == C); }
    CHECK(hasB && hasC);
}

// ---- for_each_chunk typed per-chunk iteration ---------------------
void test_for_each_chunk() {
    auto w = ms::World::create();
    const u32 vb = w->register_component<ms::CVelocity>("V");
    constexpr u32 N = 50;
    for (u32 i = 0; i < N; ++i) {
        const ms::EntityId e = w->create_entity();
        ms::CVelocity v; v.speed = static_cast<float>(i);
        w->add<ms::CVelocity>(e, v);
    }
    double sum = 0.0;
    u32 seen = 0;
    w->for_each_chunk(1u << vb,
        [&](u32 cnt, const ms::EntityId*,
            const cardinal::vector<u8*>& ptrs) {
            const auto* vs = reinterpret_cast<const ms::CVelocity*>(ptrs[vb]);
            CHECK(vs != nullptr);
            for (u32 r = 0; r < cnt; ++r) { sum += vs[r].speed; ++seen; }
        });
    CHECK(seen == N);
    CHECK(sum == static_cast<double>(N * (N - 1) / 2));   // 0+1+…+49 = 1225
}

// ---- 1024-entity chunk split + Stats ------------------------------
void test_chunk_split_and_stats() {
    auto w = ms::World::create();
    const u32 tb = w->register_component<ms::CTransform>("T");
    constexpr u32 N = 1025;                       // kEntitiesPerChunk = 1024
    for (u32 i = 0; i < N; ++i) {
        const ms::EntityId e = w->create_entity();
        ms::CTransform t;
        w->add<ms::CTransform>(e, t);
    }
    CHECK(w->entity_count() == N);
    u32 n = 0;
    w->for_each(1u << tb, [&](ms::EntityId) { ++n; });
    CHECK(n == N);                                // visited across both chunks

    const auto s = w->stats();
    CHECK(s.entities == N);
    CHECK(s.component_count == 1u);
    CHECK(s.archetype_count >= 2u);               // archetype 0 + {T}
    CHECK(s.chunk_count >= 2u);                   // {T} spilled to a 2nd chunk
    CHECK(s.entities_created_total == N);
}

}  // namespace

int main() {
    test_register();
    test_lifecycle();
    test_components();
    test_for_each();
    test_for_each_chunk();
    test_chunk_split_and_stats();

    if (g_fail == 0) {
        cardinal::log::infof("masstest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("masstest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
