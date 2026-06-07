// =============================================================================
// Cardinal — built-in starter prefabs implementation.
// =============================================================================
#include <cardinal/actor/builtin_prefabs.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::actor {

namespace {

// Build a detached prototype actor (id 0) with a Transform already on it,
// ready for the caller to adopt type-specific components onto.
cardinal::unique_ptr<Actor> make_proto(const char* name) {
    auto a = cardinal::make_unique<Actor>(0u, name);
    a->adopt_component(cardinal::make_unique<TransformComponent>());
    return a;
}

// Register `proto` as `name` only if the world doesn't already have a prefab
// by that name (don't clobber designer-captured content). Returns 1 if added.
u32 add_if_absent(World& w, const char* name, cardinal::unique_ptr<Actor> proto) {
    if (w.has_prefab(name)) return 0;
    w.add_prefab(name, cardinal::move(proto));
    return 1;
}

}  // namespace

u32 register_builtin_prefabs(World& world) {
    u32 added = 0;

    // ---- Lights -------------------------------------------------------
    {
        auto a = make_proto("Point Light");
        auto l = cardinal::make_unique<LightComponent>();
        l->kind = LightKind::Point;
        l->color = {1.0f, 1.0f, 1.0f};
        l->intensity = 4.0f;
        l->range = 15.0f;
        a->adopt_component(cardinal::move(l));
        added += add_if_absent(world, "Point Light", cardinal::move(a));
    }
    {
        auto a = make_proto("Directional Light");
        auto l = cardinal::make_unique<LightComponent>();
        l->kind = LightKind::Directional;
        l->color = {1.0f, 0.96f, 0.86f};   // warm key light
        l->intensity = 1.6f;
        a->adopt_component(cardinal::move(l));
        added += add_if_absent(world, "Directional Light", cardinal::move(a));
    }
    {
        auto a = make_proto("Spot Light");
        auto l = cardinal::make_unique<LightComponent>();
        l->kind = LightKind::Spot;
        l->color = {1.0f, 1.0f, 1.0f};
        l->intensity = 6.0f;
        l->range = 25.0f;
        l->spot_inner_cos = 0.95f;
        l->spot_outer_cos = 0.85f;
        a->adopt_component(cardinal::move(l));
        added += add_if_absent(world, "Spot Light", cardinal::move(a));
    }

    // ---- Physics prop -------------------------------------------------
    {
        auto a = make_proto("Physics Cube");
        auto m = cardinal::make_unique<MeshComponent>();
        m->asset_id = "primitives/box";
        m->tint = {0.8f, 0.8f, 0.82f};
        a->adopt_component(cardinal::move(m));
        auto rb = cardinal::make_unique<RigidBodyComponent>();
        rb->mass = 1.0f;
        rb->use_gravity = true;
        a->adopt_component(cardinal::move(rb));
        added += add_if_absent(world, "Physics Cube", cardinal::move(a));
    }

    // ---- Camera -------------------------------------------------------
    {
        auto a = make_proto("Camera");
        a->adopt_component(cardinal::make_unique<CameraComponent>());
        added += add_if_absent(world, "Camera", cardinal::move(a));
    }

    // ---- Gameplay markers (tagged, transform-only) --------------------
    {
        auto a = make_proto("Trigger Volume");
        auto t = cardinal::make_unique<TagComponent>();
        t->add("trigger");
        a->adopt_component(cardinal::move(t));
        added += add_if_absent(world, "Trigger Volume", cardinal::move(a));
    }
    {
        auto a = make_proto("Player Start");
        auto t = cardinal::make_unique<TagComponent>();
        t->add("player_start");
        a->adopt_component(cardinal::move(t));
        added += add_if_absent(world, "Player Start", cardinal::move(a));
    }

    cardinal::log::infof("actor/world",
        "register_builtin_prefabs: %u starter prefab(s) installed", added);
    return added;
}

}  // namespace cardinal::actor
