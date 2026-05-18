#pragma once

// =============================================================================
// Cardinal — Game class reflection.
//
// A GameActor subclass declares reflected properties that the editor can
// inspect, save, and load. We do NOT use a C++ reflection library — instead
// we bundle everything into a registration thunk emitted by the
// CARDINAL_REGISTER_GAME_CLASS macro. The thunk gets called once at static
// init time and adds a ClassDef to the global ClassRegistry.
//
// PropertyDef points at a member field on a single instance. The editor
// reads/writes via the void* pointer + Kind tag. Min/max/tooltip travel
// alongside so the inspector can render meaningful sliders.
//
// Example:
//   class TurretActor : public cardinal::game::GameActor {
//   public:
//       float fire_rate_hz = 2.0f;
//       float detection_range = 30.0f;
//       cardinal::scene::Vec3 muzzle_offset{0, 1.5f, 0};
//
//       void begin_play() override { ... }
//       void on_tick(float dt) override { ... }
//   };
//
//   CARDINAL_REGISTER_GAME_CLASS(TurretActor, "AI/Turret",
//       PROP_FLOAT(fire_rate_hz, 0.1f, 20.0f, "Shots per second")
//       PROP_FLOAT(detection_range, 1.0f, 200.0f, "Detection radius")
//       PROP_VEC3 (muzzle_offset, "Local-space muzzle position"));
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cardinal::game {

class GameActor;

enum class PropertyKind : u32 {
    Float, Int, Bool, Vec3, Color, String,
};

struct PropertyDef {
    std::string  name;
    std::string  tooltip;
    PropertyKind kind{PropertyKind::Float};
    void*        ptr{nullptr};        // typed pointer into the instance
    float        fmin{0.0f}, fmax{1.0f};
    int          imin{0},    imax{100};
};

struct ClassDef {
    std::string  name;                // "TurretActor"
    std::string  category;            // "AI/Turret" - slash-separated for the picker
    // Factory + property descriptor pair. Both bind the same instance.
    std::function<std::unique_ptr<GameActor>()>            create;
    std::function<std::vector<PropertyDef>(GameActor*)>    describe_properties;
};

class ClassRegistry {
public:
    static ClassRegistry& instance();

    void                      register_class(ClassDef def);
    const ClassDef*           find(const std::string& name) const;
    std::vector<std::string>  all_names() const;
    std::vector<const ClassDef*> all_in_category(const std::string& cat_prefix) const;
    usize                     size() const noexcept;

private:
    ClassRegistry() = default;
    std::vector<ClassDef> classes_;
};

// ---------------------------------------------------------------------------
// Macros — the developer-facing surface.
//
// CARDINAL_REGISTER_GAME_CLASS expands to a static struct whose constructor
// runs at process start (or first include of the TU). It builds a ClassDef
// and pushes it into the registry.
//
// PROP_* expand to expressions that build a PropertyDef inside the
// describe_properties lambda.
// ---------------------------------------------------------------------------
#define CARDINAL_GAME_CONCAT_(a, b) a##b
#define CARDINAL_GAME_REG_NAME_(line) CARDINAL_GAME_CONCAT_(_cardinal_game_reg_, line)

#define CARDINAL_REGISTER_GAME_CLASS(Type, Category, ...)                   \
    namespace {                                                             \
        struct CARDINAL_GAME_REG_NAME_(__LINE__) {                          \
            CARDINAL_GAME_REG_NAME_(__LINE__)() {                           \
                ::cardinal::game::ClassDef d;                               \
                d.name     = #Type;                                         \
                d.category = (Category);                                    \
                d.create   = []() -> std::unique_ptr<                       \
                    ::cardinal::game::GameActor> {                          \
                    return std::make_unique<Type>();                        \
                };                                                          \
                d.describe_properties = [](::cardinal::game::GameActor*     \
                                            base) {                         \
                    auto* self = static_cast<Type*>(base);                  \
                    (void)self;                                             \
                    std::vector<::cardinal::game::PropertyDef> props;       \
                    __VA_ARGS__                                             \
                    return props;                                           \
                };                                                          \
                ::cardinal::game::ClassRegistry::instance()                 \
                    .register_class(std::move(d));                          \
            }                                                               \
        } CARDINAL_GAME_REG_NAME_(__LINE__){};                              \
    }

#define PROP_FLOAT(field, fmin_, fmax_, tooltip_)                           \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::Float;                     \
        p.ptr  = static_cast<void*>(&self->field);                          \
        p.fmin = (fmin_); p.fmax = (fmax_);                                 \
        props.push_back(std::move(p));                                      \
    }

#define PROP_INT(field, imin_, imax_, tooltip_)                             \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::Int;                       \
        p.ptr  = static_cast<void*>(&self->field);                          \
        p.imin = (imin_); p.imax = (imax_);                                 \
        props.push_back(std::move(p));                                      \
    }

#define PROP_BOOL(field, tooltip_)                                          \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::Bool;                      \
        p.ptr  = static_cast<void*>(&self->field);                          \
        props.push_back(std::move(p));                                      \
    }

#define PROP_VEC3(field, tooltip_)                                          \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::Vec3;                      \
        p.ptr  = static_cast<void*>(&self->field);                          \
        props.push_back(std::move(p));                                      \
    }

#define PROP_COLOR(field, tooltip_)                                         \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::Color;                     \
        p.ptr  = static_cast<void*>(&self->field);                          \
        props.push_back(std::move(p));                                      \
    }

#define PROP_STRING(field, tooltip_)                                        \
    {                                                                       \
        ::cardinal::game::PropertyDef p;                                    \
        p.name = #field; p.tooltip = (tooltip_);                            \
        p.kind = ::cardinal::game::PropertyKind::String;                    \
        p.ptr  = static_cast<void*>(&self->field);                          \
        props.push_back(std::move(p));                                      \
    }

}  // namespace cardinal::game
