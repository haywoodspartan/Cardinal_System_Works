#include <cardinal/game/reflection.hpp>
#include <cardinal/game/game_actor.hpp>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/utility.hpp>
#include <cardinal/core/diag/log.hpp>

namespace cardinal::game {

ClassRegistry& ClassRegistry::instance() {
    // Process-singleton with deterministic destruction (order-of-static-init
    // safe because the macro registrations write into this on first use).
    static ClassRegistry s;
    return s;
}

void ClassRegistry::register_class(ClassDef def) {
    // Last-wins on duplicate name (matches the cppscript reload pattern).
    for (auto& c : classes_) {
        if (c.name == def.name) { c = cardinal::move(def); return; }
    }
    classes_.push_back(cardinal::move(def));
}

const ClassDef* ClassRegistry::find(const cardinal::string& name) const {
    for (const auto& c : classes_) if (c.name == name) return &c;
    return nullptr;
}

cardinal::vector<cardinal::string> ClassRegistry::all_names() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(classes_.size());
    for (const auto& c : classes_) r.push_back(c.name);
    cardinal::sort(r.begin(), r.end());
    return r;
}

cardinal::vector<const ClassDef*> ClassRegistry::all_in_category(
    const cardinal::string& cat_prefix) const
{
    cardinal::vector<const ClassDef*> r;
    for (const auto& c : classes_) {
        if (c.category.compare(0, cat_prefix.size(), cat_prefix) == 0) {
            r.push_back(&c);
        }
    }
    cardinal::sort(r.begin(), r.end(),
        [](const ClassDef* a, const ClassDef* b){
            if (a->category != b->category) return a->category < b->category;
            return a->name < b->name;
        });
    return r;
}

usize ClassRegistry::size() const noexcept { return classes_.size(); }

// ---------------------------------------------------------------------------
// GameActor::clone — re-create the registered subclass + copy reflected
// property values. Lives here (not in the header) because it needs the
// ClassRegistry + PropertyDef layout, and game_actor.hpp must stay free of
// the reflection dependency.
// ---------------------------------------------------------------------------
cardinal::unique_ptr<cardinal::actor::Component> GameActor::clone() const {
    const ClassDef* def = ClassRegistry::instance().find(class_name_);
    if (def == nullptr || !def->create) {
        cardinal::log::warnf("game/actor",
            "GameActor::clone(): class '%s' not registered — prefab will "
            "omit this GameActor component", class_name_.c_str());
        return nullptr;
    }

    cardinal::unique_ptr<GameActor> copy = def->create();
    if (!copy) return nullptr;
    copy->set_class_name(class_name_);
    // owner_ / playing_ deliberately left at defaults — a stamped instance
    // is unattached + stopped until the world + Game wire it up.

    // Copy reflected property values source -> clone, matched by index.
    // Both instances are the SAME concrete class, so describe_properties
    // returns identical field layouts in the same order.
    if (def->describe_properties) {
        // const_cast is safe: describe_properties only reads the source's
        // field addresses; we copy OUT of them, never mutate the source.
        auto src_props = def->describe_properties(const_cast<GameActor*>(this));
        auto dst_props = def->describe_properties(copy.get());
        const usize n = (src_props.size() < dst_props.size())
                            ? src_props.size() : dst_props.size();
        for (usize i = 0; i < n; ++i) {
            const PropertyDef& s = src_props[i];
            PropertyDef&       d = dst_props[i];
            if (s.kind != d.kind || s.ptr == nullptr || d.ptr == nullptr) continue;
            switch (s.kind) {
                case PropertyKind::Float:
                    *static_cast<float*>(d.ptr) = *static_cast<const float*>(s.ptr);
                    break;
                case PropertyKind::Int:
                    *static_cast<int*>(d.ptr) = *static_cast<const int*>(s.ptr);
                    break;
                case PropertyKind::Bool:
                    *static_cast<bool*>(d.ptr) = *static_cast<const bool*>(s.ptr);
                    break;
                case PropertyKind::Vec3:
                case PropertyKind::Color: {
                    // Both stored as 3 contiguous floats.
                    float* dv = static_cast<float*>(d.ptr);
                    const float* sv = static_cast<const float*>(s.ptr);
                    dv[0] = sv[0]; dv[1] = sv[1]; dv[2] = sv[2];
                    break;
                }
                case PropertyKind::String:
                    *static_cast<cardinal::string*>(d.ptr) =
                        *static_cast<const cardinal::string*>(s.ptr);
                    break;
            }
        }
    }

    return copy;   // unique_ptr<GameActor> -> unique_ptr<Component> (upcast)
}

}  // namespace cardinal::game
