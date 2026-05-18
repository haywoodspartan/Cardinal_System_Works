// =============================================================================
// Cardinal — deterministic game-reflection regression suite.
//
// ClassRegistry + CARDINAL_REGISTER_GAME_CLASS are the editor /
// serialization backbone. Each registered class carries a factory
// (create) and a property descriptor list (describe_properties): name,
// kind, min/max, tooltip, and a void* pointing into the LIVE instance so
// the inspector reads/writes fields and save/load round-trips them. This
// suite pins both the macro path (static-init registration; create()
// builds the right type; each PropertyDef.ptr aliases that instance's
// field so a write through it mutates the object) and the direct
// registry API (last-wins replace, find, sorted all_names, category-
// prefix filter + (category,name) sort, size). The registry is a
// process singleton with no clear(), so tests are baseline-relative and
// use unique class names. Pure CPU, deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/game/reflection.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/core/log.hpp>

#include <memory>
#include <string>
#include <vector>

namespace gm = cardinal::game;
using Vec3   = cardinal::scene::Vec3;

// File-scope reflected test class + macro registration (runs at static
// init, before main()).
struct ReflTestActor : public gm::GameActor {
    float                 speed = 2.0f;
    int                   hp    = 100;
    bool                  alive = true;
    cardinal::scene::Vec3 pos{1.0f, 2.0f, 3.0f};
    std::string           label = "hello";
    cardinal::u32         col   = 0xAABBCCDDu;
};

CARDINAL_REGISTER_GAME_CLASS(ReflTestActor, "Test/Refl",
    PROP_FLOAT (speed, 0.5f, 9.5f, "spd")
    PROP_INT   (hp,    1,    200,  "health")
    PROP_BOOL  (alive,             "is alive")
    PROP_VEC3  (pos,               "position")
    PROP_STRING(label,             "name")
    PROP_COLOR (col,               "colour"))

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("refltest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-5f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

gm::ClassDef make_def(const char* name, const char* category) {
    gm::ClassDef d;
    d.name     = name;
    d.category = category;
    d.create   = []() -> std::unique_ptr<gm::GameActor> {
        return std::make_unique<ReflTestActor>();
    };
    d.describe_properties = [](gm::GameActor*) {
        return std::vector<gm::PropertyDef>{};
    };
    return d;
}

// ---- PropertyDef defaults + PropertyKind ordering -----------------
void test_property_def() {
    gm::PropertyDef p;
    CHECK(p.name.empty() && p.tooltip.empty());
    CHECK(p.kind == gm::PropertyKind::Float);
    CHECK(p.ptr == nullptr);
    CHECK(ap(p.fmin, 0.0f) && ap(p.fmax, 1.0f));
    CHECK(p.imin == 0 && p.imax == 100);

    using K = gm::PropertyKind;
    CHECK(static_cast<cardinal::u32>(K::Float)  == 0u);
    CHECK(static_cast<cardinal::u32>(K::Int)    == 1u);
    CHECK(static_cast<cardinal::u32>(K::Bool)   == 2u);
    CHECK(static_cast<cardinal::u32>(K::Vec3)   == 3u);
    CHECK(static_cast<cardinal::u32>(K::Color)  == 4u);
    CHECK(static_cast<cardinal::u32>(K::String) == 5u);
}

// ---- macro registration: factory + property descriptors -----------
void test_macro_registration() {
    auto& reg = gm::ClassRegistry::instance();
    const gm::ClassDef* d = reg.find("ReflTestActor");
    CHECK(d != nullptr);
    if (!d) return;
    CHECK(d->name == "ReflTestActor");
    CHECK(d->category == "Test/Refl");
    CHECK(static_cast<bool>(d->create));
    CHECK(static_cast<bool>(d->describe_properties));

    std::unique_ptr<gm::GameActor> inst = d->create();
    CHECK(inst != nullptr);
    if (!inst) return;
    auto* ta = static_cast<ReflTestActor*>(inst.get());
    // Default ctor ran on the created instance.
    CHECK(ap(ta->speed, 2.0f) && ta->hp == 100 && ta->alive == true);
    CHECK(ta->label == "hello" && ta->col == 0xAABBCCDDu);

    std::vector<gm::PropertyDef> props = d->describe_properties(inst.get());
    CHECK(props.size() == sz(6));               // declaration order preserved
    if (props.size() != sz(6)) return;

    using K = gm::PropertyKind;
    CHECK(props[0].name == "speed" && props[0].kind == K::Float);
    CHECK(props[0].tooltip == "spd");
    CHECK(ap(props[0].fmin, 0.5f) && ap(props[0].fmax, 9.5f));
    CHECK(props[0].ptr == static_cast<void*>(&ta->speed));

    CHECK(props[1].name == "hp" && props[1].kind == K::Int);
    CHECK(props[1].tooltip == "health");
    CHECK(props[1].imin == 1 && props[1].imax == 200);
    CHECK(props[1].ptr == static_cast<void*>(&ta->hp));

    CHECK(props[2].name == "alive" && props[2].kind == K::Bool);
    CHECK(props[2].tooltip == "is alive");
    CHECK(props[2].ptr == static_cast<void*>(&ta->alive));

    CHECK(props[3].name == "pos" && props[3].kind == K::Vec3);
    CHECK(props[3].ptr == static_cast<void*>(&ta->pos));

    CHECK(props[4].name == "label" && props[4].kind == K::String);
    CHECK(props[4].ptr == static_cast<void*>(&ta->label));

    CHECK(props[5].name == "col" && props[5].kind == K::Color);
    CHECK(props[5].ptr == static_cast<void*>(&ta->col));

    // The void* genuinely aliases the instance — a write lands on the field.
    *static_cast<float*>(props[0].ptr)        = 7.5f;
    *static_cast<int*>(props[1].ptr)          = 42;
    *static_cast<bool*>(props[2].ptr)         = false;
    static_cast<Vec3*>(props[3].ptr)->x       = 9.0f;
    *static_cast<std::string*>(props[4].ptr)  = "world";
    *static_cast<cardinal::u32*>(props[5].ptr) = 1u;
    CHECK(ap(ta->speed, 7.5f) && ta->hp == 42 && ta->alive == false);
    CHECK(ap(ta->pos.x, 9.0f) && ta->label == "world" && ta->col == 1u);

    // describe_properties binds the SPECIFIC instance passed in.
    std::unique_ptr<gm::GameActor> inst2 = d->create();
    auto* ta2 = static_cast<ReflTestActor*>(inst2.get());
    std::vector<gm::PropertyDef> props2 = d->describe_properties(inst2.get());
    CHECK(props2.size() == sz(6));
    CHECK(props2[0].ptr == static_cast<void*>(&ta2->speed));
    CHECK(props2[0].ptr != props[0].ptr);       // distinct instances
    CHECK(ap(ta2->speed, 2.0f));                 // untouched by inst's writes
}

// ---- direct registry API: last-wins / find / sort / category ------
void test_direct_registry() {
    auto& reg = gm::ClassRegistry::instance();
    const cardinal::usize base = reg.size();

    CHECK(reg.find("ZZ_alpha") == nullptr);
    reg.register_class(make_def("ZZ_alpha", "ZCat/Sub"));
    CHECK(reg.size() == base + sz(1));
    const gm::ClassDef* a = reg.find("ZZ_alpha");
    CHECK(a != nullptr && a->category == "ZCat/Sub");
    CHECK(reg.find("ZZ_nope") == nullptr);

    // Re-register same name → last-wins replace, NOT a second entry.
    reg.register_class(make_def("ZZ_alpha", "ZCat/Changed"));
    CHECK(reg.size() == base + sz(1));           // size unchanged
    a = reg.find("ZZ_alpha");
    CHECK(a != nullptr && a->category == "ZCat/Changed");

    reg.register_class(make_def("ZZ_beta",  "ZCat/Sub"));
    reg.register_class(make_def("ZZ_gamma", "ZAlt"));
    CHECK(reg.size() == base + sz(3));

    // all_names() is globally sorted ascending and contains ours.
    auto names = reg.all_names();
    CHECK(names.size() == reg.size());
    bool sorted = true;
    for (cardinal::usize i = 1; i < names.size(); ++i)
        if (names[i - 1] > names[i]) sorted = false;
    CHECK(sorted);
    bool saw_a = false, saw_b = false, saw_g = false, saw_macro = false;
    for (const auto& n : names) {
        if (n == "ZZ_alpha")      saw_a = true;
        if (n == "ZZ_beta")       saw_b = true;
        if (n == "ZZ_gamma")      saw_g = true;
        if (n == "ReflTestActor") saw_macro = true;
    }
    CHECK(saw_a && saw_b && saw_g && saw_macro);

    // all_in_category("ZCat/") → only ZCat/-prefixed, sorted by
    // (category, name): "ZCat/Changed"(alpha) < "ZCat/Sub"(beta).
    auto cat = reg.all_in_category("ZCat/");
    CHECK(cat.size() == sz(2));
    if (cat.size() == sz(2)) {
        CHECK(cat[0]->name == "ZZ_alpha" && cat[0]->category == "ZCat/Changed");
        CHECK(cat[1]->name == "ZZ_beta"  && cat[1]->category == "ZCat/Sub");
    }
    // "ZAlt" must NOT match the "ZCat/" prefix.
    auto zalt = reg.all_in_category("ZAlt");
    bool zalt_has_gamma = false;
    for (const auto* c : zalt) if (c->name == "ZZ_gamma") zalt_has_gamma = true;
    CHECK(zalt_has_gamma);
    for (const auto* c : zalt) CHECK(c->name != "ZZ_alpha");

    // Empty prefix → every class; non-matching prefix → none.
    CHECK(reg.all_in_category("").size() == reg.size());
    CHECK(reg.all_in_category("__definitely_no_such_prefix__").empty());

    // all_in_category result is sorted by (category, then name).
    auto all = reg.all_in_category("");
    bool cat_sorted = true;
    for (cardinal::usize i = 1; i < all.size(); ++i) {
        const auto& A = *all[i - 1];
        const auto& B = *all[i];
        if (A.category > B.category ||
            (A.category == B.category && A.name > B.name)) cat_sorted = false;
    }
    CHECK(cat_sorted);

    // A directly-registered create()/describe still works.
    const gm::ClassDef* g = reg.find("ZZ_gamma");
    CHECK(g != nullptr);
    if (g) {
        auto inst = g->create();
        CHECK(inst != nullptr);
        CHECK(g->describe_properties(inst.get()).empty());   // make_def → none
    }
}

}  // namespace

int main() {
    test_property_def();
    test_macro_registration();
    test_direct_registry();

    if (g_fail == 0) {
        cardinal::log::infof("refltest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("refltest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
