#include <cardinal/serial/serial.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/sky/sky.hpp>

#include <cardinal/core/cstdio.hpp>      // cardinal::snprintf/sscanf
#include <cardinal/core/cstdlib.hpp>     // cardinal::atof/atoi
#include <cardinal/core/cstring.hpp>     // cardinal raw byte ops
#include <cardinal/core/fstream.hpp>     // cardinal::ifstream/ofstream/ios/getline
#include <cardinal/core/sstream.hpp>     // cardinal::stringstream
#include <cardinal/core/utility.hpp>     // cardinal::move
#include <cardinal/core/containers.hpp>  // cardinal::vector

namespace cardinal::serial {

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------
namespace text {

void emit_kv(cardinal::string& out, const char* key, const char* value) {
    out += "  ";
    out += key;
    out += " = \"";
    out += value;
    out += "\"\n";
}

void emit_kf(cardinal::string& out, const char* key, float value) {
    char buf[64];
    cardinal::snprintf(buf, sizeof(buf), "  %s = %.6f\n", key, value);
    out += buf;
}

void emit_kv3(cardinal::string& out, const char* key, float x, float y, float z) {
    char buf[96];
    cardinal::snprintf(buf, sizeof(buf), "  %s = (%.6f, %.6f, %.6f)\n", key, x, y, z);
    out += buf;
}

bool parse_kv(const cardinal::string& line, cardinal::string* key, cardinal::string* value) {
    const auto eq = line.find('=');
    if (eq == cardinal::string::npos) return false;
    auto trim = [](cardinal::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' '  || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.pop_back();
        return s;
    };
    *key   = trim(line.substr(0, eq));
    *value = trim(line.substr(eq + 1));
    if (value->size() >= 2 && value->front() == '"' && value->back() == '"') {
        *value = value->substr(1, value->size() - 2);
    }
    return true;
}

}  // namespace text

namespace {

// ---------------------------------------------------------------------------
// Property emit / apply
// ---------------------------------------------------------------------------
void emit_prop(cardinal::string& out, const cardinal::game::PropertyDef& p) {
    using K = cardinal::game::PropertyKind;
    char buf[256];
    switch (p.kind) {
        case K::Float: {
            const float* f = static_cast<const float*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf), "  prop float %s = %.6f\n",
                          p.name.c_str(), *f);
            out += buf;
            break;
        }
        case K::Int: {
            const int* i = static_cast<const int*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf), "  prop int %s = %d\n",
                          p.name.c_str(), *i);
            out += buf;
            break;
        }
        case K::Bool: {
            const bool* b = static_cast<const bool*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf), "  prop bool %s = %s\n",
                          p.name.c_str(), *b ? "true" : "false");
            out += buf;
            break;
        }
        case K::Vec3: {
            const float* v = static_cast<const float*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf),
                "  prop vec3 %s = (%.6f, %.6f, %.6f)\n",
                p.name.c_str(), v[0], v[1], v[2]);
            out += buf;
            break;
        }
        case K::Color: {
            const float* v = static_cast<const float*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf),
                "  prop color %s = (%.6f, %.6f, %.6f)\n",
                p.name.c_str(), v[0], v[1], v[2]);
            out += buf;
            break;
        }
        case K::String: {
            const cardinal::string* s = static_cast<const cardinal::string*>(p.ptr);
            cardinal::snprintf(buf, sizeof(buf), "  prop string %s = \"%s\"\n",
                          p.name.c_str(), s->c_str());
            out += buf;
            break;
        }
    }
}

bool apply_prop(cardinal::game::PropertyDef& p,
                const cardinal::string& kind,
                const cardinal::string& value)
{
    using K = cardinal::game::PropertyKind;
    auto parse_vec3 = [&](float& x, float& y, float& z) {
        return cardinal::sscanf(value.c_str(), " (%f , %f , %f", &x, &y, &z) == 3;
    };
    if (kind == "float" && p.kind == K::Float) {
        *static_cast<float*>(p.ptr) = static_cast<float>(cardinal::atof(value.c_str()));
        return true;
    }
    if (kind == "int" && p.kind == K::Int) {
        *static_cast<int*>(p.ptr) = cardinal::atoi(value.c_str());
        return true;
    }
    if (kind == "bool" && p.kind == K::Bool) {
        *static_cast<bool*>(p.ptr) = (value == "true" || value == "1");
        return true;
    }
    if (kind == "vec3" && p.kind == K::Vec3) {
        float x, y, z;
        if (parse_vec3(x, y, z)) {
            float* v = static_cast<float*>(p.ptr);
            v[0] = x; v[1] = y; v[2] = z;
            return true;
        }
    }
    if (kind == "color" && p.kind == K::Color) {
        float x, y, z;
        if (parse_vec3(x, y, z)) {
            float* v = static_cast<float*>(p.ptr);
            v[0] = x; v[1] = y; v[2] = z;
            return true;
        }
    }
    if (kind == "string" && p.kind == K::String) {
        // emit_prop writes string props quoted; strip a single matched
        // pair (same rule as text::parse_kv) so values round-trip
        // exactly — otherwise every save/load cycle accreted another
        // layer of quotes, silently corrupting the value.
        cardinal::string v = value;
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
            v = v.substr(1, v.size() - 2);
        *static_cast<cardinal::string*>(p.ptr) = cardinal::move(v);
        return true;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// World save
// ---------------------------------------------------------------------------
cardinal::string serialize_world(const cardinal::actor::World& world,
                                 SaveStats* stats_out)
{
    SaveStats s{};
    cardinal::string out;
    out.reserve(8192);
    out += "# Cardinal save v1\n";

    for (const auto& aptr : world.actors()) {
        if (!aptr->alive()) continue;
        out += "actor \"";
        out += aptr->name();
        out += "\" {\n";

        // Class — empty for non-GameActor instances.
        cardinal::string cls;
        cardinal::game::GameActor* ga = nullptr;
        for (const auto& c : aptr->components()) {
            if (auto* g = dynamic_cast<cardinal::game::GameActor*>(c.get())) {
                ga  = g; cls = g->class_name(); break;
            }
        }
        text::emit_kv(out, "class", cls.c_str());

        // Enabled — emitted AFTER class (load re-creates classed actors on
        // the `class =` line, so an earlier flag would be lost) and ONLY
        // when disabled (absent = enabled, so old saves load active + new
        // saves stay compact).
        if (!aptr->enabled()) {
            text::emit_kv(out, "enabled", "false");
        }

        // Transform.
        if (auto* tr = aptr->get_component<cardinal::actor::TransformComponent>()) {
            text::emit_kv3(out, "position",
                tr->translation.x, tr->translation.y, tr->translation.z);
            text::emit_kv3(out, "rotation",
                tr->rotation_euler.x, tr->rotation_euler.y, tr->rotation_euler.z);
            text::emit_kv3(out, "scale",
                tr->scale.x, tr->scale.y, tr->scale.z);
        }

        // Reflected props.
        if (ga) {
            const auto* def = cardinal::game::ClassRegistry::instance().find(ga->class_name());
            if (def && def->describe_properties) {
                auto props = def->describe_properties(ga);
                for (auto& p : props) {
                    emit_prop(out, p);
                    ++s.properties_written;
                }
            }
        }

        // Full component set — every component EXCEPT Transform (already
        // emitted as position/rotation/scale) and GameActor (emitted as
        // class + reflected props). This is what makes a world save a
        // complete LEVEL save: Mesh / Light / RigidBody / Tag / Camera /
        // AudioEmitter / Script / PlayerController / PrefabLink all
        // round-trip via Component::serialize_fields. Older saves without
        // these blocks still load (the loader only adds what it sees).
        for (const auto& c : aptr->components()) {
            const cardinal::string ctype = c->type_name();
            if (ctype == "Transform" || ctype == "GameActor") continue;
            out += "  component \""; out += ctype; out += "\" {\n";
            c->serialize_fields(out);
            out += "  }\n";
        }

        out += "}\n\n";
        ++s.actors_written;
    }

    if (stats_out) *stats_out = s;
    return out;
}

SaveStats save_world(const cardinal::actor::World& world,
                     const cardinal::string& path,
                     cardinal::string* error_out)
{
    SaveStats s{};
    const cardinal::string out = serialize_world(world, &s);

    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) {
        if (error_out) *error_out = "could not open file for write: " + path;
        return s;
    }
    f.write(out.data(), static_cast<cardinal::streamsize>(out.size()));
    s.bytes_written = static_cast<u64>(out.size());
    cardinal::log::infof("serial",
        "saved world: %u actors, %u props -> %s (%llu bytes)",
        s.actors_written, s.properties_written, path.c_str(),
        static_cast<unsigned long long>(s.bytes_written));
    return s;
}

// ---------------------------------------------------------------------------
// World load — a single stream parser drives both the file path
// (load_world) and the in-memory string path (deserialize_world).
// ---------------------------------------------------------------------------
template <class Stream>
static LoadStats parse_world_stream_(cardinal::game::Game& game,
                                     Stream& f,
                                     bool replace_existing,
                                     const char* src_label)
{
    LoadStats st{};

    if (replace_existing) {
        cardinal::vector<u32> ids;
        for (const auto& aptr : game.world().actors())
            if (aptr->alive()) ids.push_back(aptr->id());
        for (auto id : ids) game.world().destroy(id);
        game.world().sweep();
    }

    cardinal::string line;
    cardinal::actor::Actor*           cur_actor = nullptr;
    cardinal::game::GameActor*         cur_ga    = nullptr;
    cardinal::string                       cur_actor_name;
    cardinal::vector<cardinal::game::PropertyDef> cur_props;
    // Full-component-set extension: when inside a `component "Type" {` block,
    // field lines route to this component's deserialize_field instead of the
    // actor-level handlers. `in_comp` is the in-block sentinel (set whenever a
    // `component "..." {` is consumed, regardless of whether a component was
    // built); `cur_serialized_comp` is the target (null for a skipped /
    // unknown / GameActor / Transform-less block). Closing on the POINTER
    // alone would mis-route a skipped block's `}` to close the ACTOR and
    // silently truncate the rest of it — so close on the sentinel.
    cardinal::actor::Component*        cur_serialized_comp = nullptr;
    bool                              in_comp = false;
    auto refresh_props = [&]() {
        cur_props.clear();
        if (cur_ga == nullptr) return;
        const auto* def = cardinal::game::ClassRegistry::instance().find(cur_ga->class_name());
        if (def && def->describe_properties) cur_props = def->describe_properties(cur_ga);
    };

    while (cardinal::getline(f, line)) {
        // Strip leading whitespace.
        usize i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= line.size() || line[i] == '#') continue;

        // Block open.
        if (line.compare(i, 6, "actor ") == 0) {
            const auto q1 = line.find('"', i);
            const auto q2 = (q1 == cardinal::string::npos) ? cardinal::string::npos : line.find('"', q1 + 1);
            cardinal::string name = (q1 != cardinal::string::npos && q2 != cardinal::string::npos)
                ? line.substr(q1 + 1, q2 - q1 - 1) : "Actor";
            // We can't spawn yet — class isn't known. Defer until "class = ...".
            // Keep the parsed name so the "class =" branch can pass it to
            // spawn_class — otherwise classed actors lose their saved name
            // and come back named after their class.
            cur_actor      = game.world().spawn(name);
            cur_actor_name = name;
            cur_ga         = nullptr;
            cur_serialized_comp = nullptr;
            in_comp        = false;
            cur_props.clear();
            continue;
        }
        if (line[i] == '}') {
            // A `}` closes the innermost open block: a component sub-block
            // first (tracked by `in_comp`, true even for a skipped block),
            // otherwise the actor block.
            if (in_comp) {
                in_comp = false;
                cur_serialized_comp = nullptr;
            } else {
                cur_actor = nullptr; cur_ga = nullptr; cur_props.clear();
                cur_actor_name.clear();
            }
            continue;
        }

        if (cur_actor == nullptr) continue;

        // component "Type" { — opens a full-component-set sub-block. Build
        // the component via the factory + adopt it onto the actor; field
        // lines until the matching `}` go through deserialize_field.
        if (line.compare(i, 10, "component ") == 0) {
            const auto q1 = line.find('"', i);
            const auto q2 = (q1 == cardinal::string::npos)
                ? cardinal::string::npos : line.find('"', q1 + 1);
            cardinal::string ctype = (q1 != cardinal::string::npos && q2 != cardinal::string::npos)
                ? line.substr(q1 + 1, q2 - q1 - 1) : cardinal::string{};
            in_comp = true;   // sentinel: we are inside a component block now
            // Transform / GameActor are emitted via the legacy lines, never
            // as component blocks — but be defensive: route a Transform
            // block to the existing component, skip a GameActor block.
            if (ctype == "Transform") {
                cur_serialized_comp = cur_actor->get_component<cardinal::actor::TransformComponent>();
            } else if (ctype == "GameActor" || ctype.empty()) {
                cur_serialized_comp = nullptr;   // unknown / handled elsewhere
            } else {
                auto made = cardinal::actor::make_component_by_name(ctype);
                if (made) {
                    cur_serialized_comp = cur_actor->adopt_component(cardinal::move(made));
                } else {
                    cur_serialized_comp = nullptr;   // unknown type: skip its fields
                    ++st.errors;
                    st.warnings.push_back("unknown component type: " + ctype);
                }
            }
            continue;
        }

        // Inside a component block: route "key = value" to the component.
        // A skipped/unknown block (in_comp but no target) drops its field
        // lines here instead of letting them fall through to the
        // actor-level class/enabled/position/prop handlers.
        if (in_comp) {
            if (cur_serialized_comp != nullptr) {
                cardinal::string key, val;
                if (text::parse_kv(line.substr(i), &key, &val)) {
                    cur_serialized_comp->deserialize_field(key, val);
                }
            }
            continue;
        }

        // Property lines start with "prop <kind> <name> = <value>".
        if (line.compare(i, 5, "prop ") == 0) {
            cardinal::string rest = line.substr(i + 5);
            // Format: "<kind> <name> = <value>"
            const auto sp1 = rest.find(' ');
            const auto eq  = rest.find('=', sp1);
            if (sp1 == cardinal::string::npos || eq == cardinal::string::npos) {
                ++st.errors;
                continue;
            }
            cardinal::string kind = rest.substr(0, sp1);
            cardinal::string name = rest.substr(sp1 + 1, eq - sp1 - 1);
            cardinal::string val  = rest.substr(eq + 1);
            // trim
            auto trim = [](cardinal::string s) {
                while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
                while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' || s.back() == '\r')) s.pop_back();
                return s;
            };
            name = trim(name); val = trim(val);
            bool applied = false;
            for (auto& p : cur_props) {
                if (p.name == name) {
                    if (apply_prop(p, kind, val)) {
                        ++st.properties_applied;
                        applied = true;
                    }
                    break;
                }
            }
            if (!applied) ++st.errors;
            continue;
        }

        // Key=value lines.
        cardinal::string key, val;
        if (!text::parse_kv(line.substr(i), &key, &val)) continue;

        if (key == "class") {
            // Re-create the actor as the right class. Drop the placeholder
            // (only TransformComponent so far) by destroy+respawn via game.
            if (!val.empty()) {
                const u32 old_id = cur_actor->id();
                game.world().destroy(old_id);
                game.world().sweep();
                cur_actor = game.spawn_class(val, cur_actor_name);
                if (cur_actor == nullptr) {
                    ++st.actors_skipped;
                    cur_ga = nullptr;
                    continue;
                }
                for (const auto& c : cur_actor->components()) {
                    if (auto* g = dynamic_cast<cardinal::game::GameActor*>(c.get())) {
                        cur_ga = g;
                        break;
                    }
                }
                refresh_props();
                ++st.actors_spawned;
            } else {
                ++st.actors_spawned;   // anonymous transform-only actor counts too
            }
            continue;
        }

        if (key == "enabled") {
            cur_actor->set_enabled(!(val == "false" || val == "0"));
            continue;
        }

        if (key == "position" || key == "rotation" || key == "scale") {
            float x = 0, y = 0, z = 0;
            cardinal::sscanf(val.c_str(), " (%f , %f , %f", &x, &y, &z);
            if (auto* tr = cur_actor->get_component<cardinal::actor::TransformComponent>()) {
                if (key == "position")   tr->translation    = {x, y, z};
                if (key == "rotation")   tr->rotation_euler = {x, y, z};
                if (key == "scale")      tr->scale          = {x, y, z};
            }
        }
    }

    cardinal::log::infof("serial",
        "loaded world: %u spawned, %u props applied, %u errors <- %s",
        st.actors_spawned, st.properties_applied, st.errors, src_label);
    return st;
}

LoadStats load_world(cardinal::game::Game& game,
                     const cardinal::string& path,
                     bool replace_existing,
                     cardinal::string* error_out)
{
    cardinal::ifstream f(path);
    if (!f) {
        if (error_out) *error_out = "could not open file: " + path;
        return {};
    }
    return parse_world_stream_(game, f, replace_existing, path.c_str());
}

LoadStats deserialize_world(cardinal::game::Game& game,
                            const cardinal::string& text,
                            bool replace_existing)
{
    cardinal::istringstream ss(text);
    return parse_world_stream_(game, ss, replace_existing, "<memory>");
}

// ---------------------------------------------------------------------------
// Prefab library save / load
// ---------------------------------------------------------------------------
// Emit one component block per component on `proto` into `out`, at the given
// indent. Shared by single-prefab + group-member serialization.
static u32 emit_proto_components_(cardinal::string& out,
                                  const cardinal::actor::Actor& proto,
                                  const char* indent) {
    u32 n = 0;
    for (const auto& c : proto.components()) {
        const cardinal::string type = c->type_name();
        out += indent; out += "component \""; out += type; out += "\" {\n";
        if (type == "GameActor") {
            const auto* ga = static_cast<const cardinal::game::GameActor*>(c.get());
            out += indent; out += "  class = "; out += ga->class_name(); out += "\n";
            const auto* def =
                cardinal::game::ClassRegistry::instance().find(ga->class_name());
            if (def && def->describe_properties) {
                auto* mut = const_cast<cardinal::game::GameActor*>(ga);
                for (auto& p : def->describe_properties(mut)) emit_prop(out, p);
            }
        } else {
            c->serialize_fields(out);   // "  key = val" lines
        }
        out += indent; out += "}\n";
        ++n;
    }
    return n;
}

PrefabSaveStats save_prefabs(const cardinal::actor::World& world,
                             const cardinal::string& path,
                             cardinal::string* error_out)
{
    PrefabSaveStats s{};
    cardinal::string out;
    out.reserve(8192);
    out += "# Cardinal prefab library v1\n\n";

    for (const auto& name : world.prefab_names()) {
        const cardinal::actor::Actor* proto = world.prefab_prototype(name);
        if (proto == nullptr) continue;
        out += "prefab \""; out += name; out += "\" {\n";
        s.components_written += emit_proto_components_(out, *proto, "  ");
        out += "}\n\n";
        ++s.prefabs_written;
    }

    // Group prefabs — each a `group` block of `member "Name" rel x y z { ... }`
    // sub-blocks (each member is a prototype, same component format).
    for (const auto& gname : world.group_prefab_names()) {
        out += "group \""; out += gname; out += "\" {\n";
        const u32 mc = world.group_prefab_member_count(gname);
        for (u32 mi = 0; mi < mc; ++mi) {
            const cardinal::actor::Actor* mp = world.group_member_proto(gname, mi);
            if (mp == nullptr) continue;
            const cardinal::scene::Vec3 rel = world.group_member_rel(gname, mi);
            char hdr[160];
            cardinal::snprintf(hdr, sizeof(hdr),
                "  member \"%s\" rel %.6g %.6g %.6g {\n",
                mp->name().c_str(), static_cast<double>(rel.x),
                static_cast<double>(rel.y), static_cast<double>(rel.z));
            out += hdr;
            s.components_written += emit_proto_components_(out, *mp, "    ");
            out += "  }\n";
        }
        out += "}\n\n";
        ++s.prefabs_written;   // groups count toward the saved-template tally
    }

    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) {
        if (error_out) *error_out = "could not open file for write: " + path;
        return s;
    }
    f.write(out.data(), static_cast<cardinal::streamsize>(out.size()));
    s.bytes_written = static_cast<u64>(out.size());
    cardinal::log::infof("serial",
        "saved prefab library: %u prefabs, %u components -> %s (%llu bytes)",
        s.prefabs_written, s.components_written, path.c_str(),
        static_cast<unsigned long long>(s.bytes_written));
    return s;
}

PrefabLoadStats load_prefabs(cardinal::actor::World& world,
                             const cardinal::string& path,
                             bool replace_existing,
                             cardinal::string* error_out)
{
    PrefabLoadStats st{};
    cardinal::ifstream f(path);
    if (!f) {
        if (error_out) *error_out = "could not open file: " + path;
        return st;
    }

    // Parser state: a prefab block holds component sub-blocks; a component
    // sub-block holds field lines. Brace depth disambiguates the close.
    cardinal::string line;
    cardinal::string cur_prefab_name;
    cardinal::unique_ptr<cardinal::actor::Actor> proto;   // building prototype
    cardinal::string cur_comp_type;
    cardinal::actor::Component*  cur_comp = nullptr;       // built-in being filled
    cardinal::game::GameActor*   cur_ga   = nullptr;       // game-class being filled
    cardinal::vector<cardinal::game::PropertyDef> cur_props;
    // Group-prefab state: `proto` doubles as the member-being-built; in_member
    // routes its close `}` to add_group_member, in_group tracks the enclosing
    // group block so a group's own close doesn't get mistaken for a prefab.
    cardinal::string         cur_group_name;
    bool                     in_group  = false;
    bool                     in_member = false;
    cardinal::scene::Vec3    cur_member_rel{0, 0, 0};

    auto finish_component = [&]() {
        cur_comp = nullptr; cur_ga = nullptr; cur_comp_type.clear();
        cur_props.clear();
    };
    auto finish_prefab = [&]() {
        if (proto && !cur_prefab_name.empty()) {
            world.add_prefab(cur_prefab_name, cardinal::move(proto));
            ++st.prefabs_loaded;
        }
        proto.reset();
        cur_prefab_name.clear();
    };
    auto finish_member = [&]() {
        if (proto && !cur_group_name.empty()) {
            world.add_group_member(cur_group_name, cardinal::move(proto), cur_member_rel);
        }
        proto.reset();
        in_member = false;
    };

    auto strip = [](const cardinal::string& l, usize& i) {
        while (i < l.size() && (l[i] == ' ' || l[i] == '\t')) ++i;
    };
    auto extract_quoted = [](const cardinal::string& l, usize from) -> cardinal::string {
        const auto q1 = l.find('"', from);
        if (q1 == cardinal::string::npos) return {};
        const auto q2 = l.find('"', q1 + 1);
        if (q2 == cardinal::string::npos) return {};
        return l.substr(q1 + 1, q2 - q1 - 1);
    };

    while (cardinal::getline(f, line)) {
        usize i = 0; strip(line, i);
        if (i >= line.size() || line[i] == '#') continue;

        // Close brace — closes whichever level is innermost: component,
        // then member, then group, then single prefab.
        if (line[i] == '}') {
            if (cur_comp != nullptr || cur_ga != nullptr || !cur_comp_type.empty()) {
                finish_component();
            } else if (in_member) {
                finish_member();
            } else if (in_group) {
                in_group = false;
                cur_group_name.clear();
            } else {
                finish_prefab();
            }
            continue;
        }

        // prefab "Name" {
        if (line.compare(i, 7, "prefab ") == 0) {
            finish_prefab();   // safety: close any unterminated prior block
            cur_prefab_name = extract_quoted(line, i);
            proto = cardinal::make_unique<cardinal::actor::Actor>(0u, cur_prefab_name);
            if (replace_existing && world.has_prefab(cur_prefab_name)) {
                world.remove_prefab(cur_prefab_name);
            }
            continue;
        }

        // group "Name" {  — opens a multi-actor group block.
        if (line.compare(i, 6, "group ") == 0) {
            finish_prefab();
            cur_group_name = extract_quoted(line, i);
            in_group = true;
            if (replace_existing && world.has_group_prefab(cur_group_name)) {
                world.remove_group_prefab(cur_group_name);
            }
            continue;
        }

        // member "Name" rel x y z {  — one actor inside the current group.
        if (in_group && line.compare(i, 7, "member ") == 0) {
            finish_member();   // safety: close any unterminated prior member
            const cardinal::string mname = extract_quoted(line, i);
            proto = cardinal::make_unique<cardinal::actor::Actor>(0u, mname);
            in_member = true;
            // Parse "rel x y z" after the closing quote.
            cur_member_rel = {0, 0, 0};
            const auto rel = line.find("rel", i);
            if (rel != cardinal::string::npos) {
                float x = 0, y = 0, z = 0;
                cardinal::sscanf(line.c_str() + rel + 3, " %f %f %f", &x, &y, &z);
                cur_member_rel = { x, y, z };
            }
            continue;
        }

        // component "Type" {
        if (line.compare(i, 10, "component ") == 0) {
            finish_component();
            cur_comp_type = extract_quoted(line, i);
            if (!proto) continue;
            if (cur_comp_type == "GameActor") {
                // Defer construction until we see "class = ..." (need the
                // registry name to pick the subclass).
                cur_comp = nullptr; cur_ga = nullptr;
            } else {
                auto made = cardinal::actor::make_component_by_name(cur_comp_type);
                if (made) {
                    cur_comp = proto->adopt_component(cardinal::move(made));
                    ++st.components_loaded;
                } else {
                    ++st.components_skipped;
                    st.warnings.push_back("unknown component type: " + cur_comp_type);
                }
            }
            continue;
        }

        // Field line within a component block: "key = value".
        // GameActor "class = X" creates the instance; "prop <kind> name = v"
        // applies a reflected property; built-ins use deserialize_field.
        if (cur_comp_type == "GameActor") {
            // class = ClassName
            if (line.compare(i, 5, "class") == 0) {
                cardinal::string key, val;
                if (text::parse_kv(line.substr(i), &key, &val) && key == "class") {
                    const auto* def =
                        cardinal::game::ClassRegistry::instance().find(val);
                    if (def && def->create && proto) {
                        auto ga = def->create();
                        ga->set_class_name(val);
                        cur_ga = ga.get();
                        proto->adopt_component(cardinal::move(ga));
                        ++st.components_loaded;
                        cur_props.clear();
                        if (def->describe_properties)
                            cur_props = def->describe_properties(cur_ga);
                    } else {
                        ++st.components_skipped;
                        st.warnings.push_back("unregistered game class: " + val);
                    }
                }
                continue;
            }
            // prop <kind> <name> = <value>
            if (line.compare(i, 5, "prop ") == 0 && cur_ga != nullptr) {
                // tokenise: prop <kind> <name> = <value>
                cardinal::string rest = line.substr(i + 5);
                // kind
                usize k0 = 0; while (k0 < rest.size() && rest[k0] == ' ') ++k0;
                usize k1 = rest.find(' ', k0);
                if (k1 == cardinal::string::npos) continue;
                cardinal::string kind = rest.substr(k0, k1 - k0);
                // name
                usize n0 = k1; while (n0 < rest.size() && rest[n0] == ' ') ++n0;
                usize eq = rest.find('=', n0);
                if (eq == cardinal::string::npos) continue;
                usize n1 = eq; while (n1 > n0 && rest[n1 - 1] == ' ') --n1;
                cardinal::string pname = rest.substr(n0, n1 - n0);
                cardinal::string val   = rest.substr(eq + 1);
                usize v0 = 0; while (v0 < val.size() && val[v0] == ' ') ++v0;
                val = val.substr(v0);
                // Strip trailing CR/whitespace (mirrors the world loader) so
                // a CRLF-terminated value doesn't leave apply_prop's quote
                // strip seeing '\r' as the last char — which would store a
                // string prop with literal quotes + a carriage return.
                while (!val.empty() &&
                       (val.back() == '\r' || val.back() == ' ' || val.back() == '\t'))
                    val.pop_back();
                for (auto& p : cur_props) {
                    if (p.name == pname) { apply_prop(p, kind, val); break; }
                }
                continue;
            }
            continue;
        }

        // Built-in component field.
        if (cur_comp != nullptr) {
            cardinal::string key, val;
            if (text::parse_kv(line.substr(i), &key, &val)) {
                cur_comp->deserialize_field(key, val);
            }
        }
    }
    // Close any trailing block at EOF.
    if (in_member) finish_member();
    finish_prefab();

    cardinal::log::infof("serial",
        "loaded prefab library: %u prefabs, %u components (%u skipped) <- %s",
        st.prefabs_loaded, st.components_loaded, st.components_skipped,
        path.c_str());
    return st;
}

// ---------------------------------------------------------------------------
// Sky save / load
// ---------------------------------------------------------------------------
bool save_sky(const cardinal::sky::Sky& sky, const cardinal::string& path,
              cardinal::string* error_out)
{
    cardinal::string out;
    out += "# Cardinal sky v1\n";
    out += "sky {\n";
    text::emit_kf(out, "hour",        sky.hour());
    text::emit_kf(out, "time_scale",  sky.time_scale());
    text::emit_kv(out, "frozen",      sky.frozen() ? "true" : "false");
    out += "  keys {\n";
    for (const auto& k : sky.keys()) {
        char buf[256];
        cardinal::snprintf(buf, sizeof(buf),
            "    key h=%.4f ze=(%.4f,%.4f,%.4f) ho=(%.4f,%.4f,%.4f) "
            "sc=(%.4f,%.4f,%.4f) si=%.4f am=(%.4f,%.4f,%.4f)\n",
            k.hour,
            k.zenith.x, k.zenith.y, k.zenith.z,
            k.horizon.x, k.horizon.y, k.horizon.z,
            k.sun_color.x, k.sun_color.y, k.sun_color.z,
            k.sun_intensity,
            k.ambient.x, k.ambient.y, k.ambient.z);
        out += buf;
    }
    out += "  }\n}\n";
    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) {
        if (error_out) *error_out = "could not open: " + path;
        return false;
    }
    f.write(out.data(), static_cast<cardinal::streamsize>(out.size()));
    return true;
}

bool load_sky(cardinal::sky::Sky& sky, const cardinal::string& path,
              cardinal::string* error_out)
{
    cardinal::ifstream f(path);
    if (!f) {
        if (error_out) *error_out = "could not open: " + path;
        return false;
    }
    sky.keys().clear();
    cardinal::string line;
    while (cardinal::getline(f, line)) {
        usize i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= line.size() || line[i] == '#') continue;

        if (line.compare(i, 4, "key ") == 0) {
            cardinal::sky::SkyKey k{};
            cardinal::sscanf(line.c_str() + i,
                "key h=%f ze=(%f,%f,%f) ho=(%f,%f,%f) sc=(%f,%f,%f) si=%f am=(%f,%f,%f)",
                &k.hour,
                &k.zenith.x, &k.zenith.y, &k.zenith.z,
                &k.horizon.x, &k.horizon.y, &k.horizon.z,
                &k.sun_color.x, &k.sun_color.y, &k.sun_color.z,
                &k.sun_intensity,
                &k.ambient.x, &k.ambient.y, &k.ambient.z);
            sky.keys().push_back(k);
            continue;
        }

        cardinal::string key, val;
        if (!text::parse_kv(line.substr(i), &key, &val)) continue;
        if (key == "hour")        sky.set_hour(static_cast<float>(cardinal::atof(val.c_str())));
        if (key == "time_scale")  sky.set_time_scale(static_cast<float>(cardinal::atof(val.c_str())));
        if (key == "frozen")      sky.set_frozen(val == "true" || val == "1");
    }
    sky.sort_keys();
    return true;
}

// =============================================================================
// Scene save / load — round-trips scene::Entity[] (rendering primitives).
// Mesh refs persist as Mesh::name(); host provides a registry on load.
// Format mirrors the world saver — one block per entity.
//
//   # Cardinal scene save v1
//   entity "Cube" {
//     mesh     = "primitives/box"
//     position = (0.000000, 1.500000, 0.000000)
//     rotation = (0.000000, 0.000000, 0.000000)
//     scale    = (1.000000, 1.000000, 1.000000)
//     tint     = (1.000000, 1.000000, 1.000000)
//     visible  = true
//   }
// =============================================================================
SceneSaveStats save_scene(const cardinal::scene::Scene& scene,
                          const cardinal::string& path,
                          cardinal::string* error_out)
{
    SceneSaveStats stats{};
    cardinal::string out;
    out.reserve(scene.entities().size() * 256);
    out += "# Cardinal scene save v1\n";

    for (const auto& e : scene.entities()) {
        if (!e.mesh) {
            ++stats.entities_skipped_meshless;
            continue;
        }
        if (e.mesh->name().empty()) {
            ++stats.entities_skipped_unnamed;
            continue;
        }

        out += "entity \"";
        out += e.name;
        out += "\" {\n";
        text::emit_kv (out, "mesh",     e.mesh->name().c_str());
        text::emit_kv3(out, "position",
            e.transform.translation.x, e.transform.translation.y, e.transform.translation.z);
        text::emit_kv3(out, "rotation",
            e.transform.rotation_euler.x, e.transform.rotation_euler.y, e.transform.rotation_euler.z);
        text::emit_kv3(out, "scale",
            e.transform.scale.x, e.transform.scale.y, e.transform.scale.z);
        text::emit_kv3(out, "tint", e.tint.x, e.tint.y, e.tint.z);
        text::emit_kv (out, "visible", e.visible ? "true" : "false");
        // Per-entity PBR material (added with the storage-buffer
        // material arc). Four scalars; keys are optional on load so
        // pre-material save files still parse (loader keeps the
        // default-constructed Material).
        text::emit_kf(out, "mat_spec_intensity", e.material.specular_intensity);
        text::emit_kf(out, "mat_spec_power",     e.material.specular_power);
        text::emit_kf(out, "mat_roughness",      e.material.roughness);
        text::emit_kf(out, "mat_metalness",      e.material.metalness);
        // vgeom attachment state — both keys optional, only emit if the
        // mesh has an attached hierarchy. Older save files (pre-v2)
        // simply omit them and the loader leaves vgeom alone.
        if (cardinal::scene::vgeom_attached(*e.mesh)) {
            text::emit_kv(out, "vgeom_attached", "true");
            text::emit_kv(out, "vgeom_enabled",
                cardinal::scene::vgeom_enabled(*e.mesh) ? "true" : "false");
        }
        out += "}\n";
        ++stats.entities_written;
    }

    cardinal::ofstream f(path, cardinal::ios::binary | cardinal::ios::trunc);
    if (!f) {
        if (error_out) *error_out = "could not open: " + path;
        return stats;
    }
    f.write(out.data(), static_cast<cardinal::streamsize>(out.size()));
    stats.bytes_written = out.size();
    return stats;
}

SceneLoadStats load_scene(cardinal::scene::Scene& scene,
                          const MeshRegistry& mesh_registry,
                          const cardinal::string& path,
                          bool replace_existing,
                          cardinal::string* error_out)
{
    SceneLoadStats stats{};
    cardinal::ifstream f(path);
    if (!f) {
        if (error_out) *error_out = "could not open: " + path;
        ++stats.errors;
        return stats;
    }

    if (replace_existing) {
        // Snapshot ids first so we don't mutate the vector mid-walk.
        cardinal::vector<u32> ids;
        ids.reserve(scene.entities().size());
        for (const auto& e : scene.entities()) ids.push_back(e.id);
        for (u32 id : ids) scene.remove_entity(id);
    }

    auto parse_vec3 = [](const cardinal::string& v, float& x, float& y, float& z) {
        return cardinal::sscanf(v.c_str(), " (%f , %f , %f", &x, &y, &z) == 3;
    };

    cardinal::string line;
    cardinal::scene::Entity* current = nullptr;
    cardinal::string current_mesh_name;
    bool current_vgeom_attached = false;
    bool current_vgeom_enabled  = true;
    while (cardinal::getline(f, line)) {
        usize i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= line.size() || line[i] == '#') continue;

        if (line.compare(i, 7, "entity ") == 0) {
            // entity "Name" {
            const auto q1 = line.find('"', i);
            const auto q2 = (q1 != cardinal::string::npos) ? line.find('"', q1 + 1) : cardinal::string::npos;
            if (q1 == cardinal::string::npos || q2 == cardinal::string::npos) {
                ++stats.errors;
                continue;
            }
            const cardinal::string name = line.substr(q1 + 1, q2 - q1 - 1);
            current = &scene.add_entity(name);
            current_mesh_name.clear();
            current_vgeom_attached = false;
            current_vgeom_enabled  = true;
            continue;
        }
        if (line[i] == '}') {
            if (current != nullptr) {
                if (current_mesh_name.empty()) {
                    // No mesh declared — leave entity meshless (still spawned).
                    ++stats.entities_spawned;
                } else {
                    auto it = mesh_registry.find(current_mesh_name);
                    if (it != mesh_registry.end()) {
                        current->mesh = it->second;
                        // Restore vgeom attachment state if requested. Cook
                        // is no-op when the mesh already has a hierarchy
                        // (idempotent), so reloading a scene multiple times
                        // doesn't re-cook.
                        if (current_vgeom_attached &&
                            !cardinal::scene::vgeom_attached(*it->second))
                        {
                            cardinal::scene::vgeom_attach(*it->second);
                        }
                        if (current_vgeom_attached) {
                            cardinal::scene::vgeom_set_enabled(
                                *it->second, current_vgeom_enabled);
                        }
                        ++stats.entities_spawned;
                    } else {
                        // Mesh unknown — drop the entity, log a warning.
                        scene.remove_entity(current->id);
                        ++stats.entities_skipped_unknown_mesh;
                        stats.warnings.push_back(
                            "unknown mesh '" + current_mesh_name + "' — entity dropped");
                    }
                }
            }
            current = nullptr;
            continue;
        }
        if (current == nullptr) continue;

        cardinal::string key, val;
        if (!text::parse_kv(line.substr(i), &key, &val)) continue;

        if (key == "mesh") {
            // val is quoted "name"
            if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
                current_mesh_name = val.substr(1, val.size() - 2);
            else
                current_mesh_name = val;
        } else if (key == "position") {
            float x = 0, y = 0, z = 0;
            if (parse_vec3(val, x, y, z)) {
                current->transform.translation = cardinal::scene::Vec3{x, y, z};
            }
        } else if (key == "rotation") {
            float x = 0, y = 0, z = 0;
            if (parse_vec3(val, x, y, z)) {
                current->transform.rotation_euler = cardinal::scene::Vec3{x, y, z};
            }
        } else if (key == "scale") {
            float x = 0, y = 0, z = 0;
            if (parse_vec3(val, x, y, z)) {
                current->transform.scale = cardinal::scene::Vec3{x, y, z};
            }
        } else if (key == "tint") {
            float r = 1, g = 1, b = 1;
            if (parse_vec3(val, r, g, b)) {
                current->tint = cardinal::scene::Vec3{r, g, b};
            }
        } else if (key == "visible") {
            current->visible = (val == "true" || val == "1");
        } else if (key == "mat_spec_intensity") {
            float mf = 0; if (cardinal::sscanf(val.c_str(), " %f", &mf) == 1)
                current->material.specular_intensity = mf;
        } else if (key == "mat_spec_power") {
            float mf = 0; if (cardinal::sscanf(val.c_str(), " %f", &mf) == 1)
                current->material.specular_power = mf;
        } else if (key == "mat_roughness") {
            float mf = 0; if (cardinal::sscanf(val.c_str(), " %f", &mf) == 1)
                current->material.roughness = mf;
        } else if (key == "mat_metalness") {
            float mf = 0; if (cardinal::sscanf(val.c_str(), " %f", &mf) == 1)
                current->material.metalness = mf;
        } else if (key == "vgeom_attached") {
            current_vgeom_attached = (val == "true" || val == "1");
        } else if (key == "vgeom_enabled") {
            current_vgeom_enabled = (val == "true" || val == "1");
        }
    }
    return stats;
}

}  // namespace cardinal::serial
