// =============================================================================
// Cardinal — XSI-style Render Passes & Partitions implementation.
// =============================================================================
#include <cardinal/render/render_passes.hpp>

#include <cardinal/core/std/cstdio.hpp>    // cardinal::snprintf
#include <cardinal/core/std/cstdlib.hpp>   // cardinal::strtod
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::render::rp {

// -----------------------------------------------------------------------------
// PassDef
// -----------------------------------------------------------------------------
Partition& PassDef::assign(u32 entity, const cardinal::string& pname) {
    unassign(entity);   // disjoint within the pass
    Partition* p = find_partition(pname);
    if (p == nullptr) {
        Partition np;
        np.name = pname;
        partitions.push_back(cardinal::move(np));
        p = &partitions.back();
    }
    p->entities.push_back(entity);
    return *p;
}

void PassDef::unassign(u32 entity) {
    for (auto& p : partitions) {
        for (usize i = 0; i < p.entities.size(); ++i) {
            if (p.entities[i] == entity) {
                p.entities.erase(p.entities.begin() + static_cast<isize>(i));
                return;   // disjoint invariant: it can appear at most once
            }
        }
    }
}

// -----------------------------------------------------------------------------
// PassSet
// -----------------------------------------------------------------------------
PassDef& PassSet::add_pass(cardinal::string name, scene::ViewMode vm) {
    PassDef d;
    d.name      = cardinal::move(name);
    d.view_mode = vm;
    passes_.push_back(cardinal::move(d));
    if (current_ < 0) current_ = 0;
    return passes_.back();
}

bool PassSet::remove_pass(const cardinal::string& name) {
    for (usize i = 0; i < passes_.size(); ++i) {
        if (passes_[i].name != name) continue;
        passes_.erase(passes_.begin() + static_cast<isize>(i));
        if (current_ >= static_cast<int>(passes_.size()))
            current_ = static_cast<int>(passes_.size()) - 1;
        return true;
    }
    return false;
}

PassDef* PassSet::find(const cardinal::string& name) {
    for (auto& p : passes_) if (p.name == name) return &p;
    return nullptr;
}

Applied PassSet::apply(scene::Scene& s) {
    PassDef* cur = current();
    if (cur == nullptr) return Applied{};
    return apply(s, *cur);
}

Applied PassSet::apply(scene::Scene& s, const PassDef& pass) const {
    Applied a;
    if (!pass.enabled) return a;
    for (const auto& part : pass.partitions) {
        const bool touches = part.ovr.override_visibility || part.ovr.override_tint;
        if (!touches) continue;
        for (u32 id : part.entities) {
            scene::Entity* e = s.find_by_id(id);
            if (e == nullptr) continue;   // stale id — partition edits survive deletes
            Applied::Saved sv;
            sv.id      = id;
            sv.visible = e->visible;
            sv.tint    = e->tint;
            a.saved.push_back(sv);
            if (part.ovr.override_visibility) e->visible = part.ovr.visible;
            if (part.ovr.override_tint)       e->tint    = part.ovr.tint;
        }
    }
    a.active = true;
    return a;
}

void PassSet::restore(scene::Scene& s, const Applied& a) const {
    if (!a.active) return;
    // Reverse order so an entity saved twice (can't happen within one pass —
    // partitions are disjoint — but cheap insurance) restores its FIRST state.
    for (usize i = a.saved.size(); i > 0; --i) {
        const Applied::Saved& sv = a.saved[i - 1];
        if (scene::Entity* e = s.find_by_id(sv.id)) {
            e->visible = sv.visible;
            e->tint    = sv.tint;
        }
    }
}

// -----------------------------------------------------------------------------
// Serialization — line-based text. Names may contain spaces, so every line
// puts the name LAST. Floats round-trip via %.9g.
//
//   passes v1
//   current <idx>
//   pass <enabled> <view> <name...>
//   part <ovis> <vis> <otint> <r> <g> <b> <n> <id0> ... <idN-1> <name...>
// -----------------------------------------------------------------------------
namespace {

inline void put_line_header(cardinal::string& out, const char* tag) {
    out += tag;
}
inline void put_num(cardinal::string& out, double v) {
    char b[48];
    cardinal::snprintf(b, sizeof(b), " %.9g", v);
    out += b;
}
inline void put_u(cardinal::string& out, u32 v) {
    char b[24];
    cardinal::snprintf(b, sizeof(b), " %u", v);
    out += b;
}

// Minimal whitespace tokenizer over one line.
struct Cursor {
    const char* p;
    const char* end;
    bool ok {true};

    void skip_ws() { while (p < end && (*p == ' ' || *p == '\t')) ++p; }
    // Next whitespace-delimited token as double (also used for ints).
    double num() {
        skip_ws();
        if (p >= end) { ok = false; return 0.0; }
        char* stop = nullptr;
        const double v = cardinal::strtod(p, &stop);
        if (stop == p) { ok = false; return 0.0; }
        p = stop;
        return v;
    }
    u32 uint() { return static_cast<u32>(num()); }
    // Everything remaining (trimmed of one leading space) = the name.
    cardinal::string rest() {
        skip_ws();
        return cardinal::string(p, static_cast<usize>(end - p));
    }
};

}  // namespace

cardinal::string PassSet::serialize() const {
    cardinal::string out = "passes v1\n";
    out += "current";
    put_num(out, current_);
    out += "\n";
    for (const auto& pass : passes_) {
        put_line_header(out, "pass");
        put_u(out, pass.enabled ? 1u : 0u);
        put_u(out, static_cast<u32>(pass.view_mode));
        out += " ";
        out += pass.name;
        out += "\n";
        for (const auto& part : pass.partitions) {
            put_line_header(out, "part");
            put_u(out, part.ovr.override_visibility ? 1u : 0u);
            put_u(out, part.ovr.visible ? 1u : 0u);
            put_u(out, part.ovr.override_tint ? 1u : 0u);
            put_num(out, part.ovr.tint.x);
            put_num(out, part.ovr.tint.y);
            put_num(out, part.ovr.tint.z);
            put_u(out, static_cast<u32>(part.entities.size()));
            for (u32 id : part.entities) put_u(out, id);
            out += " ";
            out += part.name;
            out += "\n";
        }
    }
    return out;
}

PassSet PassSet::deserialize(const cardinal::string& text, bool* ok) {
    PassSet ps;
    bool good = true;
    int  want_current = -1;

    const char* p   = text.c_str();
    const char* end = p + text.size();
    bool first = true;
    while (p < end) {
        const char* nl = p;
        while (nl < end && *nl != '\n') ++nl;
        const char* line_end = (nl > p && nl[-1] == '\r') ? nl - 1 : nl;

        if (line_end > p) {
            Cursor c{p, line_end};
            auto starts = [&](const char* tag, usize len) {
                if (static_cast<usize>(line_end - p) < len) return false;
                for (usize i = 0; i < len; ++i) if (p[i] != tag[i]) return false;
                return true;
            };
            if (first) {
                if (!starts("passes v1", 9)) { good = false; break; }
                first = false;
            } else if (starts("current", 7)) {
                c.p = p + 7;
                want_current = static_cast<int>(c.num());
                if (!c.ok) good = false;
            } else if (starts("pass", 4) && (p[4] == ' ' || p[4] == '\t')) {
                c.p = p + 4;
                const u32 en = c.uint();
                const u32 vm = c.uint();
                if (!c.ok) { good = false; break; }
                PassDef d;
                d.enabled   = en != 0;
                d.view_mode = static_cast<scene::ViewMode>(vm > 5u ? 0u : vm);
                d.name      = c.rest();
                ps.passes_.push_back(cardinal::move(d));
            } else if (starts("part", 4) && (p[4] == ' ' || p[4] == '\t')) {
                if (ps.passes_.empty()) { good = false; break; }
                c.p = p + 4;
                Partition part;
                part.ovr.override_visibility = c.uint() != 0;
                part.ovr.visible             = c.uint() != 0;
                part.ovr.override_tint       = c.uint() != 0;
                part.ovr.tint.x              = static_cast<float>(c.num());
                part.ovr.tint.y              = static_cast<float>(c.num());
                part.ovr.tint.z              = static_cast<float>(c.num());
                const u32 n = c.uint();
                for (u32 i = 0; i < n && c.ok; ++i) part.entities.push_back(c.uint());
                if (!c.ok) { good = false; break; }
                part.name = c.rest();
                ps.passes_.back().partitions.push_back(cardinal::move(part));
            }
            // Unknown lines are skipped (forward compatibility).
        }
        p = (nl < end) ? nl + 1 : nl;
    }

    ps.current_ = -1;
    if (want_current >= 0 && want_current < static_cast<int>(ps.passes_.size()))
        ps.current_ = want_current;
    else if (!ps.passes_.empty())
        ps.current_ = 0;

    if (ok) *ok = good;
    return ps;
}

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
PassSet make_default_pass_set() {
    PassSet ps;
    ps.add_pass("Beauty",    scene::ViewMode::Solid);
    ps.add_pass("Wireframe", scene::ViewMode::Wireframe);
    ps.add_pass("Normals",   scene::ViewMode::Normals);
    ps.set_current(0);
    return ps;
}

}  // namespace cardinal::render::rp
