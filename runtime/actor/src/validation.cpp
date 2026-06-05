// =============================================================================
// Cardinal — scene validation implementation.
// =============================================================================
#include <cardinal/actor/validation.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/core/cstdio.hpp>      // cardinal::snprintf
#include <cardinal/core/cmath.hpp>       // cardinal::isfinite

namespace cardinal::actor {

const char* severity_name(Severity s) noexcept {
    switch (s) {
        case Severity::Info:    return "Info";
        case Severity::Warning: return "Warning";
        case Severity::Error:   return "Error";
    }
    return "?";
}

namespace {

constexpr float kBoundsLimit = 1.0e5f;   // |coord| beyond this = "far outside"

bool coord_bad(float v) {
    return !cardinal::isfinite(v) || v > kBoundsLimit || v < -kBoundsLimit;
}

void add(cardinal::vector<ValidationIssue>& out, Severity sev, const Actor& a,
         cardinal::string msg) {
    out.push_back(ValidationIssue{ sev, a.id(), a.name(), cardinal::move(msg) });
}

}  // namespace

cardinal::vector<ValidationIssue> validate_world(const World& world) {
    cardinal::vector<ValidationIssue> out;

    u32 active_cameras = 0;

    // ---- Per-actor rules (in actor order) ----------------------------
    for (const auto& aptr : world.actors()) {
        const Actor& a = *aptr;
        if (!a.alive()) continue;

        const auto& comps = a.components();

        // Empty actor — nothing but the default Transform (or no comps).
        bool only_transform = true;
        for (const auto& c : comps) {
            if (cardinal::strcmp(c->type_name(), "Transform") != 0) { only_transform = false; break; }
        }
        if (comps.empty() || only_transform) {
            add(out, Severity::Info, a, "actor has no components beyond Transform");
        }

        // Transform bounds / finiteness.
        if (const auto* t = a.get_component<TransformComponent>()) {
            if (coord_bad(t->translation.x) || coord_bad(t->translation.y) ||
                coord_bad(t->translation.z)) {
                add(out, Severity::Warning, a,
                    "actor position is non-finite or far outside the world bounds");
            }
            if (t->scale.x == 0.0f || t->scale.y == 0.0f || t->scale.z == 0.0f) {
                add(out, Severity::Warning, a, "actor has a zero scale axis (invisible/degenerate)");
            }
        }

        // Mesh without an asset.
        if (const auto* m = a.get_component<MeshComponent>()) {
            if (m->asset_id.empty()) {
                add(out, Severity::Warning, a, "Mesh component has no asset_id");
            }
        }

        // Light with no contribution.
        if (const auto* l = a.get_component<LightComponent>()) {
            if (l->intensity <= 0.0f) {
                add(out, Severity::Warning, a, "Light has zero or negative intensity");
            }
            if (l->kind != LightKind::Directional && l->range <= 0.0f) {
                add(out, Severity::Warning, a, "Point/Spot light has zero range");
            }
        }

        // Audio emitter without a cue.
        if (const auto* ae = a.get_component<AudioEmitterComponent>()) {
            if (ae->cue_id.empty()) {
                add(out, Severity::Warning, a, "AudioEmitter has no cue_id");
            }
        }

        // Active camera tally (scene-level rule below).
        if (const auto* cam = a.get_component<CameraComponent>()) {
            if (cam->active) ++active_cameras;
        }
    }

    // ---- Scene-level rules -------------------------------------------
    // More than one active camera is ambiguous.
    if (active_cameras > 1) {
        char buf[96];
        cardinal::snprintf(buf, sizeof(buf),
            "%u cameras are marked active (expected at most 1)", active_cameras);
        out.push_back(ValidationIssue{ Severity::Warning, 0, "(scene)", buf });
    }

    // Duplicate actor names — count alive actors per name, report each dup
    // group once (deterministic: first-seen order of the duplicated name).
    cardinal::vector<cardinal::string> seen;       // names already reported
    for (const auto& aptr : world.actors()) {
        if (!aptr->alive()) continue;
        const cardinal::string& nm = aptr->name();
        // already reported?
        bool reported = false;
        for (const auto& s : seen) if (s == nm) { reported = true; break; }
        if (reported) continue;
        // count duplicates
        u32 cnt = 0;
        for (const auto& b : world.actors()) {
            if (b->alive() && b->name() == nm) ++cnt;
        }
        if (cnt > 1) {
            seen.push_back(nm);
            char buf[160];
            cardinal::snprintf(buf, sizeof(buf),
                "%u actors share the name '%s'", cnt, nm.c_str());
            out.push_back(ValidationIssue{ Severity::Info, 0, "(scene)", buf });
        }
    }

    return out;
}

u32 count_issues(const cardinal::vector<ValidationIssue>& issues, Severity min_severity) {
    u32 n = 0;
    for (const auto& i : issues) {
        if (static_cast<u32>(i.severity) >= static_cast<u32>(min_severity)) ++n;
    }
    return n;
}

}  // namespace cardinal::actor
