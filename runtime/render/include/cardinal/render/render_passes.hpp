#pragma once

// =============================================================================
// Cardinal — XSI-style Render Passes & Partitions.
//
// Softimage|XSI's signature rendering workflow, adapted to Cardinal:
//
//   * A scene carries a set of named RENDER PASSES (Beauty, Wireframe,
//     Normals, Matte, custom...). Each pass renders the same scene through
//     its own lens: a pass-level ViewMode plus per-partition overrides.
//   * Each pass owns PARTITIONS — disjoint groups of scene entities (an
//     entity belongs to at most ONE partition per pass; XSI enforced the
//     same). Entities in no partition ("background partition") keep their
//     scene state untouched.
//   * Partitions carry OVERRIDES: visibility (hide a group in this pass)
//     and tint (flat-colour a group — the classic matte/ID-pass workflow).
//   * The CURRENT pass drives the viewport, exactly like XSI's current-pass
//     preview: hosts call apply() before rendering and restore() after —
//     application is non-destructive (a snapshot of the touched entities is
//     taken and restored), so the authoritative scene state never drifts.
//
// The model is pure CPU + headless-testable (tests/render_passes); the
// Studio panel and per-pass AOV output land on top of this.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/scene/scene.hpp>

namespace cardinal::render::rp {

// Per-partition overrides applied while a pass is active.
struct PartitionOverride {
    bool        override_visibility {false};
    bool        visible             {true};
    bool        override_tint       {false};
    scene::Vec3 tint                {1.0f, 1.0f, 1.0f};
};

// A named, disjoint group of entities within one pass.
struct Partition {
    cardinal::string      name;
    cardinal::vector<u32> entities;      // scene entity ids
    PartitionOverride     ovr;

    bool contains(u32 id) const noexcept {
        for (u32 e : entities) if (e == id) return true;
        return false;
    }
};

// One render pass: a view mode + its partitions.
struct PassDef {
    cardinal::string           name;
    bool                       enabled   {true};
    scene::ViewMode            view_mode {scene::ViewMode::Solid};
    cardinal::vector<Partition> partitions;

    Partition* find_partition(const cardinal::string& pname) {
        for (auto& p : partitions) if (p.name == pname) return &p;
        return nullptr;
    }
    const Partition* partition_of(u32 entity) const {
        for (const auto& p : partitions) if (p.contains(entity)) return &p;
        return nullptr;
    }
    // Assign an entity to a partition (created if missing). DISJOINT within
    // the pass: the entity is removed from any other partition first (XSI
    // semantics — an object sits in exactly one partition per pass).
    Partition& assign(u32 entity, const cardinal::string& pname);
    // Drop the entity from whichever partition holds it (-> background).
    void unassign(u32 entity);
};

// Snapshot of entity state taken by apply(); hand back to restore().
struct Applied {
    struct Saved {
        u32         id      {0};
        bool        visible {true};
        scene::Vec3 tint    {1.0f, 1.0f, 1.0f};
    };
    cardinal::vector<Saved> saved;
    bool                    active {false};
};

// The scene's pass list + current pass (XSI's "current pass").
class PassSet {
public:
    PassDef& add_pass(cardinal::string name,
                      scene::ViewMode vm = scene::ViewMode::Solid);
    bool     remove_pass(const cardinal::string& name);
    PassDef* find(const cardinal::string& name);

    cardinal::vector<PassDef>&       passes()       noexcept { return passes_; }
    const cardinal::vector<PassDef>& passes() const noexcept { return passes_; }

    int      current_index() const noexcept { return current_; }
    void     set_current(int i) noexcept {
        current_ = (i >= 0 && i < static_cast<int>(passes_.size())) ? i : -1;
    }
    PassDef* current() {
        return (current_ >= 0 && current_ < static_cast<int>(passes_.size()))
             ? &passes_[static_cast<usize>(current_)] : nullptr;
    }

    // Non-destructive application of a pass's partition overrides onto the
    // scene. Every touched entity's prior {visible, tint} is snapshotted in
    // the returned Applied; restore() puts them back. apply() with no pass
    // uses current(); a null/disabled pass yields an inactive Applied.
    Applied apply(scene::Scene& s);
    Applied apply(scene::Scene& s, const PassDef& pass) const;
    void    restore(scene::Scene& s, const Applied& a) const;

    // Text round-trip (project save/load). Format is line-based; names may
    // contain spaces (they terminate each line).
    cardinal::string serialize() const;
    static PassSet   deserialize(const cardinal::string& text, bool* ok = nullptr);

private:
    cardinal::vector<PassDef> passes_;
    int                       current_ {-1};
};

// XSI-style starter set: Beauty (Solid), Wireframe, Normals. Current = Beauty.
PassSet make_default_pass_set();

}  // namespace cardinal::render::rp
