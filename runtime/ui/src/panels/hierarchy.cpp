// =============================================================================
// Studio — scene hierarchy panel implementation.
// =============================================================================
#include "hierarchy.hpp"

#include <cardinal/scene/scene.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cctype.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::ui::panels {

using cardinal::u32;
using cardinal::usize;

void draw(cardinal::scene::Scene& scene,
          u32* selected_id_inout,
          const char* title,
          bool* p_open,
          HierarchyPanelState& state)
{
    if (!ImGui::Begin(title ? title : "Hierarchy", p_open)) { ImGui::End(); return; }

    // Header row: count + filter.
    ImGui::Text("Entities: %zu", scene.entities().size());
    ImGui::SameLine();
    if (ImGui::SmallButton("Expand all"))   state.expand_pulse = +1;
    ImGui::SameLine();
    if (ImGui::SmallButton("Collapse all")) state.expand_pulse = -1;

    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##h_filter", "filter (name)",
                             state.filter, sizeof(state.filter));
    ImGui::Separator();

    // Pre-compute parent → children adjacency and a name-filter mask
    // once per frame so the recursive draw is O(N) instead of O(N²).
    // Adjacency is keyed by parent_id; root entities live under key 0.
    cardinal::unordered_map<u32, cardinal::vector<u32>> kids;
    const auto& ents = scene.entities();
    kids.reserve(ents.size() + 1);
    cardinal::unordered_map<u32, const cardinal::scene::Entity*> by_id;
    by_id.reserve(ents.size());
    for (const auto& e : ents) {
        kids[e.parent_id].push_back(e.id);
        by_id[e.id] = &e;
    }
    // If `parent_id` points at a missing entity (e.g. the parent was
    // deleted in a previous frame and the child wasn't reparented to
    // root), surface those as roots so they remain visible & editable.
    for (const auto& e : ents) {
        if (e.parent_id != 0u && by_id.find(e.parent_id) == by_id.end()) {
            kids[0].push_back(e.id);
        }
    }

    // Filter mask: an entity passes if it matches the filter OR any
    // descendant does (so collapsed branches with hidden hits don't
    // disappear). filter_buf empty → everything passes.
    const bool has_filter = (state.filter[0] != '\0');
    cardinal::unordered_map<u32, bool> visible_in_filter;
    if (has_filter) {
        // First pass: literal match (case-insensitive substring).
        cardinal::string needle = state.filter;
        for (auto& c : needle) c = static_cast<char>(cardinal::tolower(static_cast<unsigned char>(c)));
        for (const auto& e : ents) {
            cardinal::string hay = e.name;
            for (auto& c : hay) c = static_cast<char>(cardinal::tolower(static_cast<unsigned char>(c)));
            visible_in_filter[e.id] = (hay.find(needle) != cardinal::string::npos);
        }
        // Propagate "true" up the chain so ancestors of matches stay
        // visible (the user wants to see the path to the hit).
        for (const auto& e : ents) {
            if (!visible_in_filter[e.id]) continue;
            u32 p = e.parent_id;
            const usize cap = ents.size() + 1u;
            for (usize i = 0; i < cap && p != 0u; ++i) {
                visible_in_filter[p] = true;
                auto it = by_id.find(p);
                if (it == by_id.end()) break;
                p = it->second->parent_id;
            }
        }
    }

    // Drop target for the root level — drag an entity to "root drop
    // zone" to clear its parent.
    ImGui::Dummy(ImVec2(-1, 4));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CARDINAL_ENTITY_ID")) {
            if (payload->DataSize == sizeof(u32)) {
                u32 dragged = *static_cast<const u32*>(payload->Data);
                scene.set_parent(dragged, 0u);
            }
        }
        ImGui::EndDragDropTarget();
    }

    // Recursive lambda — renders one entity row + its children. Captures
    // are by reference; recursion is bounded by real scene depth.
    cardinal::function<void(u32, int)> draw_node = [&](u32 entity_id, int depth) {
        auto it = by_id.find(entity_id);
        if (it == by_id.end()) return;
        const cardinal::scene::Entity& e = *it->second;
        if (has_filter && !visible_in_filter[e.id]) return;

        ImGui::PushID(static_cast<int>(e.id));

        const bool has_children = (kids.find(e.id) != kids.end()) && !kids[e.id].empty();
        const bool sel = (selected_id_inout && *selected_id_inout == e.id);

        ImGuiTreeNodeFlags flags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
            ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen;
        if (sel)              flags |= ImGuiTreeNodeFlags_Selected;
        if (!has_children)    flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        // Apply Expand all / Collapse all pulse — overrides the open
        // state for one frame, then resets so the user's manual
        // arrow-clicks stick afterwards.
        if (state.expand_pulse != 0) {
            ImGui::SetNextItemOpen(state.expand_pulse > 0);
        }

        char label[192];
        cardinal::snprintf(label, sizeof(label), "%s %s",
                      e.visible ? "" : "(hidden)", e.name.c_str());
        const bool open = ImGui::TreeNodeEx("##node", flags, "%s", label);

        // Click-to-select on the node row (TreeNodeEx already swallows
        // the click event for us via flags).
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (selected_id_inout) *selected_id_inout = e.id;
        }

        // Drag source — dragging an entity row pushes its id as the
        // payload so other rows can accept it as a reparent target.
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            u32 id_copy = e.id;
            ImGui::SetDragDropPayload("CARDINAL_ENTITY_ID", &id_copy, sizeof(id_copy));
            ImGui::Text("Reparent: %s", e.name.c_str());
            ImGui::EndDragDropSource();
        }
        // Drop target — drop entity X onto this row to make X a child
        // of this entity. Cycle prevention lives in scene.set_parent.
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CARDINAL_ENTITY_ID")) {
                if (payload->DataSize == sizeof(u32)) {
                    u32 dragged = *static_cast<const u32*>(payload->Data);
                    scene.set_parent(dragged, e.id);
                }
            }
            ImGui::EndDragDropTarget();
        }

        // Right-click context menu — quick actions per entity.
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Select")) {
                if (selected_id_inout) *selected_id_inout = e.id;
            }
            if (ImGui::MenuItem("Unparent (move to root)")) {
                scene.set_parent(e.id, 0u);
            }
            if (ImGui::MenuItem(e.visible ? "Hide" : "Show")) {
                if (auto* m = scene.find_by_id(e.id)) m->visible = !m->visible;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Delete entity")) {
                if (selected_id_inout && *selected_id_inout == e.id) {
                    *selected_id_inout = 0u;
                }
                scene.remove_entity(e.id);
            }
            ImGui::EndPopup();
        }

        // Children — recurse if the node is open. NoTreePushOnOpen
        // (used for leaves) means we don't pair this with TreePop.
        if (open && has_children) {
            auto kit = kids.find(e.id);
            if (kit != kids.end()) {
                for (u32 cid : kit->second) draw_node(cid, depth + 1);
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    };

    // Render every root entity. Roots = parent_id 0 (or invalid parent
    // surfaced earlier).
    auto root_it = kids.find(0u);
    if (root_it != kids.end()) {
        for (u32 root_id : root_it->second) draw_node(root_id, 0);
    }

    // Reset the one-frame Expand/Collapse pulse so subsequent frames
    // honour the user's per-row arrow toggles again.
    state.expand_pulse = 0;

    ImGui::End();
}

}  // namespace cardinal::ui::panels
