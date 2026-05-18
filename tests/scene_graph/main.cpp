// =============================================================================
// Cardinal — deterministic scene-graph hierarchy regression suite.
//
// Scene's add/remove/find + parent/children + cycle-prevention back the
// editor outliner tree, drag-to-reparent, and every hierarchy walk. The
// load-bearing guarantees: set_parent() can NEVER create a cycle (a
// regression hangs/corrupts traversal), and remove_entity() reparents
// the deleted node's children to root (a regression leaves dangling
// parent_ids → walks chase freed ids). These ops are pure CPU (no Mesh
// / rhi::Device) → fully headless + deterministic. An independent cycle
// oracle cross-checks the implementation. Exit 0 = all pass.
// =============================================================================

#include <cardinal/scene/scene.hpp>
#include <cardinal/core/log.hpp>

#include <vector>

namespace {

namespace sc = cardinal::scene;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("scenetest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Independent oracle: every parent_id is 0 or an existing entity, and
// every parent chain reaches root within <= N steps (no cycle).
bool graph_is_acyclic_and_sound(sc::Scene& s) {
    auto& es = s.entities();
    const cardinal::usize n = es.size();
    for (auto& e : es) {
        if (e.parent_id != 0u && s.find_by_id(e.parent_id) == nullptr)
            return false;                              // dangling parent
        u32 w = e.parent_id;
        cardinal::usize steps = 0;
        while (w != 0u) {
            if (steps++ > n) return false;             // cycle
            sc::Entity* p = s.find_by_id(w);
            if (p == nullptr) return false;
            w = p->parent_id;
        }
    }
    return true;
}

bool vec_eq(const std::vector<u32>& a, std::initializer_list<u32> b) {
    if (a.size() != b.size()) return false;
    cardinal::usize i = 0;
    for (u32 x : b) if (a[i++] != x) return false;
    return true;
}

// ---- add / find / id monotonicity ----------------------------------
void test_add_find() {
    sc::Scene s;
    CHECK(s.entities().empty());

    // add_entity returns Entity& to entities_.back(); a later add_entity
    // can reallocate the vector, so capture ids and re-fetch by id
    // (this is exactly why the engine keys on ids, not cached ptrs).
    const u32 ia = s.add_entity("A").id;
    const u32 ib = s.add_entity("B").id;
    const u32 ic = s.add_entity("C").id;
    CHECK(ia == 1u && ib == 2u && ic == 3u);         // next_id_ starts at 1
    CHECK(s.find_by_id(ia)->parent_id == 0u);        // new = root
    CHECK(s.find_by_id(ia)->name == "A" &&
          s.find_by_id(ic)->name == "C");
    CHECK(s.entities().size() == sz(3));

    CHECK(s.find_by_id(1) != nullptr && s.find_by_id(1)->name == "A");
    CHECK(s.find_by_id(2)->name == "B");
    CHECK(s.find_by_id(0)   == nullptr);             // 0 = invalid
    CHECK(s.find_by_id(999) == nullptr);
}

// ---- remove: reparent-to-root, no id recycling ---------------------
void test_remove() {
    sc::Scene s;
    s.add_entity("A");                 // 1
    s.add_entity("B");                 // 2
    s.add_entity("C");                 // 3
    CHECK(s.remove_entity(2));
    CHECK(s.find_by_id(2) == nullptr);
    CHECK(s.entities().size() == sz(2));
    CHECK(s.find_by_id(1)->name == "A");             // survivors intact
    CHECK(s.find_by_id(3)->name == "C");
    CHECK(!s.remove_entity(2));                       // already gone
    CHECK(!s.remove_entity(999));                     // never existed

    // Ids are monotonic — a remove never frees an id for reuse.
    sc::Entity& d = s.add_entity("D");
    CHECK(d.id == 4u);

    // Removing a parent reparents its children to ROOT (no dangling).
    sc::Scene t;
    t.add_entity("P");                 // 1
    t.add_entity("C1");                // 2
    t.add_entity("C2");                // 3
    t.add_entity("G");                 // 4 (grandchild under C1)
    CHECK(t.set_parent(2, 1));
    CHECK(t.set_parent(3, 1));
    CHECK(t.set_parent(4, 2));
    CHECK(t.remove_entity(1));                        // delete the parent
    CHECK(t.find_by_id(1) == nullptr);
    CHECK(t.find_by_id(2)->parent_id == 0u);          // child → root
    CHECK(t.find_by_id(3)->parent_id == 0u);
    CHECK(t.find_by_id(4)->parent_id == 2u);          // grandchild untouched
    CHECK(graph_is_acyclic_and_sound(t));
}

// ---- children_of ----------------------------------------------------
void test_children_of() {
    sc::Scene s;
    s.add_entity("A");   // 1
    s.add_entity("B");   // 2
    s.add_entity("C");   // 3
    s.add_entity("D");   // 4
    s.add_entity("E");   // 5
    CHECK(s.set_parent(2, 1));   // B,C under A
    CHECK(s.set_parent(3, 1));
    CHECK(s.set_parent(4, 2));   // D,E under B
    CHECK(s.set_parent(5, 2));
    // Order == entities_ insertion order (no removes performed).
    CHECK(vec_eq(s.children_of(1), { 2u, 3u }));
    CHECK(vec_eq(s.children_of(2), { 4u, 5u }));
    CHECK(s.children_of(3).empty());
    CHECK(vec_eq(s.children_of(0), { 1u }));          // roots: just A
    CHECK(s.children_of(999).empty());
}

// ---- set_parent rules ----------------------------------------------
void test_set_parent_rules() {
    sc::Scene s;
    s.add_entity("A");   // 1
    s.add_entity("B");   // 2
    CHECK(s.set_parent(2, 1));
    CHECK(s.find_by_id(2)->parent_id == 1u);
    CHECK(s.set_parent(2, 0));                        // reparent to root OK
    CHECK(s.find_by_id(2)->parent_id == 0u);

    CHECK(!s.set_parent(0, 1));                       // entity 0 invalid
    CHECK(!s.set_parent(999, 1));                     // entity not found
    CHECK(!s.set_parent(2, 999));                     // parent not found
    CHECK(s.find_by_id(2)->parent_id == 0u);          // unchanged on reject
}

// ---- would_create_cycle / set_parent cycle prevention (keystone) ---
void test_cycle_prevention() {
    sc::Scene s;
    s.add_entity("A");   // 1
    s.add_entity("B");   // 2
    s.add_entity("C");   // 3
    s.add_entity("D");   // 4
    CHECK(s.set_parent(2, 1));   // A>B
    CHECK(s.set_parent(3, 2));   // A>B>C
    CHECK(s.set_parent(4, 3));   // A>B>C>D

    // Self-parent is a cycle.
    CHECK(s.would_create_cycle(1, 1));
    CHECK(!s.set_parent(1, 1));
    // Reparent an ancestor under its own descendant → cycle, rejected.
    CHECK(s.would_create_cycle(1, 4));               // A under D
    CHECK(!s.set_parent(1, 4));
    CHECK(s.find_by_id(1)->parent_id == 0u);          // A still root
    CHECK(s.would_create_cycle(2, 4));               // B under D
    CHECK(!s.set_parent(2, 4));
    CHECK(s.find_by_id(2)->parent_id == 1u);          // unchanged
    // Reparent-to-root never cycles.
    CHECK(!s.would_create_cycle(2, 0));
    CHECK(s.set_parent(2, 0));
    CHECK(graph_is_acyclic_and_sound(s));
    // Non-cyclic moves still allowed.
    CHECK(!s.would_create_cycle(4, 1));              // D under A (legal)
    CHECK(s.set_parent(4, 1));
    CHECK(graph_is_acyclic_and_sound(s));

    // Robustness: a corrupt pre-existing cycle (forced by writing
    // parent_id directly, bypassing set_parent) must NOT hang
    // would_create_cycle — it's bounded by entities_.size()+1.
    sc::Scene k;
    k.add_entity("X"); // 1
    k.add_entity("Y"); // 2
    k.add_entity("Z"); // 3
    k.find_by_id(1)->parent_id = 2u;                  // X→Y
    k.find_by_id(2)->parent_id = 1u;                  // Y→X  (forced cycle)
    const bool r = k.would_create_cycle(3, 1);        // must just RETURN
    CHECK(r == true || r == false);                   // (no infinite loop)
}

// ---- fuzz: the public API can NEVER produce a cycle ----------------
u32 xs(u32& st) { st ^= st<<13; st ^= st>>17; st ^= st<<5; return st; }

void run_fuzz(u32 seed, std::vector<u32>& trace) {
    sc::Scene s;
    u32 st = seed ? seed : 1u;
    for (int step = 0; step < 1500; ++step) {
        const u32 op = xs(st) % 10u;
        auto& es = s.entities();
        if (op < 5u || es.empty()) {
            s.add_entity("e");
        } else if (op < 8u) {
            const u32 ei = es[xs(st) % es.size()].id;
            // parent: 0 (root) ~1/4 of the time, else a random entity.
            const u32 pi = (xs(st) % 4u == 0u)
                ? 0u : es[xs(st) % es.size()].id;
            (void)s.set_parent(ei, pi);
        } else {
            (void)s.remove_entity(es[xs(st) % es.size()].id);
        }
        // INVARIANT: the public API can never make the graph cyclic or
        // leave a dangling parent_id — must hold after EVERY op.
        if (!graph_is_acyclic_and_sound(s))
            ::check_impl(false, "fuzz: graph invariant violated",
                         __LINE__);
        trace.push_back(static_cast<u32>(s.entities().size()));
    }
    // Final canonical fingerprint (sorted id,parent pairs flattened).
    auto& es = s.entities();
    std::vector<u32> ids;
    for (auto& e : es) ids.push_back(e.id);
    for (cardinal::usize i = 0; i < ids.size(); ++i)
        for (cardinal::usize j = i + 1; j < ids.size(); ++j)
            if (ids[j] < ids[i]) { const u32 t = ids[i]; ids[i] = ids[j]; ids[j] = t; }
    for (u32 id : ids) {
        trace.push_back(id);
        trace.push_back(s.find_by_id(id)->parent_id);
    }
}

void test_fuzz() {
    std::vector<u32> a, b;
    run_fuzz(0xBEEF1234u, a);
    run_fuzz(0xBEEF1234u, b);
    CHECK(a == b);                                    // deterministic
    CHECK(!a.empty());
}

}  // namespace

int main() {
    test_add_find();
    test_remove();
    test_children_of();
    test_set_parent_rules();
    test_cycle_prevention();
    test_fuzz();

    if (g_fail == 0) {
        cardinal::log::infof("scenetest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("scenetest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
