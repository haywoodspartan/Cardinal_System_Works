// =============================================================================
// Cardinal — XSI-style Render Passes & Partitions regression suite.
//
// Pure CPU, headless: partition disjointness, non-destructive apply/restore
// against a real scene::Scene, current-pass selection, stale-id tolerance,
// and the text serialize/deserialize round-trip. Exit 0 = all pass.
// =============================================================================

#include <cardinal/render/render_passes.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace rp = cardinal::render::rp;
using cardinal::u32;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("rptest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-6f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}

// ---- partition assignment stays disjoint within a pass ---------------------
void test_partition_disjoint() {
    rp::PassDef pass;
    pass.name = "Matte";
    pass.assign(7, "red");
    pass.assign(9, "red");
    pass.assign(7, "blue");            // must MOVE 7 out of "red"

    const rp::Partition* red  = pass.partition_of(9);
    const rp::Partition* blue = pass.partition_of(7);
    CHECK(red  != nullptr && red->name  == "red");
    CHECK(blue != nullptr && blue->name == "blue");
    CHECK(!pass.find_partition("red")->contains(7));
    CHECK(pass.find_partition("red")->entities.size() == 1);

    pass.unassign(9);
    CHECK(pass.partition_of(9) == nullptr);   // -> background
}

// ---- apply/restore is non-destructive against a live scene -----------------
void test_apply_restore() {
    cardinal::scene::Scene s;
    auto& a = s.add_entity("A");
    const u32 ida = a.id;
    a.tint = {0.2f, 0.3f, 0.4f};
    auto& b = s.add_entity("B");
    const u32 idb = b.id;
    auto& c = s.add_entity("C");       // background — untouched
    const u32 idc = c.id;

    rp::PassSet ps;
    rp::PassDef& matte = ps.add_pass("Matte", cardinal::scene::ViewMode::Solid);
    rp::Partition& hot = matte.assign(ida, "hot");
    hot.ovr.override_tint = true;
    hot.ovr.tint = {1.0f, 0.0f, 0.0f};
    rp::Partition& hidden = matte.assign(idb, "hidden");
    hidden.ovr.override_visibility = true;
    hidden.ovr.visible = false;

    ps.set_current(0);
    rp::Applied ap_state = ps.apply(s);
    CHECK(ap_state.active);
    CHECK(ap_state.saved.size() == 2);
    CHECK(ap(s.find_by_id(ida)->tint.x, 1.0f) && ap(s.find_by_id(ida)->tint.y, 0.0f));
    CHECK(s.find_by_id(idb)->visible == false);
    CHECK(s.find_by_id(idc)->visible == true);      // background untouched

    ps.restore(s, ap_state);
    CHECK(ap(s.find_by_id(ida)->tint.x, 0.2f));      // original tint back
    CHECK(s.find_by_id(idb)->visible == true);

    // Disabled pass -> inactive application, scene untouched.
    matte.enabled = false;
    rp::Applied a2 = ps.apply(s);
    CHECK(!a2.active);
    CHECK(ap(s.find_by_id(ida)->tint.x, 0.2f));

    // Stale entity id in a partition must not crash or block the rest.
    matte.enabled = true;
    matte.assign(9999u, "hot");
    rp::Applied a3 = ps.apply(s);
    CHECK(a3.active);
    CHECK(ap(s.find_by_id(ida)->tint.x, 1.0f));
    ps.restore(s, a3);
}

// ---- current-pass selection -------------------------------------------------
void test_current_pass() {
    rp::PassSet ps = rp::make_default_pass_set();
    CHECK(ps.passes().size() == 3);
    CHECK(ps.current_index() == 0);
    CHECK(ps.current() != nullptr && ps.current()->name == "Beauty");
    CHECK(ps.current()->view_mode == cardinal::scene::ViewMode::Solid);

    ps.set_current(1);
    CHECK(ps.current()->name == "Wireframe");
    ps.set_current(99);                    // out of range -> deselect
    CHECK(ps.current() == nullptr);

    ps.set_current(2);
    CHECK(ps.remove_pass("Normals"));      // removing the current pass clamps
    CHECK(ps.current_index() == 1);
    CHECK(!ps.remove_pass("Nope"));
}

// ---- serialize round-trip ----------------------------------------------------
void test_serialize_roundtrip() {
    rp::PassSet ps;
    rp::PassDef& beauty = ps.add_pass("Beauty Pass", cardinal::scene::ViewMode::Solid);
    (void)beauty;
    rp::PassDef& matte = ps.add_pass("Matte Layer 2", cardinal::scene::ViewMode::Wireframe);
    matte.enabled = false;
    rp::Partition& part = matte.assign(4, "hero group");
    matte.assign(7, "hero group");
    part.ovr.override_visibility = true;
    part.ovr.visible = false;
    part.ovr.override_tint = true;
    part.ovr.tint = {0.25f, 0.5f, 0.75f};
    ps.set_current(1);

    const cardinal::string text = ps.serialize();
    bool ok = false;
    rp::PassSet back = rp::PassSet::deserialize(text, &ok);
    CHECK(ok);
    CHECK(back.passes().size() == 2);
    CHECK(back.current_index() == 1);
    CHECK(back.passes()[0].name == "Beauty Pass");
    CHECK(back.passes()[0].enabled);
    rp::PassDef& m2 = back.passes()[1];
    CHECK(m2.name == "Matte Layer 2");
    CHECK(!m2.enabled);
    CHECK(m2.view_mode == cardinal::scene::ViewMode::Wireframe);
    CHECK(m2.partitions.size() == 1);
    const rp::Partition& p2 = m2.partitions[0];
    CHECK(p2.name == "hero group");
    CHECK(p2.entities.size() == 2 && p2.entities[0] == 4 && p2.entities[1] == 7);
    CHECK(p2.ovr.override_visibility && !p2.ovr.visible);
    CHECK(p2.ovr.override_tint);
    CHECK(ap(p2.ovr.tint.x, 0.25f) && ap(p2.ovr.tint.y, 0.5f) && ap(p2.ovr.tint.z, 0.75f));

    // Round-trip once more: serialize(deserialize(text)) must be identical.
    CHECK(back.serialize() == text);

    // Garbage input -> ok=false, empty-but-sane set.
    bool ok2 = true;
    rp::PassSet junk = rp::PassSet::deserialize("not a pass file", &ok2);
    CHECK(!ok2);
    CHECK(junk.passes().empty());
    CHECK(junk.current() == nullptr);
}

}  // namespace

int main() {
    test_partition_disjoint();
    test_apply_restore();
    test_current_pass();
    test_serialize_roundtrip();

    if (g_fail == 0) {
        cardinal::log::infof("rptest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("rptest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
