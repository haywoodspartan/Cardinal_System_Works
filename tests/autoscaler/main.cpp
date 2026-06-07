// =============================================================================
// Cardinal — deterministic memory-pressure / autoscaler regression suite.
//
// Pins three stacked pure pieces:
//
//   memory::classify  — used% → Pressure with INCLUSIVE (>=) boundaries
//                       and monotone tiering (default 60/80/92, plus a
//                       custom-threshold case). pressure_name /
//                       domain_name round-trips; enum order Low<…<Crit.
//   budget::Broker     — register ids are monotonic-from-1; report_used
//                       + subsystem_reports round-trip; debug_force_*
//                       drives tiers WITHOUT the OS Monitor; the
//                       on_pressure_change callback fires ONLY on a tier
//                       transition (never steady-state) and ONLY for the
//                       subsystem's own Domain; deregister stops it.
//   budget::AutoScaler — registers + applies value_low at construction,
//                       picks the per-tier value on every transition
//                       (current()/current_tier()), and deregisters in
//                       its destructor.
//
// Broker is constructed fresh per test (not a singleton) and every tier
// is pinned via debug_force_pressure on BOTH domains, so the suite is
// fully deterministic — the real memory::Monitor is never consulted.
// Threshold comparisons use exact double literals (no FP fragility).
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/sync/autoscaler.hpp>
#include <cardinal/core/budget/budget.hpp>
#include <cardinal/core/budget/memory.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace mem = cardinal::memory;
namespace bg  = cardinal::budget;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("astest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool apf(float a, float b, float e = 1e-6f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }

const bg::SubsystemReport* find_report(
    const std::vector<bg::SubsystemReport>& v, cardinal::u32 id) {
    for (const auto& r : v) if (r.id == id) return &r;
    return nullptr;
}

// ---- memory::classify — inclusive boundaries + monotone tiering ----
void test_classify() {
    const mem::PressureThresholds def{};   // 60 / 80 / 92
    CHECK(def.medium_at_used_pct == 60.0);
    CHECK(def.high_at_used_pct == 80.0);
    CHECK(def.critical_at_used_pct == 92.0);

    using P = mem::Pressure;
    CHECK(mem::classify(0.0,    def) == P::Low);
    CHECK(mem::classify(-5.0,   def) == P::Low);     // clamp-ish: below all
    CHECK(mem::classify(59.999, def) == P::Low);
    CHECK(mem::classify(60.0,   def) == P::Medium);  // >= inclusive
    CHECK(mem::classify(60.001, def) == P::Medium);
    CHECK(mem::classify(79.999, def) == P::Medium);
    CHECK(mem::classify(80.0,   def) == P::High);    // >= inclusive
    CHECK(mem::classify(91.999, def) == P::High);
    CHECK(mem::classify(92.0,   def) == P::Critical);// >= inclusive
    CHECK(mem::classify(100.0,  def) == P::Critical);
    CHECK(mem::classify(1000.0, def) == P::Critical);

    // Custom thresholds shift every boundary; still inclusive.
    const mem::PressureThresholds ct{10.0, 20.0, 30.0};
    CHECK(mem::classify(9.999, ct) == P::Low);
    CHECK(mem::classify(10.0,  ct) == P::Medium);
    CHECK(mem::classify(19.99, ct) == P::Medium);
    CHECK(mem::classify(20.0,  ct) == P::High);
    CHECK(mem::classify(29.99, ct) == P::High);
    CHECK(mem::classify(30.0,  ct) == P::Critical);

    // Names + monotone enum ordering.
    CHECK(streq(mem::pressure_name(P::Low),      "Low"));
    CHECK(streq(mem::pressure_name(P::Medium),   "Medium"));
    CHECK(streq(mem::pressure_name(P::High),     "High"));
    CHECK(streq(mem::pressure_name(P::Critical), "Critical"));
    CHECK(static_cast<cardinal::u32>(P::Low) == 0u);
    CHECK(static_cast<cardinal::u32>(P::Low) < static_cast<cardinal::u32>(P::Medium));
    CHECK(static_cast<cardinal::u32>(P::Medium) < static_cast<cardinal::u32>(P::High));
    CHECK(static_cast<cardinal::u32>(P::High) < static_cast<cardinal::u32>(P::Critical));

    CHECK(streq(bg::domain_name(bg::Domain::System), "System"));
    CHECK(streq(bg::domain_name(bg::Domain::Gpu),    "GPU"));
}

// ---- Broker registry: monotonic ids, report_used, reports ----------
void test_broker_registry() {
    bg::Broker b;
    // Pin both domains so no OS-derived transition ever fires here.
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Low);
    b.debug_force_pressure(bg::Domain::Gpu,    mem::Pressure::Low);

    bg::Subsystem s1{}; s1.name = "a"; s1.domain = bg::Domain::Gpu;
    bg::Subsystem s2{}; s2.name = "cache"; s2.domain = bg::Domain::System;
    s2.advisory_min_bytes = 1000u; s2.advisory_max_bytes = 5000u;
    bg::Subsystem s3{}; s3.name = "c"; s3.domain = bg::Domain::Gpu;

    const cardinal::u32 id1 = b.register_subsystem(s1);
    const cardinal::u32 id2 = b.register_subsystem(s2);
    const cardinal::u32 id3 = b.register_subsystem(s3);
    CHECK(id1 == 1u && id2 == 2u && id3 == 3u);   // monotonic from 1

    b.report_used(id2, 3333u);
    b.report_used(99999u, 7u);                    // unknown id → no-op

    auto rep = b.subsystem_reports();
    CHECK(rep.size() == sz(3));
    const bg::SubsystemReport* r2 = find_report(rep, id2);
    CHECK(r2 != nullptr);
    CHECK(r2->name == "cache");
    CHECK(r2->domain == bg::Domain::System);
    CHECK(r2->used_bytes == 3333u);
    CHECK(r2->advisory_min_bytes == 1000u);
    CHECK(r2->advisory_max_bytes == 5000u);
    CHECK(r2->last_seen_tier == mem::Pressure::Low);   // no transition yet
    const bg::SubsystemReport* r1 = find_report(rep, id1);
    CHECK(r1 != nullptr && r1->used_bytes == 0u);      // never reported

    b.deregister_subsystem(id1);
    auto rep2 = b.subsystem_reports();
    CHECK(rep2.size() == sz(2));
    CHECK(find_report(rep2, id1) == nullptr);          // gone
    CHECK(find_report(rep2, id3) != nullptr);          // others intact
    b.deregister_subsystem(99999u);                    // unknown → no-op
    CHECK(b.subsystem_reports().size() == sz(2));
}

// ---- Broker: transition-only callbacks + domain isolation ----------
void test_broker_transitions() {
    bg::Broker b;
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Low);
    b.debug_force_pressure(bg::Domain::Gpu,    mem::Pressure::Low);

    std::vector<mem::Pressure> got_gpu;
    std::vector<mem::Pressure> got_sys;

    bg::Subsystem g{}; g.name = "gpu"; g.domain = bg::Domain::Gpu;
    g.on_pressure_change = [&got_gpu](mem::Pressure p){ got_gpu.push_back(p); };
    bg::Subsystem s{}; s.name = "sys"; s.domain = bg::Domain::System;
    s.on_pressure_change = [&got_sys](mem::Pressure p){ got_sys.push_back(p); };
    const cardinal::u32 gid = b.register_subsystem(g);
    b.register_subsystem(s);

    // Steady Low → no transition → no callbacks.
    b.refresh();
    CHECK(got_gpu.empty() && got_sys.empty());

    // GPU Low→High: only the GPU subsystem is notified.
    b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::High);
    b.refresh();
    CHECK(got_gpu.size() == sz(1) && got_gpu[0] == mem::Pressure::High);
    CHECK(got_sys.empty());                           // System unchanged

    // Re-tick with the SAME forced tier → steady-state, no new callback.
    b.refresh();
    b.refresh();
    CHECK(got_gpu.size() == sz(1));

    // GPU High→Medium: another transition.
    b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::Medium);
    b.refresh();
    CHECK(got_gpu.size() == sz(2) && got_gpu[1] == mem::Pressure::Medium);

    // last_seen_tier in the report tracks the latest tier for that id.
    // (Bind the returned vector to a NAMED local — find_report points
    // into it, so it must outlive the deref.)
    {
        const auto rg = b.subsystem_reports();
        const bg::SubsystemReport* gr = find_report(rg, gid);
        CHECK(gr != nullptr && gr->last_seen_tier == mem::Pressure::Medium);
    }

    // System Low→Critical: only the System subsystem fires; GPU (still
    // steady Medium) does not.
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Critical);
    b.refresh();
    CHECK(got_sys.size() == sz(1) && got_sys[0] == mem::Pressure::Critical);
    CHECK(got_gpu.size() == sz(2));                   // unaffected

    // Deregister the GPU sub → it no longer receives transitions.
    b.deregister_subsystem(gid);
    b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::Critical);
    b.refresh();                                      // Medium→Critical
    CHECK(got_gpu.size() == sz(2));                   // frozen post-deregister
    const auto rgone = b.subsystem_reports();
    CHECK(find_report(rgone, gid) == nullptr);
}

// ---- AutoScaler<float>: ctor-apply Low, per-tier pick, dtor dereg --
void test_autoscaler_float() {
    bg::Broker b;
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Low);
    b.debug_force_pressure(bg::Domain::Gpu,    mem::Pressure::Low);

    std::vector<float>          vals;
    std::vector<mem::Pressure>  tiers;

    cardinal::u32 saved_id = 0;
    {
        bg::AutoScaler<float>::Config cfg;
        cfg.name           = "render_scale";
        cfg.domain         = bg::Domain::Gpu;
        cfg.value_low      = 1.00f;
        cfg.value_medium   = 0.75f;
        cfg.value_high     = 0.50f;
        cfg.value_critical = 0.25f;
        cfg.apply = [&](const float& v, mem::Pressure t) {
            vals.push_back(v); tiers.push_back(t);
        };
        bg::AutoScaler<float> as(b, std::move(cfg));
        saved_id = as.broker_id();

        // Constructor registers AND applies Low immediately.
        CHECK(saved_id == 1u);                         // fresh broker
        CHECK(vals.size() == sz(1));
        CHECK(apf(vals[0], 1.00f) && tiers[0] == mem::Pressure::Low);
        CHECK(apf(as.current(), 1.00f));
        CHECK(as.current_tier() == mem::Pressure::Low);

        // Steady Low refresh → no extra apply.
        b.refresh();
        CHECK(vals.size() == sz(1));

        // Low→High → value_high.
        b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::High);
        b.refresh();
        CHECK(vals.size() == sz(2));
        CHECK(apf(vals[1], 0.50f) && tiers[1] == mem::Pressure::High);
        CHECK(apf(as.current(), 0.50f));
        CHECK(as.current_tier() == mem::Pressure::High);

        // High→Critical → value_critical.
        b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::Critical);
        b.refresh();
        CHECK(apf(as.current(), 0.25f));
        CHECK(as.current_tier() == mem::Pressure::Critical);

        // Critical→Medium → value_medium.
        b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::Medium);
        b.refresh();
        CHECK(vals.size() == sz(4));
        CHECK(apf(as.current(), 0.75f) && as.current_tier() == mem::Pressure::Medium);

        const auto reps = b.subsystem_reports();      // named: outlives deref
        const bg::SubsystemReport* r = find_report(reps, saved_id);
        CHECK(r != nullptr);
        CHECK(r->name == "render_scale" && r->domain == bg::Domain::Gpu);
        CHECK(r->last_seen_tier == mem::Pressure::Medium);
    }   // ~AutoScaler → deregister

    const auto reps_after = b.subsystem_reports();
    CHECK(find_report(reps_after, saved_id) == nullptr);
    const cardinal::usize before = vals.size();
    b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::High);
    b.refresh();                                       // no live subscriber
    CHECK(vals.size() == before);                      // dtor truly deregistered
}

// ---- AutoScaler<u32>: System domain + integer knob ----------------
void test_autoscaler_u32() {
    bg::Broker b;
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Low);
    b.debug_force_pressure(bg::Domain::Gpu,    mem::Pressure::Low);

    std::vector<cardinal::u32> vals;

    bg::AutoScaler<cardinal::u32>::Config cfg;
    cfg.name           = "slot_pool";
    cfg.domain         = bg::Domain::System;
    cfg.value_low      = 4096u;
    cfg.value_medium   = 2048u;
    cfg.value_high     = 1024u;
    cfg.value_critical = 256u;
    cfg.apply = [&](const cardinal::u32& v, mem::Pressure){ vals.push_back(v); };
    bg::AutoScaler<cardinal::u32> as(b, std::move(cfg));

    CHECK(vals.size() == sz(1) && vals[0] == 4096u);   // ctor applied Low
    CHECK(as.current() == 4096u);

    // GPU pressure must NOT affect a System-domain scaler.
    b.debug_force_pressure(bg::Domain::Gpu, mem::Pressure::Critical);
    b.refresh();
    CHECK(vals.size() == sz(1));                        // ignored — wrong domain
    CHECK(as.current() == 4096u);

    // System Low→Critical drives it.
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::Critical);
    b.refresh();
    CHECK(vals.size() == sz(2) && vals[1] == 256u);
    CHECK(as.current() == 256u);
    CHECK(as.current_tier() == mem::Pressure::Critical);

    // System Critical→High.
    b.debug_force_pressure(bg::Domain::System, mem::Pressure::High);
    b.refresh();
    CHECK(as.current() == 1024u && as.current_tier() == mem::Pressure::High);
}

}  // namespace

int main() {
    test_classify();
    test_broker_registry();
    test_broker_transitions();
    test_autoscaler_float();
    test_autoscaler_u32();

    if (g_fail == 0) {
        cardinal::log::infof("astest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("astest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
