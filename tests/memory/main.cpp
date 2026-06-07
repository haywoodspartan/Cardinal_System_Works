// =============================================================================
// Cardinal — deterministic memory-pressure regression suite.
//
// classify(used_pct, thresholds) is the budget-decision keystone: every
// subsystem's grow / hold / trim / evict reaction and the editor memory
// panel hinge on the used_pct → Pressure mapping. A boundary regression
// silently mis-budgets — never reaching Critical (→ OOM) or always
// Critical (→ refuses allocs). The contract is `>=` per tier, cascaded
// Critical→High→Medium→Low; this suite exhaustively pins it. The OS
// snapshot + wall-clock Monitor have no fixed values (host RAM varies),
// so they get invariant + formula-self-consistency + interval-gate
// checks instead. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/budget/memory.hpp>
#include <cardinal/core/diag/log.hpp>

namespace {

namespace mem = cardinal::memory;
using mem::Pressure;
using mem::PressureThresholds;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("memtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool dbl_close(double a, double b, double eps) {
    const double d = (a > b) ? (a - b) : (b - a);
    return d <= eps;
}
cardinal::u32 tier(Pressure p) { return static_cast<cardinal::u32>(p); }

// ---- classify(): the keystone — exhaustive boundary + monotone ------
void test_classify() {
    const PressureThresholds d{};   // defaults: 60 / 80 / 92

    // Below medium → Low (incl. defensive negative).
    CHECK(mem::classify(-5.0, d)  == Pressure::Low);
    CHECK(mem::classify(0.0,  d)  == Pressure::Low);
    CHECK(mem::classify(59.999, d) == Pressure::Low);
    // Boundaries are INCLUSIVE on the lower edge (>=).
    CHECK(mem::classify(60.0, d)  == Pressure::Medium);   // exactly medium
    CHECK(mem::classify(60.0001, d) == Pressure::Medium);
    CHECK(mem::classify(79.999, d) == Pressure::Medium);
    CHECK(mem::classify(80.0, d)  == Pressure::High);     // exactly high
    CHECK(mem::classify(91.999, d) == Pressure::High);
    CHECK(mem::classify(92.0, d)  == Pressure::Critical); // exactly critical
    CHECK(mem::classify(100.0, d) == Pressure::Critical);
    CHECK(mem::classify(1.0e9, d) == Pressure::Critical);

    // Custom thresholds — same `>=` cascade with different numbers.
    const PressureThresholds c{ 10.0, 20.0, 30.0 };
    CHECK(mem::classify(9.99, c) == Pressure::Low);
    CHECK(mem::classify(10.0, c) == Pressure::Medium);
    CHECK(mem::classify(20.0, c) == Pressure::High);
    CHECK(mem::classify(30.0, c) == Pressure::Critical);
    CHECK(mem::classify(25.0, c) == Pressure::High);

    // Degenerate equal thresholds: at the shared value the cascade's
    // first check (Critical) wins.
    const PressureThresholds eq{ 50.0, 50.0, 50.0 };
    CHECK(mem::classify(49.999, eq) == Pressure::Low);
    CHECK(mem::classify(50.0, eq)   == Pressure::Critical);

    // Caller-misconfigured non-monotone thresholds: documented contract
    // is "Critical checked first" — lock the observable behaviour so a
    // refactor can't silently reorder the cascade.
    const PressureThresholds bad{ 60.0, 80.0, 50.0 };  // critical < medium
    CHECK(mem::classify(55.0, bad) == Pressure::Critical);  // 55 >= 50
    CHECK(mem::classify(49.0, bad) == Pressure::Low);       // < all

    // Sweep: tier is monotonically non-decreasing in used_pct and
    // matches the expected piecewise mapping at every sample.
    cardinal::u32 prev = 0;
    bool mono = true, exact = true;
    for (int k = 0; k <= 2000; ++k) {
        const double u = static_cast<double>(k) * 0.05;   // 0 … 100
        const Pressure p = mem::classify(u, d);
        if (tier(p) < prev) mono = false;
        prev = tier(p);
        const Pressure want =
            (u >= 92.0) ? Pressure::Critical :
            (u >= 80.0) ? Pressure::High     :
            (u >= 60.0) ? Pressure::Medium   : Pressure::Low;
        if (p != want) exact = false;
    }
    CHECK(mono);
    CHECK(exact);
}

// ---- pressure_name() incl. out-of-range ----------------------------
void test_pressure_name() {
    CHECK(cardinal::string(mem::pressure_name(Pressure::Low))      == "Low");
    CHECK(cardinal::string(mem::pressure_name(Pressure::Medium))   == "Medium");
    CHECK(cardinal::string(mem::pressure_name(Pressure::High))     == "High");
    CHECK(cardinal::string(mem::pressure_name(Pressure::Critical)) == "Critical");
    // Out-of-range enum → documented "?" fallback (never UB / empty).
    CHECK(cardinal::string(
        mem::pressure_name(static_cast<Pressure>(99u))) == "?");
}

// ---- snapshots: invariants + load_percent self-consistency ---------
void test_snapshots() {
    const mem::SystemSnapshot s = mem::query_system();
    CHECK(s.total_bytes > 0u);                       // host has RAM
    CHECK(s.available_bytes <= s.total_bytes);
    CHECK(s.load_percent >= 0.0 && s.load_percent <= 100.0);
    if (s.total_bytes > 0u) {
        const double expect = 100.0 *
            (1.0 - static_cast<double>(s.available_bytes) /
                   static_cast<double>(s.total_bytes));
        // Deterministic relative to the snapshot's OWN numbers, even
        // though absolute RAM varies host-to-host.
        CHECK(dbl_close(s.load_percent, expect, 0.5));
    }

    const mem::ProcessSnapshot p = mem::query_process();
    CHECK(p.working_set_bytes > 0u);                 // we're running
    CHECK(p.peak_working_set_bytes >= p.working_set_bytes);
    CHECK(p.private_bytes > 0u);                     // committed memory
}

// ---- Monitor: pre-sample / refresh / interval-gate / accessors -----
void test_monitor() {
    mem::Monitor m;
    // Pre-sample: cached snapshot is default-zero; system_pressure()
    // classifies load_percent 0 → Low (no syscall, no crash).
    CHECK(m.last_system().total_bytes == 0u);
    CHECK(m.system_pressure() == Pressure::Low);

    m.refresh();                                     // force a sample
    const mem::SystemSnapshot s = m.last_system();
    CHECK(s.total_bytes > 0u);
    CHECK(m.last_process().working_set_bytes > 0u);
    // Accessor cross-check: system_pressure() == classify(cached, th).
    const PressureThresholds th{};
    CHECK(m.system_pressure(th) ==
          mem::classify(s.load_percent, th));
    const PressureThresholds th2{ 1.0, 2.0, 3.0 };
    CHECK(m.system_pressure(th2) ==
          mem::classify(s.load_percent, th2));

    // Interval gate: a tick() whose interval hasn't elapsed must NOT
    // resample — the cached snapshot stays byte-identical.
    m.tick(3600000u);                                // 1h: never elapsed
    const mem::SystemSnapshot s2 = m.last_system();
    CHECK(s2.total_bytes == s.total_bytes);
    CHECK(s2.available_bytes == s.available_bytes);
    CHECK(dbl_close(s2.load_percent, s.load_percent, 1e-9));

    // tick(0): interval satisfied → it resamples (no crash, still sane).
    m.tick(0u);
    CHECK(m.last_system().total_bytes > 0u);

    // A fresh Monitor's accessor uses the (zeroed) cache, not a live
    // syscall — classify(0, custom) is still Low here.
    mem::Monitor m2;
    CHECK(m2.system_pressure(th2) == Pressure::Low);
}

}  // namespace

int main() {
    test_classify();
    test_pressure_name();
    test_snapshots();
    test_monitor();

    if (g_fail == 0) {
        cardinal::log::infof("memtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("memtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
