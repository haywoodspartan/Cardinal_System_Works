// =============================================================================
// Cardinal — deterministic Result<T,E> regression suite.
//
// Result is the engine's exception-free value/error return, used across
// net / cook / serial / pack / import. T and E share a hand-managed
// union, so the headline property is LIFETIME correctness: each
// contained object constructed exactly once and destroyed exactly once
// across copy / move / assign / scope-exit. A regression there silently
// leaks or double-frees on every error path. An instrumented payload
// makes the accounting exact; results that hold the engine's own
// cardinal::string / cardinal::unique_ptr exercise non-trivial + move-
// only behaviour. Exit 0 = all pass.
// =============================================================================

#include <cardinal/core/result.hpp>
#include <cardinal/core/log.hpp>

#include <utility>          // std::move

namespace {

namespace co = cardinal::core;
template <class T, class E> using Result = co::Result<T, E>;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("resulttest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

// Instrumented payload — exact construct/destruct accounting. Result
// manages the union via placement-new + explicit dtor (its operator=
// does ~Result then placement-new), so only ctor / copy-ctor / move-
// ctor / dtor are exercised; `live()` must return to 0.
struct Trk {
    static int ctor, cpy, mov, dtor;
    int  v{0};
    bool moved{false};
    Trk()              noexcept { ++ctor; }
    explicit Trk(int x) noexcept : v(x) { ++ctor; }
    Trk(const Trk& o)  noexcept : v(o.v) { ++cpy; }
    Trk(Trk&& o)       noexcept : v(o.v) { ++mov; o.moved = true; o.v = -1; }
    Trk& operator=(const Trk&) = delete;   // Result never assigns the payload
    Trk& operator=(Trk&&)      = delete;
    ~Trk() { ++dtor; }
    static int  live()  { return ctor + cpy + mov - dtor; }
    static void reset() { ctor = cpy = mov = dtor = 0; }
};
int Trk::ctor = 0, Trk::cpy = 0, Trk::mov = 0, Trk::dtor = 0;

// ---- basic ok / err / inspection -----------------------------------
void test_basic() {
    Result<int, cardinal::string> a(co::ok, 42);
    CHECK(a.is_ok() && !a.is_err());
    CHECK(static_cast<bool>(a));
    CHECK(a.value() == 42);
    CHECK(*a == 42);
    CHECK(a.value_or(7) == 42);

    Result<int, cardinal::string> b(co::err, cardinal::string("boom"));
    CHECK(b.is_err() && !b.is_ok());
    CHECK(!static_cast<bool>(b));
    CHECK(b.error() == "boom");
    CHECK(b.value_or(7) == 7);

    Result<cardinal::string, int> s(co::ok, cardinal::string("hello"));
    CHECK(s.is_ok());
    CHECK(*s == "hello");
    CHECK(s->size() == 5u);                 // operator->
    const auto& cs = s;
    CHECK(cs.value() == "hello");           // const value()

    // Implicit value-side construction (E not constructible from the arg).
    Result<int, cardinal::string> imp = 99;
    CHECK(imp.is_ok() && *imp == 99);
    Result<cardinal::string, int> imp2 = cardinal::string("x");
    CHECK(imp2.is_ok() && *imp2 == "x");
}

// ---- copy: deep + independent + balanced ---------------------------
void test_copy() {
    Trk::reset();
    {
        Result<Trk, int> a(co::ok, Trk(5));
        CHECK(Trk::live() == 1);
        CHECK(a.value().v == 5);
        {
            Result<Trk, int> b = a;             // copy-ctor (ok side)
            CHECK(Trk::live() == 2);
            CHECK(b.is_ok() && b.value().v == 5);
            b.value().v = 123;                  // independent storage
            CHECK(a.value().v == 5);
        }
        CHECK(Trk::live() == 1);                // b destroyed cleanly
        // Self-copy is guarded (this != &o) — value preserved, balanced.
        a = a;
        CHECK(a.value().v == 5 && Trk::live() == 1);
    }
    CHECK(Trk::live() == 0);
    CHECK(Trk::dtor == Trk::ctor + Trk::cpy + Trk::mov);   // no leak/dup

    // Copy of an err-side Result copies E, never T.
    Trk::reset();
    {
        Result<int, Trk> e(co::err, Trk(8));
        Result<int, Trk> e2 = e;
        CHECK(e2.is_err() && e2.error().v == 8);
        CHECK(Trk::live() == 2);
    }
    CHECK(Trk::live() == 0);
}

// ---- move: transfers, source valid, balanced -----------------------
void test_move() {
    Trk::reset();
    {
        Result<Trk, int> a(co::ok, Trk(11));
        Result<Trk, int> b = std::move(a);      // move-ctor
        CHECK(b.is_ok() && b.value().v == 11);
        CHECK(a.is_ok());                       // discriminant unchanged
        CHECK(a.value().moved);                 // payload was moved-from
        CHECK(Trk::mov >= 1);
    }
    CHECK(Trk::live() == 0);                     // both destroyed once

    // Move-only payload via the engine's own vocab.
    {
        Result<cardinal::unique_ptr<int>, int> r(
            co::ok, cardinal::make_unique<int>(7));
        CHECK(r.is_ok() && r.value() != nullptr && *r.value() == 7);
        Result<cardinal::unique_ptr<int>, int> r2 = std::move(r);
        CHECK(r2.is_ok() && r2.value() != nullptr && *r2.value() == 7);
        CHECK(r.value() == nullptr);            // ownership transferred
    }
}

// ---- assignment across the active member ---------------------------
void test_assign() {
    Trk::reset();
    {
        Result<Trk, Trk> a(co::ok, Trk(1));
        CHECK(a.is_ok() && a.value().v == 1);
        a = Result<Trk, Trk>(co::err, Trk(2));  // ok → err: old T dtor'd
        CHECK(a.is_err() && a.error().v == 2);
        a = Result<Trk, Trk>(co::ok, Trk(3));   // err → ok: old E dtor'd
        CHECK(a.is_ok() && a.value().v == 3);
        Result<Trk, Trk> mv(co::err, Trk(4));
        a = std::move(mv);                      // move-assign, ok → err
        CHECK(a.is_err() && a.error().v == 4);
    }
    CHECK(Trk::live() == 0);
    CHECK(Trk::dtor == Trk::ctor + Trk::cpy + Trk::mov);
}

// ---- Result<void, E> -----------------------------------------------
void test_void() {
    Result<void, cardinal::string> ok(co::ok);
    CHECK(ok.is_ok() && static_cast<bool>(ok));

    Result<void, cardinal::string> e(co::err, cardinal::string("nope"));
    CHECK(e.is_err() && !static_cast<bool>(e));
    CHECK(e.error() == "nope");

    Trk::reset();
    {
        Result<void, Trk> a(co::err, Trk(5));
        Result<void, Trk> b = a;                // copy err
        CHECK(b.is_err() && b.error().v == 5);
        Result<void, Trk> c = std::move(a);     // move err
        CHECK(c.is_err() && c.error().v == 5);
        Result<void, Trk> okv(co::ok);          // ok: no E constructed
        b = okv;                                // err → ok (E dtor'd)
        CHECK(b.is_ok());
        b = Result<void, Trk>(co::err, Trk(6)); // ok → err
        CHECK(b.is_err() && b.error().v == 6);
    }
    CHECK(Trk::live() == 0);
    CHECK(Trk::dtor == Trk::ctor + Trk::cpy + Trk::mov);
}

}  // namespace

int main() {
    test_basic();
    test_copy();
    test_move();
    test_assign();
    test_void();

    // Global ledger: no payload leaked or double-destroyed anywhere.
    const bool balanced =
        (Trk::dtor == Trk::ctor + Trk::cpy + Trk::mov);
    ::check_impl(balanced, "global Trk lifetime balanced", __LINE__);
    ::check_impl(Trk::live() == 0, "global Trk live()==0", __LINE__);

    if (g_fail == 0) {
        cardinal::log::infof("resulttest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("resulttest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
