// =============================================================================
// Cardinal — hang-watchdog regression suite.
//
// The watchdog exists because freezes never reach the crash filter: Windows
// logs Event 1002 ("stopped interacting") with no dump and no stack. Here we
// prove (1) a silent heartbeat fires exactly one dump, (2) a poked heartbeat
// never fires, (3) restart re-arms. Wall time ~5 s (inherent to a timer test).
// =============================================================================

#include <cardinal/core/diag/crash.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <chrono>
#include <filesystem>
#include <thread>

namespace {

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("wdtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

}  // namespace

int main() {
    namespace cr = cardinal::crash;
    namespace fs = std::filesystem;

    const fs::path dump_dir = fs::temp_directory_path() / "cardinal_wd_test";
    std::error_code ec;
    fs::create_directories(dump_dir, ec);

    cr::CrashConfig cfg;
    cfg.dump_dir = dump_dir.string();
    cfg.detail   = cr::CrashConfig::Detail::Small;   // keep the test dump tiny
    cr::install(cfg);

    // 1) Silent heartbeat -> fires once, dump lands on disk.
    CHECK(cr::watchdog_start(1));
    CHECK(!cr::watchdog_fired());
    {
        // Poll up to 6 s — the stall threshold is 1 s + 250 ms sampling.
        bool fired = false;
        for (int i = 0; i < 60 && !fired; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            fired = cr::watchdog_fired();
        }
        CHECK(fired);
    }
#if CARDINAL_PLATFORM_WINDOWS
    CHECK(!cr::last_dump_path().empty());
    CHECK(fs::exists(cr::last_dump_path()));
#endif
    cr::watchdog_stop();

    // 2) Poked heartbeat -> never fires (2.4 s of life signs vs 1 s stall).
    CHECK(cr::watchdog_start(1));                    // restart re-arms (fired reset)
    CHECK(!cr::watchdog_fired());
    for (int i = 0; i < 16; ++i) {
        cr::watchdog_poke();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    CHECK(!cr::watchdog_fired());
    cr::watchdog_stop();

    // 3) stop() is idempotent; poke after stop is harmless.
    cr::watchdog_stop();
    cr::watchdog_poke();

    cr::uninstall();
    fs::remove_all(dump_dir, ec);                    // best-effort cleanup

    if (g_fail == 0) {
        cardinal::log::infof("wdtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("wdtest", "%d / %d checks FAILED", g_fail, g_checks);
    return 1;
}
