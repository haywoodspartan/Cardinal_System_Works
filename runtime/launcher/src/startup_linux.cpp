// =============================================================================
// Cardinal startup menu — POSIX stub.
//
// Until WSI lands on Linux, the launcher just returns Auto. A proper GTK or
// Qt-less native dialog (or a tiny SDL2-based one) will replace this when
// Linux WSI is wired in.
// =============================================================================

#include <cardinal/launcher/startup.hpp>
#include <cardinal/core/platform.hpp>

#if !CARDINAL_PLATFORM_WINDOWS

namespace cardinal::launcher {

BackendChoice show_startup_menu(const StartupOptions& /*opts*/) {
    return BackendChoice::Auto;
}

}  // namespace cardinal::launcher

#endif
