#pragma once

// =============================================================================
// Cardinal — RAII scope guard for last-line cleanup.
//
//   auto guard = cardinal::core::on_scope_exit([&]{ free(buf); });
//
// Use the CARDINAL_DEFER macro for the common pattern of "do this when the
// current scope ends, no matter how it ends":
//
//   void f() {
//       FILE* fp = fopen(path, "rb");
//       CARDINAL_DEFER([&]{ if (fp) fclose(fp); });
//       ...
//   }
//
// The guard can be `dismiss()`-ed to suppress execution (for transactional
// "commit on success" patterns).
// =============================================================================

#include <utility>        // std::move

namespace cardinal::core {

template <typename Fn>
class ScopeGuard {
public:
    explicit ScopeGuard(Fn&& fn) noexcept : fn_(std::move(fn)) {}
    ScopeGuard(ScopeGuard&& o) noexcept
        : fn_(std::move(o.fn_)), live_(o.live_) { o.live_ = false; }
    ~ScopeGuard() noexcept { if (live_) fn_(); }

    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&)      = delete;

    void dismiss() noexcept { live_ = false; }

private:
    Fn   fn_;
    bool live_{true};
};

template <typename Fn>
[[nodiscard]] inline ScopeGuard<Fn> on_scope_exit(Fn&& fn) noexcept {
    return ScopeGuard<Fn>(std::forward<Fn>(fn));
}

}  // namespace cardinal::core

// Token-paste helper so two CARDINAL_DEFER calls in one scope don't collide
// on the local-variable name.
#define CARDINAL_DEFER_IMPL2(line, fn) \
    auto _cardinal_defer_##line = ::cardinal::core::on_scope_exit(fn)
#define CARDINAL_DEFER_IMPL1(line, fn) CARDINAL_DEFER_IMPL2(line, fn)
#define CARDINAL_DEFER(fn)             CARDINAL_DEFER_IMPL1(__LINE__, fn)
