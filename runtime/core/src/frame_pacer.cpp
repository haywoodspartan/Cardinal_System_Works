// =============================================================================
// Cardinal — frame pacer implementation.
// =============================================================================
#include <cardinal/core/frame_pacer.hpp>
#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    // WIN32_LEAN_AND_MEAN comes from the build system already.
    #include <Windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace cardinal::core {

namespace {

using Clock = std::chrono::steady_clock;

class FramePacerImpl final : public FramePacer {
public:
    void set_window(void* native_handle) noexcept override {
#if CARDINAL_PLATFORM_WINDOWS
        hwnd_ = reinterpret_cast<HWND>(native_handle);
#else
        (void)native_handle;
#endif
    }

    void set_limit(PaceContextId ctx, const FrameLimit& limit) override {
        limits_[ctx] = limit;
    }
    FrameLimit limit(PaceContextId ctx) const override {
        auto it = limits_.find(ctx);
        return it != limits_.end() ? it->second : FrameLimit{};
    }
    void clear_limit(PaceContextId ctx) override {
        limits_.erase(ctx);
    }

    bool is_foreground() const noexcept override { return foreground_; }
    float effective_fps_cap() const noexcept override { return effective_cap_; }

    u64 wait_for_next_frame() override {
        // Refresh foreground state. Non-Windows: always foreground.
        foreground_ = check_foreground();

        // Pick the strictest (smallest non-zero) cap across all contexts.
        // Each context contributes its FG or BG cap depending on state.
        float best_cap = 0.0f;
        for (const auto& kv : limits_) {
            const float c = foreground_ ? kv.second.fps_foreground
                                        : kv.second.fps_background;
            if (c <= 0.0f) continue;                 // uncapped contributes nothing
            if (best_cap == 0.0f || c < best_cap) best_cap = c;
        }
        effective_cap_ = best_cap;

        const auto now = Clock::now();
        if (best_cap <= 0.0f) {
            last_frame_start_ = now;
            return 0;
        }

        // Budget = 1 / cap. Deadline is last_frame_start_ + budget. If
        // we're already past the deadline (the last frame overran the
        // budget) just bail with zero sleep so we don't pile up debt.
        const auto budget_ns = std::chrono::nanoseconds(
            static_cast<long long>(1.0e9 / best_cap));
        const auto deadline  = last_frame_start_ + budget_ns;
        u64 slept_us = 0;
        if (now < deadline) {
            const auto wait = deadline - now;
            // Coarse sleep — leave the last 500us for a tight spin so we
            // don't undershoot due to OS scheduler granularity.
            constexpr auto kSpinBudget = std::chrono::microseconds(500);
            if (wait > kSpinBudget) {
                std::this_thread::sleep_for(wait - kSpinBudget);
            }
            // Spin out the remaining sub-millisecond.
            while (Clock::now() < deadline) {
                std::this_thread::yield();
            }
            slept_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(wait).count());
        }
        last_frame_start_ = Clock::now();
        return slept_us;
    }

private:
    bool check_foreground() const noexcept {
#if CARDINAL_PLATFORM_WINDOWS
        if (hwnd_ == nullptr) return true;
        const HWND fg = ::GetForegroundWindow();
        if (fg == nullptr) return false;
        if (fg == hwnd_)   return true;

        // Treat ANY top-level window owned by OUR process as "foreground" —
        // this is what we care about. ImGui multi-viewport torn-out panels
        // are WS_POPUP windows with no owner; GetAncestor(GA_ROOTOWNER)
        // returns the popup itself, so the previous walk-the-owner logic
        // wrongly classified them as "background" and the pacer applied
        // the 30 FPS background cap whenever the user clicked an Inspector
        // or Render Pipeline secondary window. Cross-checking the process
        // id is the correct test: any of our windows in front → foreground.
        DWORD fg_pid = 0;
        ::GetWindowThreadProcessId(fg, &fg_pid);
        DWORD self_pid = 0;
        ::GetWindowThreadProcessId(hwnd_, &self_pid);
        if (fg_pid != 0 && fg_pid == self_pid) return true;

        // Fallback to the older owner-walk for any case we missed.
        HWND walk = fg;
        for (int hops = 0; hops < 8 && walk != nullptr; ++hops) {
            if (walk == hwnd_) return true;
            HWND parent = ::GetAncestor(walk, GA_ROOTOWNER);
            if (parent == walk) break;
            walk = parent;
        }
        return false;
#else
        return true;
#endif
    }

#if CARDINAL_PLATFORM_WINDOWS
    HWND hwnd_{nullptr};
#endif
    std::unordered_map<PaceContextId, FrameLimit> limits_;
    Clock::time_point last_frame_start_{Clock::now()};
    bool              foreground_{true};
    float             effective_cap_{0.0f};
};

}  // namespace

std::unique_ptr<FramePacer> FramePacer::create() {
    return std::make_unique<FramePacerImpl>();
}

}  // namespace cardinal::core
