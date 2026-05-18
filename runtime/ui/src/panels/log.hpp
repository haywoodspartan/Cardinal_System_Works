#pragma once

// =============================================================================
// Studio — log panel + bounded ring sink (free-function helper).
//
// Lifted out of studio.cpp so log-panel UX changes (filtering, grouping,
// per-level colours, file export) edit in isolation. The studio owns one
// LogPanelState across frames and:
//   - registers `state.store` as a cardinal::log::Sink at startup
//   - removes it at shutdown
//   - forwards its `draw_log_panel` virtual call to `panels::log::draw`
//
// Self-contained: the sink is a thread-safe ring buffer, snapshotted into
// scratch vectors on each draw so the UI never holds the sink's mutex.
// =============================================================================

#include <cardinal/core/log.hpp>
#include <cardinal/core/types.hpp>

#include <imgui.h>

#include <cardinal/core/containers.hpp>
#include <cardinal/core/thread.hpp>

namespace cardinal::ui::panels::log_panel {

// Bounded ring sink — receives every cardinal::log emission, keeps the
// latest kCapacity entries, drops older ones. Thread-safe (engine logging
// happens on worker threads; UI snapshots from the main thread).
class Store final : public cardinal::log::Sink {
public:
    struct Entry {
        cardinal::log::Level lvl{cardinal::log::Level::Info};
        cardinal::string          category;
        cardinal::string          message;
    };

    Store() : entries_(kCapacity) {}

    void on_message(cardinal::log::Level lvl, const char* cat, const char* msg) override;
    void snapshot(cardinal::vector<Entry>& out) const;
    void clear();

private:
    static constexpr size_t kCapacity = 4096;

    mutable cardinal::mutex mutex_;
    cardinal::vector<Entry> entries_;
    size_t             head_{0};
    size_t             size_{0};
};

struct State {
    Store                cstore;             // the cardinal::log sink
    ImGuiTextFilter      filter;             // category / message substring
    cardinal::log::Level min_level{cardinal::log::Level::Trace};
    bool                 auto_scroll{true};
    // Scratch buffers reused across frames so steady-state draws don't
    // hit the allocator.
    cardinal::vector<Store::Entry> scratch_entries;
    cardinal::vector<size_t>       scratch_filtered;
};

void draw(const char* title, bool* p_open, State& state);

}  // namespace cardinal::ui::panels::log_panel
