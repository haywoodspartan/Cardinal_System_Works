#pragma once

// =============================================================================
// Cardinal — chrome://tracing JSON exporter.
//
// FrameScope (cardinal::async::FrameScope) already records per-phase
// timings into the Profiler panel. This layer captures those samples on
// a rolling per-frame basis and dumps them as a chrome://tracing JSON
// file the user can open in:
//   - chrome://tracing
//   - https://ui.perfetto.dev/   (drag the JSON in)
//   - Speedscope                  (https://www.speedscope.app/)
//
// Usage:
//
//     cardinal::trace::begin_capture(120);   // record next 120 frames
//     ... run the engine for ~2 seconds at 60 FPS ...
//     cardinal::trace::end_capture("trace.json");   // writes file
//
// Or capture a single frame on demand:
//
//     {
//         cardinal::trace::FrameCapture _fc;     // RAII
//         engine.tick_one_frame(app);
//     }   // dumps last frame only on exit if dump path was set
//
// Output is the chrome-tracing "Trace Event Format" — duration events
// (ph:"X") with ts (microseconds), dur, name (the FrameScope name),
// pid (process id), tid (worker thread id when available).
// =============================================================================

#include <cardinal/core/types.hpp>

#include <string>

namespace cardinal::trace {

// Begin recording. Subsequent FrameScope samples are buffered until
// `end_capture` is called or the frame budget is reached. The buffer
// is bounded — once it fills, additional samples are dropped (we log
// once that drops are happening). Keep frame_budget reasonable —
// 120 frames at ~50 phases each = ~6000 events = ~600 KiB JSON.
//
// `frame_budget = 0` means "until end_capture is called explicitly".
void begin_capture(u32 frame_budget = 120);

// Finalise + write to disk. Returns true on success. Closing the file
// also resets the in-memory buffer for the next capture.
bool end_capture(const std::string& output_path);

// True between begin_capture and end_capture.
bool capturing() noexcept;

// RAII single-frame capture — start at construction, stop at destruction.
// If `dump_path` is non-empty, writes the JSON on destruction.
class FrameCapture {
public:
    explicit FrameCapture(std::string dump_path = {}) noexcept;
    ~FrameCapture();
    FrameCapture(const FrameCapture&)            = delete;
    FrameCapture& operator=(const FrameCapture&) = delete;
private:
    std::string path_;
    bool        was_capturing_;
};

// Internal — called by FrameScope's destructor when capturing is on.
// Records (name, start_ns, end_ns, tid). Cheap (one mutex + emplace).
namespace detail {
void record_scope(const char* name, i64 start_ns, i64 end_ns, u32 tid) noexcept;
void note_frame_boundary() noexcept;
}  // namespace detail

}  // namespace cardinal::trace
