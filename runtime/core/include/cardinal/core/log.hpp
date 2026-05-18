#pragma once

// =============================================================================
// Cardinal logging.
//
// Single global log bus with pluggable sinks. The engine never writes
// directly to stdout/stderr — every diagnostic flows through here so the
// editor can mirror it into the Studio Log panel, files, network, etc.
//
// Default sink is stderr (installed in the first call into the log API).
// Add additional sinks via add_sink(); remove with remove_sink().
//
// All variadic calls are printf-style. Category is a short tag
// ("rhi", "ui/studio", "jobs") used for filtering in the UI.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cstdarg>

namespace cardinal::log {

enum class Level : u32 {
    Trace = 0,   // very verbose, dev-only
    Info  = 1,   // normal lifecycle messages
    Warn  = 2,   // unexpected but recoverable
    Error = 3,   // failure path
};

const char* level_name(Level l) noexcept;

// Implement to receive log messages. Sinks are dispatched on the calling
// thread under a global mutex — keep on_message() short.
class Sink {
public:
    virtual ~Sink() = default;
    virtual void on_message(Level lvl, const char* category, const char* msg) = 0;
};

// Sink lifetime is the caller's. Adding the same pointer twice is a no-op.
void add_sink(Sink* sink);
void remove_sink(Sink* sink);

// Drop messages whose level is below this. Default: Trace (everything).
void set_min_level(Level l);

// Variadic emission — formats once, then dispatches to all sinks.
void emitf (Level lvl, const char* category, const char* fmt, ...);
void vemitf(Level lvl, const char* category, const char* fmt, va_list args);

// Convenience wrappers.
inline void tracef(const char* cat, const char* fmt, ...) {
    va_list a; va_start(a, fmt); vemitf(Level::Trace, cat, fmt, a); va_end(a);
}
inline void infof (const char* cat, const char* fmt, ...) {
    va_list a; va_start(a, fmt); vemitf(Level::Info,  cat, fmt, a); va_end(a);
}
inline void warnf (const char* cat, const char* fmt, ...) {
    va_list a; va_start(a, fmt); vemitf(Level::Warn,  cat, fmt, a); va_end(a);
}
inline void errorf(const char* cat, const char* fmt, ...) {
    va_list a; va_start(a, fmt); vemitf(Level::Error, cat, fmt, a); va_end(a);
}

}  // namespace cardinal::log
