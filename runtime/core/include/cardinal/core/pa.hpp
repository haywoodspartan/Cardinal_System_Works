#pragma once

// =============================================================================
// Cardinal core — pa umbrella. Pulls in every pa::* sub-header so a porting
// site can write `#include <cardinal/core/pa.hpp>` and get the full
// PaLock / PaFile / PaDirectory / PaQueue / PaSeh / PaString / PaThread /
// PaTime / PaTimer surface modernised onto C++20.
//
// Per-component headers live under cardinal/core/pa/ so individual modules
// can pull just the slice they need.
// =============================================================================

#include <cardinal/core/pa/lock.hpp>
#include <cardinal/core/pa/access.hpp>
#include <cardinal/core/pa/string.hpp>
#include <cardinal/core/pa/queue.hpp>
#include <cardinal/core/pa/wall_time.hpp>
#include <cardinal/core/pa/stopwatch.hpp>
#include <cardinal/core/pa/file.hpp>
#include <cardinal/core/pa/directory.hpp>
#include <cardinal/core/pa/thread.hpp>
#include <cardinal/core/pa/seh.hpp>
