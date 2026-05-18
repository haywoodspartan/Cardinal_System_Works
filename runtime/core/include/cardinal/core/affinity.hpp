#pragma once

#include <cardinal/core/types.hpp>

namespace cardinal::affinity {

// Pin the calling thread to a single logical core (OS-assigned id).
// Returns true on success.
bool pin_current_thread(u32 logical_core_id);

}  // namespace cardinal::affinity
