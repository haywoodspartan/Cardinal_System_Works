#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_LINUX

#include <cardinal/core/sync/affinity.hpp>

#include <pthread.h>
#include <sched.h>

namespace cardinal::affinity {

bool pin_current_thread(u32 logical_core_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(logical_core_id), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

}  // namespace cardinal::affinity

#endif  // CARDINAL_PLATFORM_LINUX
