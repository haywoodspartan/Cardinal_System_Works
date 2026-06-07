#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#include <cardinal/core/sync/affinity.hpp>

#include <Windows.h>

namespace cardinal::affinity {

bool pin_current_thread(u32 logical_core_id) {
    GROUP_AFFINITY a{};
    a.Group = static_cast<WORD>(logical_core_id / 64u);
    a.Mask  = static_cast<KAFFINITY>(1ULL) << (logical_core_id % 64u);
    return SetThreadGroupAffinity(GetCurrentThread(), &a, nullptr) != 0;
}

}  // namespace cardinal::affinity

#endif  // CARDINAL_PLATFORM_WINDOWS
