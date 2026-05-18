// Single TU that compiles VMA. Isolated so we can:
//   1) crank down warnings on the noisy header without affecting the rest
//   2) avoid bloating rhi_vulkan.cpp's compile time with VMA's ~10K LOC

#include <cardinal/core/platform.hpp>

#if CARDINAL_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR 1
#elif CARDINAL_PLATFORM_LINUX
    #define VK_USE_PLATFORM_XCB_KHR 1
#endif

#include <volk.h>

#if CARDINAL_COMPILER_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4100)  // unreferenced formal parameter
    #pragma warning(disable: 4127)  // conditional expression is constant
    #pragma warning(disable: 4189)  // local var initialized but not referenced
    #pragma warning(disable: 4324)  // padding due to alignas
    #pragma warning(disable: 4505)  // unreferenced local function
    #pragma warning(disable: 4701)  // potentially uninitialized local
    #pragma warning(disable: 4703)  // potentially uninitialized local pointer
    #pragma warning(disable: 4244)  // conversion narrowing
    #pragma warning(disable: 4267)  // size_t→u32
    #pragma warning(disable: 4365)  // signed/unsigned mismatch
    #pragma warning(disable: 4623)  // default ctor implicitly deleted
    #pragma warning(disable: 4625)
    #pragma warning(disable: 4626)
    #pragma warning(disable: 5026)
    #pragma warning(disable: 5027)
#endif

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS  0   // we provide via volk
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#if CARDINAL_COMPILER_MSVC
    #pragma warning(pop)
#endif
