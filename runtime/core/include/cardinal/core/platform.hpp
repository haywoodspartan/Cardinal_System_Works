#pragma once

// ----- Platform -------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
    #define CARDINAL_PLATFORM_WINDOWS 1
#else
    #define CARDINAL_PLATFORM_WINDOWS 0
#endif

#if defined(__linux__)
    #define CARDINAL_PLATFORM_LINUX 1
#else
    #define CARDINAL_PLATFORM_LINUX 0
#endif

// ----- Compiler -------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
    #define CARDINAL_COMPILER_MSVC 1
#else
    #define CARDINAL_COMPILER_MSVC 0
#endif

#if defined(__clang__)
    #define CARDINAL_COMPILER_CLANG 1
#else
    #define CARDINAL_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #define CARDINAL_COMPILER_GCC 1
#else
    #define CARDINAL_COMPILER_GCC 0
#endif

// ----- Architecture ---------------------------------------------------------
#if defined(_M_X64) || defined(__x86_64__)
    #define CARDINAL_ARCH_X64 1
#else
    #define CARDINAL_ARCH_X64 0
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
    #define CARDINAL_ARCH_ARM64 1
#else
    #define CARDINAL_ARCH_ARM64 0
#endif

// ----- Hot/cold inline ------------------------------------------------------
#if CARDINAL_COMPILER_MSVC
    #define CARDINAL_FORCEINLINE __forceinline
    #define CARDINAL_NOINLINE    __declspec(noinline)
#else
    #define CARDINAL_FORCEINLINE inline __attribute__((always_inline))
    #define CARDINAL_NOINLINE    __attribute__((noinline))
#endif

// ----- Branch hints ---------------------------------------------------------
#define CARDINAL_LIKELY   [[likely]]
#define CARDINAL_UNLIKELY [[unlikely]]

// ----- Cache line -----------------------------------------------------------
namespace cardinal {
inline constexpr unsigned long long cache_line_size = 64;
}  // namespace cardinal

// ----- Cache locality annotations -------------------------------------------
//
// CARDINAL_READ_MOSTLY
//   For globals + class fields that are written ONCE at boot/init and read
//   on every hot path. Borrowed from the Linux kernel pattern: place such
//   variables in a dedicated section so the cache lines they occupy are
//   never invalidated by writes to nearby (frequently-mutated) variables.
//
//     - GCC/Clang : __attribute__((section(".data.cardinal_rmo")))
//                   The linker groups everything in this section together,
//                   so the read-mostly cache lines pack tightly and stay
//                   warm. Limitation: only works on variables with static
//                   storage duration. No-op when applied elsewhere.
//
//     - MSVC      : __declspec(allocate(".cardinal_rmo")) requires a
//                   preceding `#pragma section`. Restricting the macro to
//                   compile-time-initialised variables on MSVC is awkward,
//                   so we degrade to CARDINAL_CACHE_ALIGNED there. Same
//                   intent (no false sharing); not the same packing trick.
//
// CARDINAL_CACHE_ALIGNED
//   alignas(cache_line_size). Use on individual hot variables (or whole
//   structs) that need their own line so adjacent writes don't bounce them
//   between cores. Works everywhere.
//
// CARDINAL_HOT / CARDINAL_COLD
//   Function-level hints. Tell the compiler "this is on every frame's
//   critical path" or "rarely called — okay to optimise for size, place
//   in cold-text section away from the hot icache." MSVC ignores both
//   gracefully.
//
// Usage:
//   CARDINAL_READ_MOSTLY static const OsInfo info = build_os_info();
//   CARDINAL_CACHE_ALIGNED std::atomic<u64> hot_counter{0};
//   CARDINAL_HOT  void parallel_for_chunk_runner(...);
//   CARDINAL_COLD void rarely_called_diagnostics(...);
// =============================================================================
#if CARDINAL_COMPILER_GCC || CARDINAL_COMPILER_CLANG
    #define CARDINAL_READ_MOSTLY __attribute__((section(".data.cardinal_rmo")))
    #define CARDINAL_HOT         __attribute__((hot))
    #define CARDINAL_COLD        __attribute__((cold))
#elif CARDINAL_COMPILER_MSVC
    // No-op section attribute on MSVC: __declspec(allocate) is brittle for
    // runtime-initialised globals (which most of ours are). Fall back to
    // alignas — still buys us "no false sharing with neighbouring writes."
    #define CARDINAL_READ_MOSTLY alignas(::cardinal::cache_line_size)
    #define CARDINAL_HOT
    #define CARDINAL_COLD
#else
    #define CARDINAL_READ_MOSTLY
    #define CARDINAL_HOT
    #define CARDINAL_COLD
#endif

#define CARDINAL_CACHE_ALIGNED alignas(::cardinal::cache_line_size)

// Prefetch hint — for places where you know the next loop iteration's
// address well in advance (e.g. walking a linked list). Cheap on x86,
// no-op-safe on platforms without it.
#if CARDINAL_COMPILER_GCC || CARDINAL_COMPILER_CLANG
    #define CARDINAL_PREFETCH(addr) __builtin_prefetch((addr))
#elif CARDINAL_COMPILER_MSVC
    #include <intrin.h>
    #define CARDINAL_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
    #define CARDINAL_PREFETCH(addr) ((void)(addr))
#endif
