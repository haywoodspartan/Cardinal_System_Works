set(CARDINAL_CORE_HEADERS
    # ---------------------------------------------------------------------
    # Foundation (root) — types, platform, precompile, umbrella. These are
    # the most heavily-included files in the engine; everything else lives
    # one directory down so the tree is navigable.
    # ---------------------------------------------------------------------
    include/cardinal/core/types.hpp
    include/cardinal/core/typedefs.hpp
    include/cardinal/core/platform.hpp
    include/cardinal/core/precompile.hpp
    include/cardinal/core/std.hpp                # umbrella over std/

    # ---------------------------------------------------------------------
    # std/ — std vocabulary wrappers (single sanctioned include sites per
    # the FOUNDATION RULE). Non-core code uses cardinal:: aliases only.
    # ---------------------------------------------------------------------
    include/cardinal/core/std/algorithm.hpp
    include/cardinal/core/std/any.hpp
    include/cardinal/core/std/atomic.hpp
    include/cardinal/core/std/bit.hpp
    include/cardinal/core/std/cassert.hpp
    include/cardinal/core/std/cctype.hpp
    include/cardinal/core/std/charconv.hpp
    include/cardinal/core/std/chrono.hpp
    include/cardinal/core/std/climits.hpp
    include/cardinal/core/std/cmath.hpp
    include/cardinal/core/std/containers.hpp
    include/cardinal/core/std/cstdarg.hpp
    include/cardinal/core/std/cstdio.hpp
    include/cardinal/core/std/cstdlib.hpp
    include/cardinal/core/std/cstring.hpp
    include/cardinal/core/std/ctime.hpp
    include/cardinal/core/std/filesystem.hpp
    include/cardinal/core/std/fstream.hpp
    include/cardinal/core/std/future.hpp
    include/cardinal/core/std/hash.hpp
    include/cardinal/core/std/limits.hpp
    include/cardinal/core/std/new.hpp
    include/cardinal/core/std/sstream.hpp
    include/cardinal/core/std/string_view.hpp
    include/cardinal/core/std/thread.hpp
    include/cardinal/core/std/traits.hpp
    include/cardinal/core/std/utility.hpp

    # ---------------------------------------------------------------------
    # alloc/ — bump / pool / arena allocators for hot transient paths.
    # ---------------------------------------------------------------------
    include/cardinal/core/alloc/linear_allocator.hpp
    include/cardinal/core/alloc/pool_allocator.hpp
    include/cardinal/core/alloc/arena_allocator.hpp

    # ---------------------------------------------------------------------
    # container/ — cardinal-specific associative containers + concurrent
    # queues.
    # ---------------------------------------------------------------------
    include/cardinal/core/container/flat_set.hpp
    include/cardinal/core/container/flat_map.hpp
    include/cardinal/core/container/queue.hpp

    # ---------------------------------------------------------------------
    # sync/ — concurrency primitives: jobs/async/affinity/topology +
    # autoscaler + frame_pacer + lock/access/worker_thread.
    # ---------------------------------------------------------------------
    include/cardinal/core/sync/jobs.hpp
    include/cardinal/core/sync/async.hpp
    include/cardinal/core/sync/affinity.hpp
    include/cardinal/core/sync/topology.hpp
    include/cardinal/core/sync/autoscaler.hpp
    include/cardinal/core/sync/frame_pacer.hpp
    include/cardinal/core/sync/lock.hpp
    include/cardinal/core/sync/access.hpp
    include/cardinal/core/sync/worker_thread.hpp

    # ---------------------------------------------------------------------
    # clock/ — time / wall_time / stopwatch / RepeatableTimer.
    # ---------------------------------------------------------------------
    include/cardinal/core/clock/time.hpp
    include/cardinal/core/clock/wall_time.hpp
    include/cardinal/core/clock/stopwatch.hpp

    # ---------------------------------------------------------------------
    # os/ — OS abstraction (HAL, IO dispatcher, file watcher, Windows
    # kernel-object attributes, sync file/directory/SEH).
    # ---------------------------------------------------------------------
    include/cardinal/core/os/hal.hpp
    include/cardinal/core/os/io.hpp
    include/cardinal/core/os/file_watcher.hpp
    include/cardinal/core/os/kernel_object_attribute.hpp
    include/cardinal/core/os/file.hpp
    include/cardinal/core/os/directory.hpp
    include/cardinal/core/os/seh.hpp

    # ---------------------------------------------------------------------
    # math/ — math / simd_math / geometry / spatial.
    # ---------------------------------------------------------------------
    include/cardinal/core/math/math.hpp
    include/cardinal/core/math/simd_math.hpp
    include/cardinal/core/math/geom.hpp
    include/cardinal/core/math/spatial.hpp

    # ---------------------------------------------------------------------
    # string/ — fixed-capacity stack strings.
    # ---------------------------------------------------------------------
    include/cardinal/core/string/fixed_string.hpp

    # ---------------------------------------------------------------------
    # diag/ — log, crash reporter, trace export.
    # ---------------------------------------------------------------------
    include/cardinal/core/diag/log.hpp
    include/cardinal/core/diag/crash.hpp
    include/cardinal/core/diag/trace_export.hpp

    # ---------------------------------------------------------------------
    # service/ — service locator + shared_object intern cache.
    # ---------------------------------------------------------------------
    include/cardinal/core/service/global_object_manager.hpp
    include/cardinal/core/service/shared_object.hpp

    # ---------------------------------------------------------------------
    # budget/ — pressure-tier budgets + OS memory snapshot.
    # ---------------------------------------------------------------------
    include/cardinal/core/budget/budget.hpp
    include/cardinal/core/budget/memory.hpp

    # ---------------------------------------------------------------------
    # handle/ — opaque handle types + id generation.
    # ---------------------------------------------------------------------
    include/cardinal/core/handle/handle.hpp
    include/cardinal/core/handle/id_gen.hpp

    # ---------------------------------------------------------------------
    # error/ — Result type + scope-guard.
    # ---------------------------------------------------------------------
    include/cardinal/core/error/result.hpp
    include/cardinal/core/error/scope_guard.hpp

    # ---------------------------------------------------------------------
    # Backwards-compat forwarding stubs at the OLD root-level paths so
    # pre-reorg engine #includes resolve to the new canonical locations
    # without code churn. Remove once consumers migrate.
    # ---------------------------------------------------------------------
    # std wrappers (28 stubs)
    include/cardinal/core/algorithm.hpp
    include/cardinal/core/any.hpp
    include/cardinal/core/atomic.hpp
    include/cardinal/core/bit.hpp
    include/cardinal/core/cassert.hpp
    include/cardinal/core/cctype.hpp
    include/cardinal/core/charconv.hpp
    include/cardinal/core/chrono.hpp
    include/cardinal/core/climits.hpp
    include/cardinal/core/cmath.hpp
    include/cardinal/core/containers.hpp
    include/cardinal/core/cstdarg.hpp
    include/cardinal/core/cstdio.hpp
    include/cardinal/core/cstdlib.hpp
    include/cardinal/core/cstring.hpp
    include/cardinal/core/ctime.hpp
    include/cardinal/core/filesystem.hpp
    include/cardinal/core/fstream.hpp
    include/cardinal/core/future.hpp
    include/cardinal/core/hash.hpp
    include/cardinal/core/limits.hpp
    include/cardinal/core/new.hpp
    include/cardinal/core/sstream.hpp
    include/cardinal/core/string_view.hpp
    include/cardinal/core/thread.hpp
    include/cardinal/core/traits.hpp
    include/cardinal/core/utility.hpp
    # diag/
    include/cardinal/core/log.hpp
    include/cardinal/core/crash.hpp
    include/cardinal/core/trace_export.hpp
    # math/
    include/cardinal/core/math.hpp
    include/cardinal/core/simd_math.hpp
    include/cardinal/core/geom.hpp
    include/cardinal/core/spatial.hpp
    # sync/
    include/cardinal/core/jobs.hpp
    include/cardinal/core/async.hpp
    include/cardinal/core/affinity.hpp
    include/cardinal/core/topology.hpp
    include/cardinal/core/autoscaler.hpp
    include/cardinal/core/frame_pacer.hpp
    # os/
    include/cardinal/core/hal.hpp
    include/cardinal/core/io.hpp
    include/cardinal/core/file_watcher.hpp
    include/cardinal/core/kernel_object_attribute.hpp
    # clock/
    include/cardinal/core/time.hpp
    # budget/
    include/cardinal/core/budget.hpp
    include/cardinal/core/memory.hpp
    # handle/
    include/cardinal/core/handle.hpp
    include/cardinal/core/id_gen.hpp
    # error/
    include/cardinal/core/result.hpp
    include/cardinal/core/scope_guard.hpp
    # alloc / container / service (from prior reorg pass)
    include/cardinal/core/linear_allocator.hpp
    include/cardinal/core/pool_allocator.hpp
    include/cardinal/core/arena_allocator.hpp
    include/cardinal/core/flat_set.hpp
    include/cardinal/core/flat_map.hpp
    include/cardinal/core/global_object_manager.hpp
    include/cardinal/core/shared_object.hpp
)

set(CARDINAL_CORE_SOURCES
    # diag/  (log.cpp embeds the static-init AbortSilencer that kills the
    # modal "Debug Error! abort() has been called" / assert popup in debug
    # builds — folded into log.cpp because that TU is always linked, while
    # a standalone silencer.cpp inside a static lib gets dropped by the
    # linker when nothing references its symbols.)
    src/diag/log.cpp
    src/diag/crash.cpp
    src/diag/trace_export.cpp
    # math/ — one TU per ISA tier; per-source /arch flags below.
    src/math/geom.cpp
    src/math/simd_math_scalar.cpp
    src/math/simd_math_sse42.cpp
    src/math/simd_math_avx.cpp
    src/math/simd_math_avx2.cpp
    src/math/simd_math_avx512.cpp
    src/math/simd_math_dispatch.cpp
    # sync/
    src/sync/jobs.cpp
    src/sync/async.cpp
    src/sync/topology.cpp
    src/sync/frame_pacer.cpp
    src/sync/lock.cpp                       # ProcessLock cross-process named mutex
    src/sync/worker_thread.cpp              # Thread + ThreadManager
    # os/
    src/os/hal.cpp
    src/os/io.cpp
    src/os/file_watcher.cpp
    src/os/kernel_object_attribute.cpp
    src/os/file.cpp                         # File / SyncReadFile / SyncWriteFile
    src/os/directory.cpp                    # Directory enumeration
    src/os/seh.cpp                          # SehManager + minidump bridge
    # clock/
    src/clock/time.cpp
    src/clock/wall_time.cpp                 # Time / CpuTime / CpuUsage
    # budget/
    src/budget/budget.cpp
    src/budget/memory.cpp
    # service/
    src/service/global_object_manager.cpp
)

if(WIN32)
    list(APPEND CARDINAL_CORE_SOURCES
        src/sync/topology_windows.cpp
        src/sync/affinity_windows.cpp
        src/budget/memory_windows.cpp
    )
elseif(UNIX AND NOT APPLE)
    list(APPEND CARDINAL_CORE_SOURCES
        src/sync/topology_linux.cpp
        src/sync/affinity_linux.cpp
        src/budget/memory_linux.cpp
    )
endif()

add_library(cardinal_core STATIC ${CARDINAL_CORE_SOURCES} ${CARDINAL_CORE_HEADERS})
add_library(cardinal::core ALIAS cardinal_core)

target_include_directories(cardinal_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cardinal_core PUBLIC cardinal::compile_flags)

# Per-platform extras for HAL.
if(WIN32)
    # psapi: HAL OS info / process metrics.
    # version: os/file.cpp uses GetFileVersionInfo / VerQueryValue (Win32
    # Version Resource scan) for File::get_version / get_my_version.
    # crash.cpp resolves dbghelp.dll dynamically via LoadLibrary so we
    # don't need to link it here — keeps the core lib free of a hard
    # dependency on dbghelp for hosts that opt out of crash reporting.
    target_link_libraries(cardinal_core PUBLIC psapi version)
elseif(UNIX AND NOT APPLE)
    # pthread for affinity APIs + std::thread + dl for dynamic loader.
    find_package(Threads REQUIRED)
    target_link_libraries(cardinal_core PUBLIC Threads::Threads dl)
endif()

set_target_properties(cardinal_core PROPERTIES
    FOLDER runtime
)

# ----- SIMD math: per-source /arch flags ------------------------------------
#
# Each ISA TU MUST be compiled with the right /arch (MSVC) or -m flag
# (GCC/Clang) so the compiler can emit the corresponding instructions.
# Mixing AVX-512 code into a TU built only with /arch:AVX2 doesn't
# compile (intrinsics aren't in scope); mixing AVX-512 instructions
# into a /arch:AVX2 TU at the wrong source line crashes on CPUs
# without the ISA. Per-source flags keep the boundaries clean.
#
# MSVC /arch flags (each is the FLOOR — code can use everything from
# baseline up through the named ISA):
#   /arch:SSE2     — implicit on x64 (no flag emitted)
#   /arch:AVX      — enables AVX
#   /arch:AVX2     — enables AVX2 + FMA3
#   /arch:AVX512   — enables AVX-512 F + others MSVC's CRT recognises
#
# GCC/Clang use -mavx, -mavx2 -mfma, -mavx512f -mavx512vl, etc. The
# scalar TU stays at baseline x86_64 (SSE2) so it runs on EVERY x64 CPU.
if(MSVC)
    set_source_files_properties(src/math/simd_math_avx.cpp    PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    set_source_files_properties(src/math/simd_math_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/math/simd_math_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/math/simd_math_avx.cpp    PROPERTIES COMPILE_OPTIONS "-mavx")
    set_source_files_properties(src/math/simd_math_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/math/simd_math_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
endif()
