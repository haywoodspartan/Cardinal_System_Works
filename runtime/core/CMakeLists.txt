set(CARDINAL_CORE_HEADERS
    # Engine-wide precompiled header. Any module can opt-in via:
    #   target_precompile_headers(<target> PRIVATE
    #       <cardinal/core/precompile.hpp>)
    # Foundation rule still holds: std headers stay INSIDE cardinal::core,
    # this header re-exports the foundation's container / utility / math /
    # algorithm surface so downstream modules don't restate it per .cpp.
    include/cardinal/core/precompile.hpp
    include/cardinal/core/types.hpp
    include/cardinal/core/platform.hpp
    include/cardinal/core/topology.hpp
    include/cardinal/core/affinity.hpp
    include/cardinal/core/jobs.hpp
    include/cardinal/core/log.hpp
    include/cardinal/core/frame_pacer.hpp
    include/cardinal/core/handle.hpp
    include/cardinal/core/time.hpp
    include/cardinal/core/result.hpp
    include/cardinal/core/scope_guard.hpp
    include/cardinal/core/hash.hpp
    include/cardinal/core/id_gen.hpp
    include/cardinal/core/math.hpp
    include/cardinal/core/spatial.hpp
    include/cardinal/core/memory.hpp
    include/cardinal/core/budget.hpp
    include/cardinal/core/async.hpp
    include/cardinal/core/autoscaler.hpp
    # HAL (file I/O, dynamic library, OS info, mono clock, CPU features) +
    # IO Dispatcher (priority async I/O on top of the worker pool) — folded
    # into core so every dependent gets them transitively without separate
    # cardinal::hal / cardinal::io link targets.
    include/cardinal/core/hal.hpp
    include/cardinal/core/io.hpp
    # Geometry math (AABB / OBB / Sphere / Capsule / Plane / Frustum /
    # Triangle / Polygon2D / Ray / Hit / AabbBvh / DynamicMesh) lives here
    # in cardinal::core::geom. Used by physics / partition / level / nav /
    # ai — every one of those modules already depends on core.
    include/cardinal/core/geom.hpp
    # Crash reporter — Win32 SEH unhandled-exception filter that writes a
    # minidump + text log on uncaught exceptions. Stub on Linux for now.
    include/cardinal/core/crash.hpp
    # Trace export — chrome://tracing-compatible JSON dump of FrameScope
    # samples for offline profiling.
    include/cardinal/core/trace_export.hpp
    # File watcher — event-driven file change notifications (Win32
    # ReadDirectoryChangesW today; inotify on Linux when WSI lands).
    include/cardinal/core/file_watcher.hpp
    # SIMD math — runtime-dispatched vector ops + std::vector aliases.
    include/cardinal/core/simd_math.hpp
    include/cardinal/core/typedefs.hpp
    # Windows kernel-object attribute wrapper — RAII over OBJECT_ATTRIBUTES
    # for NT API calls (NtCreateFile/Event/Mutant/Key, etc.). Builds as an
    # empty no-op type on non-Windows so cross-platform code can hold it
    # unconditionally without #ifdef ladders at the use site.
    include/cardinal/core/kernel_object_attribute.hpp
    # Flat (sorted-vector-backed) associative containers — cache-tight
    # std::set / std::map drop-ins for build-once-read-many tables, small
    # N, and iteration-heavy paths. Modernised port of Pearl Abyss
    # BinarySet/BinaryMap (binary-search lookup on a flat array).
    include/cardinal/core/flat_set.hpp
    include/cardinal/core/flat_map.hpp
    # Bump / pool allocators — per-frame scratch arena + fixed-size slab
    # pool. The bread-and-butter allocators for hot transient paths
    # (graph nodes, RHI command staging, particle pool, etc.). Header-only.
    include/cardinal/core/linear_allocator.hpp
    include/cardinal/core/pool_allocator.hpp
    # STL-allocator adapter that forwards allocate() to a LinearAllocator
    # — used to back std::vector / cardinal::vector with arena memory
    # transparently. deallocate() is a no-op; reset the arena en masse.
    include/cardinal/core/arena_allocator.hpp
    # Service locator for engine-wide singletons + shared_object<T>
    # flyweight cache (find-or-create with weak_ptr-backed auto-eviction).
    include/cardinal/core/global_object_manager.hpp
    include/cardinal/core/shared_object.hpp
    # Pearl Abyss Pa* surface modernised onto C++20. Umbrella + per-component
    # sub-headers under cardinal/core/pa/ — used as the porting bridge when
    # bringing CrimsonDesert-style server / tooling code into cardinal::core.
    include/cardinal/core/pa.hpp
    include/cardinal/core/pa/lock.hpp
    include/cardinal/core/pa/access.hpp
    include/cardinal/core/pa/string.hpp
    include/cardinal/core/pa/queue.hpp
    include/cardinal/core/pa/wall_time.hpp
    include/cardinal/core/pa/stopwatch.hpp
    include/cardinal/core/pa/file.hpp
    include/cardinal/core/pa/directory.hpp
    include/cardinal/core/pa/thread.hpp
    include/cardinal/core/pa/seh.hpp
)

set(CARDINAL_CORE_SOURCES
    src/topology.cpp
    src/jobs.cpp
    src/log.cpp
    src/frame_pacer.cpp
    src/time.cpp
    src/memory.cpp
    src/budget.cpp
    src/async.cpp
    src/hal.cpp
    src/io.cpp
    src/geom.cpp
    src/crash.cpp
    src/trace_export.cpp
    src/file_watcher.cpp
    # SIMD math — one TU per ISA tier, each compiled with its own
    # /arch flag below. Dispatcher binds the highest-tier ops the
    # CPU supports at static init.
    src/simd_math_scalar.cpp
    src/simd_math_sse42.cpp
    src/simd_math_avx.cpp
    src/simd_math_avx2.cpp
    src/simd_math_avx512.cpp
    src/simd_math_dispatch.cpp
    # KernelObjectAttribute — single TU with internal #if CARDINAL_PLATFORM_WINDOWS
    # gate (Windows path = full impl over OBJECT_ATTRIBUTES; non-Windows path =
    # no-op stubs so the symbol set is identical across platforms).
    src/kernel_object_attribute.cpp
    # Pa* port — non-template TUs (lock-process / file-IO / directory-fs /
    # thread-impl / wall-time / SEH-minidump). Templates stay header-only.
    src/pa/lock.cpp
    src/pa/file.cpp
    src/pa/directory.cpp
    src/pa/thread.cpp
    src/pa/wall_time.cpp
    src/pa/seh.cpp
    # GlobalObjectManager — service locator for engine-wide singletons.
    src/global_object_manager.cpp
)

if(WIN32)
    list(APPEND CARDINAL_CORE_SOURCES
        src/topology_windows.cpp
        src/affinity_windows.cpp
        src/memory_windows.cpp
    )
elseif(UNIX AND NOT APPLE)
    list(APPEND CARDINAL_CORE_SOURCES
        src/topology_linux.cpp
        src/affinity_linux.cpp
        src/memory_linux.cpp
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
    # version: pa/file.cpp uses GetFileVersionInfo / VerQueryValue (Win32
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
    # SSE4.2 + SSE4A use intrinsics that compile under the x64 baseline,
    # so no extra /arch flag (would set a higher floor than we want).
    set_source_files_properties(src/simd_math_avx.cpp    PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    set_source_files_properties(src/simd_math_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/simd_math_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/simd_math_avx.cpp    PROPERTIES COMPILE_OPTIONS "-mavx")
    set_source_files_properties(src/simd_math_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/simd_math_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
endif()
