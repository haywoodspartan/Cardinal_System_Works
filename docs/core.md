# Cardinal Core — Foundation Reference

`cardinal::core` is the bottom of the stack — the single module every other
module depends on. It owns the engine's vocabulary types, memory, containers,
math, OS/timing/sync primitives, and a small set of hand-rolled, dependency-free
utilities. Public headers live in `runtime/core/include/cardinal/core/`.

For the overall architecture see [engine.md](engine.md).

---

## 1. The Foundation Rule

> **Base C++ headers are referenced in exactly one place — `cardinal::core`.**
> Every other module uses `cardinal::` typedefs and the wrappers in
> `cardinal/core/std/`, never `<vector>`/`<string>`/`<cmath>` directly.

This keeps a single canonical layout/ABI across the engine and makes
allocator/standard-library swaps a one-module change. The wrappers are thin
aliases:

```cpp
#include <cardinal/core/types.hpp>          // u8/u16/u32/u64, i*, f32/f64, usize
#include <cardinal/core/std/containers.hpp> // cardinal::vector, unordered_map, ...
#include <cardinal/core/std/string.hpp>     // cardinal::string
#include <cardinal/core/std/cmath.hpp>      // cardinal::sqrt, sin, ...

cardinal::vector<cardinal::u32> ids;
cardinal::string name = "actor";
```

`cardinal/core/std/` mirrors the standard headers it funnels:
`containers`, `string`, `cmath`, `cstdio`, `cstdlib`, `cstring`, `cctype`,
`filesystem`, `fstream`, `sstream`, `algorithm`, `utility`, `memory`,
`future`, etc. (`cardinal::fs` aliases `std::filesystem`).

---

## 2. Vocabulary types

`types.hpp` / `typedefs.hpp` provide the fixed-width integer + float typedefs
(`u8 u16 u32 u64 i8…i64 f32 f64 usize`) and the smart-pointer/string aliases
(`cardinal::unique_ptr`, `shared_ptr`, `make_unique`, `string`). Use these
everywhere instead of the raw `std` names.

Small vocabulary helpers live alongside as single headers: `flags.hpp`
(type-safe bit flags), `inplace_function.hpp` (fixed-capacity `std::function`),
`easing.hpp`, `noise.hpp`, `rng.hpp`.

---

## 3. Memory — `alloc/`

Explicit, data-oriented allocation. Two primary allocators:

- **`LinearAllocator`** — atomic bump arena. O(1) allocation, no per-object
  free; reclaim with `ScopedMarker` (RAII save/restore) or a full `reset()`.
  Used for per-frame scratch (e.g. RHI barrier staging).
- **`PoolAllocator<T>`** — mutex-guarded slab + free list for fixed-size `T`;
  stable addresses, fast acquire/release.

```cpp
cardinal::core::LinearAllocator scratch{64 * 1024};
{
    cardinal::core::LinearAllocator::ScopedMarker m{scratch};
    void* p = scratch.allocate(256, 16);
    // ... use p ...
}   // marker pops — memory reclaimed
```

Budget tracking lives in `budget/`.

---

## 4. Containers — `container/` + single headers

Beyond the `std` aliases, core ships data-oriented containers:
`dense_map.hpp`, `slot_map.hpp`, `sparse_set.hpp`, `small_vector.hpp`
(SBO vector), `spsc_ring.hpp` (single-producer/consumer ring),
`string_builder.hpp`. `handle/` provides generational handles
(index + generation) for stable, ABA-safe references into pools.

---

## 5. Math — `math/` (multi-ISA SIMD)

A single canonical math layout (`Vec2/3/4`, `Mat4`, quaternions) shared across
every module — **no conversions at boundaries**. The SIMD backend is compiled
once per ISA and dispatched at runtime:

```
scalar · SSE4.2 · AVX · AVX2 · AVX-512
```

Each ISA TU (`simd_math_<isa>.cpp`) is built with its own compiler flags; the
active backend is chosen at startup via `cardinal::simd::active_backend()`
(and `get_scalar/get_sse42/...`). Closed-form values are pinned by the
deterministic math test suite.

---

## 6. Compression — `compress/`

Hand-rolled, zero-dependency codecs (the engine had no reachable inflate):

- **`inflate_raw` / `inflate_zlib`** — RFC 1951 / 1950 DEFLATE decompressor.
  The keystone that unblocks FBX (compressed property arrays), USDZ (deflate
  zip entries), and PNG (IDAT). Bounds-checked for untrusted input.
- **`decode_png`** — PNG decoder built on inflate: chunk walk
  (IHDR/IDAT/IEND) → `inflate_zlib` → reverse the per-scanline filters
  (None/Sub/Up/Average/Paeth). 8/16-bit grayscale / RGB / GA / RGBA.

```cpp
#include <cardinal/core/compress/png.hpp>
cardinal::vector<cardinal::u8> px;
cardinal::u32 w, h, ch, bits;
if (cardinal::core::compress::decode_png(bytes, n, px, w, h, ch, bits)) { /* ... */ }
```

---

## 7. Services — `service/`

Process-wide wiring without globals:

- **`GlobalObjectManager`** — a service locator (register/lookup engine
  singletons by type).
- **`shared_object<T>`** — an intern-table factory (dedup shared instances by
  key).

---

## 8. Platform — `os/`, `clock/`, `sync/`

- **`os/`** — HAL: file I/O, a coalescing async-I/O dispatcher, a file
  watcher, process/path helpers.
- **`clock/`** — frame timing, high-resolution timers.
- **`sync/`** — threading + the **job system** (fiber-based work-stealing),
  CPU topology + thread-affinity (`topology_*`, `affinity_*`), plus standard
  sync primitives via the `std/` wrappers.

> Note: the job system keeps `#pragma optimize off` in `jobs.cpp` on MSVC
> pending a proper assembly fiber switch (an `/O2` optimizer interaction).

---

## 9. Diagnostics + errors

- **`diag/`** — logging (`cardinal::log::infof/warnf/errorf`, category-tagged)
  and assertions. Studio's Log/Console panels are sinks on top of this.
- **`error/`** — result/error types for fallible APIs (`error_code` is aliased
  from `<system_error>` in `std/filesystem.hpp`).
- **`trace/`** (separate module) — execution tracing / profiling surfaced in
  the Studio Profiler.

---

## 10. Conventions

- Prefer `cardinal::` typedefs + `std/` wrappers; never include base C++
  headers outside `cardinal::core`.
- Prefer explicit memory (arenas/pools) on hot paths over ad-hoc `new`.
- Keep new code deterministic + testable — add a console suite under `tests/`
  (exit 0 = pass) and register it with CTest.
- One canonical math layout — pass `Vec*/Mat4` by value/reference, no
  per-boundary conversions.

See `tests/` for executable, authoritative examples of each subsystem.
