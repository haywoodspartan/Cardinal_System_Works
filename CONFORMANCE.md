# Cardinal Foundation Conformance

## The FOUNDATION RULE

> Base C++ standard-library headers are `#include`d **only** inside
> `cardinal::core`. The Foundation restructures them into the `cardinal::`
> vocabulary so ownership / allocator / ABI / small-buffer / container
> policy is changeable in **one place**. Non-core code never includes a
> `std` header directly — it uses the `cardinal::` alias.

The rule's canonical statement lives in
`runtime/core/include/cardinal/core/types.hpp` (the master typedef header).
Every `cardinal::core` std-wrapper additionally `#include`s
`<cardinal/core/types.hpp>` first, so the master typedefs are always present.

## Status: Phase 1 COMPLETE — every runtime module is conformant

`cardinal::core` ships the full restructured `std` surface — 26 granular
wrapper headers plus the `<cardinal/core/std.hpp>` umbrella and the
`tests/foundation_std` verification suite.

Every other module under `runtime/` consumes that vocabulary: **zero
`std::` usages and zero direct `std` includes remain outside
`cardinal::core`** (and the documented vendored exceptions below).

Conformant runtime modules (one self-contained, build-clean,
56/56-regression-green, committed+pushed migration pass each):

```
vm  edit  trace  nav  navmesh  particles  anim  sky  cine  console
input  ai  physics  partition  serial  cook  pack  asset  project
world  actor  level  sim  net  import  vt  scene  vgeom  game  rhi
ui  hud  shader  audio  window  launcher  mass  plugin  script
cppscript  engine  compile  render  sandbox
```

## Sanctioned exceptions

The sweep deliberately excludes — these are correct, not violations:

1. **`runtime/core/` itself** — the wrapper layer. It is *defined* by
   referencing the `std` headers and re-exporting them as `cardinal::`
   aliases. This is the single sanctioned `std` include site.
2. **Vendored third-party under `*/external/`** — the Dear ImGui
   platform/renderer backends in `runtime/ui/external/imgui_backends/`
   are upstream sources (`#include "imgui.h"` by design); never rewritten.
3. **Third-party C APIs** — the Lua C headers
   (`<lua.h>`/`<lualib.h>`/`<lauxlib.h>`, inside `extern "C"`) are a
   library API, not the C++ stdlib.
4. **OS / GPU SDK headers** — `<Windows.h>`, `<vulkan/vulkan.h>`,
   `<volk.h>`, `<d3d12.h>`, `<dxgi.h>`, `<dxcapi.h>`, `<gdiplus.h>`,
   `<mmdeviceapi.h>`, the NVIDIA SDK headers, etc. These are platform /
   vendor surfaces, not `std`.
5. **The one sanctioned `namespace std` use** — the `hash<>`
   specialisation for `cardinal::vt::TileKey` in `runtime/vt/types.hpp`.
   Specialising `std::hash` for a user key type is the only standard
   mechanism to make it usable in `cardinal::unordered_map/set`; there is
   no alternative, so it stays (it contains no `std::` *token*).

## ImGui master reference template

The same single-sanctioned-include discipline is applied to the editor's
ImGui dependency: `runtime/ui/include/cardinal/ui/imgui.hpp` is the only
place `<imgui.h>` is referenced for the Studio function surface (the
`StudioImpl` facade + every `src/panels/*` draw function). Backend headers
(`<imgui_internal.h>`, `<imgui_impl_*.h>`) stay explicit in the two
backend-wiring TUs that need them; vendored backends are untouched.
`imgui::imgui` is linked PRIVATE on `cardinal_ui`, so the blast radius is
exactly the UI library.

## Verifying conformance

Two ripgrep sweeps over `runtime/`, excluding `runtime/core/` and any
`*/external/` path. Both must return **no files**:

```sh
# A. std:: usages
grep -rlE --include='*.cpp' --include='*.hpp' --include='*.h' '\bstd::' \
    runtime | grep -v '/core/' | grep -v '/external/'

# B. direct std-library includes
grep -rlE --include='*.cpp' --include='*.hpp' --include='*.h' \
    '^[[:space:]]*#[[:space:]]*include[[:space:]]*<(algorithm|array|vector|memory|functional|cstring|cstdio|cstdarg|cstdlib|string|string_view|map|set|unordered_map|unordered_set|cmath|chrono|mutex|thread|atomic|limits|queue|deque|list|optional|utility|sstream|fstream|cctype|ctime|cassert|climits|new|bit|charconv|any|typeindex|filesystem|type_traits|variant|future|condition_variable|ratio|tuple|numeric|cstddef|cstdint|system_error|shared_mutex|iterator|span)>' \
    runtime | grep -v '/core/' | grep -v '/external/'
```

Plus the build gate every pass enforced: `build.bat release msvc` clean
under `/W4 /WX`, and all 56 `Cardinal_Test_*` regression suites green.

## Untrusted-input trust boundaries

A second conformance invariant, hardened in the same per-pass discipline
(one boundary per pass: build-clean, 56/56, a deterministic hostile-input
regression, committed+pushed):

> Any size / count / length / offset that originates **outside** the
> engine's trust boundary is validated in 64-bit (`usize`) arithmetic and
> bounded against the real container/buffer **before** it drives an
> allocation, `resize`/`reserve`, `memcpy`, or index. One named `kMax*`
> constant is the single source of truth for each bound so a producer and
> consumer cannot drift apart.

| Boundary | Entry point | Guard | Commit |
|---|---|---|---|
| Cooked-asset codec | `asset::codec::decode_{texture,mesh,shader,material}` | size fields widened to `usize` (no 32-bit wrap) | `ab18470` |
| Cooked-asset container | `cook::CookedAsset::deserialize` | `usize(kCookHeaderBytes)+size` | `58b3453` |
| Pack archive index | `pack::Archive::open` | `reserve` ≤ `(sz−index_offset)/30` | `58b3453` |
| Pack entry load | `pack::Archive::load_blocking`/`load_async` | `offset ≤ fsz && size ≤ fsz−offset` | `68d85e1` |
| VM bytecode module | `vm::load` | `kMaxCodeLen` (16 MiB) in the descriptor scan | `e6f71a9` |
| glTF accessor | `import` `read_accessor` | `cnt ≤ buf.size()`, `size_t` index | `431f5b8` |
| glTF primitive indices | `import` mesh assembly | per-triangle range-check vs vertex count; UB-safe `float→idx` | `b8ffe90` |
| UDP reorder buffer | `net` UDP `poll` | `kMaxReorderBuffered` (256) | `ccfbf7d` |
| UDP peer table | `net` UDP `Connect` | `kMaxPeers` (4096) | `6158744` |
| UDP unacked backlog | `net` UDP `send_to_` | `kMaxUnacked` (1024) + dead-peer reap | `83416f8` |
| Recent-projects store | `project::RecentProjects::load` | `kMaxRecent` (16), shared with `add` | `8ed046c` |

Audited and already defensive (verified, no change required): the OBJ
importer (`emit()` bounds-checks every index with a zero fallback), the
`net` UDP short-datagram path (`got < sizeof(PktHeader)` drop), the
`vm::load` exact-size check (`len == header + table + body`), and the
`serial` / `project` text parsers (every `find` is npos-guarded before
`substr`). The in-process `net.cpp` loopback transport is fed only by the
local app and is **not** a trust boundary.

Every boundary above carries a deterministic regression that feeds
hostile input (UINT32/64-max counts, negative / NaN, out-of-range
indices, oversized stores) and asserts a clean bounded result — no
crash, OOB read, or over-allocation — under the standing `/W4 /WX` +
56/56 gate.

## Integration coverage & headless-testability boundary

A third invariant, established by the same per-pass discipline (one
cross-module seam per pass: build-clean, full-regression-green, a
deterministic headless suite, committed+pushed). The asset lifecycle —
the longest cross-module path in the engine — is now pinned end to end:

| Seam (cross-module) | Suite | Commit |
|---|---|---|
| codec → cook::CookedAsset → pack::Builder → asset::Registry | `asset_pipeline` | `0ae029a` |
| on-disk OBJ → importer → MeshCooker → pack → Registry | `content_pipeline` | `3fafcb0` |
| `cook::cook_all` driver + hash manifest (skip/recook/force) | `cook_incremental` | `9ff607f` |
| cook_all → `pack::distribute` → Archive / cooked-dir → Registry (dev==shipping) | `shipping_pipeline` | `93a1cd6` |
| `pack::Archive::load_async` ↔ `async` ↔ `JobSystem` (256 concurrent) | `pack_streaming` | `6253bc2` |

Other deterministic/headless integration seams were surveyed and found
**already covered** by their existing suites — pinning them again would
be box-ticking, not coverage: world save/load round-trip
(`serial_world`), sky save/load (`serial`), undo/redo composite +
capacity↔cursor↔redo-truncation (`undo`), `LevelManager`↔`World` +
HLOD (`level`), `Game::tick`↔`SimWorld`↔deferred begin_play/sweep
(`game`), the fixed-step accumulator (`sim`), physics determinism
(`physics`), grid/polygon pathfinding pipelines (`nav`, `navmesh`),
and the bytecode VM incl. its verifier/trap boundary (`vm`).

> **Headless-testability boundary.** Several remaining seams are
> *correctly deferred*, not gaps: there is **no null/headless
> `rhi::Device` backend**, and `scene::Mesh` can only be constructed
> through factories that take an `rhi::Device&` (and it owns a GPU
> `rhi::Buffer`). So `serial::save_scene`/`load_scene`,
> `scene_graph` `entity_world_aabb`/`pick_entity`, and any
> render-data round-trip are not deterministically headless-testable
> today. The `serial`/`scene_graph` suites name this deferral by
> design. The single highest-leverage unlock is a minimal **null RHI
> backend** (no-op buffers, CPU-side data) — it would make the scene
> serialization round-trip and render-data seams headless-pinnable;
> until it exists, autonomous effort pivots to targeted pure-CPU
> latent-bug review (the pattern that fixed the `mass` typeid-hash
> truncation and `audio::play_2d`), not redundant integration suites.

## Note on history

An interim commit (`b2d53bb`) prematurely declared "Phase 1 COMPLETE"
before `render` and `sandbox` were migrated — the Phase N sweep caught the
gap, both were migrated (`97a814c`, and the sandbox commit alongside this
doc), and conformance is now genuinely verified clean by the sweep above.
