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

## Note on history

An interim commit (`b2d53bb`) prematurely declared "Phase 1 COMPLETE"
before `render` and `sandbox` were migrated — the Phase N sweep caught the
gap, both were migrated (`97a814c`, and the sandbox commit alongside this
doc), and conformance is now genuinely verified clean by the sweep above.
