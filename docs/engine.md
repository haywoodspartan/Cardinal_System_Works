# Cardinal Engine — Architecture

This document describes how the Cardinal engine is structured: the module
stack, how the layers depend on each other, the render and gameplay paths, and
the project/build pipeline. For build commands and requirements see the root
[README](../README.md); for the editor see [studio.md](studio.md); for the
foundation library see [core.md](core.md).

> Status: foundation (`v0.0.1`). C++23, dual RHI (Vulkan + D3D12),
> Windows + Linux, vcpkg manifest mode, CMake ≥ 3.27 + Ninja.

---

## 1. Big picture

Cardinal is a **layered set of static libraries** under `runtime/`, each
exposing `cardinal::<name>` with public headers in
`runtime/<name>/include/cardinal/<name>/`. Two roll-up targets sit on top:

- **`cardinal::engine`** — the aggregator a shipped game links. It transitively
  pulls in ~40 runtime modules (core, rhi, render, scene, game, sim, physics,
  audio, nav, project, cook, pack, buildtool, …).
- **`cardinal::ui`** — the Studio editor layer (ImGui panels + `StudioEngine`)
  built on top of `cardinal::engine`.

Executables live in `apps/` (`cardinal_launcher`, `cardinal_import`,
`cardinal_sandbox_runner`) and `samples/` (`03_studio` is the editor,
`02_window`, `11_stress_world`, etc.).

```
apps / samples            cardinal_launcher · 03_studio (Studio) · cardinal_import
        │
   cardinal::ui           Studio panels, StudioEngine, ImGui + ImGuizmo
        │
 cardinal::engine         aggregator (links everything below)
        │
 ┌──────┴───────────────────────────────────────────────────────────┐
 gameplay / sim     actor · world · sim · game · physics · particles ·
                    anim · nav · navmesh · ai · mass · input · hud ·
                    cine · audio · net
 graphics           scene · render · postfx · lighting · vt · vgeom · sky
 tooling / assets   serial · asset · import · cook · pack · buildtool ·
                    project · shader · edit · level · cmd · partition · trace
 scripting          cppscript · vm · sandbox · script · console · plugin
 └──────┬───────────────────────────────────────────────────────────┘
        │
 foundation          cardinal::core · cardinal::window · cardinal::rhi
```

**Dependency rule:** modules only depend *downward*. `cardinal::core` is the
root — the single place base C++ headers are referenced (see
[core.md](core.md)). Nothing in `runtime/` may link `cardinal::ui`.

---

## 2. Foundation layer

| Module | Responsibility |
|---|---|
| `cardinal::core` | Types, allocators, containers, math (multi-ISA SIMD), `std/` wrappers, jobs, compress (DEFLATE/PNG), service locator, OS/clock/sync. See [core.md](core.md). |
| `cardinal::window` | OS window abstraction (Win32 / X11). |
| `cardinal::rhi` | Rendering Hardware Interface — thin abstraction over **Vulkan** and **D3D12**. |

### RHI

The RHI (`cardinal::rhi`) is a thin device/swapchain/command abstraction with
two backends selected at launch:

- **Vulkan** (`src/vulkan/`) — Windows + Linux. Split into `device` /
  `swapchain` / `commands` / `compute` translation units behind
  `vulkan_internal.hpp`.
- **D3D12** (`src/d3d12/`) — Windows only. Per-class split (`device`,
  `swapchain`, `commands`, `compute`).

Both backends implement graphics + **compute pipelines** (`create_compute_pipeline`,
`dispatch`/`dispatch_indirect`, UAV/storage-buffer binding, barriers, async
compute queue, variable-rate shading). Device capabilities
(`rhi::GpuCapabilities`) are probed per backend — mesh shaders, ray tracing,
VRS, bindless, FP8/FP4 math (D3D12 SM6.9 / Vulkan `VK_KHR_shader_float8`),
DirectStorage — and drive feature gating up the stack.

Optional **NVIDIA SDKs** (Reflex, DLSS/Streamline, NRD, RTXMU, NTC, …) are
detected by directory existence and degrade gracefully when absent — see
[../THIRD_PARTY.md](../THIRD_PARTY.md).

---

## 3. Rendering — AEGIS Pipeline 2.0

The renderer is organised as a **render graph** (`cardinal::render`,
`graph::Graph`) executed by a selectable backend:

| Backend | Use |
|---|---|
| `NullBackend` | Topology only — per-frame stats, zero buffer allocation. |
| `CpuBackend` / `ThreadedCpuBackend` | Full virtual-GPU simulation on the host. |
| `RhiBackend` | Real GPU compute dispatch through the RHI. |

**AEGIS Render Pipeline 2.0** (`PipelineId::Aegis`, the Studio default) is a
graph-driven GPU pipeline: virtual-geometry classify → meshlet build/cull →
screen-space-error LOD → frustum/Hi-Z cull → V-Buffer → tile light cull →
ReSTIR DI (opt-in) → TAA → Tonemap → Composite. An **adaptive math-division
tier** (`GeometryTier` = FP32 / FP16 / FP8 / FP4) escalates precision based on
device caps + the requested ceiling.

The bridge (`aegis_pipeline_bridge.cpp`) exposes the pipeline's settings as
editor *knobs* (see [studio.md](studio.md)) and clamps requested features
against real device caps (`aegis_resolve_config`). On-screen drawing currently
delegates to `scene::ForwardRenderer`; the graph runs alongside for telemetry.

> **GPU compute dispatch is opt-in** (`RhiBackend::set_gpu_execute`, default
> **off**). The default path is recording-only telemetry + the ForwardRenderer,
> which is the validated, stable path. The experimental GPU dispatch is exposed
> via the Studio's *GPU Compute Execute* knob for on-device validation.

---

## 4. Gameplay + simulation

A game is authored by subclassing **`cardinal::game::GameActor`** and
registering the class so the engine can spawn it by name and drive its
reflected properties:

```cpp
#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>

class Spinner final : public cardinal::game::GameActor {
public:
    float speed{45.0f};
    void begin_play() override { /* ... */ }
    void on_tick(float dt) override { /* ... */ }
};

CARDINAL_REGISTER_GAME_CLASS(Spinner, "Game/Spinner",
    PROP_FLOAT(speed, 0.0f, 720.0f, "Spin speed (deg/s)"))
```

The runtime pieces:

- **`cardinal::actor`** — `World`, `Actor`, `Component` (+ builtin components:
  Transform, Mesh, Camera, Light, AudioEmitter, RigidBody, Tag,
  PlayerController, Script, PrefabLink). Components are clonable and
  serialisable; prefabs capture/instantiate actor subtrees.
- **`cardinal::sim`** — `SimWorld` drives ordered tick groups (PreUpdate →
  PrePhysics → Physics → PostPhysics → Update → LateUpdate → Render).
- **`cardinal::game`** — `Game` orchestrates the play state machine
  (`start_play` → `tick` → `stop_play`), fires `begin_play`/`on_tick`/`end_play`,
  and spawns registered classes via `spawn_class`.

Worlds serialise to the text `# Cardinal save v1` format
(`cardinal::serial::save_world` / `load_world`).

---

## 5. Assets, projects, and the build pipeline

### Import (`cardinal::import`)
`import_file(path)` auto-detects by extension: **OBJ / glTF / GLB / FBX /
USDA / USDZ / Megascans**, plus heightmaps (RAW16 / BMP / TGA / **PNG**) for
terrain. Returns an `ImportScene` (meshes, materials, nodes). The
`cardinal_import` CLI wraps this for scripted/CI import.

### Cook + Pack
- **`cardinal::cook`** — `cook_all()` bakes source assets in `assets/` into
  `cooked/<rel>.cooked` (texture / mesh / shader / audio cookers; hash-skips
  up-to-date files).
- **`cardinal::pack`** — `distribute()` gathers `cooked/`, builds a `.cpk`
  pack + a `distribution.cardinal` manifest, and stages the exe/DLLs/saves
  into a shippable bundle.

### Projects (`cardinal::project`)
A project is a directory with a `project.cardinal` manifest +
`src/ assets/ cooked/ pack/ shaders/ save/`. `instantiate_template()`
scaffolds one of four templates (**Blank / FirstPerson / TopDown / Cinematic**)
with distinct, runnable starter `GameActor` code, a default world, and a full
build setup (`generate_build_files()` writes a `CMakeLists.txt` that
`add_subdirectory()`s the engine + a `src/main.cpp` game-runner entry +
`build.bat`/`build.sh`).

### BuildCookRun (`cardinal::buildtool`)
The orchestrator (Cardinal's analog of Unreal's BuildCookRun) runs the staged
pipeline over a project:

```
Build → Cook → Pack/Stage → Archive
```

`run_pipeline(project, options)` with `BuildConfig` (Debug / Development /
Shipping → `CMAKE_BUILD_TYPE` Debug / RelWithDebInfo / Release) and
`BuildTarget` (Editor / Game / Server). Drivable from the Studio **Build** menu
and the Project panel's **Package** section.

---

## 6. Build & test

See the [README](../README.md) for full commands. In short:

```
build.bat                              # interactive (Windows)
build.bat release msvc                 # RelWithDebInfo, x64
build.bat release msvc Cardinal_System_Studio   # one target
cmake --preset windows-msvc-release && cmake --build --preset windows-msvc-release
ctest --test-dir build/windows-msvc-release --output-on-failure
```

Presets: `windows-msvc-{debug,release}[-arm64]`,
`windows-clang-{debug,release}[-arm64]`, `linux-clang-{debug,release}[-arm64]`,
`default`. Output lands in `build/<preset>/bin`; CMake also writes a `run/`
launcher per executable. Every module has a deterministic CTest suite under
`tests/` (no GPU/clock/network — identical results on any machine).

---

## 7. Where to look

| You want… | Start at |
|---|---|
| The foundation / utilities | [core.md](core.md), `runtime/core/` |
| The editor | [studio.md](studio.md), `samples/03_studio/`, `runtime/ui/` |
| The renderer | `runtime/render/` (`graph.*`, `gpu_aegis.*`, `aegis_pipeline_bridge.cpp`) |
| The RHI | `runtime/rhi/src/{vulkan,d3d12}/` |
| Gameplay | `runtime/game/`, `runtime/actor/`, `runtime/sim/` |
| Projects + packaging | `runtime/project/`, `runtime/buildtool/`, `runtime/cook/`, `runtime/pack/` |
| Importing assets | `runtime/import/`, `apps/cardinal_import/` |
