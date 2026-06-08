# Cardinal Studio — Editor Guide

Cardinal Studio is the engine's editor: a docked, multi-viewport ImGui
application for building worlds, importing assets, tuning the renderer, and
packaging shippable builds. This guide covers launching it, the layout, the
menus and panels, and the common workflows. For engine internals see
[engine.md](engine.md).

The Studio is the `03_studio` sample (`Cardinal_System_Studio`), built on the
`cardinal::ui` editor layer (`StudioEngine` + the panels in
`runtime/ui/src/panels/`).

---

## 1. Launch

Build, then run via the generated launcher (or directly):

```
build.bat release msvc Cardinal_System_Studio
run\Cardinal_System_Studio.cmd          # Windows
run/Cardinal_System_Studio              # Linux
```

`cardinal_launcher` shows a backend picker (Vulkan / D3D12) and spawns the
Studio with `--backend=<choice>`. The backend can also be hot-swapped at
runtime; Studio re-binds its ImGui backend across the swap.

---

## 2. Layout

The Studio opens as a dockspace with a menu bar and a default panel layout.
Panels are dockable, closable, and toggled from the **View** menu. Multiple
3D **viewports** are supported (add/remove from the View menu), each with its
own camera, FPS quota, and view mode.

- **Fly camera**: hold the mouse in a viewport and use WASD + mouse-look
  (driven through ImGui state so it cooperates with the editor).
- **Gizmos**: ImGuizmo translate/rotate/scale handles render in the active
  viewport for the selected actor.

---

## 3. Menu bar

| Menu | Contents |
|---|---|
| **File** | Import 3D Asset… · Import Megascans… · Import Heightmap (Terrain)… · Quit |
| **Create** | Spawn primitives / registered game classes by category · Terrain Grid… |
| **View** | Add/Remove Viewport · per-viewport toggles · show/hide every panel |
| **Settings** | VSync, foreground/background FPS caps, placement tool |
| **Build** | **Cook Content** · **Package: Development** · **Package: Shipping (+archive)** — runs the `buildtool` pipeline on the current project; reports the artifact dir inline |
| **Help** | About · ImGui Demo |

The **Build** menu is the UE5-`BuildCookRun` analog — see
[§7 Packaging](#7-packaging-a-build).

---

## 4. Panels

Grouped as they appear under `runtime/ui/src/panels/`:

**Scene**
- **Outliner / Hierarchy** — the actor tree; search/filter, multi-select, drag,
  and a right-click context menu (rename, duplicate, delete, focus).
- **Inspector** — components + reflected properties of the selection.
- **Prefab** — capture/instantiate/revert/apply prefabs; a starter library.
- **Class Picker** — registered `GameActor` classes to spawn.

**Assets / Project**
- **Project** — create/open projects, recents (right-click: Copy path / Remove),
  source + cooked asset lists (filter, Copy path), engine-root + **Generate
  build files**, and the **Package** section (config combo + archive toggle +
  *Build, Cook, Run*).
- **Cook & Pack** — cook/pack the current project.
- **Save / Load** — world snapshots.
- **Asset Palette** — placeable assets (right-click context menu).

**Rendering**
- **Render Pipeline** — the AEGIS settings (see [§6](#6-aegis-render-settings)).
- **Sky / Time of Day**, **Particles**, **Virtual Textures**, **Virtual
  Geometry**.

**Diagnostics**
- **Stats** — adapter/caps/render-settings/frame; right-click → **Copy GPU +
  perf summary** (handy for bug reports).
- **Log** / **Console** — right-click → copy / clear.
- **Memory & Budgets**, **Profiler**, **SIMD**, **Options / Settings**.

**Gameplay**
- **Game** (PIE bar — Play/Pause/Stop), **Simulation**, **Input**,
  **Navigation**, **World Systems**, **Actors**.

**Animation**: Sequencer, Mixer, Curve Editor. **Scripting**: Code Sandbox
(`cppscript`), Shader Compiler. **Tools**: Editor Modes, Brush, Mesh Tools,
Texture Tools.

---

## 5. Common workflows

### Create a project
**Project** panel → *New project* → set Path / Name / Author / Template
(Blank / First Person / Top-Down / Cinematic) / Engine root → **Create
project**. This scaffolds runnable starter `GameActor` code, a default world,
and a full build setup. (See [engine.md §5](engine.md#5-assets-projects-and-the-build-pipeline).)

### Import an asset
**File → Import 3D Asset…** (OBJ / glTF / GLB / FBX / USD / Megascans), or
from the shell with the CLI:

```
run\cardinal_import.cmd path\to\model.glb
```

It prints meshes / materials / nodes / verts / tris and returns the failed
count (so it works as a CI validation step).

### Build a level
Spawn from **Create** or the **Asset Palette** (Place tool), arrange with
gizmos + the layout tools (align/distribute, array, grid snap), tune
components in the **Inspector**, and capture reusable groups as **Prefabs**.
Validate with the scene **Problems**/validation tools. Save via **Save / Load**.

### Play in editor
The **Game** bar drives the play state: **Play** snapshots the world and fires
`begin_play`/`on_tick`; **Stop** restores the snapshot. Undo/redo is
snapshot-based per edit.

---

## 6. AEGIS render settings

The **Render Pipeline** panel exposes the AEGIS pipeline as grouped *knobs*
(checkbox / slider / combo), each with a tooltip; hardware-gated knobs grey
out with a reason on unsupported GPUs. Groups:

- **Visualisation** (view mode), **AEGIS** (graph backend, max geometry tier,
  virtual geometry / meshlet / Hi-Z, **GPU Compute Execute (experimental)**)
- **Quality**, **Shadows**, **Anti-Aliasing**, **Reflections**, **Lighting**,
  **Global Illumination**, **Post-FX**, **Color Grading**, **Atmosphere**,
  **Detail & Streaming**, **Frame Pacing**, **Upscaler**, **GPU Features**,
  **Debug**.

Knobs that map to real config (max tier, FP8/FP4, async compute, VRS, bindless,
DirectStorage) drive the pipeline; the rest persist + display and take effect
as the renderer consumes them.

> **GPU Compute Execute** is experimental + off by default. Enabling it routes
> the AEGIS graph through real GPU compute dispatch (`RhiBackend`); leave it off
> for the stable ForwardRenderer path.

---

## 7. Packaging a build

Two equivalent entry points run the `buildtool` BuildCookRun pipeline on the
current project:

- **Build menu** → *Package: Development* / *Package: Shipping (+archive)* /
  *Cook Content*.
- **Project panel → Package** section → Configuration combo + *Build, Cook,
  Run*.

Stages: **Build** (generate build files + compose the cmake command) →
**Cook** (`assets/` → `cooked/`) → **Pack/Stage** (`.cpk` + manifest + staged
exe in `dist/<config>/`) → **Archive** (optional copy). The actual engine
*compile* is run separately via the generated `build.bat` (it builds the engine
from source under your vcpkg toolchain).

---

## 8. Tips

- **Context menus everywhere** — right-click the Outliner, viewport, asset
  lists, Log/Console, Stats, and recents for quick actions.
- **Copy for bug reports** — Stats → *Copy GPU + perf summary*, and Log →
  *Copy visible lines* paste straight into an issue.
- **Per-viewport FPS caps** live in **Settings**; background windows can be
  throttled to save power.
