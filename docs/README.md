# Cardinal Documentation

Developer documentation for the Cardinal engine, editor, and foundation
library. Start here, then dive into the area you need.

| Doc | What it covers | Audience |
|---|---|---|
| **[engine.md](engine.md)** | Architecture: the module stack + layering, the RHI (Vulkan/D3D12), the AEGIS render pipeline, gameplay/sim, and the asset → cook → pack → package pipeline. | Engine + gameplay devs |
| **[studio.md](studio.md)** | The Cardinal Studio editor: launch, layout, menus, panels, and the common workflows (create a project, import assets, build a level, play, package). | Editor users / tool devs |
| **[core.md](core.md)** | `cardinal::core` reference: the Foundation Rule, vocabulary types, allocators, containers, multi-ISA SIMD math, compression, services, and platform primitives. | Anyone writing engine code |

## Also see

- **[../README.md](../README.md)** — project overview, requirements, build &
  test commands, repository layout, and license.
- **[../THIRD_PARTY.md](../THIRD_PARTY.md)** — optional NVIDIA SDK integration.

## Quick orientation

- **Build:** `build.bat` (Windows) / `./build.sh` (Linux), or a CMake preset
  (`windows-msvc-release`, `linux-clang-release`, `default`, …). Output in
  `build/<preset>/bin`, launchers in `run/`.
- **Run the editor:** `run/Cardinal_System_Studio`.
- **Import an asset (CLI):** `run/cardinal_import <file.glb|gltf|obj|fbx|usd>`.
- **Tests:** `ctest --test-dir build/<preset> --output-on-failure` — one
  deterministic suite per module.

## Layout recap

```
runtime/    engine modules (cardinal::core … cardinal::engine, cardinal::ui)
apps/       host executables (cardinal_launcher, cardinal_import, …)
samples/    example programs (03_studio is the editor)
tests/      one regression suite per module
docs/       this documentation
```
