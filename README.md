# Cardinal System Works

A greenfield, performance-first foundation engine in modern C++ — dual
backend (Vulkan + DirectX 12), cross-platform (Windows + Linux), built with a
deliberate zero-dependency ethos.

> Status: early / foundation (`v0.0.1`). The runtime, editor scaffolding,
> sample apps and a broad deterministic test net are in place; the renderer
> and higher-level systems are under active development.

## Design principles

- **Performance first.** Data-oriented layouts, explicit memory, job-system
  parallelism, and a single canonical math layout shared across every module
  (no conversions at boundaries).
- **Zero-dependency ethos.** The engine and its test harness avoid external
  libraries wherever practical. Base C++ headers are funnelled through one
  place (`cardinal::core` types); engine code uses `cardinal::` typedefs.
- **Deterministic + testable.** A hand-rolled, dependency-free test harness:
  every module has its own console suite that returns non-zero on failure and
  is registered with CTest. The suites pin exact, reproducible behavior
  (closed-form values, compile-time `static_assert` goldens where possible).
- **Dual RHI.** A thin RHI abstracts Vulkan and D3D12; optional NVIDIA SDKs
  are integrated opt-in and degrade gracefully when absent.

## Requirements

| Platform | Toolchain |
|---|---|
| Windows | Visual Studio 2022 (any edition) or clang-cl, CMake ≥ 3.27, Ninja |
| Linux   | clang/clang++, CMake ≥ 3.27, Ninja |

C++23. [vcpkg](https://github.com/microsoft/vcpkg) is used in manifest mode
(set `VCPKG_ROOT`) for the small set of unavoidable third-party headers.
NVIDIA SDKs are **not** bundled — see [THIRD_PARTY.md](THIRD_PARTY.md).

## Building

The build wrappers configure and build a CMake preset in one step.

**Interactive** (run with no arguments — prompts for OS / architecture /
build type / toolchain):

```
build.bat            (Windows)
./build.sh           (Linux)
```

**Classic / CI** (positional args, unchanged and scriptable):

```
build.bat release msvc                       Windows, RelWithDebInfo, x64
build.bat release msvc Cardinal_System_Studio  build a single target
build.bat release clang arm64                clang-cl, arm64
build.bat clean                              wipe build/ and run/
./build.sh release                           Linux, RelWithDebInfo
```

Build output lands in `build/<preset>/bin/`. CMake also generates a
git-ignored `run/` directory with one launcher per executable, so any
program can be started from the repo root:

```
run\Cardinal_System_Studio.cmd        (Windows)
run/Cardinal_System_Studio            (Linux)
```

Direct CMake also works: `cmake --preset windows-msvc-release` (or
`default`), then `cmake --build --preset windows-msvc-release`.

## Tests

Every suite is a standalone console executable (exit 0 = pass) registered
with CTest:

```
ctest --test-dir build/windows-msvc-release --output-on-failure
```

or run them straight from the launchers (`run/Cardinal_Test_*`). The suites
are pure and deterministic — no GPU, no clock, no network — so they pass
identically on any machine.

## Project layout

```
runtime/      engine modules (core, render, rhi, scene, edit, physics,
              nav, anim, audio, ai, world, sim, ui/hud, net, ...)
apps/         shippable host executables (launcher, sandbox runner)
samples/      focused example programs (topology, jobs, window, studio, ...)
tests/        one deterministic regression suite per module
cmake/        build glue (incl. existence-gated NVIDIA SDK integration)
shaders/      shader sources
build.bat /   interactive + scriptable build wrappers
build.sh
```

`runtime/core` is the foundation every other module depends on; it is the
single place base C++ headers are referenced.

## Documentation

Developer docs live in [`docs/`](docs/README.md):

- [docs/engine.md](docs/engine.md) — engine architecture: module stack, RHI,
  the AEGIS render pipeline, gameplay/sim, and the asset/build pipeline.
- [docs/studio.md](docs/studio.md) — the Cardinal Studio editor guide.
- [docs/core.md](docs/core.md) — the `cardinal::core` foundation reference.

## Third-party SDKs

Optional NVIDIA SDKs (RTXMU, NRD, RTXDI, RTXGI, NTC, Reflex, Streamline,
RTX Path Tracing) integrate opt-in and are detected by directory existence.
None are committed to this repository and none are required to build — see
[THIRD_PARTY.md](THIRD_PARTY.md) for what each enables and how to obtain it.

## License

Cardinal System Works License v1.0 — a permissive, MIT-style license with a
**trust-based, non-binding 1% revenue share** (an honor-system request, not a
legal condition; personal / hobby / educational / research / non-profit use
is always exempt). See [LICENSE](LICENSE).
