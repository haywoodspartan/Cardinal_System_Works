# Third-Party SDKs (NVIDIA)

Cardinal can integrate several NVIDIA SDKs under `third_party/nvidia/`.

**None of them are committed to this repository, and none are required to
build the engine.** NVIDIA's SDKs are governed by NVIDIA's own licenses and
are not redistributable here, so the entire `third_party/` tree is
git-ignored. You must obtain each SDK yourself from NVIDIA and drop it at the
expected path; nothing here is uploaded or re-distributed.

`cmake/cardinal_third_party.cmake` detects each SDK purely by directory
existence and only enables the matching feature (and its `CARDINAL_HAS_*`
define) when present. A clean clone with an empty `third_party/` configures
and builds normally — the related render features are simply compiled out.

## SDKs and expected paths

Obtain each from the NVIDIA Developer site / the relevant NVIDIA GitHub
project, then place it so the listed file exists:

| Path | SDK | CMake detects via | Exposes |
|---|---|---|---|
| `third_party/nvidia/rtxmu/`   | RTX Memory Utility            | `include/rtxmu/VkAccelStructManager.h` | `cardinal::tp::rtxmu` |
| `third_party/nvidia/nrd/`     | NVIDIA Real-Time Denoiser     | `Include/NRD.h` | `cardinal::tp::nrd` |
| `third_party/nvidia/rtxdi/`   | RTXDI (ReSTIR) + NVAPI mirror | `External/NVAPI/nvapi.h` | `cardinal::tp::nvapi` |
| `third_party/nvidia/rtxgi/`   | RTXGI (SHARC / NRC)           | `Libraries/{Nrc,Sharc}/CMakeLists.txt` | sample libs |
| `third_party/nvidia/ntc-sdk/` | Neural Texture Compression SDK| `include/libntc/ntc.h` | `cardinal::tp::ntc_sdk` |
| `third_party/nvidia/reflex/`  | NVIDIA Reflex SDK (Vulkan)    | `Reflex_Vulkan/inc/NvLowLatencyVk.h` | `cardinal::tp::reflex` |
| `third_party/nvidia/ngx/`     | Streamline / NGX (DLSS family)| `include/sl.h` + `lib/x64/sl.interposer.lib` | `cardinal::tp::streamline` |
| `third_party/nvidia/ntc/`     | NTC tool + runtime binaries   | `bin/windows-x64/{libntc.dll,ntc-cli.exe}` | `CARDINAL_NTC_TOOL`, NTC DLL deploy |
| `third_party/nvidia/rtxpt/`   | RTX Path Tracing sample (opt) | `Rtxpt.exe` | `CARDINAL_RTXPT_BIN` (reference only) |

> The NVAPI headers/libs the engine uses for D3D Reflex live **inside** the
> RTXDI distribution (`third_party/nvidia/rtxdi/External/NVAPI`), so there is
> no separate NVAPI download.

## Verifying detection

After placing any SDK, reconfigure and watch the `[third_party]` log lines:

```
build configure release            (Windows)
./build.sh configure release       (Linux)
```

```
-- [third_party] RTXMU      ✓ Vulkan-only static lib
-- [third_party] Streamline ✓ DLSS family (N DLLs)
...
```

A `✓` means detected and wired. An SDK that prints nothing was not found and
its feature is disabled — a fully supported configuration. Do **not** add any
`third_party/` content to git; it stays local-only by design.
