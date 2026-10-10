# Cardinal Rendering Roadmap — Luminous-class image, Nanite / Lumen-class cost

This document pins the renderer's target and the order of work toward it. The
bar is explicit: the **image quality of Square Enix's Luminous Engine** (its
grounded, filmic look: physically based shading, soft dynamic global
illumination, convincing skin / hair / eyes, volumetrics, temporal stability)
at the **per-frame cost model of Unreal 5's Nanite and Lumen** (virtualised
geometry culled and rasterised on the GPU; dynamic GI gathered from cheap
caches with screen-space traces first and world-space fallbacks second;
everything temporally amortised).

It is a program, not a patch. Each slice below is a self-contained commit
with a CPU reference, an HLSL kernel validated on both backends by
`tests/gpu_compute`, and an editor knob. For the engine layering see
[engine.md](engine.md); for the editor see [studio.md](studio.md).

---

## 1. Where AEGIS stands

The AEGIS pipeline (`runtime/render`, `graph::Graph` executed by `RhiBackend`
on the async compute queue) already has the *shape* of both systems. What
exists today, and whether it runs on the GPU:

| Area | Pass | GPU kernel | Notes |
|---|---|---|---|
| Geometry (Nanite track) | `GeometryClassifyPass`, `MeshletBuildPass`, `MeshletCullPass`, `ScreenSpaceErrorPass` | ✓ | Meshlet build + frustum / cone cull + screen-space-error LOD selection. |
| | `DrawIndirectGenPass` | ✓ | LDS-scan compaction, order-preserving. |
| | `TriangleBinPass` + binned `VisibilityBufferPass` | ✓ | Tile-binned V-Buffer raster, 160–230× the brute loop, byte-identical. |
| | `HiZBuildPass`, `HiZOcclusionTestPass` | ✓ (this slice) | One pass per mip; the occlusion test reads last frame's persistent pyramid and gates `TriangleBinPass`. Single-pass (no re-test of the culled set yet). See §4. |
| | `AdaptiveGeometryPass` | stub | FP32 / FP16 / FP8 / FP4 tier selection. |
| Shading (Lumen track) | `TiledLightCullPass`, `VBufResolvePass` | ✓ | Tile-lit Cook-Torrance PBR; 2×2 content-adaptive shading rate opt-in. |
| | **`ScreenSpaceGIPass`** | ✓ (this slice) | Screen-space AO + one-bounce diffuse over the resolve. See §3. |
| | `ReSTIRSamplePass` / `Spatial` / `Temporal` | CPU only | Direct-light reservoir reuse; GPU port blocked on `splitmix32` bit parity. |
| | `RadianceCachePass` (lighting module) | CPU only | Wraps the CPU `ProbeVolumeBaker` (6-axis ambient cubes) as a graph pass. |
| Post | `MotionVectorPass`, `TonemapPass`, `CompositePresentPass` | ✓ | |
| | `TAAPass`, `VolumetricFogPass`, `ColorGradingPass`, `DepthOfFieldPass` | CPU only | Passes exist but are **not wired into `AegisPipeline::build`**. |
| Scene feed | Bridge zero-copy streaming | ✓ | 200k-triangle stream at ~0.9 ms/frame into unified memory. |

Not present anywhere yet: image-based lighting / environment maps,
screen-space reflections, a world-space distance field, hardware ray-tracing
pipelines in the RHI (BLAS skeleton only), subsurface / hair / eye shading
models.

---

## 2. The two tracks

### Nanite track — geometry cost

Goal: triangle cost independent of scene size; overdraw bounded by the
screen, not the content.

1. **Hi-Z occlusion, first pass** (landed, §4): `HiZBuildPass` is one
   graph pass per mip, `HiZOcclusionTestPass` reads last frame's pyramid
   (a persistent graph buffer) through last frame's view-proj, adds a
   screen-rect frustum reject, and its per-meshlet visibility gates
   `TriangleBinPass`, so culled meshlets are never set up or rasterised.
   **Remaining: the second pass** — re-test the meshlets culled in pass 1
   against THIS frame's pyramid and rasterise the newly visible ones, which
   removes the one-frame pop on disocclusion. Acceptance: a camera sweep in
   `tests/aegis_bench` with zero V-Buffer difference against the unculled
   reference.
2. **GPU vertex transform**: stream model-space vertices + entity matrices
   instead of re-uploading world-space triangles each frame
   (`VertexTransformPass` exists, unwired). Cuts the 36 B/tri/frame stream.
3. **Cluster LOD streaming**: tie `vgeom` cluster hierarchies to the SSE pass
   so LOD is picked per cluster on the GPU, with a persistent-threads cull.
4. **Software raster for micro-triangles**: route sub-pixel triangles through
   a 64-bit atomic depth+id path inside the bin kernel.

### Lumen track — lighting cost

Goal: dynamic diffuse GI + specular at a fixed per-frame budget, amortised
over time.

1. **Screen-space final gather** (landed: `ScreenSpaceGIPass`, §3).
2. **TAA wired as history** (next after Hi-Z): put `TAAPass` into
   `AegisPipeline::build` with a bridge-persistent history buffer and the
   real `MotionVectorPass` output. Every later GI stage accumulates through
   it, so it lands before any probe work.
3. **World-space radiance cache on the GPU**: port `RadianceCachePass` so the
   probe volume is updated by compute (a few probes per frame, round-robin)
   and sampled by the GI pass for taps that leave the screen.
4. **Global distance field + cone traces**: voxelise cooked meshes into
   per-mesh SDFs at cook time, composite a clipmap global field on the GPU,
   and trace cones from the GI pass for long-range occlusion / sky
   visibility. This is Lumen's software path; no RT hardware required.
5. **Specular**: screen-space reflections against the V-Buffer, falling back
   to the radiance cache; hardware RT (RTXDI / NRD already vendored) once the
   RHI gains TLAS + RT pipelines.
6. **ReSTIR DI on the GPU**: resolve the `splitmix32` parity and wire the
   reservoirs into the resolve.

### Luminous look — surface and atmosphere

Independent of cost model; each is a shading-model or post slice:

- Image-based lighting (prefiltered environment + split-sum BRDF LUT).
- Shadows in the AEGIS path (cascaded for the sun, PCSS; the forward path
  already has a directional map).
- `VolumetricFogPass` wired, with shadowed scattering.
- Skin (pre-integrated / screen-space subsurface), eye (cornea refraction),
  hair (Marschner / Kajiya-Kay) material models in the resolve.
- Bloom, physical camera exposure, filmic grading via `ColorGradingPass`.

---

## 3. Landed in this slice: `ScreenSpaceGIPass`

`runtime/render/gpu_ssgi.{hpp,cpp}`, wired after `VBufResolvePass` when the
PBR pair (inverse view-proj + camera position) is present. One dispatch at
8×8 threads; per pixel it marches 8 compass directions × `steps` integer
pixel offsets (default 4 → 32 taps) through the V-Buffer depth, reconstructs
each sample's world position with the resolve's inverse view-proj, and for
every sample within `radius` that rises above the tangent plane
(`cos θ > 0.05`) accumulates a weight from `ssgi_sample_weight(dist, cos θ,
radius)`:

```
ao       = 1 − ao_strength · Σw / taps
radiance = resolve + albedo·ambient·(ao − 1) + albedo·(1 − metal)·intensity·Σ(w·L_sample) / taps
```

Sample positions are integer so CPU and GPU march the same pixels; the only
float work is reconstruction and dot products, which is why the harness can
compare the GPU bytes against the CPU reference (`EpsPbr` on radiance,
`EpsF32` on AO). Editor knobs live under *Global Illumination* in the
pipeline panel; `AegisConfig::screen_space_gi` / `AegisConfig::ssgi` carry
the same values for hosts.

The weighting function is the single place the look is decided — distance
falloff, cosine lobe, thickness heuristic — and is the first thing to tune
once the pass is on screen.

## 4. Landed: Hi-Z occlusion + visibility-gated binning

`graph::BufferDesc::persistent` lets a buffer keep its bytes across executes
on every backend (the CPU backends stop re-zeroing it; `RhiBackend` already
retained its cache). The Hi-Z pyramid is declared persistent, so the frame's
`HiZOcclusionTestPass` reads **last frame's** pyramid before this frame's
`HiZBuildPass` passes (one per mip, 11 dispatches at 1920×1080) overwrite it
from the new V-Buffer depth — the graph's write-after-read ordering runs the
test first with no host copies. The test projects each meshlet's bounds
twice: through the current view-proj for a screen-rect frustum reject, then
through `view_proj_prev` (the matrix that built the pyramid) to pick the mip
whose texel covers the rect and compare nearest depth against its max. A
host-fed `hiz_history_valid` flag keeps the first execute after any rebuild
or backend switch frustum-only. The output byte per meshlet gates
`TriangleBinPass`, which now skips culled 64-triangle groups entirely.

Validated by `tests/gpu_compute` on both backends with a two-frame scenario
(frame 2 samples frame 1's pyramid, visibility bytes exact), and by a
render-graph test that proves culling an occluded meshlet leaves the
V-Buffer byte-identical. Knob: *Hi-Z Occlusion Culling* under AEGIS.
