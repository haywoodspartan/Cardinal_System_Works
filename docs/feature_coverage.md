# Feature coverage — the Luminous Engine list against Cardinal

Target set by the user (2026-10-13): the published Luminous Engine feature
list, delivered on AEGIS at Nanite/Lumen cost. This page is the honest map of
what exists, what is half there, and what is missing, with the order the
missing pieces should land in. Status words:

- **HAVE** — runs in the engine on the GPU path a user sees (AEGIS or the
  forward fallback), with tests.
- **PARTIAL** — exists but is CPU-only, unwired (never added by
  `AegisPipeline::build`), knob-only (a Studio setting nothing reads), or
  covers a fraction of the feature. The row says which.
- **MISSING** — nothing beyond a comment or a roadmap line.

Paths are relative to the repo root. Audited 2026-10-13 against commit
3112f9c plus the per-pass GPU timestamp slice.

## 1. Coverage

| Feature | Status | Where | Evidence |
|---|---|---|---|
| SSAO | HAVE (GPU, wired) | `runtime/render/src/gpu_ssgi.cpp` | AO term of `ScreenSpaceGIPass`. The separate `ssao_enabled` knob is knob-only. |
| Spherical-harmonics ambient | MISSING | `runtime/lighting/include/cardinal/lighting/bake.hpp` | 6-axis ambient cubes only; CPU; tests only. |
| Skeletal animation | PARTIAL | `runtime/scene/src/skin.cpp`, `runtime/render/src/gpu_skin.cpp`, `runtime/import/src/import_gltf.cpp` | CPU skinning in ForwardRenderer + bridge; GPU `SkinPass` unwired; glTF skins import but no animation channels; nothing writes `Entity::pose`. |
| Animation state machines / blend trees | MISSING | `runtime/anim/include/cardinal/anim/anim.hpp` | Generic `Curve`/`Clip`/`Player` only. |
| Full-body IK, procedural animation | MISSING | — | |
| Facial (muscle) animation, blendshapes | MISSING | `runtime/import/src/import_fbx.cpp` | FBX blendshapes deferred; glTF morph targets not imported. |
| Motion capture / facial capture import | MISSING | — | |
| MSAA | MISSING | `runtime/rhi/src/vulkan/vulkan.cpp`, `runtime/rhi/src/d3d12/device.cpp` | Sample count hard-coded to 1; `msaa_samples` knob-only. |
| FXAA / image-based AA | PARTIAL (CPU, unwired) | `runtime/postfx/src/postfx.cpp` | `make_fxaa_pass`; `postfx::Chain` used only by `tests/postfx`. |
| TAA | PARTIAL (CPU-only, unwired) | `runtime/render/src/gpu_postvol.cpp` | `TAAPass` has no `compute_hlsl`. Motion vectors ARE wired (`MotionVectorPass`). |
| Reactive AI (perception) | HAVE (basic) | `runtime/ai/include/cardinal/ai/ai.hpp` | Sight-cone + hearing sensors, stimuli. |
| Behaviour trees | HAVE (core nodes) | same | Sequence/Selector/Inverter/Wait/Custom + Blackboard; Parallel/Repeater declared, not implemented. No FSM. |
| GPGPU AI | MISSING | `runtime/mass` | Mass ECS boids are CPU. |
| Deferred / light pre-pass | HAVE (as visibility buffer) | `runtime/render/src/gpu_visbuf.cpp`, `gpu_aegis.cpp` | V-Buffer (depth/prim/mat/normal/motion) + tiled `VBufResolvePass` = deferred texturing; forward fallback in `scene::ForwardRenderer`. |
| Level editor | HAVE | `runtime/ui/src/panels/scene/*`, `runtime/edit`, `samples/03_studio` | Outliner, Inspector, gizmos, prefabs, editor modes, brushes. |
| Animation editor | PARTIAL | `runtime/ui/src/panels/animation/` | Curve editor + sequencer; no rig / state-graph editor. |
| Shader authoring | PARTIAL | `runtime/ui/src/panels/scripting/shader_compiler.cpp` | HLSL text + compiler; no material node graph. |
| Blueprint / ICE | PARTIAL | `runtime/blueprint`, `runtime/ice` | DSL round-trip + VM; ICE evaluator; neither has a node canvas yet. |
| D3D12, Vulkan | HAVE | `runtime/rhi/src/{d3d12,vulkan}` | Graphics + compute + async compute, both. No DX11 (not planned). |
| Day-night cycle | HAVE | `runtime/sky`, `runtime/ui/src/panels/rendering/sky_panel.cpp` | Keyframed sun + ambient into `LightSet`; gradient sky, no physical atmosphere. |
| Volumetric clouds | MISSING | — | |
| Environment maps / cube maps / IBL | MISSING | `runtime/scene/src/renderer.cpp` | "no cubemap / IBL yet"; `rhi::TextureDesc` is 2D, one mip; `ibl_*` knobs knob-only. |
| Eye shader | MISSING | — | |
| Real-time GI | PARTIAL | `runtime/render/src/gpu_ssgi.cpp` | One-bounce screen-space diffuse (GPU, wired). No world-space GI, no glossy indirect. |
| Probe / radiance cache | PARTIAL (CPU, unwired) | `runtime/lighting/src/gpu_radiance_cache.cpp` | `RadianceCachePass`. |
| Lightmap / irradiance bake | PARTIAL (CPU, tests only) | `runtime/lighting/src/bake.cpp` | `LightmapBaker`, `ProbeVolumeBaker`; renderer never samples them. See `project_gpu_lighting_plan`. |
| Path-traced bake, ray-bundle bake | PARTIAL / MISSING | `runtime/lighting/src/raytracer.cpp`, `gpu_lighting.cpp` | CPU `PathTracer` works; DXR shader is a string. |
| ReSTIR DI | PARTIAL (CPU-only, never runs) | `runtime/render/src/gpu_restir.cpp` | Gated on `restir_world_pos` the bridge never supplies. |
| Hardware ray tracing | PARTIAL (skeleton) | `runtime/rhi/src/vulkan/vulkan.cpp`, `runtime/rhi/src/d3d12/internal.hpp` | Vulkan BLAS real; TLAS TODO; D3D12 returns null; no RT shader stages or pipelines. |
| GPGPU infrastructure | HAVE | `runtime/rhi/src/*/compute.cpp`, `tests/gpu_compute` | Async compute lane, 2 submissions deep, per-pass GPU timestamps. |
| GPU particles / hair / cloth / VFX | MISSING | `runtime/particles` | CPU particle system, never drawn in the viewport. |
| GPU physics | PARTIAL (CPU-only, unwired) | `runtime/physics/src/gpu_physics.cpp` | Integration/broadphase/contact passes as source strings. |
| Hair (sim, shading, spline geometry) | MISSING | `runtime/render/include/cardinal/render/gpu_aegis.hpp` | Only a `GeometryClass::Foliage` tag. |
| HDR lighting + tonemap | HAVE | `gpu_aegis.cpp`, `renderer.cpp`, `algos.cpp` | Float radiance, ACES, GPU `TonemapPass`. |
| Exposure | PARTIAL | `runtime/scene/include/cardinal/scene/light.hpp` | Manual only; AEGIS `exposure` knob not read into config; no auto-exposure. |
| HDR display output | MISSING | `runtime/rhi/src/vulkan/swapchain.cpp` | sRGB only. |
| Refraction, glass | MISSING | — | No transmission/IOR, no blend state. |
| Reflection (SSR, planar, cube) | MISSING | `renderer.cpp` | Hemisphere specular only; `ssr_enabled` knob-only. |
| Volumetric lighting / fog | PARTIAL (CPU-only, unwired) | `gpu_postvol.cpp` | Froxel `VolumetricFogPass`, unshadowed. |
| Per-pixel lighting, Cook-Torrance PBR | HAVE | `renderer.cpp`, `VBufResolvePass` | GGX in both paths. |
| Order-independent transparency | MISSING | `runtime/rhi/include/cardinal/rhi/rhi.hpp` | `PipelineDesc` has no blend state. |
| Subsurface scattering (skin, cloth) | MISSING | — | |
| Mesh LOD / meshlet LOD / virtual geometry | PARTIAL | `runtime/edit`, `gpu_aegis.cpp`, `runtime/vgeom` | `ScreenSpaceErrorPass` writes `out_lod` nothing reads; vgeom cluster DAG used by the forward path only. |
| glTF / GLB | HAVE | `runtime/import/src/import_gltf.cpp` | Meshes, PBR, nodes, skins; no animations. |
| FBX, USD | PARTIAL | `import_fbx.cpp`, `import_usd.cpp` | FBX binary geometry only; USDA + USDZ root, no USDC. Maya reaches us through these. |
| PBR materials | PARTIAL | `renderer.cpp` | Metallic-roughness, but only albedo sampled; normal/MR/AO/emissive not bound. |
| Hi-Z occlusion | HAVE (GPU, wired) | `gpu_visbuf.cpp` | Build + test gate `TriangleBinPass`; single pass. |
| Point-cloud / imperfect shadow maps | MISSING | — | |
| Shadow mapping + PCF | HAVE (forward path only) | `renderer.cpp` | 2048 D32 directional, 3×3 PCF. AEGIS has no shadows; no cascades, no point/spot shadows. |
| Skin shader | MISSING | — | Roadmap line only. |
| Tessellation | PARTIAL | `runtime/render/include/cardinal/render/tess.hpp`, `geo.cpp` | Factor maths + CPU PN-triangle subdivide; no hull/domain stages (mesh shaders are the intended route). |
| 2K/4K/8K textures | PARTIAL | `runtime/cook/src/cook.cpp` | Up to 16384², RGBA8, mip 0 only; BCn size maths only. |
| Virtual texturing | PARTIAL | `runtime/vt` | CPU page table + tile cache + async decode; no GPU indirection/feedback. |
| Colour grading, DoF | PARTIAL (CPU, unwired) | `runtime/render/src/gpu_color.cpp` | 3D LUT grading; CoC-driven DoF. |
| Bloom, chromatic aberration, vignette, grain | PARTIAL (CPU, unwired) | `runtime/postfx/src/postfx.cpp` | |
| Bokeh, motion blur | MISSING | — | Motion vectors exist; nothing blurs. |

Doc drift found during the audit: `docs/rendering_roadmap.md` §1 says "BLAS
skeleton only" (true for Vulkan; D3D12 has none); `engine.md` says TAA is in
the AEGIS chain (it is not); header comments in `import_gltf.cpp` (skins
"out of scope") and `scene.hpp` (roughness "not read by the PS") are stale.

## 2. Build order

Ordered by what unblocks the most visible features per slice, under the
GPU-first rule (every pass lands on AEGIS through RHI + HLSL, never as a CPU
stub) and the AEGIS cost discipline (per-pass GPU timestamps are the gate:
a slice reports its ms before it merges).

1. **RHI texture + blend foundations.** Cube / 3D / array textures with mip
   chains, BC formats in the cooker, a blend state in `PipelineDesc`, FP16
   swapchain. Everything in rows 4-7 below needs at least one of these.
2. **Lighting stack on AEGIS.** Cascaded shadow maps feeding
   `VBufResolvePass` (AEGIS currently has no shadows), IBL (prefiltered
   environment + BRDF LUT), SSR. Then TAA on the GPU (motion vectors are
   already there) and the post stack ported from the CPU passes: bloom, DoF,
   motion blur, grading, auto-exposure.
3. **Characters.** Animation runtime: glTF/FBX clip import, pose sampling,
   blend tree + state machine, IK and procedural layers, state-graph editor.
   Wire `SkinPass`. Then skin SSS, eye shader, hair strands (sim + Marschner
   shading). Blendshapes for faces.
4. **World.** GPU particles (sim + depth collisions + sorted sprites),
   cloth with wind, volumetric fog wired + shadowed, physical atmosphere and
   clouds, water with ripples and planar/SSR reflection.
5. **Ray tracing.** RT shader stages and pipelines in the RHI, TLAS on both
   backends, DXR on D3D12; then RT shadows, reflections, and world-space GI
   (radiance cache + distance field) on top of the screen-space pass.

The in-flight slice is the last row of §1 of `rendering_roadmap.md`
(per-pass GPU timestamps), which is the gate for every slice above.
