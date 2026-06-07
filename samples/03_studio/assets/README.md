# Studio sample assets

Drop-in assets for trying the Studio's **File → Import** flow.

## 3D models (Blender / Maya / any DCC)

The importer takes the interchange formats every DCC exports:

| Tool                | Export as           |
|---------------------|---------------------|
| Blender             | glTF (.glb/.gltf), OBJ |
| Maya / Autodesk     | OBJ, glTF, FBX*     |
| ZBrush / Substance  | OBJ, glTF           |
| anything else       | OBJ / glTF          |

\* FBX is recognised but its binary geometry backend is not implemented yet —
an FBX import resolves materials and reports the mesh as pending. Export OBJ or
glTF from the DCC for working geometry today.

**Try it:** `File → Import 3D Asset…`, then paste the absolute path to
`sample_pyramid.obj` (next to this file). It spawns in front of the camera as a
first-class actor (Outliner / gizmo / undo all work, like any primitive).

`sample_pyramid.obj` + `sample_pyramid.mtl` exercise the OBJ/MTL path: geometry
(`v`/`vn`/`f`, n-gon fan triangulation) plus a metallic-roughness material
(`Kd`/`Pr`/`Pm`).

## Megascans (Quixel Bridge)

`File → Import Megascans…`, then paste the path to a Megascans **asset folder**
or its manifest `.json`. The importer is tolerant to Bridge version drift:

- reads the map list from `maps` **or** `components`;
- accepts `uri` / `path` / `file` for each map;
- routes albedo / normal / roughness (gloss → inverted) / metalness /
  ao / displacement / emissive / opacity / ORM-packed maps to the material;
- for 3D assets, imports the lowest non-FBX mesh LOD;
- falls back to scanning `<name>_<RES>_<MapType>.<ext>` filenames when the JSON
  omits or stales a map.

> Note: PBR texture **maps are imported and stored on the material** today, but
> the renderer does not yet sample them (that is a later GPU phase). Imported
> geometry renders immediately with its base/vertex colour.
