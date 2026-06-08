// =============================================================================
// cardinal_import — command-line asset importer (a UE5-commandlet-style tool).
//
// Imports one or more 3D / material assets through cardinal::import (auto-detect
// by extension: OBJ / glTF / GLB / FBX / USD(A/Z) / Megascans) and prints a
// summary per file. Exit code = number of files that failed to import (0 = all
// ok), so it doubles as a CI asset-validation step.
//
//   cardinal_import <file> [<file> ...]
// =============================================================================

#include <cardinal/import/import.hpp>

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: cardinal_import <file.gltf|glb|obj|fbx|usd|usdz|...> [more...]\n");
        return 2;
    }

    int failures = 0;
    for (int i = 1; i < argc; ++i) {
        cardinal::string err;
        const cardinal::import::ImportScene scn =
            cardinal::import::import_file(argv[i], &err);

        if (!scn.ok) {
            std::printf("[FAIL] %s\n       %s\n", argv[i],
                        err.empty() ? "import failed" : err.c_str());
            ++failures;
            continue;
        }

        std::printf("[ OK ] %s\n", argv[i]);
        std::printf("       format=%s  meshes=%zu  materials=%zu  nodes=%zu  "
                    "verts=%u  tris=%u\n",
                    scn.source_format.c_str(),
                    static_cast<size_t>(scn.meshes.size()),
                    static_cast<size_t>(scn.materials.size()),
                    static_cast<size_t>(scn.nodes.size()),
                    scn.total_vertices(), scn.total_triangles());
        if (!scn.diagnostics.empty())
            std::printf("       %s\n", scn.diagnostics.c_str());
    }

    std::printf("cardinal_import: %d file(s), %d failed\n", argc - 1, failures);
    return failures == 0 ? 0 : 1;
}
