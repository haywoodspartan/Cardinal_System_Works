// =============================================================================
// Cardinal — terrain profile library round-trip smoke test.
//
// Registers the engine-shipped default profiles, dumps the entire library
// to disk, then loads it back into a fresh library and re-saves to a
// second file. Diff'ing the two files proves save/load is a faithful
// round-trip. Useful on its own for inspecting the on-disk format.
// =============================================================================
#include <cardinal/scene/terrain.hpp>

#include <cstdio>

using namespace cardinal::scene;

int main(int argc, char** argv) {
    const char* out_path  = (argc > 1) ? argv[1] : "terrain_profiles.txt";
    const char* roundtrip = (argc > 2) ? argv[2] : "terrain_profiles_roundtrip.txt";

    register_default_terrain_profiles();
    auto& lib = TerrainProfileLibrary::instance();
    std::printf("Registered %zu default profiles:\n", lib.all().size());
    for (const auto& n : lib.names()) std::printf("  - %s\n", n.c_str());

    if (!lib.save_to_file(out_path)) {
        std::fprintf(stderr, "save_to_file('%s') failed\n", out_path);
        return 1;
    }
    std::printf("\nWrote library to '%s'\n", out_path);

    // Round-trip: clear and reload.
    for (const auto& n : lib.names()) lib.remove(n.c_str());
    if (!lib.load_from_file(out_path)) {
        std::fprintf(stderr, "load_from_file failed\n");
        return 1;
    }
    if (!lib.save_to_file(roundtrip)) return 1;
    std::printf("Re-saved as '%s'  (compare with original to verify round-trip)\n",
                roundtrip);
    return 0;
}
