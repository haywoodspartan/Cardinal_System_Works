#pragma once

// =============================================================================
// Cardinal — Build-time asset cooker.
//
// Walks a project's `assets/` tree, runs an appropriate Cooker per file
// extension, writes the baked bytes to `cooked/<rel>.cooked`.
//
// Cookers are pluggable: register your own via Cooker::register_for(ext).
// Built-in cookers ship for .png/.tga (Texture), .obj/.cardmesh (Mesh),
// .hlsl (Shader). Each cooker produces a CookedAsset blob with a typed
// header so the runtime can identify it without filename guessing.
//
// Hash-based incremental: each source file's hash is recorded in a
// manifest; cook skips files whose source hash + cooker version match
// the last cook. Force a full rebuild via cook_all(force=true).
// =============================================================================

#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/std/containers.hpp>   // unordered_map/vector

namespace cardinal::project { class Project; }
namespace cardinal::shader  { class Compiler; }

namespace cardinal::cook {

// ---------------------------------------------------------------------------
// AssetType — every cooked blob starts with this so the runtime can
// dispatch to the right loader without filename guessing.
// ---------------------------------------------------------------------------
enum class AssetType : u32 {
    Unknown = 0,
    Texture = 1,
    Mesh    = 2,
    Shader  = 3,
    Audio   = 4,
    Material= 5,
};

const char* asset_type_name(AssetType t) noexcept;
AssetType   asset_type_for_extension(const cardinal::string& ext) noexcept;

// ---------------------------------------------------------------------------
// CookedAsset — typed blob produced by a Cooker.
//
// header bytes (16 total):
//   magic[4]      "COOK"
//   asset_type    u32
//   payload_size  u32  (bytes following the header)
//   cooker_ver    u32  (cooker semver — for invalidation)
// ---------------------------------------------------------------------------
inline constexpr u32 kCookHeaderBytes = 16;
inline constexpr char kCookMagic[4]   = {'C','O','O','K'};

struct CookedAsset {
    AssetType        type{AssetType::Unknown};
    u32              cooker_version{1};
    cardinal::vector<u8>  payload;        // post-header content
    cardinal::string      diagnostics;    // human-readable "what we did"

    // Build the on-disk byte sequence (header + payload).
    cardinal::vector<u8>  serialize() const;
    static bool      deserialize(const cardinal::vector<u8>& bytes, CookedAsset& out);
};

// ---------------------------------------------------------------------------
// Cooker — interface; one per AssetType.
//
// `cook(source_bytes, source_path, ctx)` consumes the source file's bytes
// and returns the cooked output. ctx carries optional knobs (compression
// quality, mip count, etc.) that the cooker may consult.
// ---------------------------------------------------------------------------
struct CookContext {
    cardinal::string project_root;
    cardinal::string asset_path;          // absolute path to source
    cardinal::shader::Compiler* shader_compiler{nullptr};   // for shader cooker
    // Optional knobs.
    u32         texture_target_width{0};      // 0 = keep source size
    bool        texture_generate_mips{true};
    bool        mesh_generate_meshlets{true};
};

class Cooker {
public:
    virtual ~Cooker() = default;
    virtual AssetType   asset_type()    const noexcept = 0;
    virtual u32         cooker_version() const noexcept { return 1; }
    virtual const char* name()          const noexcept = 0;
    virtual CookedAsset cook(const cardinal::vector<u8>& source_bytes,
                             const CookContext& ctx) = 0;
};

// Built-in factories.
cardinal::unique_ptr<Cooker> make_texture_cooker();
cardinal::unique_ptr<Cooker> make_mesh_cooker();
cardinal::unique_ptr<Cooker> make_shader_cooker();
cardinal::unique_ptr<Cooker> make_audio_cooker();

// ---------------------------------------------------------------------------
// CookerRegistry — register cookers by file extension. Multi-extension
// mappings (e.g. "png" + "jpg" -> Texture) are supported.
// ---------------------------------------------------------------------------
class CookerRegistry {
public:
    void register_cooker(const cardinal::string& extension,
                         cardinal::shared_ptr<Cooker> cooker);
    Cooker* find_for_extension(const cardinal::string& extension) const;

    // Install the built-in set.
    void register_builtin(cardinal::shader::Compiler* shader_compiler = nullptr);

private:
    cardinal::unordered_map<cardinal::string, cardinal::shared_ptr<Cooker>> by_ext_;
};

// ---------------------------------------------------------------------------
// Driver — one entry point that walks the project and cooks everything.
//
// Returns per-file results. Up-to-date files are skipped (status=Skipped).
// ---------------------------------------------------------------------------
struct CookResult {
    enum class Status : u32 { Skipped, Cooked, Failed };
    Status      status{Status::Skipped};
    cardinal::string source_relpath;
    cardinal::string cooked_relpath;
    AssetType   type{AssetType::Unknown};
    u64         source_hash{0};
    u32         payload_bytes{0};
    cardinal::string diagnostics;
    cardinal::string error_message;
};

struct CookManifest {
    // source_relpath -> source_hash from the last successful cook.
    cardinal::unordered_map<cardinal::string, u64> hashes;
    bool load_from(const cardinal::string& path);
    bool save_to  (const cardinal::string& path) const;
};

struct CookSummary {
    u32 cooked_count{0};
    u32 skipped_count{0};
    u32 failed_count{0};
    u64 total_payload_bytes{0};
};

CookSummary cook_all(const cardinal::project::Project& proj,
                     const CookerRegistry& registry,
                     cardinal::vector<CookResult>& out_results,
                     bool force = false,
                     CookContext template_ctx = {});

}  // namespace cardinal::cook
