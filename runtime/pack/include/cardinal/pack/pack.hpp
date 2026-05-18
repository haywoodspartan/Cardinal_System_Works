#pragma once

// =============================================================================
// Cardinal — Asset pack file format.
//
// Single-file streamable archive. Optimised for game shipping:
//   - One mmap-friendly contiguous file (no per-asset opens)
//   - Index lives at file END so `Builder::add(..)` can append without
//     rewriting earlier blobs
//   - Per-entry hash for integrity check + dedup at pack time
//   - Async streaming via cardinal::async::async — request returns a Future
//
// Format (binary, little-endian):
//
//   header (32 bytes):
//     magic[4]       "CPK1"
//     version        u32
//     entry_count    u32
//     index_offset   u64
//     total_size     u64
//     reserved       u32
//
//   data section: contiguous payloads
//     payload[0]: bytes
//     payload[1]: bytes
//     ...
//
//   index section (at index_offset):
//     for each entry:
//       key_len      u16
//       key (utf-8)  bytes
//       asset_type   u32
//       offset       u64
//       size         u64
//       hash         u64
// =============================================================================

#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/future.hpp>       // cardinal::future/promise
#include <cardinal/core/containers.hpp>   // unordered_map/vector
#include <cardinal/cook/cook.hpp>

namespace cardinal::pack {

inline constexpr u32  kPackVersion        = 1;
inline constexpr char kPackMagic[4]       = {'C','P','K','1'};
inline constexpr u32  kPackHeaderBytes    = 32;

struct PackEntryDesc {
    cardinal::string         key;            // logical name (path or id)
    cook::AssetType     type{cook::AssetType::Unknown};
    u64                 offset{0};
    u64                 size  {0};
    u64                 hash  {0};
};

// ---------------------------------------------------------------------------
// Builder — append entries, write atomically.
//
// Usage:
//   Builder b("game/pack/main.cpk");
//   b.add("textures/wood",  cook::AssetType::Texture, bytes);
//   b.add("meshes/teapot",  cook::AssetType::Mesh,    other_bytes);
//   b.finalize();
// ---------------------------------------------------------------------------
class Builder {
public:
    explicit Builder(cardinal::string path);
    ~Builder();
    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;

    // Returns true on success. `bytes` is copied; caller can free
    // immediately. Duplicate keys overwrite the previous entry's index slot
    // (the old payload still occupies space — this is "append plus index"
    // semantics, like a tar file).
    bool add(const cardinal::string& key, cook::AssetType type,
             const cardinal::vector<u8>& bytes);

    // Convenience: read a file from disk + add as a single entry.
    bool add_file(const cardinal::string& key, cook::AssetType type,
                  const cardinal::string& source_path);

    // Write index + header. Renames the temporary to the final path
    // atomically. Returns false on any I/O error.
    bool finalize(cardinal::string* error_out = nullptr);

    u32 entry_count() const noexcept { return static_cast<u32>(entries_.size()); }
    u64 bytes_written_so_far() const noexcept { return data_cursor_; }

private:
    cardinal::string                 path_;
    cardinal::string                 tmp_path_;
    cardinal::vector<PackEntryDesc>  entries_;
    void*                       file_handle_{nullptr};   // FILE*
    u64                         data_cursor_{0};
    bool                        finalized_{false};
};

// ---------------------------------------------------------------------------
// Archive — read side.
//
// Open() validates the header + loads the index. Per-entry payloads are
// fetched on demand; load_blocking() reads synchronously, load_async()
// returns a Future<cardinal::vector<u8>> that resolves on a worker thread.
// ---------------------------------------------------------------------------
class Archive {
public:
    static cardinal::shared_ptr<Archive> open(const cardinal::string& path,
                                         cardinal::string* error_out = nullptr);
    ~Archive();

    const cardinal::vector<PackEntryDesc>& entries() const noexcept { return entries_; }
    const PackEntryDesc*              find(const cardinal::string& key) const noexcept;
    bool                              contains(const cardinal::string& key) const noexcept;
    u32                               entry_count() const noexcept { return static_cast<u32>(entries_.size()); }
    const cardinal::string&                path() const noexcept { return path_; }

    // Synchronous blocking read of a full entry.
    bool load_blocking(const cardinal::string& key,
                       cardinal::vector<u8>& out_bytes,
                       cardinal::string* error_out = nullptr) const;

    // Async load via cardinal::async — completes on a worker thread.
    // Empty vector on failure.
    cardinal::shared_future<cardinal::vector<u8>> load_async(const cardinal::string& key) const;

    struct Stats {
        u32 entry_count{0};
        u64 file_size{0};
        u64 total_payload{0};
        u64 streams_in_flight{0};   // running async loads
        u64 bytes_streamed{0};
    };
    Stats stats() const noexcept;

private:
    Archive() = default;

    cardinal::string                        path_;
    cardinal::vector<PackEntryDesc>         entries_;
    cardinal::unordered_map<cardinal::string, u32> by_key_;   // key -> index in entries_
    u64                                file_size_{0};
};

// ---------------------------------------------------------------------------
// Project Packer — one-call client distribution build.
//
// Turns a cooked project tree into a shippable client folder:
//
//   <out_dir>/
//     <pack_name>.cpk          all cooked assets (+ optional saves)
//     distribution.cardinal    text manifest the launcher validates
//     <extra files…>           caller-supplied (game .exe, runtime DLLs)
//
// It recursively gathers `<project_root>/cooked/ * *.cooked`, keys each
// entry by its cooked-tree-relative path (sans the `.cooked` suffix) so
// asset::Registry::mount_archive resolves the same keys the dev-mode
// directory mount used — dev and shipping stay key-identical. The
// cooked blob's own type header is read back so pack entries carry the
// right cook::AssetType. Save files (scene/world/sky) are optionally
// bundled under `save/<name>` so a distributed build boots straight
// into its shipped level. extra_files are copied verbatim (basename)
// for staging the executable + NVIDIA/runtime DLLs next to the pack.
// ---------------------------------------------------------------------------
struct DistOptions {
    cardinal::string              project_root;          // holds cooked/ , save/
    cardinal::string              out_dir;               // created if absent
    cardinal::string              pack_name{"client"};   // → out_dir/<name>.cpk
    bool                     include_saves{true};
    cardinal::vector<cardinal::string> extra_files;           // abs paths, copied in
    cardinal::string              app_name{"Cardinal Game"};
    cardinal::string              engine_version{"0.1.0"};
};

struct DistResult {
    bool        ok{false};
    cardinal::string pack_path;
    cardinal::string manifest_path;
    u32         asset_count{0};
    u32         save_count{0};
    u32         extra_count{0};
    u64         total_bytes{0};
    cardinal::string error;
};

// Build the distribution. Idempotent: overwrites <out_dir> contents for
// a clean shippable bundle each call.
DistResult distribute(const DistOptions& opt);

}  // namespace cardinal::pack
