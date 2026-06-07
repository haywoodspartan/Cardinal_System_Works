#include <cardinal/pack/pack.hpp>

#include <cardinal/core/sync/async.hpp>
#include <cardinal/core/diag/log.hpp>

#include <cardinal/core/std/algorithm.hpp>    // cardinal::sort/find/...
#include <cardinal/core/std/atomic.hpp>       // cardinal::atomic
#include <cardinal/core/std/cstdio.hpp>       // cardinal::fopen/fread/...
#include <cardinal/core/std/cstring.hpp>      // cardinal::memcmp
#include <cardinal/core/std/ctime.hpp>        // cardinal::time
#include <cardinal/core/std/filesystem.hpp>   // cardinal::fs, error_code
#include <cardinal/core/std/fstream.hpp>      // cardinal::ifstream/ios/streamsize
#include <cardinal/core/std/thread.hpp>       // cardinal::mutex/lock_guard
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace fs = cardinal::fs;

namespace cardinal::pack {

namespace {

void wr_u16(cardinal::vector<u8>& o, u16 v) {
    o.push_back(static_cast<u8>(v));
    o.push_back(static_cast<u8>(v >> 8));
}
void wr_u32(cardinal::vector<u8>& o, u32 v) {
    o.push_back(static_cast<u8>(v));
    o.push_back(static_cast<u8>(v >> 8));
    o.push_back(static_cast<u8>(v >> 16));
    o.push_back(static_cast<u8>(v >> 24));
}
void wr_u64(cardinal::vector<u8>& o, u64 v) {
    for (int i = 0; i < 8; ++i) o.push_back(static_cast<u8>(v >> (i * 8)));
}
u16 rd_u16(const u8* p) { return static_cast<u16>(p[0] | (p[1] << 8)); }
u32 rd_u32(const u8* p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
u64 rd_u64(const u8* p) {
    u64 r = 0;
    for (int i = 0; i < 8; ++i) r |= static_cast<u64>(p[i]) << (i * 8);
    return r;
}

u64 fnv1a64(const u8* data, usize n) noexcept {
    u64 h = 0xcbf29ce484222325ull;
    for (usize i = 0; i < n; ++i) { h ^= data[i]; h *= 0x100000001b3ull; }
    return h;
}

cardinal::atomic<u64> g_streams_inflight{0};
cardinal::atomic<u64> g_bytes_streamed {0};

}  // namespace

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------
Builder::Builder(cardinal::string path)
    : path_(cardinal::move(path)), tmp_path_(path_ + ".tmp")
{
    cardinal::error_code ec;
    fs::create_directories(fs::path(path_).parent_path(), ec);
    file_handle_ = static_cast<void*>(cardinal::fopen(tmp_path_.c_str(), "wb"));
    if (file_handle_ == nullptr) {
        cardinal::log::errorf("pack", "could not open %s for write", tmp_path_.c_str());
        return;
    }
    // Reserve header bytes at file start; we'll seek back and write the
    // real header in finalize().
    cardinal::vector<u8> zeros(kPackHeaderBytes, 0);
    cardinal::fwrite(zeros.data(), 1, zeros.size(),
                static_cast<FILE*>(file_handle_));
    data_cursor_ = kPackHeaderBytes;
}

Builder::~Builder() {
    if (file_handle_ != nullptr && !finalized_) {
        cardinal::fclose(static_cast<FILE*>(file_handle_));
        cardinal::error_code ec;
        fs::remove(tmp_path_, ec);
    }
}

bool Builder::add(const cardinal::string& key, cook::AssetType type,
                  const cardinal::vector<u8>& bytes)
{
    if (file_handle_ == nullptr) return false;
    PackEntryDesc e{};
    e.key    = key;
    e.type   = type;
    e.offset = data_cursor_;
    e.size   = bytes.size();
    e.hash   = fnv1a64(bytes.data(), bytes.size());

    cardinal::fwrite(bytes.data(), 1, bytes.size(),
                static_cast<FILE*>(file_handle_));
    data_cursor_ += bytes.size();

    // Replace existing entry with the same key (last-wins).
    bool replaced = false;
    for (auto& existing : entries_) {
        if (existing.key == key) { existing = e; replaced = true; break; }
    }
    if (!replaced) entries_.push_back(cardinal::move(e));
    return true;
}

bool Builder::add_file(const cardinal::string& key, cook::AssetType type,
                       const cardinal::string& source_path)
{
    cardinal::ifstream f(source_path, cardinal::ios::binary);
    if (!f) return false;
    f.seekg(0, cardinal::ios::end);
    const cardinal::streamsize n = f.tellg();
    // tellg → -1 on non-seekable stream (named pipe / char device path).
    // Cast to usize wraps to SIZE_MAX → bad_alloc. The pack builder
    // takes source_path from callers across the cook/dist pipeline;
    // a corrupted asset directory with a special-file entry could
    // crash the builder. The companion slurp() helper below already
    // has this guard (line 385); fix add_file too. Same 6a1640d
    // pattern as cook/shader read_all.
    if (n < 0) return false;
    f.seekg(0, cardinal::ios::beg);
    cardinal::vector<u8> bytes(static_cast<usize>(n));
    f.read(reinterpret_cast<char*>(bytes.data()), n);
    return add(key, type, bytes);
}

bool Builder::finalize(cardinal::string* error_out) {
    if (file_handle_ == nullptr) {
        if (error_out) *error_out = "builder failed to open file";
        return false;
    }
    // Index section.
    const u64 index_offset = data_cursor_;
    cardinal::vector<u8> idx;
    for (const auto& e : entries_) {
        wr_u16(idx, static_cast<u16>(e.key.size()));
        idx.insert(idx.end(), e.key.begin(), e.key.end());
        wr_u32(idx, static_cast<u32>(e.type));
        wr_u64(idx, e.offset);
        wr_u64(idx, e.size);
        wr_u64(idx, e.hash);
    }
    cardinal::fwrite(idx.data(), 1, idx.size(), static_cast<FILE*>(file_handle_));
    const u64 total_size = data_cursor_ + idx.size();

    // Header — seek back to file start and write.
    cardinal::fseek64(static_cast<FILE*>(file_handle_), 0, SEEK_SET);
    cardinal::vector<u8> hdr;
    hdr.reserve(kPackHeaderBytes);
    hdr.insert(hdr.end(), kPackMagic, kPackMagic + 4);
    wr_u32(hdr, kPackVersion);
    wr_u32(hdr, static_cast<u32>(entries_.size()));
    wr_u64(hdr, index_offset);
    wr_u64(hdr, total_size);
    wr_u32(hdr, 0);   // reserved
    while (hdr.size() < kPackHeaderBytes) hdr.push_back(0);
    cardinal::fwrite(hdr.data(), 1, hdr.size(), static_cast<FILE*>(file_handle_));

    cardinal::fclose(static_cast<FILE*>(file_handle_));
    file_handle_ = nullptr;

    cardinal::error_code ec;
    fs::rename(tmp_path_, path_, ec);
    if (ec) {
        if (error_out) *error_out = "rename failed: " + ec.message();
        return false;
    }
    finalized_ = true;
    cardinal::log::infof("pack",
        "wrote %s: %zu entries, %llu bytes",
        path_.c_str(), entries_.size(),
        static_cast<unsigned long long>(total_size));
    return true;
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------
cardinal::shared_ptr<Archive> Archive::open(const cardinal::string& path,
                                       cardinal::string* error_out)
{
    auto a = cardinal::shared_ptr<Archive>(new Archive());
    a->path_ = path;

    FILE* f = cardinal::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        if (error_out) *error_out = "could not open " + path;
        return nullptr;
    }
    // 64-bit offsets throughout: pack archives commonly exceed 2 GiB
    // in shipping builds, and std::fseek/ftell's `long` would silently
    // truncate to 31 bits on Windows LLP64 (parked-supervised item #1).
    // cardinal::fseek64 / ftell64 dispatch to _fseeki64 / fseeko.
    cardinal::fseek64(f, 0, SEEK_END);
    const i64 sz_signed = cardinal::ftell64(f);
    if (sz_signed < 0) {
        if (error_out) *error_out = "ftell64 failed";
        cardinal::fclose(f);
        return nullptr;
    }
    const u64 sz = static_cast<u64>(sz_signed);
    cardinal::fseek64(f, 0, SEEK_SET);
    if (sz < kPackHeaderBytes) {
        if (error_out) *error_out = "file too small for header";
        cardinal::fclose(f);
        return nullptr;
    }
    cardinal::vector<u8> hdr(kPackHeaderBytes);
    cardinal::fread(hdr.data(), 1, hdr.size(), f);
    if (cardinal::memcmp(hdr.data(), kPackMagic, 4) != 0) {
        if (error_out) *error_out = "bad magic";
        cardinal::fclose(f);
        return nullptr;
    }
    const u32 version      = rd_u32(hdr.data() + 4);
    const u32 entry_count  = rd_u32(hdr.data() + 8);
    const u64 index_offset = rd_u64(hdr.data() + 12);
    a->file_size_          = rd_u64(hdr.data() + 20);
    if (version != kPackVersion) {
        if (error_out) *error_out = "unsupported pack version";
        cardinal::fclose(f);
        return nullptr;
    }
    if (index_offset >= sz) {
        if (error_out) *error_out = "index_offset out of range";
        cardinal::fclose(f);
        return nullptr;
    }

    // Read index. index_offset is u64; pass directly via fseek64 so
    // the high bits aren't lost above 2 GiB.
    cardinal::fseek64(f, static_cast<i64>(index_offset), SEEK_SET);
    // entry_count is untrusted: a corrupt header could request a
    // multi-GB reserve and OOM the process. Each on-disk entry is at
    // least 30 bytes (2 key-length + 28 fixed tail; the key may be
    // empty), so the bytes after index_offset can hold no more than
    // (sz - index_offset) / 30. The loop below still reads the real
    // count and stops at EOF — this only caps the pre-allocation.
    // (index_offset < sz is guaranteed above, so no underflow.)
    const u64 max_possible = (sz - index_offset) / 30u;
    a->entries_.reserve(static_cast<usize>(
        entry_count < max_possible ? entry_count : max_possible));
    for (u32 i = 0; i < entry_count; ++i) {
        u8 kl_buf[2];
        if (cardinal::fread(kl_buf, 1, 2, f) != 2) break;
        const u16 kl = rd_u16(kl_buf);
        cardinal::string key(kl, '\0');
        if (cardinal::fread(key.data(), 1, kl, f) != kl) break;
        u8 tail[28];
        if (cardinal::fread(tail, 1, sizeof(tail), f) != sizeof(tail)) break;
        PackEntryDesc e{};
        e.key    = cardinal::move(key);
        e.type   = static_cast<cook::AssetType>(rd_u32(tail));
        e.offset = rd_u64(tail + 4);
        e.size   = rd_u64(tail + 12);
        e.hash   = rd_u64(tail + 20);
        a->by_key_[e.key] = static_cast<u32>(a->entries_.size());
        a->entries_.push_back(cardinal::move(e));
    }
    cardinal::fclose(f);
    cardinal::log::infof("pack",
        "opened %s: %zu entries (%llu bytes total)",
        path.c_str(), a->entries_.size(),
        static_cast<unsigned long long>(a->file_size_));
    return a;
}

Archive::~Archive() = default;

const PackEntryDesc* Archive::find(const cardinal::string& key) const noexcept {
    auto it = by_key_.find(key);
    return (it == by_key_.end()) ? nullptr : &entries_[it->second];
}
bool Archive::contains(const cardinal::string& key) const noexcept {
    return by_key_.find(key) != by_key_.end();
}

bool Archive::load_blocking(const cardinal::string& key,
                            cardinal::vector<u8>& out_bytes,
                            cardinal::string* error_out) const
{
    const PackEntryDesc* e = find(key);
    if (e == nullptr) {
        if (error_out) *error_out = "no such key: " + key;
        return false;
    }
    FILE* f = cardinal::fopen(path_.c_str(), "rb");
    if (f == nullptr) {
        if (error_out) *error_out = "could not reopen " + path_;
        return false;
    }
    // e->offset / e->size come from the (untrusted) index. Validate
    // against the real file size BEFORE resize() — a corrupt entry with
    // a huge size would otherwise bad_alloc, and a bogus offset would
    // seek/read out of range. (offset <= fsz checked first so the
    // fsz - offset subtraction can't underflow.)
    cardinal::fseek64(f, 0, SEEK_END);
    const i64 endp = cardinal::ftell64(f);
    if (endp < 0) {
        cardinal::fclose(f);
        if (error_out) *error_out = "could not size " + path_;
        return false;
    }
    const u64 fsz = static_cast<u64>(endp);
    if (e->offset > fsz || e->size > fsz - e->offset) {
        cardinal::fclose(f);
        if (error_out) *error_out = "entry out of range: " + key;
        return false;
    }
    cardinal::fseek64(f, static_cast<i64>(e->offset), SEEK_SET);
    out_bytes.resize(static_cast<usize>(e->size));
    const usize got = cardinal::fread(out_bytes.data(), 1, out_bytes.size(), f);
    cardinal::fclose(f);
    if (got != out_bytes.size()) {
        if (error_out) *error_out = "short read";
        return false;
    }
    g_bytes_streamed.fetch_add(out_bytes.size(), cardinal::memory_order_relaxed);
    return true;
}

cardinal::shared_future<cardinal::vector<u8>>
Archive::load_async(const cardinal::string& key) const
{
    g_streams_inflight.fetch_add(1, cardinal::memory_order_relaxed);
    auto self_path = path_;
    auto desc_copy = find(key) ? *find(key) : PackEntryDesc{};
    auto f = cardinal::async::async([self_path, desc_copy]() {
        cardinal::vector<u8> bytes;
        if (desc_copy.size == 0) {
            g_streams_inflight.fetch_sub(1, cardinal::memory_order_relaxed);
            return bytes;
        }
        FILE* fp = cardinal::fopen(self_path.c_str(), "rb");
        if (fp == nullptr) {
            g_streams_inflight.fetch_sub(1, cardinal::memory_order_relaxed);
            return bytes;
        }
        // Same untrusted-index guard as load_blocking: bound offset+size
        // by the real file size before resize() (returns empty on a
        // corrupt entry rather than bad_alloc / OOB read).
        cardinal::fseek64(fp, 0, SEEK_END);
        const i64 endp = cardinal::ftell64(fp);
        if (endp < 0 ||
            desc_copy.offset > static_cast<u64>(endp) ||
            desc_copy.size   > static_cast<u64>(endp) - desc_copy.offset) {
            cardinal::fclose(fp);
            g_streams_inflight.fetch_sub(1, cardinal::memory_order_relaxed);
            return bytes;
        }
        cardinal::fseek64(fp, static_cast<i64>(desc_copy.offset), SEEK_SET);
        bytes.resize(static_cast<usize>(desc_copy.size));
        cardinal::fread(bytes.data(), 1, bytes.size(), fp);
        cardinal::fclose(fp);
        g_bytes_streamed.fetch_add(bytes.size(), cardinal::memory_order_relaxed);
        g_streams_inflight.fetch_sub(1, cardinal::memory_order_relaxed);
        return bytes;
    });
    // Adapt the cardinal::async::Future<T> to cardinal::shared_future<T>.
    auto p = cardinal::make_shared<cardinal::promise<cardinal::vector<u8>>>();
    auto sf = p->get_future().share();
    f.then([p](const cardinal::vector<u8>& v) {
        p->set_value(v);
    });
    return sf;
}

Archive::Stats Archive::stats() const noexcept {
    Stats s{};
    s.entry_count = static_cast<u32>(entries_.size());
    s.file_size   = file_size_;
    for (const auto& e : entries_) s.total_payload += e.size;
    s.streams_in_flight = g_streams_inflight.load(cardinal::memory_order_relaxed);
    s.bytes_streamed    = g_bytes_streamed.load(cardinal::memory_order_relaxed);
    return s;
}

// ---------------------------------------------------------------------------
// Project Packer.
// ---------------------------------------------------------------------------
namespace {

cardinal::vector<u8> slurp(const cardinal::fs::path& p) {
    cardinal::ifstream f(p, cardinal::ios::binary | cardinal::ios::ate);
    if (!f) return {};
    const cardinal::streamsize n = f.tellg();
    f.seekg(0);
    cardinal::vector<u8> v(static_cast<cardinal::size_t>(n < 0 ? 0 : n));
    if (n > 0) f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

}  // namespace

DistResult distribute(const DistOptions& opt) {
    namespace fs = cardinal::fs;
    DistResult r;

    const fs::path root  = opt.project_root;
    const fs::path cooked = root / "cooked";
    if (!fs::exists(cooked)) {
        r.error = "no cooked/ directory under '" + opt.project_root +
                  "' — cook the project first";
        return r;
    }
    cardinal::error_code ec;
    fs::create_directories(opt.out_dir, ec);

    const fs::path pack_path = fs::path(opt.out_dir) /
                               (opt.pack_name + ".cpk");
    Builder b(pack_path.string());

    u64 hash_accum = 0;
    auto add_entry = [&](const cardinal::string& key, cook::AssetType type,
                         const cardinal::vector<u8>& bytes) {
        if (!b.add(key, type, bytes)) return false;
        // Cheap content fingerprint for the manifest (FNV-1a over bytes).
        u64 h = 1469598103934665603ull;
        for (u8 c : bytes) { h ^= c; h *= 1099511628211ull; }
        for (char c : key) { h ^= static_cast<u8>(c); h *= 1099511628211ull; }
        hash_accum ^= h;
        r.total_bytes += bytes.size();
        return true;
    };

    // 1) Every cooked asset, keyed by cooked-relative path minus ".cooked".
    for (auto it = fs::recursive_directory_iterator(cooked, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file()) continue;
        const fs::path& fp = it->path();
        if (fp.extension() != ".cooked") continue;
        cardinal::string rel = fs::relative(fp, cooked, ec).generic_string();
        if (rel.size() > 7 && rel.substr(rel.size() - 7) == ".cooked")
            rel.resize(rel.size() - 7);
        const cardinal::vector<u8> bytes = slurp(fp);
        cook::AssetType type = cook::AssetType::Unknown;
        cook::CookedAsset ca;
        if (cook::CookedAsset::deserialize(bytes, ca)) type = ca.type;
        if (add_entry(rel, type, bytes)) ++r.asset_count;
    }

    // 2) Optional save files (scene/world/sky) → save/<name>.
    if (opt.include_saves) {
        const fs::path save = root / "save";
        if (fs::exists(save)) {
            for (auto it = fs::recursive_directory_iterator(save, ec);
                 !ec && it != fs::recursive_directory_iterator(); ++it) {
                if (!it->is_regular_file()) continue;
                const cardinal::string rel =
                    fs::relative(it->path(), save, ec).generic_string();
                if (add_entry("save/" + rel, cook::AssetType::Unknown,
                              slurp(it->path())))
                    ++r.save_count;
            }
        }
    }

    cardinal::string ferr;
    if (!b.finalize(&ferr)) {
        r.error = "pack finalize failed: " + ferr;
        return r;
    }
    r.pack_path = pack_path.string();

    // 3) Stage caller-supplied runtime files (game .exe, DLLs) next to
    //    the pack so <out_dir> is a self-contained shippable folder.
    for (const cardinal::string& src : opt.extra_files) {
        const fs::path s = src;
        if (!fs::exists(s)) continue;
        fs::copy_file(s, fs::path(opt.out_dir) / s.filename(),
                      fs::copy_options::overwrite_existing, ec);
        if (!ec) ++r.extra_count;
    }

    // 4) Manifest the launcher validates before mounting the pack.
    const fs::path man = fs::path(opt.out_dir) / "distribution.cardinal";
    {
        cardinal::ofstream m(man, cardinal::ios::binary | cardinal::ios::trunc);
        m << "# Cardinal distribution v1\n";
        m << "app = \""    << opt.app_name       << "\"\n";
        m << "engine = \"" << opt.engine_version << "\"\n";
        m << "pack = \""   << opt.pack_name      << ".cpk\"\n";
        m << "assets = "   << r.asset_count      << "\n";
        m << "saves = "    << r.save_count       << "\n";
        m << "extra = "    << r.extra_count      << "\n";
        m << "bytes = "    << r.total_bytes      << "\n";
        char hb[24];
        cardinal::snprintf(hb, sizeof(hb), "0x%016llx",
                      static_cast<unsigned long long>(hash_accum));
        m << "hash = "  << hb << "\n";
        m << "built = " << static_cast<long long>(cardinal::time(nullptr)) << "\n";
    }
    r.manifest_path = man.string();
    r.ok = true;

    cardinal::log::infof("pack",
        "distribute: %u assets + %u saves + %u extra → %s (%llu bytes)",
        r.asset_count, r.save_count, r.extra_count, r.pack_path.c_str(),
        static_cast<unsigned long long>(r.total_bytes));
    return r;
}

}  // namespace cardinal::pack
