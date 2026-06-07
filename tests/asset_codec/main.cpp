// =============================================================================
// Cardinal — deterministic asset-codec regression suite.
//
// asset::codec encode/decode is the typed serialiser inside every cooked
// asset payload (texture/mesh/shader/material). It is THE on-disk +
// in-memory format — a regression silently corrupts every loaded asset
// or breaks pack/save compatibility. The codec is pure and byte-exact
// (little-endian u32 headers, IEEE float via memcpy), so this suite pins
// exact byte layout, full round-trips (bit-exact floats), short-buffer
// rejection, and the documented 36-byte material back-compat path. It
// also pins the disk-free register_material/load_material cache (pointer
// identity, last-write-wins) and stats. The directory/archive paths are
// filesystem/pack-coupled and out of scope (only their deterministic
// negative queries are checked). Pure CPU, headless. Exit 0 = all pass.
// =============================================================================

#include <cardinal/asset/asset.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <vector>

namespace {

namespace as = cardinal::asset;
namespace ck = cardinal::cook;
using Vec3   = cardinal::scene::Vec3;
using B      = std::vector<cardinal::u8>;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("acodec", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool veq(const Vec3& a, const Vec3& b) {            // bit-exact (memcpy codec)
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// ---- texture: exact layout + round-trip + bounds ------------------
void test_texture() {
    as::TextureAsset t;
    t.width = 2; t.height = 1; t.channels = 4;
    t.rgba = { 10,20,30,40, 50,60,70,80 };           // 8 bytes
    B enc = as::codec::encode_texture(t);
    CHECK(enc.size() == sz(16 + 8));
    // Header is little-endian u32 ×4 then the raw rgba.
    CHECK(enc[0]==2 && enc[1]==0 && enc[2]==0 && enc[3]==0);   // width
    CHECK(enc[4]==1 && enc[5]==0 && enc[6]==0 && enc[7]==0);   // height
    CHECK(enc[8]==4 && enc[9]==0 && enc[10]==0 && enc[11]==0); // channels
    CHECK(enc[12]==8 && enc[13]==0 && enc[14]==0 && enc[15]==0);// rgba size
    CHECK(enc[16]==10 && enc[23]==80);

    as::TextureAsset out;
    CHECK(as::codec::decode_texture(enc, out));
    CHECK(out.width==2u && out.height==1u && out.channels==4u);
    CHECK(out.rgba == t.rgba);

    // Endianness: a multi-byte width lands LSB-first.
    as::TextureAsset w; w.width = 0x04030201u; w.height = 1; w.channels = 1;
    B we = as::codec::encode_texture(w);
    CHECK(we[0]==0x01 && we[1]==0x02 && we[2]==0x03 && we[3]==0x04);
    as::TextureAsset wo;
    CHECK(as::codec::decode_texture(we, wo) && wo.width == 0x04030201u);

    // Empty rgba round-trips.
    as::TextureAsset e; e.width=0; e.height=0; e.channels=4;
    B ee = as::codec::encode_texture(e);
    CHECK(ee.size() == sz(16));
    as::TextureAsset eo;
    CHECK(as::codec::decode_texture(ee, eo) && eo.rgba.empty());

    // Short buffer + lying length → reject (no crash).
    as::TextureAsset bo;
    CHECK(!as::codec::decode_texture(B{}, bo));
    CHECK(!as::codec::decode_texture(B(15, 0), bo));
    B lie(16, 0); lie[12]=100;                        // claims 100 rgba bytes
    CHECK(!as::codec::decode_texture(lie, bo));
}

// ---- mesh: round-trip + empty + bounds ----------------------------
void test_mesh() {
    as::MeshAsset m;
    as::MeshVertex v0; v0.position={1.5f,-2.25f,3.0f};
    v0.normal={0.0f,1.0f,0.0f}; v0.color={0.1f,0.2f,0.3f};
    as::MeshVertex v1; v1.position={-4.0f,5.5f,-6.75f};
    v1.normal={1.0f,0.0f,0.0f}; v1.color={0.9f,0.8f,0.7f};
    m.vertices = { v0, v1 };
    m.indices  = { 0u,1u,2u, 2u,1u,0u };
    B enc = as::codec::encode_mesh(m);
    CHECK(enc.size() == sz(4 + 2*9*4 + 4 + 6*4));     // 104

    as::MeshAsset out;
    CHECK(as::codec::decode_mesh(enc, out));
    CHECK(out.vertices.size() == sz(2));
    CHECK(veq(out.vertices[0].position, v0.position));
    CHECK(veq(out.vertices[0].normal,   v0.normal));
    CHECK(veq(out.vertices[0].color,    v0.color));
    CHECK(veq(out.vertices[1].position, v1.position));
    CHECK(veq(out.vertices[1].color,    v1.color));
    CHECK(out.indices == m.indices);

    // Empty mesh: 4 (vc=0) + 4 (ic=0).
    as::MeshAsset em;
    B ee = as::codec::encode_mesh(em);
    CHECK(ee.size() == sz(8));
    as::MeshAsset eo;
    CHECK(as::codec::decode_mesh(ee, eo));
    CHECK(eo.vertices.empty() && eo.indices.empty());

    // Bounds.
    as::MeshAsset bo;
    CHECK(!as::codec::decode_mesh(B{1,2,3}, bo));      // < 4
    B trunc(4, 0); trunc[0]=5;                          // claims 5 verts
    CHECK(!as::codec::decode_mesh(trunc, bo));
}

// ---- shader: round-trip + empty + bounds --------------------------
void test_shader() {
    as::ShaderAsset s;
    s.stage = 3u;
    s.entry_point = "main";
    s.bytecode = { 0xDEu,0xADu,0xBEu,0xEFu,0x00u };
    B enc = as::codec::encode_shader(s);
    CHECK(enc.size() == sz(4 + 4 + 4 + 4 + 5));

    as::ShaderAsset out;
    CHECK(as::codec::decode_shader(enc, out));
    CHECK(out.stage == 3u);
    CHECK(out.entry_point == "main");
    CHECK(out.bytecode == s.bytecode);

    // Empty entry + empty bytecode.
    as::ShaderAsset e; e.stage = 0u; e.entry_point=""; e.bytecode={};
    B ee = as::codec::encode_shader(e);
    CHECK(ee.size() == sz(4 + 4 + 4));
    as::ShaderAsset eo;
    CHECK(as::codec::decode_shader(ee, eo));
    CHECK(eo.entry_point.empty() && eo.bytecode.empty() && eo.stage == 0u);

    // Bounds.
    as::ShaderAsset bo;
    CHECK(!as::codec::decode_shader(B(7, 0), bo));     // < 8
    B lie(8, 0); lie[4]=200;                            // entry len 200
    CHECK(!as::codec::decode_shader(lie, bo));
}

// ---- material: round-trip + 36-byte back-compat + bounds ----------
void test_material() {
    // PBR-map extension block when all 9 map strings are empty:
    //   magic(4) + mr_packed(1) + invert(1) + 3 scalars(12) + 9*len-prefix(36).
    const cardinal::usize EXT_EMPTY = 4 + 1 + 1 + 12 + 9 * 4;   // 54

    as::MaterialAsset m;
    m.base_color = {0.1f, 0.2f, 0.3f};
    m.metallic = 0.4f; m.roughness = 0.5f;
    m.emission = {0.6f, 0.7f, 0.8f};
    m.emission_strength = 0.9f;
    m.base_color_texture = "tex.albedo";              // 10 chars
    m.metallic_roughness_texture = "t.mr";            // 4
    m.roughness_texture = "t.rough";                  // 7
    m.metallic_texture  = "t.metal";                  // 7
    m.normal_texture    = "t.normal";                 // 8
    m.occlusion_texture = "t.ao";                     // 4
    m.emissive_texture  = "t.emis";                   // 6
    m.height_texture    = "t.height";                 // 8
    m.specular_texture  = "t.spec";                   // 6
    m.opacity_texture   = "t.op";                     // 4
    m.normal_scale = 0.5f; m.occlusion_strength = 0.75f; m.height_scale = 2.0f;
    m.mr_packed = true; m.invert_roughness = true;

    B enc = as::codec::encode_material(m);
    const cardinal::usize map_bytes =
        sz(4+4)+sz(4+7)+sz(4+7)+sz(4+8)+sz(4+4)+sz(4+6)+sz(4+8)+sz(4+6)+sz(4+4);
    CHECK(enc.size() == sz(36 + 4 + 10) + sz(4 + 2 + 12) + map_bytes);

    as::MaterialAsset out;
    CHECK(as::codec::decode_material(enc, out));
    CHECK(veq(out.base_color, m.base_color));
    CHECK(out.metallic == m.metallic);                 // bit-exact
    CHECK(out.roughness == m.roughness);
    CHECK(veq(out.emission, m.emission));
    CHECK(out.emission_strength == m.emission_strength);
    CHECK(out.base_color_texture == "tex.albedo");
    // Full PBR map set round-trips exactly.
    CHECK(out.metallic_roughness_texture == "t.mr");
    CHECK(out.roughness_texture == "t.rough");
    CHECK(out.metallic_texture  == "t.metal");
    CHECK(out.normal_texture    == "t.normal");
    CHECK(out.occlusion_texture == "t.ao");
    CHECK(out.emissive_texture  == "t.emis");
    CHECK(out.height_texture    == "t.height");
    CHECK(out.specular_texture  == "t.spec");
    CHECK(out.opacity_texture   == "t.op");
    CHECK(out.normal_scale == 0.5f && out.occlusion_strength == 0.75f &&
          out.height_scale == 2.0f);
    CHECK(out.mr_packed && out.invert_roughness);

    // Empty-everything material: 36 + (4+0) base + 54-byte ext (9 empty maps).
    as::MaterialAsset noTex;
    noTex.base_color = {1.0f, 0.0f, 0.0f};
    noTex.metallic = 0.25f;
    B ne = as::codec::encode_material(noTex);
    CHECK(ne.size() == sz(40) + EXT_EMPTY);            // 94
    as::MaterialAsset no;
    CHECK(as::codec::decode_material(ne, no));
    CHECK(no.base_color_texture.empty());
    CHECK(no.normal_texture.empty() && no.metallic_roughness_texture.empty());
    CHECK(no.normal_scale == 1.0f && no.occlusion_strength == 1.0f &&
          no.height_scale == 1.0f);
    CHECK(!no.mr_packed && !no.invert_roughness);
    CHECK(veq(no.base_color, noTex.base_color) && no.metallic == 0.25f);

    // Back-compat: a legacy 36..39-byte blob (floats only) still decodes —
    // texture AND every new map cleared, scalars defaulted, floats kept.
    for (int extra = 0; extra <= 3; ++extra) {
        B legacy(ne.begin(), ne.begin() + 36 + extra);
        CHECK(legacy.size() == sz(36 + extra));
        as::MaterialAsset lo;
        lo.base_color_texture = "STALE";              // must be cleared
        lo.normal_texture     = "STALE";
        CHECK(as::codec::decode_material(legacy, lo));
        CHECK(lo.base_color_texture.empty());
        CHECK(lo.normal_texture.empty());
        CHECK(lo.normal_scale == 1.0f);
        CHECK(veq(lo.base_color, noTex.base_color));
        CHECK(lo.metallic == 0.25f);
    }

    // Back-compat: a 40-byte legacy blob (head + empty base_color_texture, NO
    // magic) decodes with the new fields defaulted (the no-extension path).
    {
        B legacy40(ne.begin(), ne.begin() + 40);
        as::MaterialAsset lo;
        CHECK(as::codec::decode_material(legacy40, lo));
        CHECK(lo.base_color_texture.empty() && lo.normal_texture.empty());
        CHECK(!lo.mr_packed && lo.height_scale == 1.0f);
    }

    // Bounds: < 36 → reject; >=40 with a lying base_color_texture length →
    // reject; a truncated ext block (magic present, body cut) → reject (no OOB).
    as::MaterialAsset bo;
    CHECK(!as::codec::decode_material(B(35, 0), bo));
    B lie(40, 0); lie[36]=50;                          // claims 50-char tex
    CHECK(!as::codec::decode_material(lie, bo));
    {
        B trunc(44, 0);                                // 36 head + 4 (len 0) + magic
        trunc[40]=0x31; trunc[41]=0x56; trunc[42]=0x50; trunc[43]=0x4D;  // 'MPV1' LE
        CHECK(!as::codec::decode_material(trunc, bo));
    }
}

// ---- in-memory register/load_material cache -----------------------
void test_registry_inmemory() {
    auto reg = as::Registry::create();
    CHECK(reg != nullptr);

    as::MaterialAsset red;  red.base_color  = {1.0f,0.0f,0.0f};
    as::MaterialAsset blue; blue.base_color = {0.0f,0.0f,1.0f};

    CHECK(reg->load_material("mat.red") == nullptr);   // not registered yet
    reg->register_material("mat.red", red);
    auto p = reg->load_material("mat.red");
    CHECK(p != nullptr && veq(p->base_color, red.base_color));

    // Cached → same shared_ptr on repeat.
    auto p2 = reg->load_material("mat.red");
    CHECK(p2 == p);

    // Re-register (last-write-wins) → new object, new pointer.
    reg->register_material("mat.red", blue);
    auto p3 = reg->load_material("mat.red");
    CHECK(p3 != nullptr && p3 != p);
    CHECK(veq(p3->base_color, blue.base_color));

    reg->register_material("mat.blue", blue);
    CHECK(reg->load_material("missing") == nullptr);   // no mounts → null
    CHECK(reg->load_texture("missing") == nullptr);
    CHECK(reg->load_mesh("missing")   == nullptr);
    CHECK(reg->load_shader("missing") == nullptr);

    as::Registry::Stats s = reg->stats();
    CHECK(s.cached_materials == 2u);                   // red + blue keys
    CHECK(s.cached_textures == 0u && s.cached_meshes == 0u);
    CHECK(s.mount_dirs == 0u && s.mount_archives == 0u);
    CHECK(s.bytes_resident == static_cast<cardinal::u64>(0)); // mats excluded
}

// ---- deterministic negative filesystem queries --------------------
void test_registry_fs_negative() {
    auto reg = as::Registry::create();
    CHECK(!reg->contains("anything"));
    CHECK(reg->type_of("anything") == ck::AssetType::Unknown);
    CHECK(reg->keys().empty());

    reg->mount_directory("__cardinal_no_such_dir__");
    CHECK(reg->stats().mount_dirs == 1u);
    CHECK(!reg->contains("anything"));                 // dir doesn't exist
    CHECK(reg->keys().empty());
    CHECK(reg->type_of("anything") == ck::AssetType::Unknown);

    reg->clear_mounts();
    CHECK(reg->stats().mount_dirs == 0u);
    CHECK(reg->stats().mount_archives == 0u);
}

// ---- size-field integer-overflow guards (untrusted cooked bytes) --
// Every size/length is attacker-controlled. A value near UINT32_MAX
// used to wrap the old 32-bit `header + n` / `n * elem` checks, slip
// past the bound test, and drive an OOB read or a multi-GB resize.
// These must now all reject cleanly — no crash, no over-allocation.
void test_overflow_guards() {
    auto put_u32 = [](B& b, cardinal::usize at, cardinal::u32 v) {
        b[at+0] = static_cast<cardinal::u8>(v);
        b[at+1] = static_cast<cardinal::u8>(v >> 8);
        b[at+2] = static_cast<cardinal::u8>(v >> 16);
        b[at+3] = static_cast<cardinal::u8>(v >> 24);
    };
    constexpr cardinal::u32 kMax = 0xFFFFFFFFu;

    // texture: rgba length wraps `16 + n`.
    {
        B t(16, 0); put_u32(t, 12, kMax);
        as::TextureAsset o;
        CHECK(!as::codec::decode_texture(t, o));
        put_u32(t, 12, 0xFFFFFFF8u);                   // 16+n wraps to 8
        CHECK(!as::codec::decode_texture(t, o));
    }
    // mesh: index count wraps `ic * 4` (vc=0 so the ic check is reached;
    // pre-fix this slipped through to a ~4 GB resize(ic)).
    {
        B m(8, 0);
        put_u32(m, 0, 0u);                             // vc = 0
        put_u32(m, 4, 0x40000000u);                    // ic*4 == 0 in u32
        as::MeshAsset o;
        CHECK(!as::codec::decode_mesh(m, o));
        put_u32(m, 4, kMax);
        CHECK(!as::codec::decode_mesh(m, o));
    }
    // shader: entry-point length wraps `8 + nl`.
    {
        B s(8, 0); put_u32(s, 4, kMax);
        as::ShaderAsset o;
        CHECK(!as::codec::decode_shader(s, o));
    }
    // material: texture-name length wraps `40 + nl` (>=40 reaches it).
    {
        B mt(40, 0); put_u32(mt, 36, kMax);
        as::MaterialAsset o;
        CHECK(!as::codec::decode_material(mt, o));
        put_u32(mt, 36, 0xFFFFFFD8u);                  // 40+nl wraps to 0
        CHECK(!as::codec::decode_material(mt, o));
    }
}

}  // namespace

int main() {
    test_texture();
    test_mesh();
    test_shader();
    test_material();
    test_overflow_guards();
    test_registry_inmemory();
    test_registry_fs_negative();

    if (g_fail == 0) {
        cardinal::log::infof("acodec", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("acodec", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
