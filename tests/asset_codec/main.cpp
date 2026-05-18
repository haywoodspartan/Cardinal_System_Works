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
#include <cardinal/core/log.hpp>

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
    as::MaterialAsset m;
    m.base_color = {0.1f, 0.2f, 0.3f};
    m.metallic = 0.4f; m.roughness = 0.5f;
    m.emission = {0.6f, 0.7f, 0.8f};
    m.emission_strength = 0.9f;
    m.base_color_texture = "tex.albedo";              // 10 chars
    B enc = as::codec::encode_material(m);
    CHECK(enc.size() == sz(36 + 4 + 10));

    as::MaterialAsset out;
    CHECK(as::codec::decode_material(enc, out));
    CHECK(veq(out.base_color, m.base_color));
    CHECK(out.metallic == m.metallic);                 // bit-exact
    CHECK(out.roughness == m.roughness);
    CHECK(veq(out.emission, m.emission));
    CHECK(out.emission_strength == m.emission_strength);
    CHECK(out.base_color_texture == "tex.albedo");

    // Empty-texture material encodes to exactly 40 bytes (36 + len 0)
    // and round-trips with an empty string (the nl==0 path).
    as::MaterialAsset noTex;
    noTex.base_color = {1.0f, 0.0f, 0.0f};
    noTex.metallic = 0.25f;
    B ne = as::codec::encode_material(noTex);
    CHECK(ne.size() == sz(40));
    as::MaterialAsset no;
    CHECK(as::codec::decode_material(ne, no));
    CHECK(no.base_color_texture.empty());
    CHECK(veq(no.base_color, noTex.base_color) && no.metallic == 0.25f);

    // Back-compat: a legacy 36..39-byte blob (floats only, no string
    // field at all) must still decode — texture cleared, floats kept.
    for (int extra = 0; extra <= 3; ++extra) {
        B legacy(ne.begin(), ne.begin() + 36 + extra);
        CHECK(legacy.size() == sz(36 + extra));
        as::MaterialAsset lo;
        lo.base_color_texture = "STALE";              // must be cleared
        CHECK(as::codec::decode_material(legacy, lo));
        CHECK(lo.base_color_texture.empty());
        CHECK(veq(lo.base_color, noTex.base_color));
        CHECK(lo.metallic == 0.25f);
    }

    // Bounds: < 36 → reject; >=40 with a lying string length → reject.
    as::MaterialAsset bo;
    CHECK(!as::codec::decode_material(B(35, 0), bo));
    B lie(40, 0); lie[36]=50;                          // claims 50-char tex
    CHECK(!as::codec::decode_material(lie, bo));
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

}  // namespace

int main() {
    test_texture();
    test_mesh();
    test_shader();
    test_material();
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
