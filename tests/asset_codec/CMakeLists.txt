# Deterministic asset-codec regression suite. asset::codec encode/decode
# is the typed serialiser that travels inside every cooked asset payload
# (texture / mesh / shader / material). It is THE on-disk + in-memory
# format: a regression silently corrupts every loaded asset or breaks
# pack/save compatibility. The codec is pure + byte-exact (little-endian
# u32, IEEE float bit-copy), so the suite pins exact byte layout, full
# round-trips, truncation/short-buffer rejection, and the documented
# 36-byte material back-compat path. Also covers the disk-free
# register_material / load_material cache (pointer identity) and stats.
# The directory/archive paths are filesystem/pack-coupled and out of
# scope (only their deterministic negative queries are checked). Pure
# CPU, headless. Exit 0 = all pass.
add_executable(Cardinal_Test_AssetCodec main.cpp)
target_link_libraries(Cardinal_Test_AssetCodec PRIVATE
    cardinal::asset
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AssetCodec PROPERTIES FOLDER tests)

add_test(NAME asset_codec COMMAND Cardinal_Test_AssetCodec)
