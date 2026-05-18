# Deterministic hashing-primitive regression suite. splitmix64 / fnv1a64
# / fxhash64 underpin content IDs, serialised keys and hash-map keying:
# if any constant, shift, or step order silently drifts, EVERY derived
# id changes and on-disk content / save compatibility breaks invisibly.
# This suite pins externally-known golden vectors — the canonical
# SplitMix64 stream outputs and the published FNV-1a-64 digests — plus
# hand-derivable fxhash primitives, AND nails them at COMPILE TIME via
# static_assert on the constexpr path so drift fails the build itself.
# It also pins the structural identities (FNV streaming chain, the
# void*/const char* overload agreement, fxhash block decomposition).
# Pure, header-only, platform-stable; the few byte-order-sensitive
# cases assume little-endian (the engine targets x64 / ARM64 LE).
# Exit 0 = all pass.
add_executable(Cardinal_Test_Hash main.cpp)
target_link_libraries(Cardinal_Test_Hash PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Hash PROPERTIES FOLDER tests)

add_test(NAME hash COMMAND Cardinal_Test_Hash)
