# Hand-rolled DEFLATE inflate (cardinal::core::compress) regression suite.
# Pins the decoder against REAL compressed vectors (fixed/dynamic/stored
# Huffman + zlib wrapper) plus the failure paths — it is the keystone the
# FBX / USDZ / PNG importers depend on, so a regression here breaks all three.
add_executable(Cardinal_Test_Compress main.cpp)
target_link_libraries(Cardinal_Test_Compress PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Compress PROPERTIES FOLDER tests)

add_test(NAME compress COMMAND Cardinal_Test_Compress)
