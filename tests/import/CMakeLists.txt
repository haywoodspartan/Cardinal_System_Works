# Deterministic asset-import regression suite. The OBJ/MTL + glTF
# backends are parse-heavy, hand-rolled, and shipped without tests —
# exactly the code that rots silently. Fixtures are generated at
# runtime into a temp dir (self-contained, no fixture files to manage,
# no path-resolution fragility), so this is pure + deterministic: no
# sockets, no timing. cardinal::import brings core+scene transitively;
# compile_flags holds it to the engine's /W4 /WX bar.
add_executable(Cardinal_Test_Import main.cpp)
target_link_libraries(Cardinal_Test_Import PRIVATE
    cardinal::import
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Import PROPERTIES FOLDER tests)

add_test(NAME import COMMAND Cardinal_Test_Import)
