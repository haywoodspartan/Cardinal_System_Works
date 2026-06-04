# Cardinal — KernelObjectAttribute regression suite.
# Exercises the cardinal::core::KernelObjectAttribute Windows kernel-object
# attribute wrapper: builder-pattern flag composition, name copy / clear /
# rebind, root-directory + security-descriptor pass-through, native()
# OBJECT_ATTRIBUTES layout (Windows-only), and move semantics (the moved-to
# OA must reference its OWN embedded UNICODE_STRING, not the source's).
add_executable(Cardinal_Test_KernelObjectAttribute main.cpp)
target_link_libraries(Cardinal_Test_KernelObjectAttribute PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_KernelObjectAttribute PROPERTIES FOLDER tests)

add_test(NAME kernel_object_attribute COMMAND Cardinal_Test_KernelObjectAttribute)
