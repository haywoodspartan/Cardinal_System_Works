# SIMD math dispatch suite. Pins every ISA tier the host supports
# (scalar / SSE4.2 / AVX / AVX2 / AVX-512, via simd::force_backend) against a
# double-precision reference — integer-valued inputs make the reductions
# order-independent and therefore EXACT across tiers, random inputs get the
# float error bound — and against each other for the element-wise ops. Then
# times the multi-chain ("software double pump") reductions against the
# original single-chain loops, which are kept verbatim in naive_*.cpp with the
# engine's own per-file arch flags, so the before/after number is reproducible
# on any machine. Tiers the CPU lacks are skipped; exit 0 = pass.
add_executable(Cardinal_Test_Simd main.cpp naive_avx2.cpp naive_avx512.cpp)
target_link_libraries(Cardinal_Test_Simd PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Simd PROPERTIES FOLDER tests)

if(MSVC)
    set_source_files_properties(naive_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(naive_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(naive_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(naive_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
endif()

add_test(NAME simd COMMAND Cardinal_Test_Simd)
