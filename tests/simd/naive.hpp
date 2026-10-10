#pragma once
// The ORIGINAL single-accumulator reductions, kept verbatim as the benchmark
// baseline. Each TU is compiled with the same per-file arch flags the engine
// uses for its tier, and main.cpp only calls into a tier the CPU reports.
#include <cardinal/core/types.hpp>

namespace naive_avx512 {
cardinal::f32 dot_f32(const cardinal::f32* a, const cardinal::f32* b, cardinal::usize n);
cardinal::f32 sum_f32(const cardinal::f32* a, cardinal::usize n);
}
namespace naive_avx2 {
cardinal::f32 dot_f32(const cardinal::f32* a, const cardinal::f32* b, cardinal::usize n);
cardinal::f32 sum_f32(const cardinal::f32* a, cardinal::usize n);
}
