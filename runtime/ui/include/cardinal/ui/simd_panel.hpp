#pragma once

// =============================================================================
// Cardinal Studio — SIMD diagnostic panel.
//
// Surfaces the runtime state of cardinal::core::simd's dispatched math:
//
//   - The CPU's vendor + brand string + which feature bits CPUID reported
//   - The active backend (the highest tier the dispatcher picked)
//   - Per-op backend (which tier each individual function-pointer landed
//     on — meaningful now that mat4_mul_array exercises per-op fallback,
//     so users can see "AVX-512 active, but mat4_mul_array fell back to
//     SSE4.2" when the AVX-512 TU declines to ship that op)
//   - A "Run benchmark" button that times each op against the active
//     backend and reports per-call ms + effective bandwidth
//
// Free function — no Studio class dependency. Any host that wants the
// panel just calls cardinal::ui::draw_simd_panel(&visible) inside its
// frame's UI block.
// =============================================================================

namespace cardinal::ui {

// Renders the panel inside an ImGui::Begin / End pair. Pass the
// visibility flag through; ImGui's window-X click flips it to false.
void draw_simd_panel(bool* p_open);

}  // namespace cardinal::ui
