# Deterministic HUD regression suite. Pins Hud's NaN-input contracts:
# bar / damage_flash / push_toast / tick_toasts each reject or
# sanitize non-finite scalar arguments so the prim & toast accumulators
# stay defined for the renderer (no UB float→u8 casts, no immortal
# toasts, no silently-skipped fills). Headless — no ImGui draw list
# created; we only exercise the state-machine side via the
# prim_count() / toast_count() accessors.
add_executable(Cardinal_Test_Hud main.cpp)
target_link_libraries(Cardinal_Test_Hud PRIVATE
    cardinal::hud
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Hud PROPERTIES FOLDER tests)

add_test(NAME hud COMMAND Cardinal_Test_Hud)
