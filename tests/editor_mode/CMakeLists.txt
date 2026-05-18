# Deterministic editor mode-state-machine regression suite.
# edit::EditorState is the Studio's single source of truth for "what is
# the user doing right now" — every panel and the viewport branch on it
# to decide whether to consume or forward an input event, and a mode
# switch fires one callback so the gizmo / brush HUD can re-arm. A
# regression here silently mis-routes input (clicks land in the wrong
# tool) or drops the re-arm callback. The machine is pure + has no
# hidden state: the name/glyph/tooltip tables, the out-of-range
# fallbacks, the change-callback firing rule (fires only on an actual
# transition, single slot / last-setter-wins, empty-fn disables), the
# brush/gizmo activation truth table and the status-text round-trip are
# all exactly pinned. Exit 0 = all pass.
add_executable(Cardinal_Test_EditorMode main.cpp)
target_link_libraries(Cardinal_Test_EditorMode PRIVATE
    cardinal::edit
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_EditorMode PROPERTIES FOLDER tests)

add_test(NAME editor_mode COMMAND Cardinal_Test_EditorMode)
