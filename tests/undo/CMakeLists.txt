# Deterministic undo/redo-stack regression suite. edit::UndoStack is the
# editor's reversibility backbone: every gizmo drag, brush stroke, mesh
# op and property edit records a Command (apply/revert pair). A
# regression silently breaks Ctrl-Z/Ctrl-Y — wrong command reverted,
# lost redo on a new edit, the capacity cap corrupting the cursor, or a
# boundary undo running off the end. The stack is pure + deterministic;
# commands here mutate an external recorder so apply/revert ORDER and the
# exact value are pinned, not just sizes. Exit 0 = all pass.
add_executable(Cardinal_Test_Undo main.cpp)
target_link_libraries(Cardinal_Test_Undo PRIVATE
    cardinal::edit
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Undo PROPERTIES FOLDER tests)

add_test(NAME undo COMMAND Cardinal_Test_Undo)
