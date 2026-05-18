# Deterministic project-module regression suite. project::Project is the
# Studio create/open/save backbone: a `project.cardinal` text manifest +
# a fixed directory layout. A regression silently corrupts project files
# (lost settings, wrong dirs) or breaks the recent-projects list. The
# manifest writer/parser, the root→subdir derivation, template name
# tables, and the RecentProjects most-recent/dedupe/cap-16 logic are
# pure + deterministic; the disk round-trips run through a unique
# std::filesystem temp dir that the suite creates and removes. Exit 0 =
# all pass.
add_executable(Cardinal_Test_Project main.cpp)
target_link_libraries(Cardinal_Test_Project PRIVATE
    cardinal::project
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Project PROPERTIES FOLDER tests)

add_test(NAME project COMMAND Cardinal_Test_Project)
