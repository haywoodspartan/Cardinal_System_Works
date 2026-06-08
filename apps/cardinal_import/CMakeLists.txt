# cardinal_import — CLI asset importer (commandlet). Console subsystem so it
# prints to the shell. Links cardinal::import (+ its scene/core deps).
add_executable(cardinal_import_exe main.cpp)
add_executable(cardinal::import_exe ALIAS cardinal_import_exe)

target_link_libraries(cardinal_import_exe PRIVATE
    cardinal::import
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)

set_target_properties(cardinal_import_exe PROPERTIES
    FOLDER apps
    OUTPUT_NAME "cardinal_import")
