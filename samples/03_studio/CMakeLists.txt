add_executable(Cardinal_System_Studio
    main.cpp
    console_setup.cpp
    sample_types.hpp)
target_link_libraries(Cardinal_System_Studio PRIVATE
    cardinal::core cardinal::window cardinal::rhi cardinal::ui
    cardinal::scene cardinal::render cardinal::script cardinal::plugin
    cardinal::edit cardinal::console cardinal::cppscript cardinal::sandbox
    cardinal::world cardinal::input cardinal::serial cardinal::particles
    cardinal::nav cardinal::hud cardinal::project cardinal::cook
    cardinal::pack cardinal::shader cardinal::asset
    # HAL + IO Dispatcher + Geometry math live inside cardinal::core now.
    cardinal::partition cardinal::level cardinal::mass
    cardinal::navmesh cardinal::ai
    cardinal::compile_flags)
set_target_properties(Cardinal_System_Studio PROPERTIES FOLDER samples)

# Studio uses ImGui directly in the sample for the demo panel; expose it.
find_package(imgui CONFIG REQUIRED)
target_link_libraries(Cardinal_System_Studio PRIVATE imgui::imgui)

cardinal_tp_deploy_runtimes(Cardinal_System_Studio)
