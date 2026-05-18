add_library(cardinal_audio STATIC
    src/audio.cpp
    src/audio_wasapi.cpp
    include/cardinal/audio/audio.hpp)
add_library(cardinal::audio ALIAS cardinal_audio)

target_include_directories(cardinal_audio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_audio
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

# WASAPI backend needs COM (CoCreateInstance / CoInitializeEx). The
# MMDevice / AudioClient interfaces are header-only from the Windows SDK.
if(WIN32)
    target_link_libraries(cardinal_audio PRIVATE ole32)
endif()

set_target_properties(cardinal_audio PROPERTIES FOLDER runtime)
