set(CARDINAL_WINDOW_HEADERS
    include/cardinal/window/window.hpp
)

set(CARDINAL_WINDOW_SOURCES)
if(WIN32)
    list(APPEND CARDINAL_WINDOW_SOURCES src/window_windows.cpp)
elseif(UNIX AND NOT APPLE)
    list(APPEND CARDINAL_WINDOW_SOURCES src/window_linux.cpp)
endif()

add_library(cardinal_window STATIC ${CARDINAL_WINDOW_SOURCES} ${CARDINAL_WINDOW_HEADERS})
add_library(cardinal::window ALIAS cardinal_window)

target_include_directories(cardinal_window PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cardinal_window
    PUBLIC  cardinal::core cardinal::compile_flags
)

if(WIN32)
    target_link_libraries(cardinal_window PUBLIC user32 gdi32)
endif()

set_target_properties(cardinal_window PROPERTIES FOLDER runtime)
