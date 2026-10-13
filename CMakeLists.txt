cmake_minimum_required(VERSION 3.27)

# vcpkg manifest mode — toolchain must be set before project()
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

project(cardinal
    VERSION 0.0.1
    DESCRIPTION "Cardinal System Works - Foundation Engine"
    LANGUAGES CXX
)

# ---- Global C++ settings ---------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Dynamic CRT — matches vcpkg's default triplet (x64-windows). Switching to
# static would force rebuilding every vcpkg dep with x64-windows-static; defer
# that until we ship.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# ---- x64 only for now ------------------------------------------------------
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Cardinal requires a 64-bit toolchain (x86_64).")
endif()

# ---- Compile-flag interface target — every cardinal target links this ------
add_library(cardinal_compile_flags INTERFACE)
add_library(cardinal::compile_flags ALIAS cardinal_compile_flags)

# NDEBUG is pinned here rather than inherited from CMAKE_CXX_FLAGS_<CONFIG>.
# CMake's per-config defaults carry /DNDEBUG for Release + RelWithDebInfo, but
# a cache whose per-config flags were blanked (seen on a dev box: both local
# trees built every "release" binary with asserts LIVE while CI had them
# compiled out) silently diverges from CI. Making it explicit keeps local ==
# CI: asserts are active only in Debug, everywhere.
target_compile_definitions(cardinal_compile_flags INTERFACE
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)

if(MSVC)
    # C++ only: the interface target also feeds ml64 (the job system's
    # fiber switch is MASM), which rejects /permissive-, /Zc:*, /EHsc ...
    target_compile_options(cardinal_compile_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX;/permissive-;/Zc:__cplusplus;/Zc:preprocessor;/Zc:inline;/Zc:throwingNew;/EHsc;/MP;/utf-8;/diagnostics:caret;/wd4324>
    )
    target_compile_definitions(cardinal_compile_flags INTERFACE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
        UNICODE _UNICODE
    )
    target_compile_options(cardinal_compile_flags INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Oi;/Ot;/GL>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2;/Oi;/Ot;/Zi>
    )
    # No embedded manifests: nothing here needs one, and skipping mt.exe
    # removes the CI flake where the manifest tool hits "Access is denied"
    # on a freshly linked binary that Defender is still scanning.
    # /DEBUG is pinned for the same reason NDEBUG is above: the cached
    # CMAKE_EXE_LINKER_FLAGS_<CONFIG> can be blank, and then RelWithDebInfo
    # objects compiled with /Zi are linked WITHOUT a PDB — every crash report
    # from a release binary resolved against a stale .pdb and printed noise.
    # /OPT:REF /OPT:ICF are spelled out so /DEBUG does not switch them off.
    target_link_options(cardinal_compile_flags INTERFACE
        /MANIFEST:NO
        $<$<NOT:$<CONFIG:Debug>>:/DEBUG:FULL /INCREMENTAL:NO /OPT:REF /OPT:ICF>
        $<$<CONFIG:Release>:/LTCG>
    )
else()
    # C++ only (the fiber switch is a GAS .S file assembled through the
    # same driver; -W* options are meaningless or "unused" errors there).
    target_compile_options(cardinal_compile_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror;-Wshadow;-Wcast-align;-Wnull-dereference;-Wdouble-promotion;-Wformat=2;-Wconversion;-Wsign-conversion>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-ffunction-sections;-fdata-sections>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O2;-g;-ffunction-sections;-fdata-sections>
    )
    target_link_options(cardinal_compile_flags INTERFACE
        $<$<CONFIG:Release>:-Wl,--gc-sections>
    )
endif()

# ---- Vendored NVIDIA SDKs (opt-in, gated on directory existence) -----------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cardinal_third_party.cmake)

# ---- Subdirectories --------------------------------------------------------
enable_testing()
add_subdirectory(runtime)
add_subdirectory(samples)
add_subdirectory(apps)
add_subdirectory(tests)

message(STATUS "Cardinal: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_BUILD_TYPE}")

# ---- run/ launchers --------------------------------------------------------
# Every executable target (apps, samples, and the Cardinal_Test_* suites)
# gets a tiny launcher in <root>/run/ that just exec's the real binary out
# of this build tree — so you can invoke any program from the repo root
# without remembering build/<preset>/bin paths. The launchers are
# generated at configure time (paths resolve via $<TARGET_FILE>), so they
# always point at whichever preset was configured last. run/ is
# git-ignored. Toggle with -DCARDINAL_GENERATE_RUN_SHORTCUTS=OFF.
option(CARDINAL_GENERATE_RUN_SHORTCUTS
       "Generate <root>/run/ launchers for every executable target" ON)

# Recursively gather every (non-imported) EXECUTABLE target in the tree.
function(_cardinal_collect_exe_targets _dir _out)
    get_property(_tgts DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_subs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    set(_acc "")
    foreach(_t IN LISTS _tgts)
        get_target_property(_ty  ${_t} TYPE)
        get_target_property(_imp ${_t} IMPORTED)
        if(_ty STREQUAL "EXECUTABLE" AND NOT _imp)
            list(APPEND _acc ${_t})
        endif()
    endforeach()
    foreach(_s IN LISTS _subs)
        _cardinal_collect_exe_targets("${_s}" _child)
        list(APPEND _acc ${_child})
    endforeach()
    set(${_out} "${_acc}" PARENT_SCOPE)
endfunction()

if(CARDINAL_GENERATE_RUN_SHORTCUTS)
    _cardinal_collect_exe_targets("${CMAKE_SOURCE_DIR}" _cardinal_exes)
    list(REMOVE_DUPLICATES _cardinal_exes)
    list(SORT _cardinal_exes)
    set(_run_dir "${CMAKE_SOURCE_DIR}/run")
    foreach(_tgt IN LISTS _cardinal_exes)
        if(WIN32)
            file(GENERATE
                OUTPUT  "${_run_dir}/${_tgt}.cmd"
                CONTENT "@echo off\r\nrem Auto-generated by CMake — points at the last-configured build tree.\r\n\"$<TARGET_FILE:${_tgt}>\" %*\r\n")
        else()
            file(GENERATE
                OUTPUT  "${_run_dir}/${_tgt}"
                CONTENT "#!/usr/bin/env sh\n# Auto-generated by CMake — points at the last-configured build tree.\nexec \"$<TARGET_FILE:${_tgt}>\" \"$@\"\n"
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                 GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        endif()
    endforeach()
    list(LENGTH _cardinal_exes _cardinal_exe_count)
    message(STATUS
        "Cardinal: run/ launchers for ${_cardinal_exe_count} executables -> ${_run_dir}")
endif()
