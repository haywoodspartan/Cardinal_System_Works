# =============================================================================
# Cardinal third-party SDK integration.
#
# Each NVIDIA / vendor SDK under third_party/nvidia/ is integrated as an opt-in
# CMake target gated on its directory existing. Engine modules consume these
# only when the corresponding RenderSettings feature is enabled at runtime.
#
# Targets exposed (only when present):
#   cardinal::tp::rtxmu       — RT memory utilities (BLAS suballoc/compact)
#   cardinal::tp::reflex      — NVIDIA Reflex (Vulkan low-latency)
#   cardinal::tp::streamline  — Streamline interposer (DLSS / DLSS-G / DLSS-D)
#   cardinal::tp::nrd         — NVIDIA Real-time Denoiser (header only here;
#                               build-the-lib opt-in via NRD_BUILD)
#   cardinal::tp::ntc_sdk     — NTC SDK headers (libntc.dll loaded at runtime)
#   cardinal::tp::rtxpt_shaders — RTXPT shader source (reference only)
#
# Cache vars set:
#   CARDINAL_NTC_TOOL         — path to ntc-cli.exe for offline texture cooking
#   CARDINAL_RTXPT_BIN        — path to Rtxpt.exe (NVIDIA's standalone PT sample)
#
# DLLs queued for deployment via cardinal_tp_deploy_runtimes(<target>) — copies
# all registered runtime DLLs next to the target's binary post-build.
# =============================================================================

set(CARDINAL_TP_DIR "${CMAKE_SOURCE_DIR}/third_party/nvidia"
    CACHE PATH "Vendored NVIDIA SDKs root")

# Track DLLs to be deployed alongside engine exes.
set_property(GLOBAL PROPERTY CARDINAL_TP_RUNTIME_DLLS "")

function(cardinal_tp_register_dll dll_path)
    if(EXISTS "${dll_path}")
        get_property(_existing GLOBAL PROPERTY CARDINAL_TP_RUNTIME_DLLS)
        list(FIND _existing "${dll_path}" _idx)
        if(_idx EQUAL -1)
            set_property(GLOBAL APPEND PROPERTY CARDINAL_TP_RUNTIME_DLLS "${dll_path}")
        endif()
    endif()
endfunction()

# Deploy all registered runtime DLLs next to <target>'s output binary.
function(cardinal_tp_deploy_runtimes target)
    get_property(_dlls GLOBAL PROPERTY CARDINAL_TP_RUNTIME_DLLS)
    foreach(_dll ${_dlls})
        get_filename_component(_name "${_dll}" NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}" "$<TARGET_FILE_DIR:${target}>/${_name}"
            COMMENT "Deploying ${_name} -> ${target}")
    endforeach()

    # MSVC Debug builds need the Debug CRT (MSVCP140D, VCRUNTIME140D,
    # VCRUNTIME140_1D) — these aren't in System32 even on a dev box, so
    # the exe fails to load with STATUS_DLL_NOT_FOUND (0xC0000135) unless
    # the Debug-CRT DLLs sit alongside it.
    #
    # Critical: deploy ONLY for Debug. Shipping debug CRTs into Release at
    # best confuses the loader and at worst causes MSVC heap-mismatch
    # crashes if any code loaded against the wrong runtime allocates with
    # one and frees with the other.
    #
    # Single-config generator (Ninja, Make): CMAKE_BUILD_TYPE is known
    # right here so we just gate the whole loop. Multi-config generator
    # (Visual Studio): CMAKE_BUILD_TYPE is empty; we still emit the copy
    # commands but wrap each one's COMMAND in a $<CONFIG:Debug> genex so
    # the file copy expands to a no-op for non-Debug configs.
    if(MSVC AND CARDINAL_DEBUG_CRT_DIR)
        set(_is_multi_config FALSE)
        get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        set(_emit_debug_crt FALSE)
        if(_is_multi_config)
            set(_emit_debug_crt TRUE)               # gated per-config below
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(_emit_debug_crt TRUE)
        endif()

        if(_emit_debug_crt)
            foreach(_crt
                msvcp140d.dll
                msvcp140_1d.dll
                msvcp140_2d.dll
                msvcp140d_atomic_wait.dll
                msvcp140d_codecvt_ids.dll
                vcruntime140d.dll
                vcruntime140_1d.dll
                vccorlib140d.dll
                concrt140d.dll)
                if(EXISTS "${CARDINAL_DEBUG_CRT_DIR}/${_crt}")
                    if(_is_multi_config)
                        # Wrap the destination path in a $<CONFIG:Debug>
                        # genex — when the active config is not Debug the
                        # path expands to empty and CMake -E copy_if_different
                        # silently no-ops on a missing arg.
                        add_custom_command(TARGET ${target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                                $<$<CONFIG:Debug>:copy_if_different>
                                $<$<CONFIG:Debug>:${CARDINAL_DEBUG_CRT_DIR}/${_crt}>
                                $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${target}>/${_crt}>
                                $<$<NOT:$<CONFIG:Debug>>:true>
                            VERBATIM
                            COMMAND_EXPAND_LISTS)
                    else()
                        add_custom_command(TARGET ${target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${CARDINAL_DEBUG_CRT_DIR}/${_crt}"
                                "$<TARGET_FILE_DIR:${target}>/${_crt}"
                            COMMENT "Deploying Debug CRT ${_crt} -> ${target}")
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Debug-CRT discovery. VS ships the Debug CRT at
#   <VS>/VC/Redist/MSVC/<version>/debug_nonredist/x64/Microsoft.VC143.DebugCRT
# but the <version> is not always the same as the toolset version. We probe
# both the toolset version and the most-recent x64 debug redist on disk.
# Sets CARDINAL_DEBUG_CRT_DIR (cache var) when found.
# ---------------------------------------------------------------------------
if(MSVC AND NOT CARDINAL_DEBUG_CRT_DIR AND CMAKE_CXX_COMPILER)
    # cl.exe lives at:
    #   <VS>/VC/Tools/MSVC/<toolset-ver>/bin/Hostx64/x64/cl.exe
    # We need to climb 6 dirs to reach <VS>/VC, then look at Redist/MSVC.
    set(_dir "${CMAKE_CXX_COMPILER}")
    foreach(_step RANGE 6)
        get_filename_component(_dir "${_dir}" DIRECTORY)
    endforeach()
    set(_redist_root "${_dir}/Redist/MSVC")

    # Capture the toolset version (used as the first guess).
    set(_tdir "${CMAKE_CXX_COMPILER}")
    foreach(_step RANGE 3)
        get_filename_component(_tdir "${_tdir}" DIRECTORY)
    endforeach()
    get_filename_component(_toolset_ver "${_tdir}" NAME)

    if(EXISTS "${_redist_root}")
        set(_candidates
            "${_redist_root}/${_toolset_ver}/debug_nonredist/x64/Microsoft.VC143.DebugCRT"
        )
        # Then every 14.x x64 debug redist on disk, newest first.
        file(GLOB _all_redists RELATIVE "${_redist_root}" "${_redist_root}/14.*")
        list(SORT _all_redists ORDER DESCENDING)
        foreach(_v ${_all_redists})
            list(APPEND _candidates
                "${_redist_root}/${_v}/debug_nonredist/x64/Microsoft.VC143.DebugCRT")
        endforeach()
        foreach(_c ${_candidates})
            if(EXISTS "${_c}/vcruntime140d.dll")
                set(CARDINAL_DEBUG_CRT_DIR "${_c}" CACHE PATH "MSVC Debug CRT redist dir")
                break()
            endif()
        endforeach()
    endif()

    if(CARDINAL_DEBUG_CRT_DIR)
        message(STATUS "[third_party] Debug CRT  ✓ ${CARDINAL_DEBUG_CRT_DIR}")
    else()
        message(STATUS "[third_party] Debug CRT  ✗ not found — Debug exes will need manual CRT deployment (probed under ${_redist_root})")
    endif()
endif()

message(STATUS "[third_party] scanning ${CARDINAL_TP_DIR}")

# Aggregated INTERFACE target — engine code links this to get CARDINAL_HAS_*
# compile-time defines for whichever SDKs were detected. Avoids a per-target
# `if(CARDINAL_HAS_X)` ladder in every consumer's CMakeLists.
add_library(cardinal_tp_defs INTERFACE)
add_library(cardinal::tp::defs ALIAS cardinal_tp_defs)

# ---------------------------------------------------------------------------
# RTXMU — RT memory utilities. Self-contained, ~5 .cpp files. We compile the
# Vulkan path only (D3D12 backend will land alongside Phase 2.6).
# ---------------------------------------------------------------------------
set(_rtxmu "${CARDINAL_TP_DIR}/rtxmu")
if(EXISTS "${_rtxmu}/include/rtxmu/VkAccelStructManager.h")
    # Vulkan-Headers comes from vcpkg manifest mode. find_package is idempotent,
    # so it's safe even if runtime/rhi/CMakeLists also calls it later.
    find_package(VulkanHeaders CONFIG REQUIRED)

    add_library(cardinal_tp_rtxmu STATIC
        ${_rtxmu}/src/Logger.cpp
        ${_rtxmu}/src/VulkanSuballocator.cpp
        ${_rtxmu}/src/VkAccelStructManager.cpp
    )
    add_library(cardinal::tp::rtxmu ALIAS cardinal_tp_rtxmu)
    target_include_directories(cardinal_tp_rtxmu PUBLIC ${_rtxmu}/include)
    target_compile_features(cardinal_tp_rtxmu PUBLIC cxx_std_17)
    target_link_libraries(cardinal_tp_rtxmu PUBLIC Vulkan::Headers)

    # RTXMU uses std::to_string without including <string>. Force-include it
    # rather than patching upstream sources. /FI<name> with NO quotes — CMake
    # would otherwise pass the literal `"string"` (with quotes) to cl.exe.
    if(MSVC)
        target_compile_options(cardinal_tp_rtxmu PRIVATE /FIstring /W3 /WX-)
    else()
        target_compile_options(cardinal_tp_rtxmu PRIVATE -include string)
    endif()

    set_target_properties(cardinal_tp_rtxmu PROPERTIES FOLDER "third_party/nvidia")
    set(CARDINAL_HAS_RTXMU TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_RTXMU=1)
    message(STATUS "[third_party] RTXMU      ✓ Vulkan-only static lib")
endif()

# ---------------------------------------------------------------------------
# Reflex (NVIDIA low-latency, Vulkan flavor). Header + import lib + DLL.
# Required by DLSS-Frame-Generation; nice to have on its own.
# ---------------------------------------------------------------------------
set(_reflex "${CARDINAL_TP_DIR}/reflex/Reflex_Vulkan")
if(EXISTS "${_reflex}/inc/NvLowLatencyVk.h")
    add_library(cardinal_tp_reflex INTERFACE)
    add_library(cardinal::tp::reflex ALIAS cardinal_tp_reflex)
    target_include_directories(cardinal_tp_reflex INTERFACE ${_reflex}/inc)
    target_link_libraries(cardinal_tp_reflex INTERFACE
        ${_reflex}/lib/NvLowLatencyVk.lib)
    cardinal_tp_register_dll("${_reflex}/lib/NvLowLatencyVk.dll")
    set(CARDINAL_HAS_REFLEX TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_REFLEX=1)
    message(STATUS "[third_party] Reflex     ✓ Vulkan low-latency")
endif()

# ---------------------------------------------------------------------------
# NvAPI — NVIDIA's general-purpose D3D-side SDK. We only consume the
# Reflex-on-D3D12 surface (NvAPI_D3D_SetSleepMode / Sleep / SetLatencyMarker
# / GetLatency). The header lives inside RTXDI's external mirror; reuse it
# rather than re-vendor.
# ---------------------------------------------------------------------------
set(_nvapi "${CARDINAL_TP_DIR}/rtxdi/External/NVAPI")
if(EXISTS "${_nvapi}/nvapi.h" AND EXISTS "${_nvapi}/amd64/nvapi64.lib")
    add_library(cardinal_tp_nvapi INTERFACE)
    add_library(cardinal::tp::nvapi ALIAS cardinal_tp_nvapi)
    target_include_directories(cardinal_tp_nvapi INTERFACE ${_nvapi})
    target_link_libraries(cardinal_tp_nvapi INTERFACE
        ${_nvapi}/amd64/nvapi64.lib)
    set(CARDINAL_HAS_NVAPI TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_NVAPI=1)
    message(STATUS "[third_party] NvAPI      ✓ D3D Reflex / latency markers")
endif()

# ---------------------------------------------------------------------------
# Streamline (NGX) — DLSS / DLSS-G / DLSS-D / Reflex / NIS / DeepDVC. The
# interposer .lib is a single static glue; runtime functionality lives in the
# nvngx_*.dll files. All DLLs in bin/x64/ are queued for deployment.
# ---------------------------------------------------------------------------
set(_sl "${CARDINAL_TP_DIR}/ngx")
if(EXISTS "${_sl}/include/sl.h" AND EXISTS "${_sl}/lib/x64/sl.interposer.lib")
    add_library(cardinal_tp_streamline INTERFACE)
    add_library(cardinal::tp::streamline ALIAS cardinal_tp_streamline)
    target_include_directories(cardinal_tp_streamline INTERFACE ${_sl}/include)
    target_link_libraries(cardinal_tp_streamline INTERFACE
        ${_sl}/lib/x64/sl.interposer.lib)
    file(GLOB _sl_dlls "${_sl}/bin/x64/*.dll")
    foreach(_dll ${_sl_dlls})
        cardinal_tp_register_dll("${_dll}")
    endforeach()
    list(LENGTH _sl_dlls _sl_dll_count)
    set(CARDINAL_HAS_STREAMLINE TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_STREAMLINE=1)
    message(STATUS "[third_party] Streamline ✓ DLSS family (${_sl_dll_count} DLLs)")
endif()

# ---------------------------------------------------------------------------
# NRD — Real-time denoiser. Has its own CMakeLists; we don't add_subdirectory
# yet because (a) the build pulls in ShaderMake + a shader pipeline and we
# don't have a renderer to feed it, (b) it would also fight our /WX. Expose
# the headers so engine code can compile against the API; flip NRD_BUILD=ON
# when we wire the actual denoiser pass in Phase 5.5.
# ---------------------------------------------------------------------------
set(_nrd "${CARDINAL_TP_DIR}/nrd")
if(EXISTS "${_nrd}/Include/NRD.h")
    add_library(cardinal_tp_nrd INTERFACE)
    add_library(cardinal::tp::nrd ALIAS cardinal_tp_nrd)
    target_include_directories(cardinal_tp_nrd INTERFACE
        ${_nrd}/Include
        ${_nrd}/Integration)
    set(CARDINAL_HAS_NRD TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_NRD=1)
    message(STATUS "[third_party] NRD        ✓ headers only (lib build deferred)")
endif()

# ---------------------------------------------------------------------------
# NTC SDK — texture decompression at runtime. libntc.dll comes from the
# adjacent ntc/ tool distribution; SDK provides headers + a forward-declared
# C API. The .lib is built by ntc-sdk's own CMakeLists; we expose just the
# headers here and load the DLL at runtime via LoadLibrary in the asset
# subsystem (Phase 3.x).
# ---------------------------------------------------------------------------
set(_ntc_sdk "${CARDINAL_TP_DIR}/ntc-sdk")
set(_ntc_dll "${CARDINAL_TP_DIR}/ntc/bin/windows-x64/libntc.dll")
if(EXISTS "${_ntc_sdk}/include/libntc/ntc.h")
    add_library(cardinal_tp_ntc_sdk INTERFACE)
    add_library(cardinal::tp::ntc_sdk ALIAS cardinal_tp_ntc_sdk)
    target_include_directories(cardinal_tp_ntc_sdk INTERFACE ${_ntc_sdk}/include)
    cardinal_tp_register_dll("${_ntc_dll}")
    cardinal_tp_register_dll("${CARDINAL_TP_DIR}/ntc/bin/windows-x64/dstorage.dll")
    cardinal_tp_register_dll("${CARDINAL_TP_DIR}/ntc/bin/windows-x64/dstoragecore.dll")
    set(CARDINAL_HAS_NTC TRUE CACHE BOOL "" FORCE)
    target_compile_definitions(cardinal_tp_defs INTERFACE CARDINAL_HAS_NTC=1)
    message(STATUS "[third_party] NTC SDK    ✓ headers + libntc.dll deploy")
endif()

# ---------------------------------------------------------------------------
# NTC tool (ntc-cli.exe) — offline texture cooking. Used at build time via
# cardinal_compress_texture(input output [args...]).
# ---------------------------------------------------------------------------
set(_ntc_cli "${CARDINAL_TP_DIR}/ntc/bin/windows-x64/ntc-cli.exe")
if(EXISTS "${_ntc_cli}")
    set(CARDINAL_NTC_TOOL "${_ntc_cli}" CACHE FILEPATH "")
    function(cardinal_compress_texture in_path out_path)
        add_custom_command(OUTPUT "${out_path}"
            COMMAND "${CARDINAL_NTC_TOOL}" --loadImages "${in_path}" --compress -o "${out_path}" ${ARGN}
            DEPENDS "${in_path}"
            COMMENT "NTC compress: ${in_path} -> ${out_path}")
    endfunction()
    message(STATUS "[third_party] ntc-cli    ✓ texture cooking")
endif()

# ---------------------------------------------------------------------------
# RTXPT — NVIDIA's standalone path tracer. We don't link it; just record the
# path so docs / launch shortcuts can find it. Useful as a reference run.
# ---------------------------------------------------------------------------
set(_rtxpt "${CARDINAL_TP_DIR}/rtxpt")
if(EXISTS "${_rtxpt}/Rtxpt.exe")
    set(CARDINAL_RTXPT_BIN "${_rtxpt}/Rtxpt.exe" CACHE FILEPATH "")
    if(EXISTS "${_rtxpt}/ShaderDynamic/Source/Rtxpt")
        add_library(cardinal_tp_rtxpt_shaders INTERFACE)
        add_library(cardinal::tp::rtxpt_shaders ALIAS cardinal_tp_rtxpt_shaders)
        target_include_directories(cardinal_tp_rtxpt_shaders INTERFACE
            ${_rtxpt}/ShaderDynamic/Source)
    endif()
    message(STATUS "[third_party] RTXPT      ✓ standalone exe + shader source")
endif()

# Optional libraries (gated on submodule init): RTXDI, RTXGI Nrc/Sharc.
foreach(_libpair
        "rtxdi/Libraries/Rtxdi"
        "rtxgi/Libraries/Nrc"
        "rtxgi/Libraries/Sharc")
    if(EXISTS "${CARDINAL_TP_DIR}/${_libpair}/CMakeLists.txt")
        message(STATUS "[third_party] ${_libpair} ✓ (sample lib present, integration TBD)")
    endif()
endforeach()

message(STATUS "[third_party] scan complete")
