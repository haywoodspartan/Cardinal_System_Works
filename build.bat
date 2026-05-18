@echo off
:: ============================================================================
:: Cardinal - Windows build script.
::
:: Wraps the CMakePresets.json configure + build in one step. Auto-locates
:: VS2022 (vswhere -> standard install paths) and runs VsDevCmd.bat so cl.exe,
:: link.exe, the Win10 SDK, ninja, and cmake are all on PATH.
::
:: Run with NO arguments for an interactive menu (target OS / architecture /
:: build type / toolchain). Pass any positional arg to skip the menu and use
:: the classic non-interactive form (unchanged - scripts/CI rely on this):
::
::   build                            - INTERACTIVE menu
::   build debug                      - debug, MSVC, x64, all targets
::   build release                    - RelWithDebInfo, MSVC, x64
::   build release msvc Cardinal_Test_Net
::                                    - build only that target
::   build release clang arm64        - RelWithDebInfo, clang-cl, arm64
::   build clean                      - wipe every build/* directory
::   build configure                  - only configure (no build)
::
::   Positional args (any order after the first):
::     debug | release          (default: debug)
::     msvc  | clang            (default: msvc)
::     x64   | arm64            (default: x64)
::     <target>                 (default: all)  - must be a real CMake target
::
:: Output: build\<preset>\bin   (x64 preset names are UNCHANGED, so
::         build\windows-msvc-release\bin still works exactly as before).
:: Exit codes: 0 = success, otherwise the underlying tool's exit code.
:: ============================================================================
setlocal enableextensions enabledelayedexpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: ---- Special verbs --------------------------------------------------------
if /i "%~1"=="clean" goto :CLEAN
if /i "%~1"=="--help" goto :HELP
if /i "%~1"=="-h"     goto :HELP
if /i "%~1"=="/?"     goto :HELP

:: ---- Defaults -------------------------------------------------------------
set "CFG=debug"
set "TC=msvc"
set "ARCH=x64"
set "TARGET="
set "CONFIGURE_ONLY=0"

:: No arguments at all -> interactive menu. Any argument -> classic parse
:: (fully back-compatible; CI / the test loop always pass arguments).
if "%~1"=="" goto :INTERACTIVE

:: ---- Parse positional args -----------------------------------------------
:PARSE
if "%~1"=="" goto :PARSED
if /i "%~1"=="debug"     ( set "CFG=debug" & shift & goto :PARSE )
if /i "%~1"=="release"   ( set "CFG=release" & shift & goto :PARSE )
if /i "%~1"=="msvc"      ( set "TC=msvc" & shift & goto :PARSE )
if /i "%~1"=="clang"     ( set "TC=clang" & shift & goto :PARSE )
if /i "%~1"=="x64"       ( set "ARCH=x64" & shift & goto :PARSE )
if /i "%~1"=="amd64"     ( set "ARCH=x64" & shift & goto :PARSE )
if /i "%~1"=="arm64"     ( set "ARCH=arm64" & shift & goto :PARSE )
if /i "%~1"=="configure" ( set "CONFIGURE_ONLY=1" & shift & goto :PARSE )
:: Anything else is the target name.
set "TARGET=%~1"
shift
goto :PARSE
:PARSED
goto :RUN

:: ---- Interactive menu -----------------------------------------------------
:INTERACTIVE
echo.
echo   ===== Cardinal interactive build =====
echo.
echo   Target OS:
echo     [1] Windows  (this host)
echo     [2] Linux
set "_os=1"
set /p "_os=  Choose [1]: "
if "%_os%"=="2" (
    echo.
    echo   Linux builds must run on a Linux host or WSL. From there run:
    echo       ./build.sh                ^(interactive^)
    echo       ./build.sh release        ^(RelWithDebInfo^)
    echo   Nothing was built on this Windows host.
    exit /b 0
)
echo.
echo   Architecture:
echo     [1] x64    ^(default^)
echo     [2] arm64
set "_a=1"
set /p "_a=  Choose [1]: "
if "%_a%"=="2" ( set "ARCH=arm64" ) else ( set "ARCH=x64" )
echo.
echo   Build type:
echo     [1] Release  ^(RelWithDebInfo^)
echo     [2] Debug
set "_c=1"
set /p "_c=  Choose [1]: "
if "%_c%"=="2" ( set "CFG=debug" ) else ( set "CFG=release" )
echo.
echo   Toolchain:
echo     [1] MSVC      ^(cl^)
echo     [2] Clang     ^(clang-cl^)
set "_t=1"
set /p "_t=  Choose [1]: "
if "%_t%"=="2" ( set "TC=clang" ) else ( set "TC=msvc" )
echo.
echo   Target ^(blank = all targets^):
set "TARGET="
set /p "TARGET=  Target: "
echo.
goto :RUN

:: ---- Resolve preset + dev-env arch ---------------------------------------
:RUN
if /i "%ARCH%"=="arm64" (
    set "PRESET=windows-%TC%-%CFG%-arm64"
    set "VSARCH=arm64"
    set "VSHOST=-host_arch=x64"
) else (
    set "PRESET=windows-%TC%-%CFG%"
    set "VSARCH=x64"
    set "VSHOST="
)

echo [build.bat] Preset: %PRESET%

:: ---- Locate VS2022 + activate dev env ------------------------------------
:: Prefer vswhere (ships with every VS install since 2017). Fall back to
:: common install dirs so the script also works on a fresh box where
:: vswhere isn't on PATH.
set "VSDEV="
where /q vswhere.exe
if not errorlevel 1 (
    for /f "usebackq delims=" %%I in (`vswhere -latest -property installationPath`) do (
        if exist "%%I\Common7\Tools\VsDevCmd.bat" set "VSDEV=%%I\Common7\Tools\VsDevCmd.bat"
    )
)
if not defined VSDEV (
    for %%E in (Enterprise Professional Community BuildTools Preview) do (
        for %%P in ("C:\Program Files\Microsoft Visual Studio\2022\%%E\Common7\Tools\VsDevCmd.bat" ^
                   "C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\Common7\Tools\VsDevCmd.bat") do (
            if not defined VSDEV if exist %%P set "VSDEV=%%~P"
        )
    )
)
if not defined VSDEV (
    echo [build.bat] ERROR: VS2022 not found. Install Visual Studio 2022 ^(any edition^).
    exit /b 1
)
echo [build.bat] VS dev env: %VSDEV% ^(-arch=%VSARCH% %VSHOST%^)
call "%VSDEV%" -arch=%VSARCH% %VSHOST% -no_logo >nul
if errorlevel 1 (
    echo [build.bat] ERROR: VsDevCmd.bat failed for arch %VSARCH%.
    exit /b %ERRORLEVEL%
)

:: ---- Sanity: cmake on PATH ------------------------------------------------
where /q cmake.exe
if errorlevel 1 (
    echo [build.bat] ERROR: cmake.exe not on PATH after VsDevCmd.
    echo [build.bat]        Install CMake or check the VS installation.
    exit /b 1
)

:: ---- Configure ------------------------------------------------------------
echo [build.bat] === Configure preset: %PRESET% ===
cmake --preset %PRESET% -S "%ROOT%"
if errorlevel 1 (
    echo [build.bat] ERROR: cmake configure failed.
    exit /b %ERRORLEVEL%
)

if "%CONFIGURE_ONLY%"=="1" (
    echo [build.bat] Configure-only: skipping build.
    exit /b 0
)

:: ---- Build ----------------------------------------------------------------
if defined TARGET (
    echo [build.bat] === Build preset: %PRESET% / target %TARGET% ===
    cmake --build --preset %PRESET% --target %TARGET%
) else (
    echo [build.bat] === Build preset: %PRESET% / all targets ===
    cmake --build --preset %PRESET%
)
set "RC=%ERRORLEVEL%"
if %RC% NEQ 0 (
    echo [build.bat] ERROR: build failed with code %RC%.
    exit /b %RC%
)
echo [build.bat] Done. Output: %ROOT%\build\%PRESET%\bin
echo [build.bat]       Launchers: %ROOT%\run\^<Target^>.cmd
exit /b 0

:: ---- clean ----------------------------------------------------------------
:CLEAN
echo [build.bat] Removing build\* and run\*
if exist "%ROOT%\build\" (
    rmdir /s /q "%ROOT%\build"
    if errorlevel 1 (
        echo [build.bat] ERROR: failed to remove build directory.
        exit /b %ERRORLEVEL%
    )
)
if exist "%ROOT%\run\" rmdir /s /q "%ROOT%\run"
echo [build.bat] Clean complete.
exit /b 0

:: ---- help -----------------------------------------------------------------
:HELP
echo Usage:
echo   build                                    Interactive menu
echo   build [debug^|release] [msvc^|clang] [x64^|arm64] [target] [configure]
echo   build clean
echo.
echo Examples:
echo   build                                    Interactive (OS/arch/cfg/tc)
echo   build release                            RelWithDebInfo, MSVC, x64
echo   build debug clang Cardinal_System_Studio Debug clang-cl, single target
echo   build release clang arm64                RelWithDebInfo, clang-cl, arm64
echo   build configure release                  Configure only, no build
echo   build clean                              Wipe build\ and run\
exit /b 0
