#!/usr/bin/env bash
# =============================================================================
# Cardinal — Linux build script.
#
# Wraps the CMakePresets.json configure + build in one step. Uses the
# linux-clang-{debug,release}[-arm64] presets (clang/clang++ from PATH).
#
# Run with NO arguments for an interactive menu (target OS / architecture /
# build type). Pass any positional arg to skip the menu and use the classic
# non-interactive form (unchanged — scripts/CI rely on this):
#
#   ./build.sh                          - INTERACTIVE menu
#   ./build.sh debug                    - debug, x64
#   ./build.sh release                  - RelWithDebInfo, x64
#   ./build.sh release arm64            - RelWithDebInfo, aarch64 (cross)
#   ./build.sh debug Cardinal_System_Studio
#                                       - debug, only that target
#   ./build.sh release -- -j 8 -v       - forward extra args to cmake --build
#   ./build.sh clean                    - rm -rf build/ run/
#   ./build.sh configure [debug|release]- only configure (no build)
#
# Positional args (any order before the optional -- target):
#   debug | release          (default: debug)
#   x64   | arm64            (default: x64)
#   <target>                 (default: all)
#
# Output: build/<preset>/bin (x64 preset names UNCHANGED). Anything after
# `--` is forwarded verbatim to `cmake --build`. Exit codes mirror the tool.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage:
  ./build.sh                                   # Interactive menu
  ./build.sh [debug|release] [x64|arm64] [target] [configure] [-- <cmake args>]
  ./build.sh clean

Examples:
  ./build.sh                              # Interactive (OS/arch/cfg)
  ./build.sh release                      # RelWithDebInfo, x64, all targets
  ./build.sh release arm64                # RelWithDebInfo, aarch64 cross
  ./build.sh debug Cardinal_System_Studio # Debug, single target
  ./build.sh release -- -j 16 -v          # Release, parallel + verbose
  ./build.sh configure release            # Configure only, no build
  ./build.sh clean                        # Wipe build/ and run/
EOF
}

# ---- Special verbs --------------------------------------------------------
case "${1:-}" in
    -h|--help|/?) usage; exit 0 ;;
    clean)
        echo "[build.sh] Removing $ROOT/build and $ROOT/run"
        rm -rf "$ROOT/build" "$ROOT/run"
        echo "[build.sh] Clean complete."
        exit 0
        ;;
esac

# ---- Defaults -------------------------------------------------------------
CFG="debug"
ARCH="x64"
TARGET=""
CONFIGURE_ONLY=0
EXTRA=()

if [[ $# -eq 0 ]]; then
    # ---- Interactive menu -------------------------------------------------
    echo
    echo "  ===== Cardinal interactive build ====="
    echo
    echo "  Target OS:"
    echo "    [1] Linux  (this host)"
    echo "    [2] Windows"
    read -r -p "  Choose [1]: " _os || true
    if [[ "${_os:-1}" == "2" ]]; then
        echo
        echo "  Windows builds must run on a Windows host. From there run:"
        echo "      build                 (interactive)"
        echo "      build release         (RelWithDebInfo)"
        echo "  Nothing was built on this Linux host."
        exit 0
    fi
    echo
    echo "  Architecture:"
    echo "    [1] x64    (default)"
    echo "    [2] arm64  (aarch64 cross — needs an aarch64 sysroot/toolchain)"
    read -r -p "  Choose [1]: " _a || true
    [[ "${_a:-1}" == "2" ]] && ARCH="arm64" || ARCH="x64"
    echo
    echo "  Build type:"
    echo "    [1] Release  (RelWithDebInfo)"
    echo "    [2] Debug"
    read -r -p "  Choose [1]: " _c || true
    [[ "${_c:-1}" == "2" ]] && CFG="debug" || CFG="release"
    echo
    read -r -p "  Target (blank = all targets): " TARGET || true
    echo
else
    # ---- Parse positional args -------------------------------------------
    while [[ $# -gt 0 ]]; do
        case "$1" in
            debug)     CFG="debug"; shift ;;
            release)   CFG="release"; shift ;;
            x64|amd64) ARCH="x64"; shift ;;
            arm64|aarch64) ARCH="arm64"; shift ;;
            configure) CONFIGURE_ONLY=1; shift ;;
            --)        shift; EXTRA=("$@"); break ;;
            -*)        EXTRA+=("$1"); shift ;;       # implicit pass-through
            *)         TARGET="$1"; shift ;;
        esac
    done
fi

if [[ "$ARCH" == "arm64" ]]; then
    PRESET="linux-clang-$CFG-arm64"
else
    PRESET="linux-clang-$CFG"
fi

# ---- Sanity: required tools on PATH ---------------------------------------
need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[build.sh] ERROR: '$1' not on PATH. Install it and retry." >&2
        echo "[build.sh]        (Debian/Ubuntu: apt install $2)" >&2
        exit 1
    fi
}
need cmake     "cmake"
need ninja     "ninja-build"
need clang     "clang"
need clang++   "clang"

# vcpkg is optional but the project's CMakeLists picks it up via VCPKG_ROOT.
# Warn (don't fail) if it isn't set — most builds need it.
if [[ -z "${VCPKG_ROOT:-}" ]]; then
    echo "[build.sh] WARN: VCPKG_ROOT not set — third-party deps may fail to find."
fi

# ---- Configure ------------------------------------------------------------
echo "[build.sh] === Configure preset: $PRESET ==="
cmake --preset "$PRESET" -S "$ROOT"

if [[ "$CONFIGURE_ONLY" == "1" ]]; then
    echo "[build.sh] Configure-only: skipping build."
    exit 0
fi

# ---- Build ----------------------------------------------------------------
BUILD_ARGS=()
if [[ -n "$TARGET" ]]; then
    BUILD_ARGS+=(--target "$TARGET")
fi
BUILD_ARGS+=("${EXTRA[@]:-}")

echo "[build.sh] === Build preset: $PRESET ${TARGET:+/ target $TARGET} ==="
cmake --build --preset "$PRESET" "${BUILD_ARGS[@]}"

echo "[build.sh] Done. Output: $ROOT/build/$PRESET/bin"
echo "[build.sh]       Launchers: $ROOT/run/<Target>"
