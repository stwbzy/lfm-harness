#!/usr/bin/env bash
# Builds the Rust core for Android and drops libharness_core.so into jniLibs.
#   GGML_CPU_ARM_ARCH  ggml ARM target (default armv8.2-a+dotprod+fp16; use
#                      armv8.6-a+dotprod+i8mm on newer SoCs, armv8-a for safety).
#                      After changing it, run in core/:
#                      cargo clean -p llama-cpp-sys-2 --target aarch64-linux-android --release
#                      (the CMake cache does not pick the new value up otherwise)
#   PROFILE            release (default) | debug
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-/opt/android-ndk}"
export GGML_CPU_ARM_ARCH="${GGML_CPU_ARM_ARCH:-armv8.2-a+dotprod+fp16}"
profile="${PROFILE:-release}"
flag=$([ "$profile" = release ] && echo --release || true)

cd "$root/core"
cargo ndk -t arm64-v8a -P 28 -o "$root/android/app/src/main/jniLibs" build --lib $flag
