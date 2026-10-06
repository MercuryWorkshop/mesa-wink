#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run the shared packaged-driver smoke tests with a native libwink.
# Usage: run-smoke.sh LIBWINK [OUT]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mesa=$(cd "$here/../../.." && pwd)
library=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
out=${2:-$(mktemp -d)}
mkdir -p "$out"
cc -O2 -I"$mesa/include" -I"$mesa/src/gallium/targets/wink-wasm" \
   "$here/../wink_smoke.c" "$library" -Wl,-rpath,"$(dirname "$library")" \
   -o "$out/wink-smoke"
"$out/wink-smoke"
