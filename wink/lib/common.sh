# Shared helpers for wink/build.sh. Sourced, never run.
#
# Everything here has to work under the bash 3.2 that macOS ships as /bin/bash:
# no mapfile, no associative arrays, no ${x,,}, no `wait -n`. Empty arrays are
# expanded as ${a[@]+"${a[@]}"} so `set -u` does not trip on them.

WINK_STAGE=${WINK_STAGE:-wink}

log()  { printf '[%s] %s\n' "$WINK_STAGE" "$*" >&2; }
warn() { printf '[%s] warning: %s\n' "$WINK_STAGE" "$*" >&2; }
die()  { printf '[%s] error: %s\n' "$WINK_STAGE" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# Run a command, echoing it first when -v was given.
run() {
  if [ "${WINK_VERBOSE:-0}" = 1 ]; then printf '[%s] + %s\n' "$WINK_STAGE" "$*" >&2; fi
  "$@"
}

ncpu() {
  getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

# Absolute, symlink-resolved path of an existing directory. `realpath` and
# `readlink -f` are not on every macOS.
absdir() { (CDPATH='' cd -- "$1" && pwd -P); }

sha512_of() {
  if have sha512sum; then sha512sum "$1" | cut -d' ' -f1
  elif have shasum; then shasum -a 512 "$1" | cut -d' ' -f1
  else "${WINK_PY:-python3}" -c 'import hashlib,sys;print(hashlib.sha512(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
  fi
}

# A short, stable hash of a string: cache keys and stamps, not security.
hash_str() {
  if have sha256sum; then printf '%s' "$1" | sha256sum | cut -c1-12
  else printf '%s' "$1" | shasum -a 256 | cut -c1-12
  fi
}

hash_file() {
  if have sha256sum; then sha256sum "$1" | cut -c1-12
  else shasum -a 256 "$1" | cut -c1-12
  fi
}

# A stamp records what a finished stage was built from. A stage whose stamp
# matches is skipped; one that fails half way never writes it, so the next run
# starts that stage over instead of trusting a partial result.
stamp_ok()    { [ -f "$1/.wink-stamp" ] && [ "$(cat "$1/.wink-stamp")" = "$2" ]; }
stamp_write() { printf '%s' "$2" > "$1/.wink-stamp"; }

# Apply a patch to a git checkout exactly once: applied, already applied, or a
# hard error -- never a silent half.
apply_patch_once() {
  local repo=$1 patch=$2
  if git -C "$repo" apply --check "$patch" >/dev/null 2>&1; then
    log "applying $(basename "$patch")"
    git -C "$repo" apply "$patch"
  elif git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
    log "$(basename "$patch") already applied"
  else
    die "$(basename "$patch") does not apply to $repo"
  fi
}

# Compare dotted versions: version_ge 3.8.2 3 -> true.
version_ge() {
  "${WINK_PY:-python3}" - "$1" "$2" <<'EOF'
import sys
def key(v):
    out = []
    for p in v.split('.'):
        n = ''.join(c for c in p if c.isdigit())
        out.append(int(n) if n else 0)
    return out
sys.exit(0 if key(sys.argv[1]) >= key(sys.argv[2]) else 1)
EOF
}

# bash 3.2 cannot parse a `case` inside $(...), so the distro is read here and
# not in a command substitution.
detect_distro() {
  local ID="" ID_LIKE=""
  # shellcheck disable=SC1091
  [ -r /etc/os-release ] && . /etc/os-release
  case " $ID $ID_LIKE " in
    *" arch "*) WINK_DISTRO=arch ;;
    *" debian "*|*" ubuntu "*) WINK_DISTRO=debian ;;
    *) WINK_DISTRO=$ID ;;
  esac
}

detect_host() {
  WINK_HOST_OS=$(uname -s)
  WINK_HOST_ARCH=$(uname -m)
  WINK_DISTRO=
  case "$WINK_HOST_OS" in
    Darwin) WINK_HOST=macos ;;
    Linux) WINK_HOST=linux; detect_distro ;;
    *) die "unsupported host $WINK_HOST_OS" ;;
  esac
}
