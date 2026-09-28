# Host tools. Sourced by wink/build.sh.
#
# Only what a Linux development machine would normally already have is asked
# of the host: git, curl, cmake, ninja, bison, flex, python3. emsdk, Tint,
# emdawnwebgpu and the Python modules Mesa's generators need are all fetched
# by later stages into the cache, so they are never the host's problem.
#
# macOS needs Homebrew for the rest. Its bison and flex are keg-only -- Apple's
# /usr/bin/bison is 2.3, which Mesa's grammars do not accept -- so their
# prefixes go on PATH ahead of /usr/bin.

WINK_BREW_FORMULAE="git bison flex cmake ninja python@3.13 node"
WINK_PACMAN_PACKAGES="base-devel git curl python cmake ninja bison flex nodejs"

deps_hint() {
  case "$WINK_HOST" in
    macos) echo "brew install $WINK_BREW_FORMULAE" ;;
    linux)
      case "$WINK_DISTRO" in
        arch) echo "sudo pacman -S --needed $WINK_PACMAN_PACKAGES" ;;
        debian) echo "sudo apt-get install build-essential git curl python3 python3-venv cmake ninja-build bison flex nodejs" ;;
        *) echo "install: git curl python3 (with venv) cmake ninja bison>=3 flex" ;;
      esac
      ;;
  esac
}

deps_install() {
  case "$WINK_HOST" in
    macos)
      have brew || die "Homebrew is required on macOS: https://brew.sh"
      # shellcheck disable=SC2086
      run brew install $WINK_BREW_FORMULAE
      ;;
    linux)
      case "$WINK_DISTRO" in
        # shellcheck disable=SC2086
        arch) run sudo pacman -S --needed --noconfirm $WINK_PACMAN_PACKAGES ;;
        *) die "--install-deps only knows Homebrew and pacman; run: $(deps_hint)" ;;
      esac
      ;;
  esac
}

deps_prepend_brew_kegs() {
  [ "$WINK_HOST" = macos ] && have brew || return 0
  local keg prefix
  for keg in bison flex; do
    prefix=$(brew --prefix "$keg" 2>/dev/null) || continue
    [ -d "$prefix/bin" ] && PATH="$prefix/bin:$PATH"
  done
  export PATH
}

# A python new enough for Mesa's generators (3.10), preferring the versions
# emsdk 3.1.56 predates the least: it was released against 3.12.
deps_find_python() {
  local c v
  for c in python3.12 python3.13 python3.11 python3.10 python3 python3.14; do
    have "$c" || continue
    v=$("$c" -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
    if version_ge "$v" 3.10; then
      WINK_HOST_PY=$(command -v "$c")
      WINK_HOST_PY_VERSION=$v
      return 0
    fi
  done
  return 1
}

deps_check() {
  WINK_STAGE=deps
  deps_prepend_brew_kegs

  local missing="" t
  for t in git curl cmake ninja bison flex; do
    have "$t" || missing="$missing $t"
  done
  deps_find_python || missing="$missing python3>=3.10"

  if [ -n "$missing" ] && [ "${WINK_INSTALL_DEPS:-0}" = 1 ]; then
    log "installing:$missing"
    deps_install
    deps_prepend_brew_kegs
    missing=""
    for t in git curl cmake ninja bison flex; do
      have "$t" || missing="$missing $t"
    done
    deps_find_python || missing="$missing python3>=3.10"
  fi
  [ -z "$missing" ] || die "missing:$missing -- run with --install-deps, or: $(deps_hint)"

  local bv
  bv=$(bison --version | sed -n '1s/.* \([0-9][0-9.]*\)$/\1/p')
  version_ge "${bv:-0}" 3.0 || die "bison $bv at $(command -v bison) is too old (need 3.x); $(deps_hint)"

  log "host $WINK_HOST${WINK_DISTRO:+ ($WINK_DISTRO)} $WINK_HOST_ARCH; python $WINK_HOST_PY_VERSION; bison $bv"
}
