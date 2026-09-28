# A private Python environment for Mesa's generators. Sourced by wink/build.sh.
#
# A venv rather than `pip install --user`: Arch and Homebrew both mark their
# system Pythons externally managed, and nothing here should need root. meson
# comes from the venv too when the host's is missing or older than Mesa's 1.4.

venv_setup() {
  WINK_STAGE=venv
  local reqs="$WINK_DIR/requirements.txt"
  local want_meson=""
  if ! have meson || ! version_ge "$(meson --version 2>/dev/null || echo 0)" 1.4.0; then
    want_meson="meson>=1.4,<2"
  fi
  WINK_VENV="$WINK_CACHE/venv-py$WINK_HOST_PY_VERSION"
  local key
  key="py=$WINK_HOST_PY_VERSION reqs=$(hash_file "$reqs") meson=$want_meson"

  if ! stamp_ok "$WINK_VENV" "$key"; then
    log "creating $WINK_VENV"
    rm -rf "$WINK_VENV"
    run "$WINK_HOST_PY" -m venv "$WINK_VENV"
    run "$WINK_VENV/bin/python" -m pip install --quiet --upgrade pip
    run "$WINK_VENV/bin/python" -m pip install --quiet -r "$reqs" ${want_meson:+"$want_meson"}
    stamp_write "$WINK_VENV" "$key"
  fi
  PATH="$WINK_VENV/bin:$PATH"
  export PATH
  WINK_PY="$WINK_VENV/bin/python"
  log "python $("$WINK_PY" -c 'import sys;print(sys.version.split()[0])'), meson $(meson --version)"
}
