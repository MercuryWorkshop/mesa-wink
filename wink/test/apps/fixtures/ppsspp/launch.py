#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Give each PPSSPP recording a fresh, writable configuration directory."""
import os
from pathlib import Path
import sys

config = Path.cwd() / "ppsspp-config"
system = config / "ppsspp/PSP/SYSTEM"
system.mkdir(parents=True)
# Avoid a first-launch controller configuration error before emulation starts.
(system / "controls.ini").write_text("[ControlMapping]\n")
(system / "ppsspp.ini").write_text("[General]\nFirstRun = false\n")
os.environ["XDG_CONFIG_HOME"] = str(config)
os.execv(sys.argv[1], sys.argv[1:])
