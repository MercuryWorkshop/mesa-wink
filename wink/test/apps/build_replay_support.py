# SPDX-License-Identifier: MIT
# Shared import for the hyphenated command-line build script.
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

spec = importlib.util.spec_from_file_location("build_replay", Path(__file__).with_name("build-replay.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HERE, MESA, run, checkout, patch = module.HERE, module.MESA, module.run, module.checkout, module.patch


def copy_artifact(source, destination):
    """Preserve independent bytes while sharing APFS blocks on the host."""
    if sys.platform == "darwin":
        result = subprocess.run(["cp", "-c", str(source), str(destination)], capture_output=True)
        if result.returncode == 0:
            return
    shutil.copy2(source, destination)
