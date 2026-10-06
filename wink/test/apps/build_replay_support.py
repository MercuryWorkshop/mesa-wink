# SPDX-License-Identifier: MIT
# Shared import for the hyphenated command-line build script.
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_replay", Path(__file__).with_name("build-replay.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HERE, MESA, run, checkout, patch = module.HERE, module.MESA, module.run, module.checkout, module.patch
