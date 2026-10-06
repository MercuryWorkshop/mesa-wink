#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build fixtures using isolated toolchains; keep all build output under build/."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from build_replay_support import HERE, MESA, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--java-home", type=Path)
    a = p.parse_args()
    a.out = a.out.resolve()
    source = a.out / "fixture-build/minecraft"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE / "fixtures/minecraft", source, dirs_exist_ok=True)
    java = a.java_home or next((a.out / "toolchains/java-macos").glob("*/Contents/Home"))
    env = dict(os.environ, JAVA_HOME=str(java), GRADLE_USER_HOME=str(a.out / "gradle-cache"))
    run([a.out / "toolchains/gradle-8.12.1/bin/gradle", "--no-daemon", "build"], cwd=source, env=env)
    jars = list((source / "build/libs").glob("*.jar"))
    jar = next(p for p in jars if not p.name.endswith("-sources.jar"))
    shutil.copy2(jar, a.out / "wink-fixture.jar")
    if not (a.out / "fixture-world").exists():
        run([os.sys.executable, HERE / "fixtures/minecraft/world.py", a.out / "fixture-world"])
    shutil.copytree(HERE / "fixtures/godot", a.out / "godot-project", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
