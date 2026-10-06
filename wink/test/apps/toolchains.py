#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install isolated, checksum-pinned Java/Gradle build tools into build/."""
import argparse
import json
from pathlib import Path
import tarfile
import zipfile
from assets import download, request, digest
from build_replay_support import MESA

JAVA = {
    "macos": ("mac", "59422c2292ae4e76b87e00d8808dbe49cffa39af731e08bb0292ddb0af4e0261"),
    "linux": ("linux", "e5c41a1ab0865ea5de9b4529bf8526005f1d4593090845387d14fe450ce39c33")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=JAVA, default="macos")
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps/toolchains")
    a = p.parse_args()
    os_name, sha = JAVA[a.platform]
    name = f"OpenJDK21U-jdk_aarch64_{os_name}_hotspot_21.0.8_9.tar.gz"
    java = {"filename": name, "url": "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.8%2B9/" + name,
            "hashes": {"sha256": sha}}
    archive = download(java, a.out)
    dest = a.out / ("java-" + a.platform)
    if not dest.exists():
        dest.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    gradle_name = "gradle-8.12.1-bin.zip"
    lock_path = a.out / "gradle.lock.json"
    if lock_path.exists():
        gradle = json.loads(lock_path.read_text())
    else:
        url = "https://services.gradle.org/distributions/" + gradle_name
        with request(url + ".sha256") as response:
            checksum = response.read().decode().strip()
        gradle = {"filename": gradle_name, "url": url, "hashes": {"sha256": checksum}}
        lock_path.write_text(json.dumps(gradle, indent=2))
    archive = download(gradle, a.out)
    if not (a.out / "gradle-8.12.1").exists():
        with zipfile.ZipFile(archive) as zip:
            zip.extractall(a.out)
        (a.out / "gradle-8.12.1/bin/gradle").chmod(0o755)
    (a.out / (a.platform + ".lock.json")).write_text(json.dumps({"schema": 1, "java": java, "gradle": gradle}, indent=2))
    print("Java", next(dest.rglob("bin/java")))
    print("Gradle", a.out / "gradle-8.12.1/bin/gradle")


if __name__ == "__main__":
    main()
