#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stage pinned sources and build a local Apple container reference image."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from build_replay_support import HERE, MESA, run, checkout, copy_artifact


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_lock(out, context, image, target, revision, pins):
    identity = json.loads(subprocess.check_output(["container", "image", "inspect", image], text=True))
    if not identity:
        raise RuntimeError("Reference image was not exported")
    lock = {"schema": 1, "mesa": revision, "pins": pins, "image": image, "target": target,
            "image_identity": identity,
            "inputs": {item.name: file_hash(item) for item in context.iterdir() if item.is_file()},
            "dockerfile_sha256": file_hash(context / "Dockerfile")}
    (out / "reference.lock.json").write_text(json.dumps(lock, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--image", default="wink-reference")
    p.add_argument("--target", choices=["reference", "applications", "extended"], help="build an intermediate image before the extended source builds")
    p.add_argument("--apps", action="store_true", help="include Godot, Qt Quick and Java for Iris recording")
    p.add_argument("--extended", action="store_true", help="also build pinned Filament and PPSSPP public-homebrew workloads")
    p.add_argument("--min-free-gib", type=float, default=12,
                   help="extended-build disk admission limit; lower only when source/build caches already exist")
    a = p.parse_args()
    if a.target == "extended" and not a.extended:
        p.error("--target extended requires --extended")
    if a.target == "applications" and not (a.apps or a.extended):
        p.error("--target applications requires --apps or --extended")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.min_free_gib < 4:
        p.error("--min-free-gib must be at least 4")
    if a.extended and shutil.disk_usage(a.out).free < a.min_free_gib * 1024**3:
        raise RuntimeError(f"Filament/PPSSPP source builds require {a.min_free_gib:g} GiB free disk space")
    pins = json.loads((HERE / "pins.json").read_text())
    stage = a.out / "reference-context"
    stage.mkdir(exist_ok=True)
    for name in ("apitrace", "gfxreconstruct"):
        source = checkout(a.out, name, pins[name])
        dest = stage / name
        source_stamp = stage / (name + ".staged-revision")
        if source_stamp.exists() and source_stamp.read_text() == pins[name]["commit"]:
            continue
        print("Staging pristine", name, flush=True)
        shutil.copytree(source, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        # The native/browser adapters patch cached sources. The reference uses
        # pristine upstream files, including every pinned submodule checkout.
        changed = subprocess.check_output(["git", "-C", source, "diff", "--name-only"], text=True).splitlines()
        for path in changed:
            original = subprocess.check_output(["git", "-C", source, "show", "HEAD:" + path])
            (dest / path).write_bytes(original)
        source_stamp.write_text(pins[name]["commit"])
    revision = subprocess.check_output(["git", "-C", MESA, "rev-parse",
                                        "HEAD" if pins["reference_mesa"] == "checkout" else pins["reference_mesa"]], text=True).strip()
    marker = stage / "mesa-revision"
    if not marker.exists() or marker.read_text().strip() != revision:
        tar = stage / "mesa.tar"
        run(["git", "-C", MESA, "archive", "--format=tar", "--output", tar, revision])
        dest = stage / "mesa"
        dest.mkdir(exist_ok=True)
        with tarfile.open(tar) as archive:
            archive.extractall(dest, filter="data")
        tar.unlink()
        marker.write_text(revision + "\n")
    for name in ("Dockerfile", "reference.py", "probe.c", "vk-probe.c"):
        shutil.copy2(HERE / "reference" / name, stage / name)
    # Apple container's incremental directory transfer can stall on large
    # source trees. Send one compressed file per pristine source checkout.
    for name in ("mesa", "apitrace", "gfxreconstruct"):
        archive_path = stage / (name + ".tar.gz")
        stamp = stage / (name + ".archive-revision")
        source_revision = revision if name == "mesa" else pins[name]["commit"]
        if name == "apitrace":
            from glreplay import adapt_wsi
            source_revision += hashlib.sha256((HERE / "glreplay.py").read_bytes() +
                (HERE / "replay/thread_context.hpp").read_bytes() +
                (HERE / "replay/egl_sync.hpp").read_bytes()).hexdigest()
            if not (stage / name).exists():
                with tarfile.open(archive_path) as archive:
                    archive.extractall(stage, filter="data")
            adapt_wsi(stage / name)
        if not archive_path.exists() or not stamp.exists() or stamp.read_text() != source_revision:
            print("Packing", name, flush=True)
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(stage / name, arcname=name)
            stamp.write_text(source_revision)
    (stage / ".dockerignore").write_text("mesa/\napitrace/\ngfxreconstruct/\n")
    if a.apps or a.extended:
        from assets import download
        godot = json.loads((HERE / "godot.lock.json").read_text())
        binary = download(godot, a.out / "app-downloads")
        shutil.copy2(binary, stage / "godot.zip")
        shutil.copytree(HERE / "fixtures/qtquick", stage / "qtquick", dirs_exist_ok=True)
        for name in ("minecraft.py", "assets.py", "build_replay_support.py", "build-replay.py", "corpus.json"):
            shutil.copy2(HERE / name, stage / name)
        with tarfile.open(stage / "app-inputs.tar.gz", "w:gz") as archive:
            for name in ("godot.zip", "qtquick", "minecraft.py", "assets.py", "build_replay_support.py", "build-replay.py", "corpus.json"):
                archive.add(stage / name, arcname=name)
    if a.extended:
        import importlib.util
        spec = importlib.util.spec_from_file_location("extended_apps", HERE / "extended-apps.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.stage_extended(a.out, stage, pins)
        # Keep application inputs in one explicitly referenced COPY archive.
        with tarfile.open(stage / "app-inputs.tar.gz", "w:gz", compresslevel=1) as archive:
            for name in ("godot.zip", "qtquick", "minecraft.py", "assets.py", "build_replay_support.py", "build-replay.py", "corpus.json",
                         "filament.tar.gz", "ppsspp.tar.gz", "pspautotests.tar.gz"):
                archive.add(stage / name, arcname=name)
    # Use an absolute context outside the ignored build directory.
    context = Path(tempfile.mkdtemp(prefix="wink-reference-context-", dir="/private/tmp"))
    inputs = ["mesa.tar.gz", "apitrace.tar.gz", "gfxreconstruct.tar.gz", "probe.c", "vk-probe.c", "reference.py"]
    if a.apps or a.extended:
        inputs.append("app-inputs.tar.gz")
    dockerfile = (stage / "Dockerfile").read_text()
    if a.apps or a.extended:
        before = "FROM ${BASE} AS reference"
        application_steps = "COPY app-inputs.tar.gz /src/app-inputs.tar.gz\nRUN mkdir -p /opt/app-inputs && tar -xzf /src/app-inputs.tar.gz -C /opt/app-inputs && rm /src/app-inputs.tar.gz\n\n"
        if a.extended:
            application_steps += "RUN mkdir -p /opt/extended-inputs && for name in filament ppsspp pspautotests; do tar -xzf \"/opt/app-inputs/$name.tar.gz\" -C /opt/extended-inputs && rm \"/opt/app-inputs/$name.tar.gz\"; done\n\n"
        dockerfile = dockerfile.replace(before, application_steps + before)
    if not a.extended:
        dockerfile = dockerfile.split("FROM applications AS application-builder")[0]
    if not (a.apps or a.extended):
        dockerfile = dockerfile.split("FROM reference AS applications")[0]
    # Content-address COPY inputs. Some older container CLI/shim combinations
    # reuse stale context contents under an unchanged source filename. Check
    # the transferred bytes inside the image before building any reference.
    for name in inputs:
        source = stage / name
        checksum = file_hash(source)
        addressed = checksum + "-" + name
        copy_artifact(source, context / addressed)
        dockerfile = re.sub(r"^COPY " + re.escape(name) + r" (.+)$",
                            lambda m: f"COPY {addressed} {m[1]}\nRUN echo '{checksum}  {m[1]}' | sha256sum -c -",
                            dockerfile, flags=re.M)
    (context / "Dockerfile").write_text(dockerfile)
    run(["container", "system", "start"])
    run(["container", "build", "--progress", "plain", *(["--target", a.target] if a.target else []), "--cpus", str(pins["resources"]["cpus"]),
         "--memory", pins["resources"]["memory"], "--build-arg", "BASE=" + pins["reference_base"],
         "--file", context / "Dockerfile", "--tag", a.image, context])
    write_lock(a.out, context, a.image, a.target, revision, pins)
    shutil.rmtree(context)


if __name__ == "__main__":
    main()
