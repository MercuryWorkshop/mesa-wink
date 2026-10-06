#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare a pinned Fabric/Iris profile and launch it in a scratch game directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile
from assets import get_json, download, digest
from build_replay_support import HERE, MESA


def library_path(name):
    group, artifact, version, *classifier = name.split(":")
    filename = f"{artifact}-{version}" + ("-" + classifier[0] if classifier else "") + ".jar"
    return "/".join([group.replace(".", "/"), artifact, version, filename])


def allowed(library, system):
    result = not library.get("rules")
    for rule in library.get("rules", []):
        os_rule = rule.get("os", {})
        if not os_rule or os_rule.get("name", system) == system:
            result = rule["action"] == "allow"
    return result


def fetch_file(cache, relative, url, sha1=None):
    path = cache / relative
    print("asset", relative, flush=True)
    if sha1 is None:
        # Maven's published checksum is locked before the JAR is fetched.
        from assets import request
        with request(url + ".sha1") as response:
            sha1 = response.read().decode().split()[0]
    return download({"filename": path.name, "url": url, "hashes": {"sha1": sha1}}, path.parent)


def prepare(a):
    corpus = json.loads((HERE / "corpus.json").read_text())
    cache = a.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / "profile.json"
    if not metadata.exists():
        manifest = get_json("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json")
        version = next(v for v in manifest["versions"] if v["id"] == corpus["minecraft"])
        mc = get_json(version["url"])
        fabric = get_json(f"https://meta.fabricmc.net/v2/versions/loader/{corpus['minecraft']}/{corpus['fabric_loader']}/profile/json")
        metadata.write_text(json.dumps({"minecraft": mc, "fabric": fabric}, indent=2) + "\n")
    profile = json.loads(metadata.read_text())
    mc, fabric = profile["minecraft"], profile["fabric"]
    system = "osx" if a.platform == "macos" else "linux"
    libraries = []
    lock = {}
    for lib in mc["libraries"] + fabric["libraries"]:
        if not allowed(lib, system):
            continue
        artifact = lib.get("downloads", {}).get("artifact")
        if artifact:
            relative, url, sha1 = artifact["path"], artifact["url"], artifact["sha1"]
            # Mojang's manifest defaults to x86 Linux native JARs. ARM64
            # classifiers come from the same LWJGL Maven release.
            if "natives-linux" in relative and not relative.endswith(("natives-linux.jar", "natives-linux-arm64.jar")):
                continue
            if relative.endswith("natives-linux.jar"):
                relative = relative.replace("natives-linux", "natives-linux-arm64")
                url = "https://repo.maven.apache.org/maven2/" + relative
                sha1 = None
            if relative.endswith("natives-macos.jar"):
                relative = relative.replace("natives-macos", "natives-macos-arm64")
                url = "https://repo.maven.apache.org/maven2/" + relative
                sha1 = None
        else:
            relative = library_path(lib["name"])
            url = lib.get("url", "https://libraries.minecraft.net/").rstrip("/") + "/" + relative
            sha1 = None
        path = fetch_file(cache / "libraries", relative, url, sha1)
        libraries.append(path)
        lock[str(path.relative_to(cache))] = digest(path, "sha1")
    client = mc["downloads"]["client"]
    libraries.append(fetch_file(cache, "minecraft-client.jar", client["url"], client["sha1"]))
    lock["minecraft-client.jar"] = client["sha1"]
    index = mc["assetIndex"]
    index_path = fetch_file(cache, f"assets/indexes/{index['id']}.json", index["url"], index["sha1"])
    objects = json.loads(index_path.read_text())["objects"].values()
    def asset(obj):
        key = obj["hash"]
        fetch_file(cache, f"assets/objects/{key[:2]}/{key}", f"https://resources.download.minecraft.net/{key[:2]}/{key}", key)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(asset, objects))
    manifest = {"schema": 1, "platform": a.platform, "profile_sha512": digest(metadata),
                "libraries": lock, "classpath": [str(p.relative_to(cache)) for p in libraries]}
    (cache / (a.platform + ".lock.json")).write_text(json.dumps(manifest, indent=2) + "\n")


def launch(a):
    cache = a.cache.resolve()
    manifest = json.loads((cache / (a.platform + ".lock.json")).read_text())
    if digest(cache / "profile.json") != manifest["profile_sha512"]:
        raise RuntimeError("Minecraft/Fabric profile checksum changed")
    profile = json.loads((cache / "profile.json").read_text())
    for name, expected in manifest["libraries"].items():
        if digest(cache / name, "sha1") != expected:
            raise RuntimeError("library checksum changed: " + name)
    if a.game.exists():
        if not a.fresh_game:
            raise RuntimeError("Game directory exists; use a new scratch directory")
        a.game.rename(a.game.with_name(a.game.name + ".previous-" + str(time.time_ns())))
    a.game.mkdir(parents=True)
    if not a.world or not a.world.is_dir():
        raise RuntimeError("--world must name the generated correctness fixture save")
    shutil.copytree(a.world, a.game / "saves/fixture", ignore=shutil.ignore_patterns("session.lock"))
    lock = json.loads((a.assets / "assets.lock.json").read_text())
    (a.game / "mods").mkdir()
    for project in ("iris", "sodium", "fabric-api"):
        asset = lock["assets"][project]
        path = a.assets / "mods" / asset["filename"]
        if digest(path) != asset["hashes"]["sha512"]:
            raise RuntimeError("mod checksum changed: " + project)
        shutil.copy2(path, a.game / "mods")
    if not a.fixture_mod:
        raise RuntimeError("--fixture-mod is required for deterministic capture")
    shutil.copy2(a.fixture_mod, a.game / "mods")
    pack = lock["assets"][a.pack]
    source = a.assets / "shaderpacks" / pack["filename"]
    if digest(source) != pack["hashes"]["sha512"]:
        raise RuntimeError("shaderpack checksum changed")
    (a.game / "shaderpacks").mkdir()
    shutil.copy2(source, a.game / "shaderpacks")
    (a.game / "config").mkdir()
    (a.game / "config/iris.properties").write_text(f"enableShaders=true\nshaderPack={pack['filename']}\n")
    (a.game / "options.txt").write_text("renderDistance:8\nsimulationDistance:5\nmaxFps:30\nenableVsync:false\nguiScale:2\nparticles:2\nfov:0.0\nonboardAccessibility:false\nsoundCategory_master:0.0\n")
    if a.preset == "stress":
        # Each pack defines its own profiles; require an explicit mapping.
        if not a.pack_settings:
            raise RuntimeError("stress preset requires --pack-settings; no silent default preset fallback")
        shutil.copy2(a.pack_settings, a.game / "shaderpacks" / (pack["filename"] + ".txt"))
    if a.resourcepack:
        resource_lock = HERE / "resourcepack.lock.json"
        if resource_lock.is_file():
            pinned = json.loads(resource_lock.read_text())
            if a.resourcepack.name == pinned["filename"] and digest(a.resourcepack) != pinned["hashes"]["sha512"]:
                raise RuntimeError("LabPBR resource pack checksum changed")
        (a.game / "resourcepacks").mkdir()
        shutil.copy2(a.resourcepack, a.game / "resourcepacks")
        with (a.game / "options.txt").open("a") as out:
            out.write('resourcePacks:["vanilla","file/' + a.resourcepack.name + '"]\n')
    if a.scene == "labpbr" and not a.resourcepack:
        raise RuntimeError("LabPBR scene requires a pinned --resourcepack")
    natives = a.game / "natives"
    natives.mkdir()
    for name in manifest["classpath"]:
        if "natives-" in name:
            with zipfile.ZipFile(cache / name) as jar:
                for member in jar.namelist():
                    if member.endswith((".so", ".dylib")):
                        (natives / Path(member).name).write_bytes(jar.read(member))
    args = [a.java, "-Xmx3G", "-Djava.library.path=" + str(natives.resolve()),
            "-Dwink.fixture.scene=" + a.scene, "-Dwink.fixture.pack=" + a.pack]
    if a.resourcepack:
        args += ["-Dwink.fixture.resourcepack=file/" + a.resourcepack.name]
    if a.platform == "macos":
        args += ["-XstartOnFirstThread"]
    if a.wink:
        if not a.glfw:
            raise RuntimeError("Native Iris on Wink requires --glfw (the GLFW to SDL bridge)")
        args += ["-Dorg.lwjgl.opengl.libname=" + str(a.wink.resolve()), "-Dorg.lwjgl.glfw.libname=" + str(a.glfw.resolve())]
    args += ["-cp", os.pathsep.join(str(cache / name) for name in manifest["classpath"]), profile["fabric"]["mainClass"],
             "--username", "WinkFixture", "--version", profile["minecraft"]["id"], "--gameDir", str(a.game.resolve()),
             "--assetsDir", str(cache / "assets"), "--assetIndex", profile["minecraft"]["assetIndex"]["id"],
             "--uuid", "0" * 32, "--accessToken", "0", "--width", "640", "--height", "360",
             "--quickPlaySingleplayer", "fixture"]
    env = dict(os.environ)
    if a.wink:
        env["SDL_VIDEO_COCOA_WINK"] = str(a.wink.resolve())
    return subprocess.call(args, cwd=a.game, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "launch"])
    p.add_argument("--platform", choices=["macos", "linux"], default="macos" if platform.system() == "Darwin" else "linux")
    p.add_argument("--cache", type=Path, default=MESA / "build/wink-apps/minecraft")
    p.add_argument("--assets", type=Path, default=MESA / "build/wink-apps/assets")
    p.add_argument("--java", default="java")
    p.add_argument("--game", type=Path)
    p.add_argument("--fresh-game", action="store_true", help="archive any previous scratch game before launching")
    p.add_argument("--world", type=Path)
    p.add_argument("--fixture-mod", type=Path)
    p.add_argument("--pack", default="complementary-reimagined")
    p.add_argument("--scene", choices=["day", "night", "rain", "water", "nether", "end", "motion", "labpbr"], default="day")
    p.add_argument("--preset", choices=["default", "stress"], default="default")
    p.add_argument("--pack-settings", type=Path)
    p.add_argument("--resourcepack", type=Path)
    p.add_argument("--wink", type=Path)
    p.add_argument("--glfw", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a)
        return 0
    if not a.game:
        p.error("launch requires --game")
    return launch(a)


if __name__ == "__main__":
    raise SystemExit(main())
