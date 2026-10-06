#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Expand the locked shader corpus into concrete local recording commands."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from assets import digest
from build_replay_support import HERE, MESA


def stress_settings(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read("shaders/shaders.properties").decode(errors="replace") if "shaders/shaders.properties" in archive.namelist() else ""
        text = text.replace("\\\n", " ")
        profiles = dict(re.findall(r"^\s*profile\.([^\s=]+)\s*=\s*(.*)$", text, re.M))
        selected = next((name for choice in ("extreme", "ultra", "veryhigh", "high") for name in profiles if name.lower().replace("_", "") == choice), None)
        options = {}
        def expand(name, seen):
            if name in seen:
                raise ValueError("cyclic pack profile")
            for token in profiles[name].split():
                if token.startswith("profile."):
                    expand(token[8:], seen | {name})
                elif token.startswith("program.") or token.startswith("!program."):
                    raise ValueError("program exclusions require a curated settings file")
                elif "=" in token or ":" in token:
                    key, value = re.split(r"[=:]", token, maxsplit=1)
                    options[key] = value
                else:
                    options[token.lstrip("!")] = "false" if token.startswith("!") else "true"
        if selected:
            expand(selected, set())
            return selected, options
        # Complementary uses the documented PROFILE option rather than
        # shaderpack profile entries. Read its allowed values from the ZIP.
        for name in archive.namelist():
            if name.startswith("shaders/") and name.endswith((".glsl", ".vsh", ".fsh", ".settings", ".inc")):
                for line in archive.read(name).decode(errors="replace").splitlines():
                    match = re.match(r"\s*#define\s+PROFILE\s+\d+.*?\[([0-9 ]+)\]", line)
                    if match:
                        return "PROFILE", {"PROFILE": str(max(map(int, match[1].split())))}
        # Packs without named profiles can still expose shadow resolution
        # through Iris's shader-option syntax. Use an actual declared value.
        for name in archive.namelist():
            if name.startswith("shaders/") and name.endswith((".glsl", ".vsh", ".fsh", ".settings", ".inc")):
                for line in archive.read(name).decode(errors="replace").splitlines():
                    match = re.search(r"(?:#define\s+|const\s+int\s+)shadowMapResolution\s*(?:=\s*)?\d+.*?\[([0-9 ]+)\]", line)
                    if match:
                        values = [v for v in map(int, match[1].split()) if v <= 4096]
                        if values:
                            return "shadow-resolution", {"shadowMapResolution": str(max(values))}
    raise ValueError("no known stress profile; supply a curated settings file")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--settings", type=Path, default=HERE / "presets", help="curated per-pack stress settings ID.txt")
    p.add_argument("--resourcepack", type=Path, help="local public LabPBR resource pack")
    p.add_argument("--tools", type=Path, help="pinned runtime tree from build-extended.py")
    a = p.parse_args()
    if not a.resourcepack:
        resourcepack = json.loads((HERE / "resourcepack.lock.json").read_text())
        candidate = a.out / "assets/resourcepacks" / resourcepack["filename"]
        if candidate.is_file() and digest(candidate) == resourcepack["hashes"]["sha512"]:
            a.resourcepack = candidate
    corpus = json.loads((HERE / "corpus.json").read_text())
    lock = json.loads((a.out / "assets/assets.lock.json").read_text())
    recipes = json.loads((HERE / "recipes.example.json").read_text())
    if a.tools:
        manifest = a.tools / "tools.lock.json"
        if not manifest.is_file():
            raise RuntimeError("Build the pinned --tools tree first")
        for name, recipe in recipes.items():
            if name.startswith(("filament-", "ppsspp-")):
                recipe["command"] = [word.replace("/opt/apps/", "/tools/") for word in recipe["command"]]
                recipe["tools_manifest_sha256"] = digest(manifest, "sha256")
                recipe.setdefault("env", {})["LD_LIBRARY_PATH"] = "/opt/reference/lib:/opt/tools/lib:/tools/lib"
    presets = a.out / "presets"
    presets.mkdir(exist_ok=True)
    missing = []
    for pack, asset in lock["assets"].items():
        if pack not in corpus["shaderpacks"] + [v["id"] for v in corpus["manual_shaderpacks"]]:
            continue
        settings = presets / (pack + ".txt")
        try:
            source = a.settings / (pack + ".txt") if a.settings else None
            if source and source.is_file():
                settings.write_bytes(source.read_bytes())
            else:
                name, options = stress_settings(a.out / "assets/shaderpacks" / asset["filename"])
                settings.write_text("# locked stress profile: " + name + "\n" + "".join(k + "=" + v + "\n" for k,v in options.items()))
        except Exception as e:
            missing.append({"pack": pack, "preset": "stress", "error": str(e)})
        for scene in corpus["scenes"]:
            for preset in corpus["presets"]:
                name = f"iris-{pack}-{scene}-{preset}"
                command = ["python3", "/suite/minecraft.py", "launch", "--platform", "linux",
                           "--cache", "/work/minecraft", "--assets", "/work/assets", "--fresh-game", "--game", "/work/games/" + name,
                           "--world", "/work/fixture-world", "--fixture-mod", "/work/wink-fixture.jar",
                           "--pack", pack, "--scene", scene, "--preset", preset]
                if preset == "stress":
                    if not settings.exists():
                        continue
                    command += ["--pack-settings", "/work/presets/" + settings.name]
                if scene == "labpbr":
                    if not a.resourcepack:
                        continue
                    import shutil
                    shutil.copy2(a.resourcepack, a.out / a.resourcepack.name)
                    command += ["--resourcepack", "/work/" + a.resourcepack.name]
                recipes[name] = {"command": command,
                    "env": {"ALSOFT_DRIVERS": "null"},
                    "required_log": ["WINK_FIXTURE Iris active " + pack + " scene=" + scene, "WINK_FIXTURE minecraft complete"],
                    "asset_sha512": asset["hashes"]["sha512"],
                    "asset_filename": asset["filename"],
                    "fixture_sha256": hashlib.sha256((a.out / "wink-fixture.jar").read_bytes()).hexdigest() if (a.out / "wink-fixture.jar").is_file() else None,
                    "world_sha256": hashlib.sha256((a.out / "fixture-world/level.dat").read_bytes()).hexdigest() if (a.out / "fixture-world/level.dat").is_file() else None,
                    "settings_sha512": digest(settings) if preset == "stress" else None,
                    "resourcepack_sha512": digest(a.resourcepack) if scene == "labpbr" else None,
                    "resourcepack_filename": a.resourcepack.name if scene == "labpbr" else None}
                if scene == "labpbr":
                    recipes[name]["required_log"].append("WINK_FIXTURE LabPBR active file/" + a.resourcepack.name)
    (a.out / "recipes.json").write_text(json.dumps(recipes, indent=2) + "\n")
    (a.out / "missing-presets.json").write_text(json.dumps(missing, indent=2) + "\n")
    print(len(recipes), "recording recipes;", len(missing), "packs need curated stress settings")


if __name__ == "__main__":
    main()
