#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resolve public publisher assets once, then verify content against a lock."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from build_replay_support import HERE, MESA

USER_AGENT = "mesa-wink-correctness/1 (local rendering regression suite)"


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=90)


def get_json(url):
    with request(url) as response:
        return json.load(response)


def digest(path, algorithm="sha512"):
    h = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(asset, directory):
    name = asset["filename"]
    if Path(name).name != name:
        raise ValueError("unsafe asset filename")
    target = directory / name
    algorithm, expected = next(iter(asset["hashes"].items()))
    if target.is_file() and digest(target, algorithm) == expected:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    try:
        with request(asset["url"]) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if digest(temporary, algorithm) != expected:
            raise RuntimeError(f"asset checksum mismatch: {name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def resolve(project, minecraft=None, prefix=None):
    from urllib.parse import urlencode
    query = "?" + urlencode({"loaders": '["fabric"]', "game_versions": json.dumps([minecraft])}) if minecraft else ""
    versions = get_json(f"https://api.modrinth.com/v2/project/{project}/version{query}")
    if prefix:
        versions = [v for v in versions if v["version_number"].startswith(prefix)]
    releases = [v for v in versions if v["version_type"] == "release"]
    versions = releases or versions
    if not versions:
        raise RuntimeError(f"no compatible release for {project}")
    version = versions[0]
    file = next((f for f in version["files"] if f.get("primary")), version["files"][0])
    return {"project": project, "version": version["id"], "version_number": version["version_number"],
            "filename": file["filename"], "url": file["url"], "hashes": {"sha512": file["hashes"]["sha512"]},
            "dependencies": version["dependencies"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps/assets")
    p.add_argument("--refresh", action="store_true", help="explicitly resolve new versions")
    p.add_argument("--manual", action="append", default=[], metavar="ID=ZIP")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((HERE / "corpus.json").read_text())
    lock_path = a.out / "assets.lock.json"
    seed = lock_path if lock_path.exists() else HERE / "assets.lock.json"
    lock = json.loads(seed.read_text()) if seed.exists() and not a.refresh else {"schema": 1, "minecraft": corpus["minecraft"], "assets": {}}
    failures = []
    for project in corpus["mods"] + corpus["shaderpacks"]:
        try:
            if project not in lock["assets"]:
                lock["assets"][project] = resolve(project, corpus["minecraft"] if project in corpus["mods"] else None,
                                                  corpus.get("mod_version_prefixes", {}).get(project))
                lock_path.write_text(json.dumps(lock, indent=2) + "\n")
            directory = a.out / ("mods" if project in corpus["mods"] else "shaderpacks")
            path = download(lock["assets"][project], directory)
            if project in corpus["shaderpacks"]:
                with zipfile.ZipFile(path) as archive:
                    if not any(n.startswith("shaders/") for n in archive.namelist()):
                        raise RuntimeError("ZIP has no shaders directory")
            print(project, lock["assets"][project]["version_number"], flush=True)
        except Exception as e:
            failures.append({"id": project, "status": "missing-assets", "error": str(e)})
    for value in a.manual:
        name, path = value.split("=", 1)
        if name not in {p["id"] for p in corpus["manual_shaderpacks"]}:
            raise ValueError(f"unknown manual pack {name}")
        source = Path(path)
        asset = {"filename": source.name, "url": source.resolve().as_uri(), "hashes": {"sha512": digest(source)}, "version": "manual"}
        download(asset, a.out / "shaderpacks")
        lock["assets"][name] = asset
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    resourcepack = json.loads((HERE / "resourcepack.lock.json").read_text())
    download(resourcepack, a.out / "resourcepacks")
    (a.out / "missing-assets.json").write_text(json.dumps(failures, indent=2) + "\n")
    for f in failures:
        print(f["id"], f["error"])
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
