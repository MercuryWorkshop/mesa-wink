#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record local CPU references and replay identical API traces through Wink."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from compare import compare_directories, compare_images
from build_replay_support import HERE, MESA, copy_artifact

ERRORS = re.compile(r"error:|(?:\[|/)ERROR\]|\[gfxrecon\] ERROR|\bFATAL\b|GL_INVALID_|WebGPU error|validation error|device lost|unsupported:|unsupported \w+ call|failed to (?:compile|link|get snapshot)|FAIL\b", re.I)
BENIGN_LOG_LINES = {"Errors from xkbcomp are not fatal to the X server"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(command, log, timeout, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        try:
            p = subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            return {"pass": False, "status": "timeout", "log": str(log)}
    errors = [line for line in log.read_text(errors="replace").splitlines()
              if line not in BENIGN_LOG_LINES and ERRORS.search(line)]
    return {"pass": rc == 0 and not errors, "status": "pass" if rc == 0 and not errors else "replay-error",
            "rc": rc, "errors": errors, "log": str(log)}


def cases(a):
    corpus = json.loads((HERE / "corpus.json").read_text())
    result = [dict(id="probe-" + api, api=api, tier="fast", probe=True, min_frames=2) for api in ("gl", "gles")]
    result.append(dict(id="probe-vulkan", api="vulkan", tier="fast", probe=True, min_frames=4, checkpoints="1,4,8,16"))
    result += corpus["applications"]
    for case in result:
        case.setdefault("min_frames", 4)
    assets = corpus["shaderpacks"] + [p["id"] for p in corpus["manual_shaderpacks"]]
    for pack in assets:
        for scene in corpus["scenes"]:
            for preset in corpus["presets"]:
                result.append(dict(id=f"iris-{pack}-{scene}-{preset}", api="gl", fixture="minecraft",
                                   pack=pack, scene=scene, preset=preset, min_frames=4,
                                   tier="fast" if pack in ("complementary-reimagined", "bsl-shaders") and scene == "day" and preset == "default" else "full"))
    overrides = json.loads(a.recipes.read_text()) if a.recipes else {}
    for case in result:
        case.update(overrides.get(case["id"], {}))
    return [c for c in result if (a.tier == "full" or c["tier"] == "fast") and (not a.case or c["id"] in a.case)]


def container_command(a, arguments, name=None):
    command = ["container", "run", "--rm", "--disable-progress-updates", "--cpus", "4", "--memory", "8g",
               "--volume", f"{a.out}:/work", "--volume", f"{HERE / 'fixtures'}:/fixtures:ro",
               "--volume", f"{HERE}:/suite:ro", "--entrypoint", "python3"]
    if name:
        command += ["--name", name]
    if a.tools:
        command += ["--volume", f"{a.tools.resolve()}:/tools:ro"]
    return [*command, a.image, "/suite/reference/reference.py", *arguments]


def case_hash(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def validate_reference(case, root):
    lock_path = root / "capture.lock.json"
    if not lock_path.exists():
        return {"status": "missing-reference", "pass": False}
    lock = json.loads(lock_path.read_text())
    trace = root / ("capture.gfxr" if case["api"] == "vulkan" else "capture.trace")
    if lock["case_sha256"] != case_hash(case) or lock["trace_sha256"] != sha256(trace):
        return {"status": "reference-incompatible", "pass": False, "error": "Case or trace changed; record a fresh reference."}
    if len(lock["checkpoints"]) < case.get("min_frames", 1) or any(
        sha256(root / "reference-0" / name) != value for name, value in lock["checkpoints"].items()
    ):
        return {"status": "reference-incompatible", "pass": False, "error": "Reference checkpoint checksum or coverage changed"}
    return {"status": "pass", "pass": True, "lock": lock, "trace": trace}


def record(a, case, root):
    if not (a.out / "reference.lock.json").is_file():
        return {"status": "missing-tools", "pass": False, "error": "Build the pinned reference image first"}
    if shutil.disk_usage(a.out).free < 4 * 1024**3:
        return {"status": "resource-limit", "pass": False, "error": "Recording requires at least 4 GiB free disk space"}
    # A refresh gets a new directory. Partial and stale captures cannot pass.
    if root.exists():
        if (root / "capture.lock.json").exists():
            if a.reuse_references:
                outcome = validate_reference(case, root)
                outcome.pop("lock", None)
                outcome.pop("trace", None)
                if outcome["pass"]:
                    outcome["status"] = "reference-reused"
                return outcome
            return {"status": "reference-exists", "pass": False, "error": "Use a fresh --out directory for an explicit reference refresh."}
        root.rename(root.with_name(root.name + ".incomplete-" + str(time.time_ns())))
    root.mkdir(parents=True)
    if not case.get("probe") and not case.get("command"):
        return {"status": "missing-assets", "pass": False, "error": "No pinned application recording command in --recipes."}
    from assets import digest
    inputs = [(case.get("asset_sha512"), a.out / "assets/shaderpacks" / case.get("asset_filename", "missing"), "sha512"),
              (case.get("resourcepack_sha512"), a.out / (case.get("resourcepack_filename") or "missing"), "sha512"),
              (case.get("fixture_sha256"), a.out / "wink-fixture.jar", "sha256"),
              (case.get("world_sha256"), a.out / "fixture-world/level.dat", "sha256")]
    for expected, path, algorithm in inputs:
        if expected and (not path.is_file() or digest(path, algorithm) != expected):
            return {"status": "missing-assets", "pass": False, "error": "Recording input missing or checksum changed: " + str(path)}
    if case.get("settings_sha512"):
        settings = a.out / "presets" / (case["pack"] + ".txt")
        if not settings.is_file() or digest(settings) != case["settings_sha512"]:
            return {"status": "missing-assets", "pass": False, "error": "Stress preset missing or checksum changed"}
    arguments = ["probe" if case.get("probe") else "record", "--api", case["api"], "--out", "/work/cases/" + case["id"]]
    if case.get("checkpoints"):
        arguments += ["--checkpoints", case["checkpoints"]]
    arguments += ["--tail-frames", str(case.get("tail_frames", 96 if case.get("fixture") == "minecraft" else 64))]
    for key, value in case.get("env", {}).items():
        arguments += ["--env", key + "=" + str(value)]
    if case.get("command"):
        arguments += ["--", *case["command"]]
    name = "wink-reference-" + str(time.time_ns())
    outcome = execute(container_command(a, arguments, name), root / "record.log", a.timeout)
    if outcome["status"] == "timeout":
        with (root / "cleanup.log").open("w") as log:
            subprocess.run(["container", "stop", name], stdout=log, stderr=subprocess.STDOUT, timeout=30)
    if not outcome["pass"]:
        return dict(outcome, status="reference-error")
    text = (root / "record.log").read_text(errors="replace")
    for marker in case.get("required_log", []):
        if marker not in text:
            return {"status": "reference-incompatible", "pass": False, "error": "Missing required application marker: " + marker}
    refs = sorted((root / "reference-0").glob("*.png"),
                  key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])
    if len(refs) < case.get("min_frames", 1):
        return {"status": "reference-incompatible", "pass": False, "error": "Too few recorded checkpoints"}
    names = [p.name for p in refs]
    if case["api"] == "vulkan" and case.get("checkpoints"):
        expected = {f"screenshot_frame_{frame}.png" for frame in case["checkpoints"].split(",")}
        if set(names) != expected:
            return {"status": "reference-incompatible", "pass": False, "error": "Missing or unexpected Vulkan checkpoints"}
    checks = []
    for repeat in (1, 2):
        checks += compare_directories(root / "reference-0", root / f"reference-{repeat}",
                                      {"tolerance": 0, "max_fraction": 0, "erode": 0}, expected=names)
    if not all(c["pass"] for c in checks):
        return {"status": "unstable-reference", "pass": False, "comparison": checks}
    # Verify the replay oracle against live application readback for the proof.
    if case.get("probe"):
        for index, image in zip(([1,4,8,16] if case["api"] == "vulkan" else [1,2]), refs):
            checks.append(compare_images(root / f"live-{index}.ppm", image,
                                         {"tolerance": 0, "max_fraction": 0, "erode": 0}))
        if not all(c["pass"] for c in checks):
            return {"status": "reference-incompatible", "pass": False, "comparison": checks}
    trace = root / ("capture.gfxr" if case["api"] == "vulkan" else "capture.trace")
    lock = {"schema": 1, "case": case, "case_sha256": case_hash(case), "trace_sha256": sha256(trace),
            "runner_sha256": sha256(HERE / "reference/reference.py"),
            "checkpoints": {p.name: sha256(p) for p in refs}, "repeatability": checks,
            "reference": json.loads((a.out / "reference.lock.json").read_text())}
    identity = json.loads(subprocess.check_output(["container", "image", "inspect", a.image], text=True))
    lock["reference"]["image"] = a.image
    lock["reference"]["image_identity"] = identity
    checkpoints = root / "checkpoints.json"
    if checkpoints.exists():
        lock["replay_checkpoints"] = ",".join(map(str, json.loads(checkpoints.read_text())["calls"]))
    (root / "capture.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return {"status": "pass", "pass": True, "checkpoints": names}


def replay(a, case, root, target):
    validated = validate_reference(case, root)
    if not validated["pass"]:
        return validated
    lock, trace = validated["lock"], validated["trace"]
    output = root / f"candidate-{target}-{time.time_ns()}"
    output.mkdir()
    api = "vulkan" if case["api"] == "vulkan" else "gl"
    executable = "gfxrecon-replay" if api == "vulkan" else "wink-glretrace"
    checkpoints = lock.get("replay_checkpoints", case.get("checkpoints", "1,8,32,64" if api == "vulkan" else "*/frame"))
    build = (a.replayers or a.out) / (api + "-" + target)
    if target == "native":
        binary = next((p for p in (build / executable, build / "bin" / executable, build / "tools/replay" / executable) if p.is_file()), None)
        if not binary:
            return {"status": "missing-tools", "pass": False, "error": "Build native " + api + " replay first"}
        if api == "vulkan":
            args = ["--wsi", "headless", "-m", "rebind", "--screenshots", checkpoints,
                    "--screenshot-format", "png", "--screenshot-dir", output]
        else:
            args = ["--singlethread", "--headless", "--debug", "-s", output / "frame-", "-S", checkpoints]
        env = dict(os.environ, MESA_DEBUG="context", WINK_VK_ABORT_ON_ERROR="1", VK_DRIVER_FILES=str(a.icd.resolve()),
                   DYLD_LIBRARY_PATH="/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", ""))
        outcome = execute([binary, *args, trace], output / "replay.log", a.timeout, env)
    else:
        from urllib.parse import urlencode
        for suffix in ("js", "wasm", "worker.js", "worker.mjs"):
            source = next((p for p in (build / (executable + "." + suffix), build / "bin" / (executable + "." + suffix), build / "tools/replay" / (executable + "." + suffix)) if p.is_file()), None)
            if source:
                copy_artifact(source, output / source.name)
            elif suffix in ("js", "wasm"):
                return {"status": "missing-tools", "pass": False, "error": "Build browser " + api + " replay first"}
        shutil.copy2(HERE / "replay/replay.html", output / "index.html")
        copy_artifact(trace, output / trace.name)
        query = urlencode({"api": case["api"], "trace": trace.name, "checkpoints": checkpoints})
        outcome = execute(["node", MESA / "wink/test/run-browser.mjs", output,
                           "--page=index.html?" + query, f"--timeout={a.timeout}", "--result=" + str(output / "browser.json")],
                          output / "replay.log", a.timeout + 30)
        result = output / "browser.json"
        if result.exists() and not json.loads(result.read_text()).get("pass"):
            outcome["pass"] = False
    if not outcome["pass"]:
        return dict(outcome, status="unsupported" if any("unsupported" in e.lower() or "VK_ERROR_EXTENSION_NOT_PRESENT" in e for e in outcome.get("errors", [])) else outcome["status"], artifacts=str(output))
    checks = compare_directories(root / "reference-0", output, case.get("policy"), output, list(lock["checkpoints"]))
    result = {"status": "pass" if all(c["pass"] for c in checks) else "render-mismatch",
              "pass": all(c["pass"] for c in checks), "comparison": checks, "artifacts": str(output)}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(a, results):
    result = {"schema": 1, "action": a.action, "tier": a.tier, "results": results,
              "pass": bool(results) and all(r["pass"] for r in results)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    junit = ET.Element("testsuite", name="wink-apps", tests=str(len(results)),
                       failures=str(sum(not r["pass"] for r in results)))
    rows = []
    for r in results:
        test = ET.SubElement(junit, "testcase", name=r["id"], classname=r.get("target") or "reference")
        if not r["pass"]:
            ET.SubElement(test, "failure", type=r["status"]).text = json.dumps(r, indent=2)
        artifact = Path(r.get("artifacts") or a.out)
        relative = os.path.relpath(artifact, a.out)
        rows.append(f'<tr><td>{html.escape(r["id"])}</td><td>{html.escape(r.get("target") or "reference")}</td><td>{html.escape(r["status"])}</td><td><a href="{html.escape(relative, quote=True)}/">artifacts</a></td></tr>')
    ET.ElementTree(junit).write(a.out / "junit.xml", encoding="unicode", xml_declaration=True)
    (a.out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Wink application correctness</title><style>body{font:16px system-ui}td,th{padding:8px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style><h1>Wink application correctness</h1><p>All missing, unsupported and unstable cases fail the coverage gate.</p><table><tr><th>Case</th><th>Target</th><th>Result</th><th>Evidence</th></tr>' + "".join(rows) + '</table><p><a href="results.json">JSON details</a></p>')
    return 0 if result["pass"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["list", "record", "run"])
    p.add_argument("--tier", choices=["fast", "full"], default="fast")
    p.add_argument("--case", action="append")
    p.add_argument("--target", action="append", choices=["native", "st", "mt"])
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--image", default="wink-reference")
    p.add_argument("--recipes", type=Path)
    p.add_argument("--tools", type=Path, help="pinned Linux ARM64 application tree mounted at /tools")
    p.add_argument("--replayers", type=Path, help="reuse retracer builds from another output directory")
    p.add_argument("--reuse-references", action="store_true", help="validate and reuse locked baselines while recording missing cases")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--icd", type=Path, default=MESA / "build/wvk-native/src/wink/vulkan/wink_devenv_icd.aarch64.json")
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.replayers:
        a.replayers = a.replayers.resolve()
    selected = cases(a)
    if a.case and set(a.case) != {c["id"] for c in selected}:
        p.error("Unknown case or case excluded by --tier; use --tier full to select full cases")
    if a.action == "list":
        print(json.dumps(selected, indent=2))
        return 0
    results = []
    for case in selected:
        root = a.out / "cases" / case["id"]
        for target in ([None] if a.action == "record" else a.target or ["native", "st"]):
            try:
                result = record(a, case, root) if a.action == "record" else replay(a, case, root, target)
            except Exception as e:
                result = {"pass": False, "status": "harness-error", "error": str(e)}
            result.update(id=case["id"], target=target)
            if root.is_dir():
                (root / (("record" if target is None else "run-" + target) + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(case["id"], target or "reference", result["status"], flush=True)
            report(a, results)
    return report(a, results)


if __name__ == "__main__":
    raise SystemExit(main())
