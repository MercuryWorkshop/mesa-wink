#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare rendered checkpoints without losing single-channel or thin errors."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter

DEFAULT_POLICY = {"tolerance": 8, "max_fraction": 0.005, "erode": 1,
                  "max_interior": 0, "alpha": False, "regions": []}


def compare_images(reference, candidate, policy=None, diff_path=None):
    policy = dict(DEFAULT_POLICY, **(policy or {}))
    if not 0 <= policy["tolerance"] <= 255 or not 0 <= policy["max_fraction"] <= 1:
        raise ValueError("invalid image tolerance")
    if policy["erode"] < 0 or policy["max_interior"] < 0:
        raise ValueError("negative erosion/interior allowance")
    mode = "RGBA" if policy["alpha"] else "RGB"
    with Image.open(reference) as a, Image.open(candidate) as b:
        a, b = a.convert(mode), b.convert(mode)
    if a.size != b.size:
        return {"pass": False, "reason": "size-mismatch", "reference": a.size, "candidate": b.size}
    worst = Image.new("L", a.size)
    for x, y in zip(a.split(), b.split()):
        worst = ImageChops.lighter(worst, ImageChops.difference(x, y))
    raw = worst.point(lambda v: 255 if v > policy["tolerance"] else 0)
    total = a.width * a.height
    differing = raw.histogram()[255]
    interior = raw.filter(ImageFilter.MinFilter(2 * policy["erode"] + 1)) if policy["erode"] else raw
    surviving = interior.histogram()[255]
    regions = []
    for region in policy["regions"]:
        x0, y0, x1, y1 = region["box"]
        if not (0 <= x0 < x1 <= a.width and 0 <= y0 < y1 <= a.height):
            raise ValueError("region outside image")
        count = raw.crop(region["box"]).histogram()[255]
        regions.append({"name": region["name"], "differing": count,
                        "pass": count <= region.get("max_pixels", 0)})
    passed = differing / total <= policy["max_fraction"] and surviving <= policy["max_interior"]
    passed = passed and all(r["pass"] for r in regions)
    if diff_path:
        diff_path = Path(diff_path)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        worst.save(diff_path)
        raw.save(diff_path.with_name(diff_path.stem + "-mask.png"))
    return {"pass": passed, "reason": "match" if passed else "render-mismatch",
            "size": a.size, "differing": differing, "fraction": differing / total,
            "interior": surviving, "max_delta": worst.getextrema()[1], "regions": regions,
            "policy": policy}


def compare_directories(reference, candidate, policy=None, out=None, expected=None):
    reference, candidate = Path(reference), Path(candidate)
    names = expected if expected is not None else sorted(p.name for p in reference.glob("*.png"))
    results = []
    if not names:
        return [{"pass": False, "reason": "missing-checkpoints"}]
    for name in names:
        a, b = reference / name, candidate / name
        if not a.is_file() or not b.is_file():
            results.append({"name": name, "pass": False, "reason": "missing-checkpoint",
                            "reference_exists": a.is_file(), "candidate_exists": b.is_file()})
        else:
            diff = Path(out) / ("diff-" + name) if out else None
            results.append(dict(name=name, **compare_images(a, b, policy, diff)))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    policy = json.loads(a.policy.read_text()) if a.policy else None
    results = compare_directories(a.reference, a.candidate, policy, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
