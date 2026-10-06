#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write a minimal vanilla 1.21.1 creative save; Minecraft generates its chunks."""
import argparse
import gzip
from pathlib import Path
import struct


def string(value):
    encoded = value.encode()
    return struct.pack(">H", len(encoded)) + encoded


def tag(name, value):
    if isinstance(value, dict):
        kind, data = 10, b"".join(tag(k, v) for k, v in value.items()) + b"\0"
    elif isinstance(value, str):
        kind, data = 8, string(value)
    else:
        kind, value = value
        data = struct.pack({1: ">b", 3: ">i", 4: ">q"}[kind], value)
    return bytes([kind]) + string(name) + data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a new output directory")
    dimensions = {}
    for name, settings, biome in (("overworld", "overworld", "multi_noise"), ("the_nether", "nether", "multi_noise"), ("the_end", "end", "the_end")):
        dimensions["minecraft:" + name] = {"type": "minecraft:" + name, "generator": {
            "type": "minecraft:noise", "settings": "minecraft:" + settings,
            "biome_source": {"type": "minecraft:" + biome, **({"preset": "minecraft:" + settings} if biome == "multi_noise" else {})}}}
    data = {"version": (3, 19133), "DataVersion": (3, 3955), "Version": {"Id": (3, 3955), "Name": "1.21.1", "Snapshot": (1, 0)},
            "LevelName": "Wink fixture", "GameType": (3, 1), "allowCommands": (1, 1), "Difficulty": (1, 0),
            "SpawnX": (3, 0), "SpawnY": (3, 80), "SpawnZ": (3, 11), "Time": (4, 6000), "DayTime": (4, 6000),
            "initialized": (1, 1), "DragonFight": {}, "WorldGenSettings": {"seed": (4, 8675309), "generate_features": (1, 1), "bonus_chest": (1, 0), "dimensions": dimensions}}
    a.out.mkdir(parents=True)
    with gzip.open(a.out / "level.dat", "wb") as out:
        out.write(tag("", {"Data": data}))


if __name__ == "__main__":
    main()
