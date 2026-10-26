#!/usr/bin/env python3
# Copyright 2026, Iacopo Catalano.
# All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Generate per-pixel semantic label PNGs for a FileDataLoader scene.

Reads `rgb_NNNNNNN.png` files from a scene directory, runs a HuggingFace
segmentation model, and writes `labels_NNNNNNN.png` (uint16) in the
`ade20k_outdoor` labelspace defined at:
    config/label_spaces/ade20k_outdoor_label_space.yaml

`ade20k_outdoor` despite its name has BOTH outdoor (Sky, Tree, Road,
Sidewalk, Vehicle, Building, Sign...) and indoor (Wall, Floor, Ceiling,
Door, Window, Table, Chair, Bed...) classes — making it suitable for mixed
scenes where the robot moves between outdoor and indoor areas.

Two source models are supported (auto-detected from the model name):

  - ADE20K  (recommended for mixed indoor+outdoor):
      `nvidia/segformer-b5-finetuned-ade-640-640`            (fast, default)
      `facebook/mask2former-swin-large-ade-semantic`         (slightly better)
  - Cityscapes (only good outdoors, fails indoors):
      `facebook/mask2former-swin-large-cityscapes-semantic`

Usage:
    python scripts/generate_labels.py /path/to/scene_0001
    python scripts/generate_labels.py /path/to/scene_0001 --overwrite
    python scripts/generate_labels.py /path/to/scene_0001 \\
        --model facebook/mask2former-swin-large-cityscapes-semantic

Requires: torch, transformers, pillow, numpy, imageio.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image


def _is_blank(label_path: pathlib.Path) -> bool:
    """True for the all-zero placeholder the scene converters write."""
    return not np.asarray(Image.open(label_path)).any()

# ---------------------------------------------------------------------------
# Cityscapes train-id (0..18, 255=ignore) -> ade20k_outdoor label id (0..36).
# Source: HF `facebook/mask2former-*-cityscapes-semantic` id2label (standard
# 19-class evaluation order).
# ---------------------------------------------------------------------------
CITYSCAPES_TO_OUTDOOR = {
    0:  6,   # road            -> Road
    1:  5,   # sidewalk        -> Sidewalk
    2:  8,   # building        -> Building
    3:  16,  # wall            -> Wall
    4:  11,  # fence           -> Fence
    5:  33,  # pole            -> Thing
    6:  9,   # traffic light   -> Signal
    7:  13,  # traffic sign    -> Sign
    8:  2,   # vegetation      -> Tree
    9:  4,   # terrain         -> Ground
    10: 1,   # sky             -> Sky
    11: 36,  # person          -> Human
    12: 36,  # rider           -> Human
    13: 7,   # car             -> Vehicle
    14: 7,   # truck           -> Vehicle
    15: 7,   # bus             -> Vehicle
    16: 7,   # train           -> Vehicle
    17: 7,   # motorcycle      -> Vehicle
    18: 7,   # bicycle         -> Vehicle
    255: 0,  # ignore          -> Unknown
}

# ---------------------------------------------------------------------------
# ADE20K class id (0..149) -> ade20k_outdoor label id (0..36).
# ADE20K class order from MMSegmentation / HuggingFace `id2label` (0=wall,
# 1=building, 2=sky, 3=floor, ...). Anything not listed -> 33 (Thing) or 0
# (Unknown) is intentional: this is a destructive remap from 150 -> 37 classes.
# ---------------------------------------------------------------------------
ADE20K_TO_OUTDOOR = [
    16,  # 0   wall                  -> Wall
    8,   # 1   building               -> Building
    1,   # 2   sky                    -> Sky
    17,  # 3   floor                  -> Floor
    2,   # 4   tree                   -> Tree
    18,  # 5   ceiling                -> Ceiling
    6,   # 6   road                   -> Road
    25,  # 7   bed                    -> Bed
    22,  # 8   windowpane             -> Window
    4,   # 9   grass                  -> Ground
    26,  # 10  cabinet                -> Storage
    5,   # 11  sidewalk               -> Sidewalk
    36,  # 12  person                 -> Human
    4,   # 13  earth                  -> Ground
    19,  # 14  door                   -> Door
    27,  # 15  table                  -> Table
    14,  # 16  mountain               -> Slope
    24,  # 17  plant                  -> Plant
    34,  # 18  curtain                -> Deformable
    28,  # 19  chair                  -> Chair
    7,   # 20  car                    -> Vehicle
    3,   # 21  water                  -> Water
    29,  # 22  painting               -> Wall_Decoration
    30,  # 23  sofa                   -> Couch
    23,  # 24  shelf                  -> Shelf
    8,   # 25  house                  -> Building
    3,   # 26  sea                    -> Water
    29,  # 27  mirror                 -> Wall_Decoration
    17,  # 28  rug                    -> Floor
    4,   # 29  field                  -> Ground
    28,  # 30  armchair               -> Chair
    28,  # 31  seat                   -> Chair
    11,  # 32  fence                  -> Fence
    27,  # 33  desk                   -> Table
    10,  # 34  rock                   -> Rock
    26,  # 35  wardrobe               -> Storage
    31,  # 36  lamp                   -> Light
    32,  # 37  bathtub                -> Appliance
    21,  # 38  railing                -> Structure
    34,  # 39  cushion                -> Deformable
    21,  # 40  base                   -> Structure
    26,  # 41  box                    -> Storage
    21,  # 42  column                 -> Structure
    13,  # 43  signboard              -> Sign
    26,  # 44  chest_of_drawers       -> Storage
    27,  # 45  counter                -> Table
    4,   # 46  sand                   -> Ground
    32,  # 47  sink                   -> Appliance
    8,   # 48  skyscraper             -> Building
    32,  # 49  fireplace              -> Appliance
    32,  # 50  refrigerator           -> Appliance
    8,   # 51  grandstand             -> Building
    5,   # 52  path                   -> Sidewalk
    20,  # 53  stairs                 -> Stairs
    6,   # 54  runway                 -> Road
    26,  # 55  case                   -> Storage
    27,  # 56  pool_table             -> Table
    34,  # 57  pillow                 -> Deformable
    19,  # 58  screen_door            -> Door
    20,  # 59  stairway               -> Stairs
    3,   # 60  river                  -> Water
    15,  # 61  bridge                 -> Bridge
    23,  # 62  bookcase               -> Shelf
    22,  # 63  blind                  -> Window
    27,  # 64  coffee_table           -> Table
    32,  # 65  toilet                 -> Appliance
    24,  # 66  flower                 -> Plant
    33,  # 67  book                   -> Thing
    14,  # 68  hill                   -> Slope
    28,  # 69  bench                  -> Chair
    27,  # 70  countertop             -> Table
    32,  # 71  stove                  -> Appliance
    2,   # 72  palm                   -> Tree
    27,  # 73  kitchen_island         -> Table
    32,  # 74  computer               -> Appliance
    28,  # 75  swivel_chair           -> Chair
    12,  # 76  boat                   -> Boat
    27,  # 77  bar                    -> Table
    32,  # 78  arcade_machine         -> Appliance
    8,   # 79  hovel                  -> Building
    7,   # 80  bus                    -> Vehicle
    34,  # 81  towel                  -> Deformable
    31,  # 82  light                  -> Light
    7,   # 83  truck                  -> Vehicle
    8,   # 84  tower                  -> Building
    31,  # 85  chandelier             -> Light
    21,  # 86  awning                 -> Structure
    31,  # 87  streetlight            -> Light
    8,   # 88  booth                  -> Building
    32,  # 89  television_receiver    -> Appliance
    7,   # 90  airplane               -> Vehicle
    4,   # 91  dirt_track             -> Ground
    34,  # 92  apparel                -> Deformable
    33,  # 93  pole                   -> Thing
    4,   # 94  land                   -> Ground
    21,  # 95  bannister              -> Structure
    20,  # 96  escalator              -> Stairs
    28,  # 97  ottoman                -> Chair
    33,  # 98  bottle                 -> Thing
    26,  # 99  buffet                 -> Storage
    29,  # 100 poster                 -> Wall_Decoration
    21,  # 101 stage                  -> Structure
    7,   # 102 van                    -> Vehicle
    12,  # 103 ship                   -> Boat
    32,  # 104 fountain               -> Appliance
    32,  # 105 conveyer_belt          -> Appliance
    21,  # 106 canopy                 -> Structure
    32,  # 107 washer                 -> Appliance
    33,  # 108 plaything              -> Thing
    3,   # 109 swimming_pool          -> Water
    28,  # 110 stool                  -> Chair
    33,  # 111 barrel                 -> Thing
    33,  # 112 basket                 -> Thing
    3,   # 113 waterfall              -> Water
    21,  # 114 tent                   -> Structure
    33,  # 115 bag                    -> Thing
    7,   # 116 minibike               -> Vehicle
    25,  # 117 cradle                 -> Bed
    32,  # 118 oven                   -> Appliance
    33,  # 119 ball                   -> Thing
    33,  # 120 food                   -> Thing
    20,  # 121 step                   -> Stairs
    33,  # 122 tank                   -> Thing
    13,  # 123 trade_name             -> Sign
    32,  # 124 microwave              -> Appliance
    24,  # 125 pot                    -> Plant
    35,  # 126 animal                 -> Dynamic_NonHuman
    7,   # 127 bicycle                -> Vehicle
    3,   # 128 lake                   -> Water
    32,  # 129 dishwasher             -> Appliance
    32,  # 130 screen                 -> Appliance
    34,  # 131 blanket                -> Deformable
    29,  # 132 sculpture              -> Wall_Decoration
    32,  # 133 hood                   -> Appliance
    31,  # 134 sconce                 -> Light
    24,  # 135 vase                   -> Plant
    9,   # 136 traffic_light          -> Signal
    33,  # 137 tray                   -> Thing
    33,  # 138 ashcan                 -> Thing
    32,  # 139 fan                    -> Appliance
    21,  # 140 pier                   -> Structure
    32,  # 141 crt_screen             -> Appliance
    33,  # 142 plate                  -> Thing
    32,  # 143 monitor                -> Appliance
    29,  # 144 bulletin_board         -> Wall_Decoration
    32,  # 145 shower                 -> Appliance
    32,  # 146 radiator               -> Appliance
    33,  # 147 glass                  -> Thing
    33,  # 148 clock                  -> Thing
    13,  # 149 flag                   -> Sign
]


def detect_mode(model_name: str) -> str:
    """Return 'cityscapes' or 'ade20k' based on model name."""
    name = model_name.lower()
    if "cityscapes" in name:
        return "cityscapes"
    if "ade" in name:
        return "ade20k"
    raise ValueError(
        f"cannot infer label space from model name '{model_name}'. "
        "Add 'ade' or 'cityscapes' to the name, or extend detect_mode()."
    )


def build_remap_lut(mode: str) -> np.ndarray:
    """Build a 256-entry LUT mapping source ids to ade20k_outdoor ids."""
    lut = np.zeros(256, dtype=np.uint16)
    if mode == "cityscapes":
        for src, dst in CITYSCAPES_TO_OUTDOOR.items():
            lut[src] = dst
    elif mode == "ade20k":
        for src, dst in enumerate(ADE20K_TO_OUTDOOR):
            lut[src] = dst
    else:
        raise ValueError(f"unknown mode: {mode}")
    return lut


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=pathlib.Path,
                    help="FileDataLoader scene dir with rgb_*.png + camera_info.yaml")
    ap.add_argument("--model",
                    default="nvidia/segformer-b5-finetuned-ade-640-640",
                    help="HF segmentation model. Must contain 'ade' or 'cityscapes'.")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (auto if omitted)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate every labels_*.png; by default existing "
                         "labels are kept and only missing or blank (all-zero "
                         "placeholder) ones are generated")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N frames (for smoke testing)")
    args = ap.parse_args()

    scene: pathlib.Path = args.scene.expanduser().resolve()
    if not scene.is_dir():
        print(f"scene not found: {scene}", file=sys.stderr)
        return 2

    rgb_files = sorted(scene.glob("rgb_*.png"))
    if not rgb_files:
        print(f"no rgb_*.png in {scene}", file=sys.stderr)
        return 2
    if args.limit:
        rgb_files = rgb_files[:args.limit]

    mode = detect_mode(args.model)
    print(f"label-space source: {mode} ({args.model})", flush=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    if mode == "cityscapes" or "mask2former" in args.model.lower():
        from transformers import (
            Mask2FormerForUniversalSegmentation as ModelCls,
            Mask2FormerImageProcessor as ProcCls,
        )
        is_mask2former = True
    else:
        from transformers import (
            SegformerForSemanticSegmentation as ModelCls,
            SegformerImageProcessor as ProcCls,
        )
        is_mask2former = False

    proc = ProcCls.from_pretrained(args.model)
    model = ModelCls.from_pretrained(args.model).to(device).eval()

    lut = build_remap_lut(mode)

    n_skip = 0
    n_done = 0
    for rgb_path in rgb_files:
        idx = rgb_path.stem.split("_")[-1]
        out = scene / f"labels_{idx}.png"
        if out.exists() and not args.overwrite and not _is_blank(out):
            n_skip += 1
            continue

        img = Image.open(rgb_path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)

        if is_mask2former:
            seg = proc.post_process_semantic_segmentation(
                outputs, target_sizes=[img.size[::-1]]  # (H, W)
            )[0].cpu().numpy().astype(np.int64)
        else:
            seg = proc.post_process_semantic_segmentation(
                outputs, target_sizes=[img.size[::-1]]
            )[0].cpu().numpy().astype(np.int64)

        seg_clipped = np.clip(seg, 0, 255).astype(np.uint8)
        labels = lut[seg_clipped]
        iio.imwrite(out, labels)

        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(rgb_files)}  last={out.name} "
                  f"unique={np.unique(labels)[:8]}", flush=True)

    print(f"done. wrote={n_done}, skipped={n_skip}, total={len(rgb_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
