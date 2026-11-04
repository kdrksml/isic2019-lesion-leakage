# -*- coding: utf-8 -*-
"""Exact and near-exact duplicate audit for the ISIC 2019 training images.

Two hashes per image:
  * md5 of the raw file bytes          -> byte-identical files;
  * md5 of a 32x32 grayscale thumbnail -> the same picture re-encoded or resized
    (e.g. the 2,074 "_downsampled" MSK copies), quantised to 4 bits so that JPEG
    noise does not break the match.

A perceptual-hash nearest-neighbour screen is deliberately NOT used as a
blocking check: on dermoscopy (dark lesion centred on light skin) its clean
nearest-neighbour distances overlap the true-duplicate distances, so it cannot
separate the two populations.

Usage:
    python duplicate_audit.py <ISIC_2019_Training_Input.zip | image_dir> <out.csv>
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def thumb_hash(data: bytes) -> str:
    im = Image.open(io.BytesIO(data)).convert("L").resize((32, 32), Image.BILINEAR)
    q = (np.asarray(im, dtype=np.uint8) >> 4).tobytes()
    return hashlib.md5(q).hexdigest()


def iter_images(src: Path):
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.lower().endswith(".jpg"):
                    yield Path(n).stem, z.read(n)
    else:
        for p in sorted(src.glob("*.jpg")):
            yield p.stem, p.read_bytes()


def main(argv):
    src, out = Path(argv[0]), Path(argv[1])
    rows = []
    for k, (name, data) in enumerate(iter_images(src), 1):
        rows.append((name, hashlib.md5(data).hexdigest(), thumb_hash(data), len(data)))
        if k % 2000 == 0:
            print(k, flush=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("image,md5_bytes,md5_thumb32q4,n_bytes\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
