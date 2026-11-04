# -*- coding: utf-8 -*-
"""Resolve ISIC Archive lesion/patient identifiers for every ISIC 2019 image.

Why: the ISIC 2019 training metadata carries a component-specific `lesion_id`
(HAM_/BCN_/MSK4_ namespaces) that cannot be joined to the official test set,
whose metadata has no lesion column at all. The ISIC Archive assigns one
archive-wide lesion identifier (IL_xxxxxxx) to every image it holds, exposed
only by the per-image detail endpoint (the list endpoint omits it). Resolving
it for train AND test images lets us measure lesion overlap between the
ISIC 2019 training set and the official test set.

Usage:
    python resolve_archive_lesion_ids.py <train_gt.csv> <test_gt.csv> <out.csv> [workers]

The output is resumable: rows already present in <out.csv> are skipped.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.isic-archive.com/api/v2/images/{}/"
FIELDS = ["image", "split", "archive_id", "lesion_id_archive", "patient_id_archive",
          "attribution", "pixels_x", "pixels_y", "diagnosis_3", "status"]


def archive_id(image: str) -> str:
    # 2,074 MSK file names carry a "_downsampled" suffix the archive does not use.
    return image.replace("_downsampled", "")


def fetch(image: str, split: str, retries: int = 4) -> dict:
    aid = archive_id(image)
    row = {"image": image, "split": split, "archive_id": aid}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(API.format(aid), timeout=30) as r:
                d = json.load(r)
            m = d.get("metadata", {})
            clin = m.get("clinical", {}) or {}
            acq = m.get("acquisition", {}) or {}
            row.update({
                "lesion_id_archive": clin.get("lesion_id", ""),
                "patient_id_archive": clin.get("patient_id", ""),
                "attribution": (d.get("attribution") or "").replace("\n", " ")[:80],
                "pixels_x": acq.get("pixels_x", ""),
                "pixels_y": acq.get("pixels_y", ""),
                "diagnosis_3": clin.get("diagnosis_3", ""),
                "status": "ok",
            })
            return row
        except urllib.error.HTTPError as e:
            if e.code == 404:
                row["status"] = "404"
                return row
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    row["status"] = "error"
    return row


def read_images(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["image"] for r in csv.DictReader(f)]


def main(argv):
    train_csv, test_csv, out_csv = argv[0], argv[1], Path(argv[2])
    workers = int(argv[3]) if len(argv) > 3 else 8
    todo = [(i, "train") for i in read_images(train_csv)] + [(i, "test") for i in read_images(test_csv)]
    done = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f) if r.get("status") in ("ok", "404")}
    todo = [t for t in todo if t[0] not in done]
    print(f"to resolve: {len(todo)} (already done: {len(done)})", flush=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futs = [ex.submit(fetch, img, sp) for img, sp in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            w.writerow(fu.result())
            if k % 500 == 0:
                f.flush()
                print(f"{k}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
