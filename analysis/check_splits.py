"""Integrity checks of the released patient-extended groups and partitions. Needs pandas only and runs in a few seconds.

    python analysis/check_splits.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "resources" / "splits_isic2019.csv")
off = pd.read_csv(ROOT / "resources" / "official_test.csv")
results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


check("25,331 unique training images", len(s) == 25331 and s.image.is_unique)
check("13,865 patient-extended groups", s.group_id.nunique() == 13865)
for col in ("p1_image_level", "p2_lesion_level"):
    sizes = s[col].value_counts().to_dict()
    check(f"{col}: 17,731 / 2533 / 5067 images", sizes == {"train": 17731, "val": 2533, "test": 5067})
diff = (pd.crosstab(s.label, s.p1_image_level) - pd.crosstab(s.label, s.p2_lesion_level)).abs().to_numpy().max()
check("class counts of the two splits differ by at most one image per class and partition", diff <= 1)
for key in ("lesion_id", "lesion_id_archive", "patient_id_archive", "md5_bytes"):
    spread = s[s[key].notna()].groupby(key).group_id.nunique()
    check(f"images sharing {key} belong to one group", (spread == 1).all())
check("no group spans two partitions of the lesion-level split",
      (s.groupby("group_id").p2_lesion_level.nunique() == 1).all())
train_groups = set(s.group_id[s.p1_image_level == "train"])
leaked = s[s.p1_image_level == "test"].group_id.isin(train_groups)
check("2926 of 5067 image-level test images (57.7%) share a group with training",
      int(leaked.sum()) == 2926 and round(100 * leaked.mean(), 1) == 57.7)
for scheme in ("lesion", "image"):
    times = sum((s[f"cv{k}_{scheme}"] == "test").astype(int) for k in range(5))
    check(f"{scheme} cross-validation: every image is a test image in exactly one fold", (times == 1).all())
for k in range(5):
    col = s[f"cv{k}_lesion"]
    tr = set(s.group_id[col == "train"])
    check(f"grouped fold {k}: no validation or test group occurs in training",
          not s.group_id[col != "train"].isin(tr).any())
check("grouped fold 0 equals the lesion-level split", (s.cv0_lesion == s.p2_lesion_level).all())
scored = off[off.scored & off.known_class]
check("5920 scored official test images of the eight classes", len(scored) == 5920)
check("no scored official test image shares an ISIC Archive lesion with the training set",
      not scored.lesion_in_train.any() and not scored.lesion_id_archive.isin(set(s.lesion_id_archive.dropna())).any())

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
