# -*- coding: utf-8 -*-
"""Split-independent near-duplicate screen over the whole training set.

For every training image, the most similar image belonging to a DIFFERENT lesion group is found (frozen ResNet-50
features, cosine). All cross-group pairs above a threshold are written for visual adjudication; adjudicated
same-lesion pairs become grouping edges in tools/isic_splits.py. Because the screen does not depend on any split,
regenerating the partitions cannot expose new candidates above the threshold.

    python neardup_global.py <project_dir> <train_zip> [threshold]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neardup_audit import contact_sheet     # noqa: E402


def main(argv):
    proj, train_zip = Path(argv[0]), Path(argv[1])
    thr = float(argv[2]) if len(argv) > 2 else 0.975
    s = pd.read_csv(proj / "resources" / "splits_isic2019.csv")
    e = np.load(proj / "_local_cache" / "probe_emb_resnet50.npy").astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    g = s["group_id"].values
    best_j = np.empty(len(s), np.int64)
    best_s = np.empty(len(s), np.float32)
    for a in range(0, len(s), 1024):
        S = e[a:a + 1024] @ e.T
        same = g[a:a + 1024][:, None] == g[None, :]
        S[same] = -1.0                                  # ignore images of the same group (incl. itself)
        best_j[a:a + 1024] = S.argmax(1)
        best_s[a:a + 1024] = S.max(1)
    rows = {}
    for i in np.where(best_s >= thr)[0]:
        j = int(best_j[i])
        key = tuple(sorted((s.image[i], s.image[j])))
        rows[key] = max(rows.get(key, 0), float(best_s[i]))
    lab = dict(zip(s.image, s.label))
    src = dict(zip(s.image, s.source))
    out = pd.DataFrame([{"query": a, "ref": b, "sim": v, "q_label": lab[a], "r_label": lab[b],
                         "q_source": src[a], "r_source": src[b]} for (a, b), v in rows.items()])
    out = out.sort_values("sim", ascending=False).reset_index(drop=True)
    out.to_csv(proj / "audit" / "neardup_global_candidates.csv", index=False)
    print(f"{len(out)} cross-group pairs with cosine >= {thr}")
    import zipfile
    z = zipfile.ZipFile(train_zip)
    for k in range(0, len(out), 40):
        contact_sheet(out.iloc[k:k + 40], z, z, proj / "audit" / f"contact_global_{k // 40 + 1:02d}.png",
                      f"Training set, cross-group pairs (cosine >= {thr}), part {k // 40 + 1}")
    print(out.head(60).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
