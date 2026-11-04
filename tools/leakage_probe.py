# -*- coding: utf-8 -*-
"""Model-agnostic nearest-neighbour leakage probe.

A frozen ImageNet backbone embeds every training image once (no fine-tuning). For each protocol, every test image is
classified by its nearest training neighbour (cosine similarity). Two numbers per protocol:
  * 1-NN macro-F1 — a classifier that can only memorise; leakage inflates it the most;
  * share of test images whose nearest training neighbour belongs to the SAME lesion group.
Under a lesion-disjoint split the second number is 0 by construction; under the image-level split it measures how
often "the closest training example is a photograph of the very same lesion".

    python leakage_probe.py <project_dir> <images.zip | image_dir> [model]
In the notebook, `probe_from_embeddings(...)` is called with embeddings computed on the GPU.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def probe_from_embeddings(emb: np.ndarray, splits: pd.DataFrame, chunk: int = 2048) -> pd.DataFrame:
    """emb rows aligned with `splits` rows. Returns one row per protocol."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    rows = []
    for col, name in (("p1_image_level", "P1 image-level"), ("p2_lesion_level", "P2 lesion-level")):
        tr = np.where(splits[col].values == "train")[0]
        te = np.where(splits[col].values == "test")[0]
        nn_idx = np.empty(len(te), dtype=np.int64)
        for s in range(0, len(te), chunk):
            sims = e[te[s:s + chunk]] @ e[tr].T
            nn_idx[s:s + chunk] = tr[sims.argmax(1)]
        y, yhat = splits["target"].values[te], splits["target"].values[nn_idx]
        same = splits["group_id"].values[te] == splits["group_id"].values[nn_idx]
        rows.append({"protocol": name, "n_test": len(te),
                     "nn_macro_f1": f1_score(y, yhat, average="macro"),
                     "nn_bacc": balanced_accuracy_score(y, yhat),
                     "nn_same_lesion_pct": 100 * same.mean(),
                     "nn_acc_when_same_lesion": float((y[same] == yhat[same]).mean()) if same.any() else float("nan"),
                     "nn_acc_when_other_lesion": float((y[~same] == yhat[~same]).mean())})
    return pd.DataFrame(rows)


def _embed_local(project: Path, src: Path, model_name: str) -> np.ndarray:
    import torch
    import timm
    from PIL import Image
    splits = pd.read_csv(project / "resources" / "splits_isic2019.csv")
    m = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    z = zipfile.ZipFile(src) if src.suffix == ".zip" else None
    names = {Path(n).stem: n for n in z.namelist()} if z else None

    def load(img):
        data = z.read(names[img]) if z else (src / f"{img}.jpg").read_bytes()
        im = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    out = []
    batch = []
    with torch.no_grad():
        for k, img in enumerate(splits["image"]):
            batch.append(load(img))
            if len(batch) == 64 or k == len(splits) - 1:
                x = (torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2) - mean) / std
                out.append(m(x).numpy().astype(np.float32))
                batch = []
            if k % 2000 == 0:
                print(k, flush=True)
    return np.concatenate(out)


def main(argv):
    project, src = Path(argv[0]), Path(argv[1])
    model = argv[2] if len(argv) > 2 else "resnet50.a1_in1k"
    cache_dir = project / "_local_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"probe_emb_{model.split('.')[0]}.npy"
    emb = np.load(cache) if cache.exists() else _embed_local(project, src, model)
    np.save(cache, emb.astype(np.float16))
    splits = pd.read_csv(project / "resources" / "splits_isic2019.csv")
    res = probe_from_embeddings(emb.astype(np.float32), splits)
    res.insert(0, "backbone", model)
    res.to_csv(project / "resources" / "leakage_probe.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
