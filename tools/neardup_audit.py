# -*- coding: utf-8 -*-
"""Content-based residual-leakage screen (complements the identifier-based grouping).

1. Exact checks: md5 of bytes and of a 4-bit 32x32 thumbnail for every official-test image, matched against training.
2. Embedding screen: frozen ImageNet ResNet-50 (same features as leakage_probe.py), cosine similarity of every
   evaluation image to its nearest training image, for (a) P2 train vs P2 val+test and (b) all training vs official
   test. Positive control: the same statistic for P1 test images whose nearest neighbour is a known sibling.
3. Contact sheets of the highest-similarity pairs for visual adjudication (a screen with human review does not need
   to separate the two populations perfectly; it ranks candidates).

    python neardup_audit.py <project_dir> <train_zip> <test_zip>
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duplicate_audit import thumb_hash     # noqa: E402


def embed_zip(zpath: Path, images, model_name="resnet50.a1_in1k", batch=64):
    import torch
    import timm
    m = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    z = zipfile.ZipFile(zpath)
    names = {Path(n).stem: n for n in z.namelist()}
    out, buf = [], []
    with torch.no_grad():
        for k, img in enumerate(images):
            im = Image.open(io.BytesIO(z.read(names[img]))).convert("RGB").resize((224, 224), Image.BILINEAR)
            buf.append(np.asarray(im, dtype=np.float32) / 255.0)
            if len(buf) == batch or k == len(images) - 1:
                x = (torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2) - mean) / std
                out.append(m(x).numpy())
                buf = []
            if k % 2000 == 0:
                print(k, flush=True)
    return np.concatenate(out).astype(np.float32)


def nn_search(q, ref, chunk=2048):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    idx, sim = np.empty(len(q), np.int64), np.empty(len(q), np.float32)
    for s in range(0, len(q), chunk):
        S = q[s:s + chunk] @ ref.T
        idx[s:s + chunk] = S.argmax(1)
        sim[s:s + chunk] = S.max(1)
    return idx, sim


def contact_sheet(pairs, zq, zr, path, title, n=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = pairs.head(n).reset_index(drop=True)
    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.3, rows * 1.45))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in pairs.iterrows():
        rr, cc = divmod(i, cols)
        for j, (zf, img, lab) in enumerate(((zq, r["query"], r["q_label"]), (zr, r["ref"], r["r_label"]))):
            names = {Path(n).stem: n for n in zf.namelist()} if not hasattr(zf, "_stem_map") else zf._stem_map
            zf._stem_map = names
            im = Image.open(io.BytesIO(zf.read(names[img]))).convert("RGB").resize((160, 160))
            ax = axes[rr, cc * 2 + j]
            ax.imshow(im)
            ax.set_title(f"{'Q' if j == 0 else 'R'} {lab}" + (f" {r['sim']:.3f}" if j == 0 else ""), fontsize=5.5)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv):
    proj, train_zip, test_zip = Path(argv[0]), Path(argv[1]), Path(argv[2])
    res = proj / "resources"
    out = proj / "audit"
    out.mkdir(exist_ok=True)
    s = pd.read_csv(res / "splits_isic2019.csv")
    off = pd.read_csv(res / "official_test.csv")
    off = off[off["scored"]].reset_index(drop=True)
    # 1 exact
    th = pd.read_csv(res / "isic2019_train_hashes.csv")
    zt = zipfile.ZipFile(test_zip)
    tnames = {Path(n).stem: n for n in zt.namelist() if n.lower().endswith(".jpg")}
    rows = []
    for img in off["image"]:
        data = zt.read(tnames[img])
        rows.append((img, hashlib.md5(data).hexdigest(), thumb_hash(data)))
    oh = pd.DataFrame(rows, columns=["image", "md5_bytes", "md5_thumb32q4"])
    oh.to_csv(res / "official_test_hashes.csv", index=False)
    exact = {"official_vs_train_byte_identical": int(oh["md5_bytes"].isin(set(th["md5_bytes"])).sum()),
             "official_vs_train_thumbnail_identical": int(oh["md5_thumb32q4"].isin(set(th["md5_thumb32q4"])).sum())}
    print(exact, flush=True)
    # 2 embeddings
    cache = proj / "_local_cache"
    emb_tr = np.load(cache / "probe_emb_resnet50.npy").astype(np.float32)          # aligned with splits rows
    f_off = cache / "probe_emb_resnet50_official.npy"
    if f_off.exists():
        emb_off = np.load(f_off).astype(np.float32)
    else:
        emb_off = embed_zip(test_zip, off["image"].tolist())
        np.save(f_off, emb_off.astype(np.float16))
    lab = s["label"].values
    summary = dict(exact)
    sheets = []
    # (a) P2 train vs P2 val+test
    tr = np.where(s.p2_lesion_level == "train")[0]
    ev = np.where(s.p2_lesion_level != "train")[0]
    i, sim = nn_search(emb_tr[ev], emb_tr[tr])
    pa = pd.DataFrame({"query": s.image.values[ev], "ref": s.image.values[tr[i]], "sim": sim,
                       "q_label": lab[ev], "r_label": lab[tr[i]],
                       "q_source": s.source.values[ev], "r_source": s.source.values[tr[i]]}).sort_values("sim", ascending=False)
    pa.to_csv(out / "neardup_p2_candidates.csv", index=False)
    # positive control: P1 test images with a sibling in P1 train — similarity to the nearest training image
    tr1 = np.where(s.p1_image_level == "train")[0]
    te1 = np.where(s.p1_image_level == "test")[0]
    i1, sim1 = nn_search(emb_tr[te1], emb_tr[tr1])
    same = s.group_id.values[te1] == s.group_id.values[tr1[i1]]
    summary.update({
        "p2_nn_cosine_quantiles": {q: float(np.quantile(sim, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
        "positive_control_same_lesion_nn_cosine_median": float(np.median(sim1[same])),
        "positive_control_same_lesion_nn_cosine_q10": float(np.quantile(sim1[same], 0.1)),
        "p2_pairs_above_positive_q10": int((sim >= np.quantile(sim1[same], 0.1)).sum()),
        "p2_pairs_above_0.95": int((sim >= 0.95).sum()), "p2_pairs_above_0.98": int((sim >= 0.98).sum()),
    })
    zr = zipfile.ZipFile(train_zip)
    contact_sheet(pa, zr, zr, out / "contact_p2_top40.png", "P2: evaluation image (Q) vs nearest training image (R)")
    # (b) train vs official
    io_, so = nn_search(emb_off, emb_tr)
    pb = pd.DataFrame({"query": off.image.values, "ref": s.image.values[io_], "sim": so,
                       "q_label": off.label.values, "r_label": lab[io_],
                       "q_source": off.source.values, "r_source": s.source.values[io_]}).sort_values("sim", ascending=False)
    pb.to_csv(out / "neardup_official_candidates.csv", index=False)
    summary.update({"official_nn_cosine_quantiles": {q: float(np.quantile(so, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                    "official_pairs_above_positive_q10": int((so >= np.quantile(sim1[same], 0.1)).sum()),
                    "official_pairs_above_0.95": int((so >= 0.95).sum()), "official_pairs_above_0.98": int((so >= 0.98).sum())})
    contact_sheet(pb, zt, zr, out / "contact_official_top40.png", "Official test (Q) vs nearest training image (R)")
    (out / "neardup_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
