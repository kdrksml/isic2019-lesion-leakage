"""Loaders for the released splits, official test table, evaluation table and per-image predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1]  # repository root
RUNS = CACHE / "results" / "runs"
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Run folder, label used in the paper. Order follows Table 3.
MAIN_MODELS = [
    ("R1_ResNet50", "ResNet-50"),
    ("R2_EfficientNetV2S", "EfficientNetV2-S"),
    ("R3_ViTS16", "ViT-S/16"),
    ("R4_SwinV2T", "SwinV2-T"),
    ("R5_ConvNeXtV2T", "ConvNeXtV2-T"),
    ("R6_MaxViTT", "MaxViT-T"),
    ("B1_SwinV2T", "SwinV2-T, shared recipe"),
    ("B1c_ConvNeXtV2T", "ConvNeXtV2-T, shared recipe"),
    ("B2_SwinV2T_GLC", "SwinV2-T, shared recipe + GLC"),
    ("B3_ConvNeXtV2T_GLC", "ConvNeXtV2-T, shared recipe + GLC"),
]
EXPLORATORY = ("C2_ConvNeXtV2T_384_GLC2", "ConvNeXtV2-T + GLC-v2, 384 px")
PROTOCOL = {"P1": "image", "P2": "lesion"}


def splits():
    return pd.read_csv(CACHE / "resources" / "splits_isic2019.csv")


def pred_path(run_dir, stem):
    """Prediction file of a run: compressed in the repository, plain when written by the training code."""
    gz = Path(run_dir) / f"{stem}.csv.gz"
    return gz if gz.exists() else Path(run_dir) / f"{stem}.csv"


def official_scored():
    off = pd.read_csv(CACHE / "resources" / "official_test.csv")
    return set(off[off.scored & off.known_class].image)


def seeds(run, protocol):
    return sorted(p.name for p in (RUNS / PROTOCOL[protocol] / run).glob("seed*") if (p / "done.json").exists())


def predictions(run, protocol, testset, seed="seed42"):
    df = pd.read_csv(pred_path(RUNS / PROTOCOL[protocol] / run / seed, f"pred_{testset}"))
    if testset == "official":
        # Keep the scored images of the eight training classes.
        scored = official_scored()
        df = df[df.image.isin(scored)].reset_index(drop=True)
    z = df[[f"z_{c}" for c in CLASSES]].to_numpy()
    df["y_pred"] = z.argmax(1)
    return df


def macro_f1(y, p):
    f1 = []
    for k in range(len(CLASSES)):
        tp = np.sum((p == k) & (y == k))
        fp = np.sum((p == k) & (y != k))
        fn = np.sum((p != k) & (y == k))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return np.array(f1)


def evaluation():
    ev = pd.read_csv(CACHE / "results" / "evaluation_long.csv")
    return ev[(ev.variant == "raw") & (~ev.tta) & (ev.checkpoint == "final")]


def macro_f1_table():
    """Seed-mean macro-F1 per model, protocol and test set, as in Table 3."""
    ev = evaluation()
    rows = []
    for run, label in MAIN_MODELS + [EXPLORATORY]:
        r = {"run": run, "label": label}
        for prot, pname in PROTOCOL.items():
            for ts in ("test", "official"):
                sel = ev[(ev.protocol == pname) & (ev.name == run) & (ev.testset == ts)]
                r[f"{prot}_{ts}"] = sel.macro_f1.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def leakage_by_class():
    lk = pd.read_csv(CACHE / "resources" / "leakage_by_class.csv")
    lk = lk[(lk.protocol == "p1_image_level") & (lk.partition == "test") & (lk["class"] != "ALL")]
    return lk.sort_values("pct_leaked", ascending=False)


def same_lesion_counts():
    """Number of P1 training images that share the lesion of each P1 test image."""
    sp = splits()
    train_per_group = sp[sp.p1_image_level == "train"].groupby("group_id").size()
    test = sp[sp.p1_image_level == "test"][["image", "group_id"]].copy()
    test["n_same"] = test.group_id.map(train_per_group).fillna(0).astype(int)
    return test.set_index("image").n_same


def dose_response():
    """P1 test accuracy of each main model (seed 42) by same-lesion training count."""
    n_same = same_lesion_counts()
    bins = pd.cut(n_same, [-1, 0, 1, 3, np.inf], labels=["0", "1", "2–3", "≥4"])
    out = {}
    for run, label in MAIN_MODELS:
        df = predictions(run, "P1", "test")
        correct = pd.Series((df.y_pred == df.y_true).to_numpy(), index=df.image)
        out[label] = correct.groupby(bins.reindex(correct.index), observed=False).mean()
    return pd.DataFrame(out).T, bins.value_counts().reindex(["0", "1", "2–3", "≥4"])


def class_f1_inflation():
    """Per-class F1, P1 minus P2 internal test, seed 42 for P1 and seed mean for P2."""
    rows = []
    for run, label in MAIN_MODELS:
        d1 = predictions(run, "P1", "test")
        f1_p1 = macro_f1(d1.y_true.to_numpy(), d1.y_pred.to_numpy())
        f1_p2 = np.mean([macro_f1(d.y_true.to_numpy(), d.y_pred.to_numpy())
                         for d in (predictions(run, "P2", "test", s) for s in seeds(run, "P2"))], axis=0)
        rows.append(f1_p1 - f1_p2)
    return pd.DataFrame(rows, columns=CLASSES, index=[m[1] for m in MAIN_MODELS])


def confusion(run, protocol, testset, seed="seed42"):
    df = predictions(run, protocol, testset, seed)
    m = np.zeros((8, 8))
    np.add.at(m, (df.y_true.to_numpy(), df.y_pred.to_numpy()), 1)
    return m / m.sum(1, keepdims=True)
