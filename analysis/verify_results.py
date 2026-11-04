"""Recompute the numbers reported in the paper from the released predictions.

Part 1 recomputes the metrics of every run used in the paper from its prediction files and compares them with
evaluation_long.csv. Part 2 recomputes the numbers quoted in the tables and the text and compares them with the
values reported in the paper. A claim passes when the recomputed value rounds to the printed one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_statistics as ra  # noqa: E402
import common as fd  # noqa: E402

EV = pd.read_csv(fd.CACHE / "results" / "evaluation_long.csv")
NAMES = ra.NAMES
ALL11 = NAMES + [fd.EXPLORATORY[0]]
RESULTS = []


def check(label, printed, value, decimals=3):
    ok = value is not None and abs(round(float(value), decimals) - printed) <= 10 ** (-decimals) * 0.51
    RESULTS.append((ok, label, printed, value))


def load(run, prot_dir, ts, seed, best=False):
    p = fd.pred_path(fd.RUNS / prot_dir / run / seed, f"pred_{ts}{'_best' if best else ''}")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if ts == "official":
        d = d[d.image.isin(fd.official_scored())].reset_index(drop=True)
    d = d[d.y_true >= 0].reset_index(drop=True)
    z = d[[f"z_{c}" for c in fd.CLASSES]].to_numpy()
    d["y_pred"] = z.argmax(1)
    return d, z


def metrics(d, z):
    y, p = d.y_true.to_numpy(), d.y_pred.to_numpy()
    cm = np.zeros((8, 8))
    np.add.at(cm, (y, p), 1)
    prob = np.exp(z - z.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    mal = [0, 2, 7]
    return {"macro_f1": ra.f1_from_cm(cm), "bacc": ra.bacc_from_cm(cm), "accuracy": ra.acc_from_cm(cm),
            "macro_auc": roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(8))),
            "mel_sensitivity": cm[0, 0] / cm[0].sum(),
            "malignant_sensitivity": cm[np.ix_(mal, mal)].sum() / cm[mal].sum(), "n": len(y)}


# ------------------------------------------------------------------ part 1
def part1():
    worst, n = {}, 0
    runs = [(p, r, s) for p in ("image", "lesion") for r in ALL11 for s in sorted(x.name for x in (fd.RUNS / p / r).glob("seed*"))]
    runs += [(f"{p}_cv{k}", "CV_R4_SwinV2T", "seed42") for p in ("image", "lesion") for k in range(5)]
    for prot, run, seed in runs:
        for ts in ("test", "official"):
            for best in (False, True):
                got = load(run, prot, ts, seed, best)
                ref = EV[(EV.protocol == prot) & (EV.name == run) & (EV.seed == int(seed[4:])) & (EV.testset == ts)
                         & (~EV.tta) & (EV.variant == ("best_ckpt" if best else "raw"))]
                if got is None or len(ref) != 1:
                    RESULTS.append((False, f"run file or row missing: {prot} {run} {seed} {ts} best={best}", None, len(ref)))
                    continue
                m = metrics(*got)
                for k, v in m.items():
                    worst[k] = max(worst.get(k, 0), abs(v - ref[k].iloc[0]))
                n += 1
    print(f"part 1: {n} run/test-set/checkpoint combinations recomputed from predictions")
    for k, v in worst.items():
        print(f"   largest difference to evaluation_long, {k}: {v:.2e}")
        RESULTS.append((v < 1e-9, f"evaluation_long reproduces {k} of every paper run", 0, v))


# ------------------------------------------------------------------ part 2
def ev_mean(prot, ts, col, names=NAMES, variant="raw"):
    vals = []
    for r in names:
        sel = EV[(EV.protocol == prot) & (EV.name == r) & (EV.testset == ts) & (~EV.tta) & (EV.variant == variant)]
        vals.append(sel[col].mean())
    return float(np.mean(vals)), vals


def part2():
    sp = fd.splits()
    off = ra.OFFICIAL[ra.OFFICIAL.scored & ra.OFFICIAL.known_class]
    # data
    check("training images", 25331, len(sp), 0)
    for src, v in (("BCN20000", 12413), ("HAM10000", 10015), ("MSK", 819)):
        check(f"{src} images", v, int(sp.source.str.contains(src).sum()), 0)
    check("NV images", 12875, int((sp.label == "NV").sum()), 0)
    check("DF images", 239, int((sp.label == "DF").sum()), 0)
    check("patient-extended groups", 13865, sp.group_id.nunique(), 0)
    spr = sp.reset_index(drop=True)
    lg = ra.lesion_groups(spr)
    ls = pd.Series(lg).value_counts()
    check("lesion groups without patient links", 13913, len(ls), 0)
    check("multi-image lesion groups", 5065, int((ls > 1).sum()), 0)
    check("images in multi-image lesion groups", 16483, int(ls[ls > 1].sum()), 0)
    check("share in multi-image lesion groups %", 65.1, 100 * ls[ls > 1].sum() / len(sp), 1)
    check("largest lesion group", 31, int(ls.max()), 0)
    check("images per lesion group", 1.82, len(sp) / len(ls), 2)
    joined = pd.Series(lg).groupby(spr.group_id.values).nunique()
    check("patient-extended groups joining lesion groups", 30, int((joined > 1).sum()), 0)
    check("lesion groups joined by patient links", 78, int(joined[joined > 1].sum()), 0)
    in_train = np.isin(lg[spr.p1_image_level.values == "test"], lg[spr.p1_image_level.values == "train"])
    check("P1 test images sharing a lesion group", 2910, int(in_train.sum()), 0)
    check("P1 test images sharing a lesion group %", 57.4, 100 * in_train.mean(), 1)
    for col in ("p1_image_level", "p2_lesion_level"):
        for part, v in (("train", 17731), ("val", 2533), ("test", 5067)):
            check(f"{col} {part} size", v, int((sp[col] == part).sum()), 0)
    check("official scored known", 5920, len(off), 0)
    check("official Barcelona", 4409, int((off.source == "Hospital Clinic (BCN)").sum()), 0)
    check("official lesions", 2415, off.lesion_id_archive.nunique(), 0)
    for c, v in (("NV", 40.1), ("MEL", 20.8), ("AK", 6.3)):
        check(f"official share {c}", v, 100 * (off.label == c).mean(), 1)
    t2 = sp[sp.p2_lesion_level == "test"]
    for c, v in (("NV", 50.8), ("MEL", 17.8), ("AK", 3.4)):
        check(f"internal share {c}", v, 100 * (t2.label == c).mean(), 1)
    n_same = fd.same_lesion_counts()
    check("leaked P1 test images", 2926, int((n_same > 0).sum()), 0)
    check("P1 test groups", 4190, sp[sp.p1_image_level == "test"].group_id.nunique(), 0)
    check("P2 test groups", 2772, t2.group_id.nunique(), 0)
    v = sp[sp.p2_lesion_level == "val"]
    check("P2 val groups", 1387, v.group_id.nunique(), 0)
    check("P2 val DF", 24, int((v.label == "DF").sum()), 0)
    check("P2 val VASC", 25, int((v.label == "VASC").sum()), 0)
    g1 = len(set(lg[spr.p1_image_level.values == "train"]))
    g2 = len(set(lg[spr.p2_lesion_level.values == "train"]))
    check("more distinct training lesions under P1 %", 12.6, 100 * (g1 / g2 - 1), 1)
    # identical files among leaked images
    md5_train = set(sp.md5_bytes[sp.p1_image_level == "train"].dropna())
    te = sp[sp.p1_image_level == "test"].set_index("image")
    leaked = te.index[te.index.map(n_same) > 0]
    ident = int(te.loc[leaked, "md5_bytes"].isin(md5_train).sum())
    check("leaked test images with an identical training file", 11, ident, 0)
    check("leaked test images with only other photographs", 2915, len(leaked) - ident, 0)

    # Table 5
    tab = fd.macro_f1_table()
    printed = {"R1_ResNet50": (0.677, 0.522, 0.495, 0.506), "R2_EfficientNetV2S": (0.807, 0.611, 0.562, 0.552),
               "R3_ViTS16": (0.805, 0.600, 0.561, 0.545), "R4_SwinV2T": (0.834, 0.644, 0.604, 0.565),
               "R5_ConvNeXtV2T": (0.852, 0.640, 0.604, 0.573), "R6_MaxViTT": (0.849, 0.644, 0.591, 0.585),
               "B1_SwinV2T": (0.852, 0.653, 0.592, 0.579), "B1c_ConvNeXtV2T": (0.836, 0.643, 0.619, 0.578),
               "B2_SwinV2T_GLC": (0.863, 0.656, 0.598, 0.583), "B3_ConvNeXtV2T_GLC": (0.850, 0.647, 0.616, 0.594),
               "C2_ConvNeXtV2T_384_GLC2": (0.867, 0.660, 0.645, 0.617)}
    ci_printed = {"R1_ResNet50": (0.155, 0.106, 0.204, 0.166, 0.104, 0.228),
                  "R2_EfficientNetV2S": (0.197, 0.147, 0.247, 0.186, 0.123, 0.250),
                  "R3_ViTS16": (0.205, 0.154, 0.257, 0.189, 0.126, 0.252),
                  "R4_SwinV2T": (0.190, 0.145, 0.235, 0.152, 0.098, 0.206),
                  "R5_ConvNeXtV2T": (0.213, 0.167, 0.259, 0.182, 0.126, 0.239),
                  "R6_MaxViTT": (0.205, 0.159, 0.252, 0.200, 0.139, 0.261),
                  "B1_SwinV2T": (0.199, 0.155, 0.243, 0.186, 0.134, 0.239),
                  "B1c_ConvNeXtV2T": (0.192, 0.151, 0.234, 0.152, 0.101, 0.203),
                  "B2_SwinV2T_GLC": (0.207, 0.163, 0.251, 0.192, 0.139, 0.245),
                  "B3_ConvNeXtV2T_GLC": (0.204, 0.162, 0.246, 0.182, 0.130, 0.234),
                  "C2_ConvNeXtV2T_384_GLC2": (0.206, 0.163, 0.250, 0.178, 0.124, 0.232)}
    for _, r in tab.iterrows():
        for k, col in enumerate(("P1_test", "P2_test", "P1_official", "P2_official")):
            check(f"Table 5 {r.run} {col}", printed[r.run][k], r[col])
    per = per_model_ci(ALL11)
    for run, vals in ci_printed.items():
        for k, key in enumerate(("infl", "infl_lo", "infl_hi", "did", "did_lo", "did_hi")):
            check(f"Table 5 {run} {key}", vals[k], per[run][key])

    # Table 6
    for label, names, other, scale, best, pv in (
            ("primary", NAMES, "official", 1, False, (0.197, 0.164, 0.230, 0.179, 0.143, 0.215)),
            ("six baselines", NAMES[:6], "official", 1, False, (0.194, 0.159, 0.230, 0.179, 0.139, 0.219)),
            ("Barcelona", NAMES, "official_bcn", 1, False, (0.197, 0.164, 0.230, 0.177, 0.137, 0.217)),
            ("ViDIR", NAMES, "official_vidir", 1, False, (0.197, 0.164, 0.230, 0.196, 0.158, 0.234)),
            ("seed SD tripled", NAMES, "official", 3, False, (0.197, 0.160, 0.233, 0.179, 0.136, 0.222)),
            ("eleven networks", ALL11, "official", 1, False, (0.198, 0.165, 0.231, 0.179, 0.142, 0.215)),
            ("best-validation checkpoints", NAMES, "official", 1, True, (0.192, 0.158, 0.226, 0.173, 0.136, 0.209))):
        r = contrasts_variant(names, other, scale, best)
        got = (r["infl"]["estimate"], *r["infl"]["ci"], r["did"]["estimate"], *r["did"]["ci"])
        for k in range(6):
            check(f"Table 6 {label} [{k}]", pv[k], got[k])

    # Table 7 means and text
    t7 = {"macro_f1": (0.823, 0.626, 0.584, 0.566), "bacc": (0.811, 0.609, 0.558, 0.543),
          "accuracy": (0.878, 0.776, 0.715, 0.709), "macro_auc": (0.979, 0.940, 0.908, 0.904),
          "mel_sensitivity": (0.792, 0.625, 0.676, 0.664), "malignant_sensitivity": (0.858, 0.746, 0.786, 0.779)}
    for col, vals in t7.items():
        got = [ev_mean(p, ts, col)[0] for p, ts in (("image", "test"), ("lesion", "test"), ("image", "official"),
                                                    ("lesion", "official"))]
        for k in range(4):
            check(f"Table 7 {col} [{k}]", vals[k], got[k])
    inf = [ev_mean("image", "test", "macro_f1")[1][i] - ev_mean("lesion", "test", "macro_f1")[1][i] for i in range(10)]
    check("inflation range low", 0.155, min(inf))
    check("inflation range high", 0.213, max(inf))
    p2off = ev_mean("lesion", "official", "macro_f1")[1]
    check("P2 official macro-F1 low", 0.51, min(p2off), 2)
    check("P2 official macro-F1 high", 0.59, max(p2off), 2)
    a1, a2 = ev_mean("image", "test_lesion_agg", "macro_f1")[0], ev_mean("lesion", "test_lesion_agg", "macro_f1")[0]
    check("lesion-aggregated P1", 0.829, a1)
    check("lesion-aggregated P2", 0.662, a2)
    check("lesion-aggregated inflation", 0.166, a1 - a2)
    # balanced accuracy on leaked / non-leaked P1 test images
    lk = set(leaked)
    b_l, b_u = [], []
    for r in NAMES:
        d = fd.predictions(r, "P1", "test")
        for sub, out in ((d[d.image.isin(lk)], b_l), (d[~d.image.isin(lk)], b_u)):
            cm = np.zeros((8, 8))
            np.add.at(cm, (sub.y_true.values, sub.y_pred.values), 1)
            out.append(ra.bacc_from_cm(cm))
    check("BACC leaked P1 test", 0.858, np.mean(b_l))
    check("BACC non-leaked P1 test", 0.669, np.mean(b_u))
    # rank agreement
    tabm = tab.set_index("run")
    for label, names, prot, tv, pv in (("six baselines P1", NAMES[:6], "P1", 0.87, 0.017),
                                       ("six baselines P2", NAMES[:6], "P2", 0.87, 0.017),
                                       ("ten models P1", NAMES, "P1", 0.47, 0.073),
                                       ("ten models P2", NAMES, "P2", 0.73, 0.002)):
        t, p = stats.kendalltau(tabm.loc[names, f"{prot}_test"], tabm.loc[names, f"{prot}_official"])
        check(f"Kendall tau {label}", tv, t, 2)
        check(f"Kendall p {label}", pv, p, 3)
    # validation dynamics of the 256-px ablation models
    for prot, (v5, v30) in (("lesion", (0.615, 0.611)), ("image", (0.761, 0.857))):
        h = [pd.read_csv(fd.RUNS / prot / r / s / "history.csv").set_index("epoch").val_macro_f1
             for r in NAMES[6:] for s in ("seed42",)]
        check(f"validation macro-F1 epoch 5 {prot}", v5, np.mean([x.loc[5] for x in h]))
        check(f"validation macro-F1 epoch 30 {prot}", v30, np.mean([x.loc[30] for x in h]))
    # best-validation versus final weights, lesion-level runs
    shared = [r for r in NAMES[6:]] + [fd.EXPLORATORY[0]]
    di, do, better = [], [], 0
    for r in shared:
        for s in fd.seeds(r, "P2"):
            q = lambda ts, v: EV[(EV.protocol == "lesion") & (EV.name == r) & (EV.seed == int(s[4:])) & (EV.testset == ts)
                                 & (~EV.tta) & (EV.variant == v)].macro_f1.iloc[0]
            di.append(q("test", "best_ckpt") - q("test", "raw"))
            do.append(q("official", "best_ckpt") - q("official", "raw"))
            better += do[-1] > 0
    check("best-checkpoint internal change, shared recipe", 0.001, np.mean(di))
    check("best-checkpoint official change, shared recipe", 0.013, np.mean(do))
    check("shared-recipe runs improved on official", 14, better, 0)
    check("shared-recipe lesion-level runs", 15, len(do), 0)
    dp = []
    for r in NAMES[:6]:
        for s in fd.seeds(r, "P2"):
            q = lambda v: EV[(EV.protocol == "lesion") & (EV.name == r) & (EV.seed == int(s[4:])) & (EV.testset == "official")
                             & (~EV.tta) & (EV.variant == v)].macro_f1.iloc[0]
            dp.append(q("best_ckpt") - q("raw"))
    check("best-checkpoint official change, plain recipe", -0.004, np.mean(dp))
    # cross-validation
    cv_checks()


def per_model_ci(names):
    """Per-model intervals: seed variance pooled over the ten main models, and over all eleven for the 384-px row."""
    eleven = ra.contrasts(names=ALL11)["per_model"]   # the original per-model table pools over all eleven
    out = {}
    for run in names:
        r = eleven[run]
        out[run] = {"infl": r["infl"], "infl_lo": r["infl_ci"][0], "infl_hi": r["infl_ci"][1],
                    "did": r["did"], "did_lo": r["did_ci"][0], "did_hi": r["did_ci"][1]}
    return out


def contrasts_variant(names, other, scale, best):
    """ra.contrasts, optionally on the best-validation predictions."""
    if not best:
        return ra.contrasts(other=other, names=names, scale_p1_sd=scale)
    old = fd.predictions

    def best_pred(run, prot, testset, seed="seed42"):
        d = pd.read_csv(fd.pred_path(fd.RUNS / fd.PROTOCOL[prot] / run / seed, f"pred_{testset}_best"))
        if testset == "official":
            d = d[d.image.isin(fd.official_scored())].reset_index(drop=True)
        d["y_pred"] = d[[f"z_{c}" for c in fd.CLASSES]].to_numpy().argmax(1)
        return d
    fd.predictions = best_pred
    ra.CACHE_CM.clear()
    try:
        return ra.contrasts(other=other, names=names, scale_p1_sd=scale)
    finally:
        fd.predictions = old
        ra.CACHE_CM.clear()


def cv_checks():
    sp = fd.splits().set_index("image")
    pooled = {}
    folds = {"image": [], "lesion": []}
    for scheme in ("image", "lesion"):
        parts = []
        for k in range(5):
            d, _ = load("CV_R4_SwinV2T", f"{scheme}_cv{k}", "test", "seed42")
            o, _ = load("CV_R4_SwinV2T", f"{scheme}_cv{k}", "official", "seed42")
            parts.append(d)
            f_int = metrics(d, d[[f"z_{c}" for c in fd.CLASSES]].to_numpy())["macro_f1"]
            f_off = metrics(o, o[[f"z_{c}" for c in fd.CLASSES]].to_numpy())["macro_f1"]
            folds[scheme].append((f_int, f_off))
        pooled[scheme] = pd.concat(parts)
    cm = {}
    for scheme, d in pooled.items():
        m = np.zeros((8, 8))
        np.add.at(m, (d.y_true.values, d.y_pred.values), 1)
        cm[scheme] = ra.f1_from_cm(m)
    check("CV pooled inflation", 0.202, cm["image"] - cm["lesion"])
    infl = [folds["image"][k][0] - folds["lesion"][k][0] for k in range(5)]
    did = [(folds["image"][k][0] - folds["image"][k][1]) - (folds["lesion"][k][0] - folds["lesion"][k][1]) for k in range(5)]
    gap1 = [folds["image"][k][0] - folds["image"][k][1] for k in range(5)]
    gap2 = [folds["lesion"][k][0] - folds["lesion"][k][1] for k in range(5)]
    check("CV mean inflation", 0.203, np.mean(infl))
    check("CV SD inflation", 0.025, np.std(infl, ddof=1))
    check("CV mean DiD", 0.192, np.mean(did))
    check("CV SD DiD", 0.042, np.std(did, ddof=1))
    check("CV image gap mean", 0.229, np.mean(gap1))
    check("CV image gap SD", 0.005, np.std(gap1, ddof=1))
    check("CV lesion gap min", -0.023, min(gap2))
    check("CV lesion gap max", 0.086, max(gap2))
    check("CV lesion gap mean", 0.037, np.mean(gap2))
    check("CV fold 0 among two largest lesion gaps", 1, float(sorted(gap2, reverse=True).index(gap2[0]) < 2), 0)


if __name__ == "__main__":
    part1()
    part2()
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    pd.DataFrame(RESULTS, columns=["passed", "quantity", "reported", "recomputed"]).to_csv(
        out / "verification.csv", index=False)
    bad = [r for r in RESULTS if not r[0]]
    print(f"\n{len(RESULTS)} checks, {len(RESULTS) - len(bad)} passed, {len(bad)} failed")
    for ok, label, printed, value in bad:
        print(f"   FAIL {label}: printed {printed}, recomputed {value}")
