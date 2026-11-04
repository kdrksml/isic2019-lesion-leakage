"""Statistical analyses of the paper. Every number is computed from the run predictions, the split table and the
audit files.

Lesion-cluster bootstrap: multinomial weights over all lesion groups shared by the two internal test sets and
independent weights over ISIC Archive lesions for the official test, B = 2000, fixed generator seeds.
Run: python analysis/paired_statistics.py   (writes analysis/statistics_results.json)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as fd  # noqa: E402

B = 2000
W_SEED = {"test": 21, "official": 33, "official_bcn": 34, "official_vidir": 35}
SOURCES = {"official_bcn": "Hospital Clinic (BCN)", "official_vidir": "ViDIR Vienna (HAM-type)"}
NAMES = [r for r, _ in fd.MAIN_MODELS]
NC = len(fd.CLASSES)
MAL = [fd.CLASSES.index(c) for c in ("MEL", "BCC", "SCC")]
OUT = {}


# ------------------------------------------------------------------ helpers
def f1_from_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = 2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0).mean(-1)


def bacc_from_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup = cm.sum(-1)
    return np.nanmean(np.where(sup > 0, tp / np.maximum(sup, 1e-12), np.nan), axis=-1)


def acc_from_cm(cm):
    return np.trace(cm, axis1=-2, axis2=-1) / cm.sum((-2, -1))


def std_acc_from_cm(w):
    """Accuracy standardised to the class weights w (recall per class weighted by w)."""
    def f(cm):
        sup = cm.sum(-1)
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / np.maximum(sup, 1e-12)
        ok = sup > 0
        return (np.where(ok, rec, 0) * w).sum(-1) / (ok * w).sum(-1)
    return f


def prevalence_f1(cm, prev):
    """Macro-F1 after rescaling the confusion rows to a target class prevalence."""
    rows = cm / np.maximum(cm.sum(-1, keepdims=True), 1e-12)
    return f1_from_cm(rows * prev[..., :, None])


def draw(n, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)


def official_group(df):
    g = df.group.astype(str).values
    fallback = np.array(["img_" + i for i in df.image.astype(str)])
    return np.where((g == "nan") | (g == ""), fallback, g)


def ci_norm(pt, se):
    return [pt - 1.96 * se, pt + 1.96 * se]


def p_norm(pt, se):
    return float(2 * (1 - stats.norm.cdf(abs(pt) / se))) if se > 0 else 0.0


# ------------------------------------------------------------------ bootstrap engine
SPLITS = fd.splits()
OFFICIAL = pd.read_csv(fd.CACHE / "resources" / "official_test.csv").set_index("image")
CODES = {"test": np.array(sorted(SPLITS.group_id.astype(str).unique()))}
WEIGHTS = {"test": draw(len(CODES["test"]), W_SEED["test"])}


def preds(run, prot, ts, seed):
    d = fd.predictions(run, prot, "test" if ts == "test" else "official", seed)
    if ts in SOURCES:
        d = d[OFFICIAL.loc[d.image, "source"].values == SOURCES[ts]].reset_index(drop=True)
    return d


def group_key(ts, d):
    return d.group.astype(str).values if ts == "test" else official_group(d)


def boot_cm(ts, d):
    """Point confusion matrix and its B bootstrap replicates for prediction frame d of test set ts."""
    g = group_key(ts, d)
    if ts not in CODES:
        CODES[ts] = np.array(sorted(set(g)))
        WEIGHTS[ts] = draw(len(CODES[ts]), W_SEED[ts])
    idx = pd.Index(CODES[ts]).get_indexer(g)
    assert (idx >= 0).all()
    M = np.zeros((len(CODES[ts]), NC, NC), np.float32)
    np.add.at(M, (idx, d.y_true.values, d.y_pred.values), 1)
    cmb = (WEIGHTS[ts] @ M.reshape(len(CODES[ts]), -1)).reshape(B, NC, NC)
    return M.sum(0).astype(float), cmb


CACHE_CM = {}


def run_cms(run, prot, ts, seed, subset=None):
    key = (run, prot, ts, seed, None if subset is None else hash(frozenset(subset)))
    if key not in CACHE_CM:
        d = preds(run, prot, ts, seed)
        if subset is not None:
            d = d[d.image.isin(subset)].reset_index(drop=True)
        CACHE_CM[key] = boot_cm(ts, d)
    return CACHE_CM[key]


def table(metric, ts, names=NAMES, subset=None):
    """Per (protocol, model): seed mean of the point metric, seed-mean bootstrap array, seed count, per-seed points."""
    out = {}
    for prot in ("P1", "P2"):
        for run in names:
            pts, boots = [], []
            for s in fd.seeds(run, prot):
                cm, cmb = run_cms(run, prot, ts, s, subset)
                pts.append(float(metric(cm)))
                boots.append(metric(cmb))
            out[(prot, run)] = {"pt": float(np.mean(pts)), "boot": np.mean(boots, 0), "k": len(pts), "seeds": pts}
    return out


def seed_var(T, prot, names=NAMES):
    num = den = 0.0
    for run in names:
        v = T[(prot, run)]["seeds"]
        if len(v) > 1:
            num += np.var(v, ddof=1) * (len(v) - 1)
            den += len(v) - 1
    return num / den if den else None


def seed_vars(T, names=NAMES):
    v1, v2 = seed_var(T, "P1", names), seed_var(T, "P2", names)
    return (v1 if v1 is not None else v2), (v2 if v2 is not None else v1)


def summarise(per_model_pt, per_model_boot, seed_terms):
    pt = float(np.mean(per_model_pt))
    se = math.sqrt(float(np.var(np.mean(per_model_boot, 0), ddof=1)) + sum(seed_terms) / len(per_model_pt) ** 2)
    return {"estimate": pt, "se": se, "ci": ci_norm(pt, se), "p": p_norm(pt, se)}


def contrasts(metric=f1_from_cm, other="official", names=NAMES, scale_p1_sd=1.0, pool=None):
    """Mean contrasts over `names`. Seed variances are pooled over the models in `pool` (default: `names`), as in the
    original analysis, where per-model intervals use the variance pooled over the whole model set."""
    pool = list(pool or names)
    every = list(dict.fromkeys(list(names) + pool))
    Ti, To = table(metric, "test", every), table(metric, other, every)
    vi1, vi2 = seed_vars(Ti, pool)
    vo1, vo2 = seed_vars(To, pool)
    vi1, vo1 = vi1 * scale_p1_sd ** 2, vo1 * scale_p1_sd ** 2
    rows = {"infl": ([], [], []), "did": ([], [], []), "gain": ([], [], [])}
    per_model = {}
    for run in names:
        i1, i2, o1, o2 = Ti[("P1", run)], Ti[("P2", run)], To[("P1", run)], To[("P2", run)]
        k1, k2 = i1["k"], i2["k"]
        terms = {"infl": (i1["pt"] - i2["pt"], i1["boot"] - i2["boot"], vi1 / k1 + vi2 / k2),
                 "did": ((i1["pt"] - o1["pt"]) - (i2["pt"] - o2["pt"]),
                         (i1["boot"] - o1["boot"]) - (i2["boot"] - o2["boot"]),
                         (vi1 + vo1) / min(k1, o1["k"]) + (vi2 + vo2) / min(k2, o2["k"])),
                 "gain": (o1["pt"] - o2["pt"], o1["boot"] - o2["boot"], vo1 / o1["k"] + vo2 / o2["k"])}
        for key, (pt, bt, vs) in terms.items():
            rows[key][0].append(pt)
            rows[key][1].append(bt)
            rows[key][2].append(vs)
        per_model[run] = {}
        for key, (pt, bt, vs) in terms.items():
            se = math.sqrt(float(np.var(bt, ddof=1)) + vs)
            per_model[run].update({key: pt, f"{key}_ci": ci_norm(pt, se), f"{key}_p": p_norm(pt, se)})
    res = {k: summarise(*v) for k, v in rows.items()}
    res["per_model"] = per_model
    res["seed_sd"] = {"P1_internal": math.sqrt(vi1), "P2_internal": math.sqrt(vi2),
                      "P1_official": math.sqrt(vo1), "P2_official": math.sqrt(vo2)}
    # share of the inflation not matched by an official gain: DiD / inflation = 1 - gain / inflation.
    # Internal and official weights are independent and the seed terms enter separately, so the covariance is zero.
    I, G = res["infl"]["estimate"], res["gain"]["estimate"]
    share = 1 - G / I
    se_share = math.sqrt(res["gain"]["se"] ** 2 / I ** 2 + G ** 2 * res["infl"]["se"] ** 2 / I ** 4)
    res["share"] = {"estimate": share, "se": se_share, "ci": ci_norm(share, se_share)}
    return res


# ------------------------------------------------------------------ 1. reproduce H1, H2 and add the official gain
def primary():
    res = contrasts()
    OUT["primary"] = res
    print("H1 inflation", fmt(res["infl"]), "| H2 DiD", fmt(res["did"]))
    print("official gain", fmt(res["gain"]), "p =", f"{res['gain']['p']:.4f}")
    print("share DiD/inflation", fmt(res["share"]))
    pos = [r for r in res["per_model"].values() if r["gain"] > 0]
    sig = [r for r in res["per_model"].values() if r["gain_ci"][0] > 0]
    print(f"models with positive official gain {len(pos)}, with CI above zero {len(sig)}")
    OUT["primary_counts"] = {"positive_gain": len(pos), "gain_ci_above_zero": len(sig)}
    for key in SOURCES:
        r = contrasts(other=key)
        OUT[f"primary_{key}"] = {k: r[k] for k in ("did", "gain", "share")}
        print(key, "DiD", fmt(r["did"]), "gain", fmt(r["gain"]), "share", fmt(r["share"]))
    # how large the image-level seed SD would have to be before the intervals reach zero
    for key in ("infl", "did"):
        lo, hi = 1.0, 200.0
        for _ in range(40):
            mid = (lo + hi) / 2
            r = contrasts(scale_p1_sd=mid)[key]
            lo, hi = (mid, hi) if r["ci"][0] > 0 else (lo, mid)
        OUT[f"break_even_sd_factor_{key}"] = lo
        print(f"break-even factor on the image-level seed SD for {key}: {lo:.1f}")


def fmt(r):
    return f"{r['estimate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"


# ------------------------------------------------------------------ 2. specificity next to the sensitivities
def binary_rates():
    def rates(cm):
        mel = fd.CLASSES.index("MEL")
        sens_mel = cm[mel, mel] / cm[mel].sum()
        spec_mel = (cm.sum() - cm[mel].sum() - cm[:, mel].sum() + cm[mel, mel]) / (cm.sum() - cm[mel].sum())
        ben = [k for k in range(NC) if k not in MAL]
        sens_mal = cm[np.ix_(MAL, MAL)].sum() / cm[MAL].sum()
        spec_mal = cm[np.ix_(ben, ben)].sum() / cm[ben].sum()
        return np.array([sens_mel, spec_mel, sens_mal, spec_mal])

    out = {}
    for ts in ("test", "official"):
        for prot in ("P1", "P2"):
            vals = [np.mean([rates(run_cms(r, prot, ts, s)[0]) for s in fd.seeds(r, prot)], 0) for r in NAMES]
            out[f"{prot}_{ts}"] = dict(zip(["mel_sens", "mel_spec", "mal_sens", "mal_spec"], np.mean(vals, 0)))
    OUT["binary_rates"] = out
    print("binary rates", {k: {m: round(v, 3) for m, v in d.items()} for k, d in out.items()})


# ------------------------------------------------------------------ 3. institutions of the official test
def institutions():
    res = {}
    off = OFFICIAL[OFFICIAL.scored & OFFICIAL.known_class]
    for key, src in SOURCES.items():
        sub = off[off.source == src]
        res[key] = {"n": int(len(sub)), "lesions": int(sub.lesion_id_archive.nunique()),
                    "class_share": (sub.label.value_counts(normalize=True).reindex(fd.CLASSES).fillna(0)).to_dict(),
                    "sites": sub.anatom_site_general.fillna("unknown").value_counts(normalize=True).to_dict()}
        for metric, name in ((f1_from_cm, "macro_f1"), (bacc_from_cm, "bacc")):
            T = table(metric, key)
            for prot in ("P1", "P2"):
                res[key][f"{prot}_{name}"] = float(np.mean([T[(prot, r)]["pt"] for r in NAMES]))
    # class recall of the lesion-level models per institution, and macro-F1 with the class mix exchanged
    prev = {k: np.array([res[k]["class_share"][c] for c in fd.CLASSES]) for k in SOURCES}
    for key in SOURCES:
        other = [k for k in SOURCES if k != key][0]
        recalls, f1_swapped = [], []
        for r in NAMES:
            for s in fd.seeds(r, "P2"):
                cm = run_cms(r, "P2", key, s)[0]
                recalls.append(np.diag(cm) / np.maximum(cm.sum(1), 1))
                f1_swapped.append(float(prevalence_f1(cm, prev[other])))
        res[key]["P2_recall"] = dict(zip(fd.CLASSES, np.mean(recalls, 0)))
        res[key]["P2_macro_f1_with_other_class_mix"] = float(np.mean(f1_swapped))
    # internal lesion-level test restricted to the matching training source
    src_of = SPLITS.set_index("image").source
    for key, src in (("official_bcn", "BCN20000"), ("official_vidir", "HAM10000")):
        imgs = set(SPLITS.image[(SPLITS.p2_lesion_level == "test") & (SPLITS.source == src)])
        T = table(f1_from_cm, "test", subset=imgs)
        res[key]["P2_internal_same_source_macro_f1"] = float(np.mean([T[("P2", r)]["pt"] for r in NAMES]))
        res[key]["P2_internal_same_source_n"] = len(imgs)
    del src_of
    OUT["institutions"] = res
    for k, v in res.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, dict)})
        print("   classes", {c: round(x, 3) for c, x in v["class_share"].items()})
        print("   P2 recall", {c: round(x, 2) for c, x in v["P2_recall"].items()})


# ------------------------------------------------------------------ 4. common test images, class balance, placebo DiD
def common_images():
    common = SPLITS[(SPLITS.p1_image_level == "test") & (SPLITS.p2_lesion_level == "test")]
    n_same = fd.same_lesion_counts()
    leaked = set(common.image[common.image.map(n_same) > 0])
    clean = set(common.image) - leaked
    lab = common.set_index("image").label
    res = {"n_leaked": len(leaked), "n_clean": len(clean),
           "class_share_leaked": lab[list(leaked)].value_counts(normalize=True).reindex(fd.CLASSES).fillna(0).to_dict(),
           "class_share_clean": lab[list(clean)].value_counts(normalize=True).reindex(fd.CLASSES).fillna(0).to_dict()}
    counts = pd.crosstab(lab, lab.index.isin(leaked)).reindex(fd.CLASSES).fillna(0)
    res["chi2_p"] = float(stats.chi2_contingency(counts.values)[1])
    w = lab.value_counts(normalize=True).reindex(fd.CLASSES).fillna(0).values
    for name, subset in (("leaked", leaked), ("clean", clean)):
        for mname, metric in (("acc", acc_from_cm), ("std_acc", std_acc_from_cm(w)), ("bacc", bacc_from_cm)):
            T = table(metric, "test", subset=subset)
            d_pt = [T[("P1", r)]["pt"] - T[("P2", r)]["pt"] for r in NAMES]
            d_bt = [T[("P1", r)]["boot"] - T[("P2", r)]["boot"] for r in NAMES]
            v1, v2 = seed_vars(T)
            r = summarise(d_pt, d_bt, [v1 / T[("P1", m)]["k"] + v2 / T[("P2", m)]["k"] for m in NAMES])
            r["P1"] = float(np.mean([T[("P1", m)]["pt"] for m in NAMES]))
            r["P2"] = float(np.mean([T[("P2", m)]["pt"] for m in NAMES]))
            res[f"{name}_{mname}"] = r
    # placebo DiD: the same contrast as H2 on the common images without leakage, where the protocols differ only in
    # the grouping constraint. Accuracy and balanced accuracy, as macro-F1 is unstable on 390 images.
    for mname, metric in (("acc", acc_from_cm), ("bacc", bacc_from_cm)):
        Ti, To = table(metric, "test", subset=clean), table(metric, "official")
        vi1, vi2 = seed_vars(Ti)
        vo1, vo2 = seed_vars(To)
        pts, bts, vs = [], [], []
        for m in NAMES:
            i1, i2, o1, o2 = Ti[("P1", m)], Ti[("P2", m)], To[("P1", m)], To[("P2", m)]
            pts.append((i1["pt"] - o1["pt"]) - (i2["pt"] - o2["pt"]))
            bts.append((i1["boot"] - o1["boot"]) - (i2["boot"] - o2["boot"]))
            vs.append((vi1 + vo1) / min(i1["k"], o1["k"]) + (vi2 + vo2) / min(i2["k"], o2["k"]))
        res[f"placebo_did_{mname}"] = summarise(pts, bts, vs)
        # the same DiD on all internal test images, for comparison
        Ta = table(metric, "test")
        pts = [(Ta[("P1", m)]["pt"] - To[("P1", m)]["pt"]) - (Ta[("P2", m)]["pt"] - To[("P2", m)]["pt"]) for m in NAMES]
        res[f"full_did_{mname}"] = float(np.mean(pts))
    OUT["common_images"] = res
    print("common images", {k: (fmt(v) if isinstance(v, dict) and "ci" in v else v) for k, v in res.items()
                            if not k.startswith("class_share")})
    print("   class share leaked", {c: round(x, 3) for c, x in res["class_share_leaked"].items()})
    print("   class share clean ", {c: round(x, 3) for c, x in res["class_share_clean"].items()})


# ------------------------------------------------------------------ 5. nearest neighbour on identical test images
EMB = fd.CACHE / "_local_cache" / "probe_emb_resnet50.npy"   # written by tools/leakage_probe.py


def nearest_neighbour():
    if not EMB.exists():
        print("skipped nearest neighbour: run tools/leakage_probe.py first")
        return
    e = np.load(EMB).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    y = SPLITS.target.values
    g = SPLITS.group_id.values
    res = {}
    for prot, col in (("P1", "p1_image_level"), ("P2", "p2_lesion_level")):
        tr = np.where(SPLITS[col] == "train")[0]
        te = np.where(SPLITS[col] == "test")[0]
        S = e[te] @ e[tr].T
        pred_all = y[tr][S.argmax(1)]
        same = g[te][:, None] == g[tr][None, :]
        S[same] = -2.0
        pred_excl = y[tr][S.argmax(1)]
        leaked = same.any(1)
        f = lambda p, m: float(fd.macro_f1(y[te][m], p[m]).mean())
        res[prot] = {"all": f(pred_all, np.ones(len(te), bool)), "excluding_same_lesion": f(pred_excl, np.ones(len(te), bool)),
                     "leaked_all": f(pred_all, leaked) if leaked.any() else None,
                     "leaked_excluding_same_lesion": f(pred_excl, leaked) if leaked.any() else None,
                     "n_leaked": int(leaked.sum())}
    OUT["nearest_neighbour"] = res
    print("1-NN", res)


# ------------------------------------------------------------------ 6. patient links, duplicates with conflicting labels
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def components(df, keys, pairs=()):
    uf = UF(len(df))
    for key in keys:
        col = df[key]
        for _, idx in df.index[col.notna() & (col.astype(str) != "")].to_series().groupby(col).groups.items():
            idx = list(idx)
            for j in idx[1:]:
                uf.union(idx[0], j)
    pos = {im: i for i, im in enumerate(df.image)}
    for a, b in pairs:
        if a in pos and b in pos:
            uf.union(pos[a], pos[b])
    return np.array([uf.find(i) for i in range(len(df))])


def lesion_groups(df):
    """Lesion groups without patient links: lesion identifiers, identical files and accepted near-duplicate pairs."""
    adj = pd.read_csv(fd.CACHE / "resources" / "neardup_adjudication.csv")
    adj = adj[adj.decision.isin(["same_lesion_confirmed", "same_lesion_probable"])]
    pairs = list(adj[["image_a", "image_b"]].itertuples(index=False, name=None))
    return components(df, ["lesion_id", "lesion_id_archive", "md5_bytes"], pairs)


def links_and_labels():
    df = SPLITS.reset_index(drop=True)
    base = components(df, ["lesion_id", "lesion_id_archive"])
    res = {"patient_images": int(df.patient_id_archive.notna().sum()),
           "patients": int(df.patient_id_archive.nunique())}
    merges = []
    for pid, sub in df[df.patient_id_archive.notna()].groupby("patient_id_archive"):
        comps = sub.groupby(base[sub.index]).agg(lesion=("lesion_id", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                                                 labels=("label", lambda s: tuple(sorted(set(s)))),
                                                 n=("image", "size"))
        if len(comps) > 1:
            with_id = comps.lesion.notna().sum()
            labels = set(l for t in comps.labels for l in t)
            merges.append({"patient": pid, "components": len(comps), "with_lesion_id": int(with_id),
                           "distinct_labels": len(labels), "images": int(comps.n.sum())})
    m = pd.DataFrame(merges)
    res["patients_joining_components"] = int(len(m))
    res["links"] = int((m.components - 1).sum())
    res["links_between_components_with_different_dataset_lesion_ids"] = int((m.with_lesion_id.clip(lower=1) - 1).sum())
    res["patients_whose_components_differ_in_diagnosis"] = int((m.distinct_labels > 1).sum())
    res["images_in_patient_merged_groups"] = int(df.group_id.isin(df.group_id[df.patient_id_archive.notna()]).sum())
    # lesion-only grouping (no patient links): how much of the P1 leakage depends on patient links
    lesion_only = lesion_groups(df)
    res["groups_lesion_only"] = int(len(set(lesion_only)))
    res["groups_with_patient_links"] = int(df.group_id.nunique())
    sizes = pd.Series(lesion_only).value_counts()
    res["lesion_only_multi_image_groups"] = int((sizes > 1).sum())
    res["lesion_only_images_in_multi_image_groups"] = int(sizes[sizes > 1].sum())
    res["lesion_only_largest_group"] = int(sizes.max())
    res["images_per_lesion_group"] = float(len(df) / len(sizes))
    joined = pd.Series(lesion_only).groupby(df.group_id.values).nunique()
    res["patient_extended_groups_joining_lesion_groups"] = int((joined > 1).sum())
    res["lesion_groups_joined_by_patient_links"] = int(joined[joined > 1].sum())
    for name, gid in (("with_patient_links", df.group_id.values), ("lesion_only", lesion_only)):
        tr = set(gid[df.p1_image_level == "train"])
        te = df.p1_image_level == "test"
        res[f"p1_test_leak_pct_{name}"] = float(100 * np.mean([x in tr for x in gid[te]]))
        tr2 = set(gid[df.p2_lesion_level == "train"])
        te2 = df.p2_lesion_level == "test"
        res[f"p2_test_leak_pct_{name}"] = float(100 * np.mean([x in tr2 for x in gid[te2]]))
    # byte-identical files
    dup = df[df.md5_bytes.notna() & df.md5_bytes.duplicated(keep=False)]
    sets = dup.groupby("md5_bytes")
    conflicts = [s for _, s in sets if s.label.nunique() > 1]
    res["byte_identical_sets"] = int(sets.ngroups)
    res["byte_identical_images"] = int(len(dup))
    res["conflict_sets"] = [{"images": list(s.image), "labels": list(s.label), "sources": list(s.source),
                             "p1": list(s.p1_image_level), "p2": list(s.p2_lesion_level),
                             "same_group": bool(s.group_id.nunique() == 1)} for s in conflicts]
    bad = set(i for s in conflicts for i in s.image)
    full = {p: np.mean([table(f1_from_cm, "test")[(p, r)]["pt"] for r in NAMES]) for p in ("P1", "P2")}
    keep = set(df.image) - bad
    red = table(f1_from_cm, "test", subset=keep)
    res["macro_f1_change_without_conflicts"] = {p: float(np.mean([red[(p, r)]["pt"] for r in NAMES]) - full[p])
                                               for p in ("P1", "P2")}
    off_conf = OFFICIAL.index.isin(bad).sum()
    res["conflict_images_in_official"] = int(off_conf)
    OUT["links"] = res
    print("links", {k: v for k, v in res.items() if k != "conflict_sets"})
    for s in res["conflict_sets"]:
        print("   conflict", s)


# ------------------------------------------------------------------ 7. near-duplicate screen: wider sensitivity analysis
def neardup_sensitivity():
    if not EMB.exists():
        print("skipped near-duplicate sensitivity: run tools/leakage_probe.py first")
        return
    e = np.load(EMB).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    df = SPLITS.reset_index(drop=True)
    tr = np.where(df.p2_lesion_level == "train")[0]
    te = np.where(df.p2_lesion_level == "test")[0]
    sim = (e[te] @ e[tr].T).max(1)
    # positive control: P1 test images and their nearest training image of the same lesion
    tr1 = np.where(df.p1_image_level == "train")[0]
    te1 = np.where(df.p1_image_level == "test")[0]
    S1 = e[te1] @ e[tr1].T
    same = df.group_id.values[te1][:, None] == df.group_id.values[tr1][None, :]
    pos = np.where(same.any(1), np.where(same, S1, -2).max(1), np.nan)
    pos = pos[~np.isnan(pos)]
    res = {"p2_test_nn_quantiles": {q: float(np.quantile(sim, q)) for q in (0.5, 0.9, 0.99)},
           "same_lesion_nn_quantiles": {q: float(np.quantile(pos, q)) for q in (0.1, 0.5, 0.9)}}
    full = table(f1_from_cm, "test")
    base = float(np.mean([full[("P2", r)]["pt"] for r in NAMES]))
    p1 = float(np.mean([full[("P1", r)]["pt"] for r in NAMES]))
    res["p2_internal_macro_f1"] = base
    res["excluded"] = []
    for q in (0.99, 0.9, 0.75, 0.5):
        thr = float(np.quantile(sim, q))
        keep = set(df.image.values[te][sim < thr])
        T = table(f1_from_cm, "test", subset=keep)
        v = float(np.mean([T[("P2", r)]["pt"] for r in NAMES]))
        share_pos = float(np.mean(pos >= thr))
        res["excluded"].append({"quantile": q, "threshold": thr, "n_excluded": int(len(te) - len(keep)),
                                "p2_macro_f1": v, "change": v - base, "inflation": p1 - v,
                                "share_of_same_lesion_pairs_above": share_pos})
    OUT["neardup"] = res
    print("near-dup", res["p2_test_nn_quantiles"], res["same_lesion_nn_quantiles"])
    for r in res["excluded"]:
        print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


# ------------------------------------------------------------------ 8. cross-validation folds
def cv_folds():
    df = SPLITS
    res = {}
    for scheme in ("lesion", "image"):
        for f in range(5):
            col = df[f"cv{f}_{scheme}"]
            sizes = col.value_counts().to_dict()
            tr = set(df.group_id[col == "train"])
            leak_test = float(100 * df.group_id[col == "test"].isin(tr).mean())
            leak_val = float(100 * df.group_id[col == "val"].isin(tr).mean())
            res[f"{scheme}_{f}"] = {**{k: int(v) for k, v in sizes.items()}, "leak_test_pct": leak_test,
                                   "leak_val_pct": leak_val,
                                   "class_max_dev_pct": float(100 * (pd.crosstab(df.label, col, normalize="columns")
                                                                     .sub(df.label.value_counts(normalize=True), axis=0)
                                                                     .abs().max().max()))}
    res["fold0_equals_p2"] = bool((df.cv0_lesion == df.p2_lesion_level).all())
    OUT["cv"] = res
    for k, v in res.items():
        print("cv", k, v)


if __name__ == "__main__":
    primary()
    binary_rates()
    institutions()
    common_images()
    nearest_neighbour()
    links_and_labels()
    neardup_sensitivity()
    cv_folds()
    Path(__file__).with_name("statistics_results.json").write_text(
        json.dumps(OUT, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)), encoding="utf-8")
