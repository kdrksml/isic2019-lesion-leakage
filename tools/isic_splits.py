# -*- coding: utf-8 -*-
"""Frozen ISIC 2019 partitions for the paired leakage study.

P1  image-level : the conventional stratified 70/10/20 split at image level
                  (train_test_split, seed 42) — reproduced exactly.
P2  lesion-level: the same proportions, but every image of a lesion lands in one
                  partition. A "lesion" is a connected component of the graph whose
                  edges join two images when they share
                    (a) the ISIC 2019 component lesion id (HAM_/BCN_/MSK4_),
                    (b) the ISIC Archive lesion id (IL_...),
                    (c) the ISIC Archive patient id (IP_..., available for few images), or
                    (d) a byte-identical or thumbnail-identical file.
                  Images with none of these links form singleton groups.
P3  official    : the ISIC 2019 test set (scored, known-class images); evaluation only.

Usage (writes resources/splits_isic2019.csv, resources/official_test.csv and
resources/split_audit.json):
    python isic_splits.py <train_gt.csv> <train_meta.csv> <project_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
SEED, TEST_SIZE, VAL_SIZE = 42, 0.20, 0.125


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def source_of(lesion_id):
    if isinstance(lesion_id, str):
        for pre, name in (("HAM_", "HAM10000"), ("BCN_", "BCN20000"), ("MSK", "MSK")):
            if lesion_id.startswith(pre):
                return name
    return "Legacy (no lesion id)"


def load_train(gt_csv, meta_csv, archive_csv=None, hash_csv=None):
    gt = pd.read_csv(gt_csv)
    md = pd.read_csv(meta_csv)
    df = gt[["image"]].copy()
    df["target"] = gt[CLASS_NAMES].values.argmax(1).astype(int)
    df["label"] = df["target"].map(lambda i: CLASS_NAMES[i])
    df = df.merge(md[["image", "lesion_id", "age_approx", "sex", "anatom_site_general"]],
                  on="image", how="left")
    df["source"] = df["lesion_id"].map(source_of)
    if archive_csv is not None and Path(archive_csv).exists():
        ar = pd.read_csv(archive_csv)
        ar = ar[ar["split"] == "train"][["image", "lesion_id_archive", "patient_id_archive", "status"]]
        ar = ar.drop_duplicates("image")
        df = df.merge(ar, on="image", how="left")
    else:
        df["lesion_id_archive"] = np.nan
        df["patient_id_archive"] = np.nan
        df["status"] = "missing"
    if hash_csv is not None and Path(hash_csv).exists():
        hs = pd.read_csv(hash_csv)
        df = df.merge(hs[["image", "md5_bytes", "md5_thumb32q4"]], on="image", how="left")
    else:
        df["md5_bytes"] = np.nan
        df["md5_thumb32q4"] = np.nan
    return df


def build_groups(df, adjudicated=None):
    """Connected components over shared component-lesion / archive-lesion / patient / file hashes, plus pairs that a
    visual review of the content-based screen judged to be the same lesion (confirmed or probable)."""
    uf = UnionFind(len(df))
    edges_by_key = {}
    for key in ("lesion_id", "lesion_id_archive", "patient_id_archive", "md5_bytes", "md5_thumb32q4"):
        col = df[key]
        n_edges = 0
        for _, idx in df.index[col.notna() & (col.astype(str) != "")].to_series().groupby(col).groups.items():
            idx = list(idx)
            for j in idx[1:]:
                if uf.find(idx[0]) != uf.find(j):
                    n_edges += 1
                uf.union(idx[0], j)
        edges_by_key[key] = n_edges  # merges contributed beyond the previous keys
    if adjudicated is not None and len(adjudicated):
        pos = {img: i for i, img in enumerate(df["image"].values)}
        n_edges = 0
        for a, b in adjudicated[["image_a", "image_b"]].itertuples(index=False):
            if a in pos and b in pos:
                if uf.find(pos[a]) != uf.find(pos[b]):
                    n_edges += 1
                uf.union(pos[a], pos[b])
        edges_by_key["visual_adjudication"] = n_edges
    roots = np.array([uf.find(i) for i in range(len(df))])
    _, gid = np.unique(roots, return_inverse=True)
    df = df.copy()
    df["group_id"] = ["G%05d" % g for g in gid]
    df["group_size"] = df.groupby("group_id")["image"].transform("size")
    return df, edges_by_key


def split_image_level(df):
    """Stratified image-level split (same calls, same order, same seed as the conventional split)."""
    tv, te = train_test_split(df.index.values, test_size=TEST_SIZE,
                              stratify=df["target"], random_state=SEED)
    tr, va = train_test_split(tv, test_size=VAL_SIZE,
                              stratify=df.loc[tv, "target"], random_state=SEED)
    s = pd.Series("train", index=df.index)
    s.loc[va] = "val"
    s.loc[te] = "test"
    return s


def split_lesion_level(df):
    """Lesion-disjoint 70/10/20 with label stratification (StratifiedGroupKFold)."""
    sgk = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=SEED)
    tv_idx, te_idx = next(sgk.split(df, df["target"], groups=df["group_id"]))
    tv = df.iloc[tv_idx]
    sgk2 = StratifiedGroupKFold(n_splits=int(round(1 / VAL_SIZE)), shuffle=True, random_state=SEED)
    tr_rel, va_rel = next(sgk2.split(tv, tv["target"], groups=tv["group_id"]))
    s = pd.Series("train", index=df.index)
    s.iloc[tv_idx[va_rel]] = "val"
    s.iloc[te_idx] = "test"
    return s


def cv_columns(df, k=5):
    """Optional k-fold estimates for one cheap model: lesion-grouped (P2-style) and image-stratified (P1-style).
    Each fold's training part is split again into train/val with the same 12.5 % validation share."""
    cols = {}
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    for f, (tv, te) in enumerate(sgk.split(df, df["target"], groups=df["group_id"])):
        s = pd.Series("train", index=df.index)
        s.iloc[te] = "test"
        tvdf = df.iloc[tv]
        inner = StratifiedGroupKFold(n_splits=int(round(1 / VAL_SIZE)), shuffle=True, random_state=SEED)
        _, va = next(inner.split(tvdf, tvdf["target"], groups=tvdf["group_id"]))
        s.iloc[tv[va]] = "val"
        cols[f"cv{f}_lesion"] = s
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for f, (tv, te) in enumerate(skf.split(df, df["target"])):
        s = pd.Series("train", index=df.index)
        s.iloc[te] = "test"
        _, va = train_test_split(tv, test_size=VAL_SIZE, stratify=df["target"].values[tv], random_state=SEED)
        s.loc[df.index[va]] = "val"
        cols[f"cv{f}_image"] = s
    return pd.DataFrame(cols)


def leakage_table(df, split_col):
    """Share of val/test images whose lesion group also has an image in train."""
    tr_groups = set(df.loc[df[split_col] == "train", "group_id"])
    rows = []
    for part in ("val", "test"):
        sub = df[df[split_col] == part]
        leak = sub["group_id"].isin(tr_groups)
        rows.append({"protocol": split_col, "partition": part, "class": "ALL",
                     "n": int(len(sub)), "n_leaked": int(leak.sum()),
                     "pct_leaked": float(100 * leak.mean()) if len(sub) else 0.0})
        for c in CLASS_NAMES:
            m = sub["label"] == c
            rows.append({"protocol": split_col, "partition": part, "class": c,
                         "n": int(m.sum()), "n_leaked": int(leak[m].sum()),
                         "pct_leaked": float(100 * leak[m].mean()) if m.any() else 0.0})
    return pd.DataFrame(rows)


def official_test(test_gt_csv, test_meta_csv, archive_csv, train_df):
    te = pd.read_csv(test_gt_csv)
    names = CLASS_NAMES + ["UNK"]
    te["label"] = [names[i] for i in te[names].values.argmax(1)]
    tm = pd.read_csv(test_meta_csv)
    te = te.merge(tm, on="image", how="left")
    if archive_csv is not None and Path(archive_csv).exists():
        ar = pd.read_csv(archive_csv)
        ar = ar[ar["split"] == "test"][["image", "lesion_id_archive", "attribution", "pixels_x", "pixels_y"]]
        te = te.merge(ar.drop_duplicates("image"), on="image", how="left")
        te["source"] = te["attribution"].map({"Hospital Clínic de Barcelona": "Hospital Clinic (BCN)",
                                              "MILK study team": "ViDIR Vienna (HAM-type)"}).fillna("other")
    else:
        te["lesion_id_archive"] = np.nan
        te["source"] = "unknown"
    train_lesions = set(train_df["lesion_id_archive"].dropna().astype(str))
    te["lesion_in_train"] = te["lesion_id_archive"].astype(str).isin(train_lesions)
    te["scored"] = te["score_weight"] > 0
    te["known_class"] = te["label"] != "UNK"
    te["target"] = te["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).fillna(-1).astype(int)
    keep = ["image", "label", "target", "score_weight", "validation_weight", "scored", "known_class",
            "lesion_id_archive", "lesion_in_train", "source", "age_approx", "sex", "anatom_site_general"]
    return te[keep]


def main(argv):
    gt_csv, meta_csv, proj = argv[0], argv[1], Path(argv[2])
    res = proj / "resources"
    archive = res / "isic2019_archive_ids.csv"
    hashes = res / "isic2019_train_hashes.csv"
    df = load_train(gt_csv, meta_csv, archive, hashes)
    adj_path = res / "neardup_adjudication.csv"
    adj = None
    if adj_path.exists():
        adj = pd.read_csv(adj_path)
        adj = adj[adj["decision"].isin(["same_lesion_confirmed", "same_lesion_probable"])]
    df, merges = build_groups(df, adj)
    df["p1_image_level"] = split_image_level(df)
    df["p2_lesion_level"] = split_lesion_level(df)

    cvc = cv_columns(df)
    df = pd.concat([df, cvc], axis=1)

    # hard guarantee: P2 (and every grouped CV fold) is lesion-disjoint
    for col in ["p2_lesion_level"] + [c for c in cvc.columns if c.endswith("_lesion")]:
        g = df.groupby("group_id")[col].nunique()
        assert int((g > 1).sum()) == 0, f"{col} is not group-disjoint"

    keep = ["image", "target", "label", "source", "lesion_id", "lesion_id_archive", "patient_id_archive", "group_id",
            "group_size", "md5_bytes", "p1_image_level", "p2_lesion_level",
            "age_approx", "sex", "anatom_site_general"] + list(cvc.columns)
    df[keep].to_csv(res / "splits_isic2019.csv", index=False)

    off = official_test(res / "ISIC_2019_Test_GroundTruth.csv", res / "ISIC_2019_Test_Metadata.csv",
                        archive, df)
    off.to_csv(res / "official_test.csv", index=False)

    leak = pd.concat([leakage_table(df, "p1_image_level"), leakage_table(df, "p2_lesion_level")])
    # sensitivity: leakage of P1 had only the shipped component lesion id been used for grouping
    naive = df.assign(group_id=df["lesion_id"].where(df["lesion_id"].notna(), "img_" + df["image"]))
    naive_p1 = leakage_table(naive, "p1_image_level").query("partition=='test' and `class`=='ALL'")["pct_leaked"].iloc[0]
    # a lesion-level split built from the shipped id only: how many of its test images are linked to training by the full grouping?
    naive["p2_naive"] = split_lesion_level(naive)
    tr_full = set(df.loc[naive["p2_naive"] == "train", "group_id"])
    naive_p2_residual = float(100 * df.loc[naive["p2_naive"] == "test", "group_id"].isin(tr_full).mean())
    leak.to_csv(res / "leakage_by_class.csv", index=False)

    dup_bytes = df["md5_bytes"].dropna().duplicated(keep=False).sum() if df["md5_bytes"].notna().any() else None
    audit = {
        "n_images": int(len(df)),
        "n_component_lesions": int(df["lesion_id"].nunique()),
        "n_images_without_component_lesion_id": int(df["lesion_id"].isna().sum()),
        "n_archive_lesion_ids": int(df["lesion_id_archive"].nunique()),
        "n_images_without_archive_lesion_id": int(df["lesion_id_archive"].isna().sum()),
        "n_groups": int(df["group_id"].nunique()),
        "n_multi_image_groups": int((df.groupby("group_id").size() > 1).sum()),
        "images_in_multi_image_groups": int((df["group_size"] > 1).sum()),
        "max_group_size": int(df["group_size"].max()),
        "merges_added_by_key": merges,
        "images_in_byte_identical_sets": None if dup_bytes is None else int(dup_bytes),
        "split_sizes": {c: df[c].value_counts().to_dict() for c in ("p1_image_level", "p2_lesion_level")},
        "official_test": {
            "n_total": int(len(off)),
            "n_scored": int(off["scored"].sum()),
            "n_scored_known": int((off["scored"] & off["known_class"]).sum()),
            "n_scored_known_lesion_in_train": int((off["scored"] & off["known_class"] & off["lesion_in_train"]).sum()),
            "n_with_archive_lesion_id": int(off["lesion_id_archive"].notna().sum()),
        },
        "leakage_p1_test_pct": float(leak.query("protocol=='p1_image_level' and partition=='test' and `class`=='ALL'")["pct_leaked"].iloc[0]),
        "leakage_p2_test_pct": float(leak.query("protocol=='p2_lesion_level' and partition=='test' and `class`=='ALL'")["pct_leaked"].iloc[0]),
        "leakage_p1_test_pct_component_id_only": float(naive_p1),
        "residual_leak_pct_of_component_id_lesion_split": naive_p2_residual,
        "official_test_by_source": off[off["scored"] & off["known_class"]]["source"].value_counts().to_dict(),
    }
    (res / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(pd.crosstab(df["label"], df["p2_lesion_level"]))


if __name__ == "__main__":
    main(sys.argv[1:])
