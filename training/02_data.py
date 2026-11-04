# ── 2 · Download (official bucket) + frozen partitions ────────────────────────
S3 = "https://isic-challenge-data.s3.amazonaws.com/2019/"
TRAIN_ZIP, TEST_ZIP = "ISIC_2019_Training_Input.zip", "ISIC_2019_Test_Input.zip"
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_FULL = ["Melanoma", "Melanocytic nevus", "Basal cell carcinoma", "Actinic keratosis",
              "Benign keratosis", "Dermatofibroma", "Vascular lesion", "Squamous cell carcinoma"]
NC = len(CLASS_NAMES)

def _fetch(name: str, dest: Path) -> Path:
    out = dest / name
    if not out.exists():
        print("downloading", name, "...")
        # -c continues a partial file, --tries/--timeout ride out short network drops
        subprocess.run(["wget", "-q", "-c", "--tries=20", "--timeout=60", "--waitretry=10",
                        "-O", str(out) + ".part", S3 + name], check=True)
        Path(str(out) + ".part").rename(out)
    return out

def _unzip_once(zpath: Path, dest: Path, marker: str) -> Path:
    flag = dest / f".{marker}_unzipped"
    if not flag.exists():
        print("unzipping", zpath.name, "...")
        subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(dest)], check=True)
        flag.touch()
        zpath.unlink()          # free local disk
    return dest

def image_dir(root: Path, sub: str) -> Path:
    for cand in (root / sub, root / sub / sub):
        if cand.is_dir() and any(cand.glob("ISIC_*.jpg")):
            return cand
    hit = next(root.rglob("ISIC_*.jpg"), None)
    if hit is None:
        raise FileNotFoundError(f"no ISIC images under {root}")
    return hit.parent

# Image caches (uint8 memory maps, INTER_AREA from the original files): 256 px (≈ 9 GB) for the 256-px runs and a
# 640-px source cache (≈ 56 GB) for the 384-px network (cell 4, cache_for). A cache saved on Drive is copied back
# instead of being rebuilt; the original images are downloaded only when a cache has to be built.
PERSIST_CACHE = (IN_COLAB and not SMOKE) or os.environ.get("ISICLEAK_TEST_PERSIST") == "1"
DRIVE_CACHE = PROJECT / "_image_cache"
LOCAL_CACHE = Path("/content/cache") if IN_COLAB else DATA_DIR / "cache"
HR_SOURCE = 640                                      # = CFG.HR_SOURCE_SIZE (asserted in cell 3)
CACHE_SIZES = (256, HR_SOURCE)

def cache_name(size: int) -> str:
    return f"isic2019_{size}{'_smoke' if SMOKE else ''}"

CACHE_NAME = cache_name(256)
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

def restore_cache(size: int) -> bool:
    """Copies a cache saved on Drive to the local disk (once). True if the local cache is complete afterwards."""
    cn = cache_name(size)
    if PERSIST_CACHE and (DRIVE_CACHE / f"{cn}.json").exists() and not (LOCAL_CACHE / f"{cn}.json").exists():
        t0 = time.time()
        print(f"restoring the {size}-px image cache from Drive (no download needed) ...")
        shutil.copyfile(DRIVE_CACHE / f"{cn}.npy", LOCAL_CACHE / f"{cn}.npy")
        shutil.copyfile(DRIVE_CACHE / f"{cn}.json", LOCAL_CACHE / f"{cn}.json")   # last: marks complete
        print(f"[cache] {size} px restored in {(time.time() - t0) / 60:.1f} min")
    return "train_dir" in _cache_meta_of(size)

def _cache_meta_of(size: int) -> Dict[str, Any]:
    f = LOCAL_CACHE / f"{cache_name(size)}.json"
    return json.loads(f.read_text()) if f.exists() else {}

def ensure_raw_images() -> None:
    """Downloads and unzips the official ISIC 2019 images (Colab; idempotent)."""
    if not IN_COLAB:
        return
    for d, z, sub in ((DATA_DIR / "train", TRAIN_ZIP, "ISIC_2019_Training_Input"),
                      (DATA_DIR / "test", TEST_ZIP, "ISIC_2019_Test_Input")):
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f".{sub}_unzipped").exists():
            _unzip_once(_fetch(z, d), d, sub)

CACHE_READY = restore_cache(HR_SOURCE)              # the source cache holds every image: no download needed
_cache_meta = _cache_meta_of(HR_SOURCE) if CACHE_READY else {}
if IN_COLAB and not SMOKE:                           # fail fast, before any download, if the VM disk is too small
    _need_gb = 5 + 9 + (0 if CACHE_READY else 26 + 58)   # checkpoints | 256-px cache | original images + 640-px cache
    _free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"[disk] {_free_gb:.0f} GB free on the VM, about {_need_gb} GB needed")
    if _free_gb < _need_gb:
        raise RuntimeError(f"the VM disk has {_free_gb:.0f} GB free but the run needs about {_need_gb} GB: choose an "
                           "A100 / H100 High-RAM runtime with a larger disk")
if not CACHE_READY:
    ensure_raw_images()
if CACHE_READY:
    TRAIN_IMG, TEST_IMG = Path(_cache_meta["train_dir"]), Path(_cache_meta["test_dir"])
else:
    TRAIN_IMG = image_dir(DATA_DIR / "train", "ISIC_2019_Training_Input")
    TEST_IMG = image_dir(DATA_DIR / "test", "ISIC_2019_Test_Input")
for _sz in CACHE_SIZES:                              # a restored cache must index the same file paths as the others
    _m = _cache_meta_of(_sz)
    if "train_dir" in _m and (_m["train_dir"], _m["test_dir"]) != (str(TRAIN_IMG), str(TEST_IMG)):
        print(f"[cache] {_sz}-px cache indexes other paths -> rebuilt from the original images")
        (LOCAL_CACHE / f"{cache_name(_sz)}.json").unlink()

# frozen partitions (built once by tools/isic_splits.py from the official CSVs + archive ids + file hashes)
SPLITS = pd.read_csv(RESOURCES / "splits_isic2019.csv")
OFFICIAL = pd.read_csv(RESOURCES / "official_test.csv")
AUDIT = json.loads((RESOURCES / "split_audit.json").read_text(encoding="utf-8"))
assert len(SPLITS) == 25331 and SPLITS["image"].is_unique
for col in ("p1_image_level", "p2_lesion_level"):
    assert set(SPLITS[col]) == {"train", "val", "test"}
crossing = SPLITS.groupby("group_id")["p2_lesion_level"].nunique()
assert int((crossing > 1).sum()) == 0, "lesion group crosses P2 partitions"

SPLITS["filepath"] = [str(TRAIN_IMG / f"{n}.jpg") for n in SPLITS["image"]]
OFFICIAL["filepath"] = [str(TEST_IMG / f"{n}.jpg") for n in OFFICIAL["image"]]
# every scored image is predicted; metrics use the eight known classes
OFFICIAL_EVAL = OFFICIAL[OFFICIAL["scored"]].reset_index(drop=True)

PROTOCOL_COL = {"image": "p1_image_level", "lesion": "p2_lesion_level"}

def protocol_column(protocol: str) -> str:
    """'image' / 'lesion' -> frozen P1 / P2 columns; 'lesion_cv3' / 'image_cv3' -> optional CV fold columns."""
    if protocol in PROTOCOL_COL:
        return PROTOCOL_COL[protocol]
    kind, fold = protocol.split("_cv")
    return f"cv{int(fold)}_{kind}"

def protocol_frames(protocol: str):
    col = protocol_column(protocol)
    parts = {k: SPLITS[SPLITS[col] == k].reset_index(drop=True) for k in ("train", "val", "test")}
    if SMOKE:   # tiny subset of whole lesion groups — pipeline test only
        parts = {k: _smoke_subset(v, SMOKE_CAP[k]) for k, v in parts.items()}
    return parts["train"], parts["val"], parts["test"]

SMOKE_CAP = dict(zip(("train", "val", "test"), map(int, os.environ.get("ISICLEAK_SMOKE_CAP", "24,6,6").split(","))))

def _smoke_subset(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    keep = []
    for _, g in df.groupby("target"):
        sizes = g.groupby("group_id").size()
        order = rng.choice(sizes.index.values, size=len(sizes), replace=False, p=(sizes / sizes.sum()).values)
        n = 0
        for gid in order:
            if n >= cap:
                break
            rows = g.index[g["group_id"] == gid][:3]
            keep.extend(rows)
            n += len(rows)
    return df.loc[sorted(keep)].reset_index(drop=True)

off_eval = OFFICIAL_EVAL if not SMOKE else OFFICIAL_EVAL.groupby("target", group_keys=False).apply(
    lambda g: g.sample(min(len(g), SMOKE_CAP["test"]), random_state=0)).reset_index(drop=True)

if SMOKE:   # only the images of the tiny subsets (main protocols + the first two CV folds)
    _smoke_protocols = ["image", "lesion"] + [f"{k}_cv{i}" for k in ("image", "lesion") for i in (0, 1)]
    _needed = sum([f["filepath"].tolist() for p in _smoke_protocols for f in protocol_frames(p)], [])
else:
    _needed = SPLITS["filepath"].tolist()
_needed = _needed + off_eval["filepath"].tolist()
missing = [] if CACHE_READY else [p for p in _needed if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"{len(missing)} image files missing, e.g. {missing[:3]}")

for proto in ("image", "lesion"):
    tr, va, te = protocol_frames(proto)
    print(f"[{proto:6s}] train={len(tr):6d} val={len(va):5d} test={len(te):5d} "
          f"| test images whose lesion is in train: {te['group_id'].isin(set(tr['group_id'])).mean():.1%}")
_known = off_eval[off_eval["known_class"]]
print(f"[official] scored images = {len(off_eval)} | known-class = {len(_known)} "
      f"(lesion seen in training: {int(_known['lesion_in_train'].sum())}) | UNK = {int((~off_eval['known_class']).sum())}")
