# ── 4 · Image cache, augmentation, datasets, loaders ──────────────────────────
from albumentations.pytorch import ToTensorV2

class ImageCache:
    """All images resized once to IMG_SIZE (INTER_AREA) in one uint8 memory map shared by fork workers."""

    def __init__(self, filepaths: Sequence[str], size: int, path: Path):
        self.paths = sorted(set(map(str, filepaths)))
        self.index = {p: i for i, p in enumerate(self.paths)}
        meta = path.with_suffix(".json")
        shape = (len(self.paths), size, size, 3)
        fresh = not (path.exists() and meta.exists()
                     and json.loads(meta.read_text()).get("paths_hash") == self._hash())
        if fresh:
            need = int(np.prod(shape)) * 1.05 + 2e9
            free = shutil.disk_usage(str(path.parent)).free
            if free < need:                   # a full disk would crash the kernel (SIGBUS) while writing the map
                raise OSError(f"not enough local disk for the {size}-px cache: {free / 1e9:.0f} GB free, "
                              f"{need / 1e9:.0f} GB needed ({path.parent})")
            arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
            def _load(i):
                img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(self.paths[i])
                arr[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
                list(ex.map(_load, range(len(self.paths))))
            arr.flush()
            del arr
            meta.write_text(json.dumps({"paths_hash": self._hash(), "n": len(self.paths)}))
            print(f"[cache] built {len(self.paths)} images in {time.time() - t0:.0f}s -> {path}")
        self.arr = np.load(path, mmap_mode="r")

    def _hash(self) -> str:
        import hashlib
        return hashlib.md5("\n".join(self.paths).encode()).hexdigest()

    def rows(self, filepaths) -> np.ndarray:
        return np.array([self.index[str(p)] for p in filepaths], dtype=np.int64)

    def get(self, row: int) -> np.ndarray:
        return np.array(self.arr[row])          # copy: augmentations may write in place

CACHES: Dict[int, ImageCache] = {}

def open_cache(size: int) -> ImageCache:
    """Opens (restores from Drive, or builds from the original images) the cache of one size, once per kernel."""
    if size not in CACHES:
        cn = cache_name(size)
        if not restore_cache(size):
            missing = [p for p in _needed if not os.path.exists(p)]
            if missing:
                ensure_raw_images()
        CACHES[size] = ImageCache(_needed, size, LOCAL_CACHE / f"{cn}.npy")
        meta_path = LOCAL_CACHE / f"{cn}.json"
        meta = json.loads(meta_path.read_text())
        if "train_dir" not in meta:   # record where the images were, so a restored cache rebuilds the same paths
            meta.update(train_dir=str(TRAIN_IMG), test_dir=str(TEST_IMG))
            meta_path.write_text(json.dumps(meta))
        if size == 256:
            persist_cache(size)       # the 640-px source cache (≈ 56 GB) is not copied to Drive: rebuilt in ≈ 20 min
    return CACHES[size]

def cache_for(size: int) -> ImageCache:
    """256-px runs read the 256-px cache; the 384-px network reads the 640-px source cache and is resized from it
    (INTER_AREA), so its local views keep native-like detail."""
    return open_cache(256 if size <= 256 else HR_SOURCE)

def persist_cache(size: int, margin_gb: float = 10.0) -> None:
    """Copies one cache to Drive once (written once, never overwritten) if Drive keeps `margin_gb` free afterwards."""
    cn = cache_name(size)
    if not PERSIST_CACHE or (DRIVE_CACHE / f"{cn}.json").exists():
        return
    nbytes = (LOCAL_CACHE / f"{cn}.npy").stat().st_size
    try:
        free = shutil.disk_usage(str(DRIVE_CACHE.parent)).free
    except OSError:
        free = None
    if free is not None and free < nbytes * 1.05 + margin_gb * 1e9:
        print(f"[cache] {size}-px cache not saved to Drive: {free / 1e9:.1f} GB free, {nbytes / 1e9:.1f} GB + "
              f"{margin_gb:.0f} GB margin needed (a new VM would download and resize the data again)")
        return
    DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copyfile(LOCAL_CACHE / f"{cn}.npy", DRIVE_CACHE / f"{cn}.npy.tmp")
    os.replace(DRIVE_CACHE / f"{cn}.npy.tmp", DRIVE_CACHE / f"{cn}.npy")
    shutil.copyfile(LOCAL_CACHE / f"{cn}.json", DRIVE_CACHE / f"{cn}.json")      # last: marks the copy complete
    print(f"[cache] {size} px saved to Drive in {(time.time() - t0) / 60:.1f} min ({nbytes / 1e9:.1f} GB): {DRIVE_CACHE}")

# the caches are opened on first use (cache_for): 256 px for the 256-px runs, 640 px for the 384-px network

_TO01 =[A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0), ToTensorV2()]
_REFLECT = cv2.BORDER_REFLECT_101

def _coarse_dropout():
    try:
        return A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.3)
    except TypeError:
        return A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3)

def _rrc(scale, ratio=(0.8, 1.25), S: Optional[int] = None, interp: Optional[int] = None):
    S = S or CFG.IMG_SIZE
    kw = {} if interp is None else {"interpolation": interp}
    try:
        return A.RandomResizedCrop(size=(S, S), scale=scale, ratio=ratio, p=1.0, **kw)
    except TypeError:
        return A.RandomResizedCrop(S, S, scale=scale, ratio=ratio, p=1.0, **kw)

def _coarse_dropout_scaled(S: int):
    """CoarseDropout with hole sizes in proportion to the input (10–20 px at 256 px)."""
    lo, hi = max(1, round(10 * S / 256)), max(2, round(20 * S / 256))
    try:
        return A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(lo, hi), hole_width_range=(lo, hi), p=0.3)
    except TypeError:
        return A.CoarseDropout(max_holes=8, max_height=hi, max_width=hi, p=0.3)

_AREA = cv2.INTER_AREA

def build_transform(kind: str, size: Optional[int] = None) -> A.Compose:
    """size = input resolution of the run. 256 (default): the transforms of the 256-px runs (input = the 256-px
    cache). Larger sizes (384-px network): the same recipe on the 640-px source cache, every down-sizing with area
    interpolation."""
    S = size or CFG.IMG_SIZE
    if S != CFG.IMG_SIZE:                     # 384-px network
        if kind == "train":
            return A.Compose([_rrc((0.6, 1.0), S=S, interp=_AREA), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                              A.Rotate(limit=90, border_mode=_REFLECT, p=0.7),
                              A.GridDistortion(border_mode=_REFLECT, p=0.3),
                              A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.HueSaturationValue(10, 15, 10, p=0.5),
                              _coarse_dropout_scaled(S)] + _TO01)
        if kind == "eval":
            return A.Compose([A.Resize(S, S, interpolation=_AREA)] + _TO01)
        if kind == "local":
            return A.Compose([A.Resize(S, S, interpolation=_AREA), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                              A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03, p=0.8),
                              A.GaussianBlur(blur_limit=(3, 5), p=0.2)] + _TO01)
        raise ValueError(kind)
    if kind == "eval":
        return A.Compose([A.Resize(S, S)] + _TO01)
    if kind == "plain":
        return A.Compose([A.Resize(S, S), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)] + _TO01)
    if kind == "local":                       # applied AFTER the lesion-centred crop (see centred_crop)
        return A.Compose([A.Resize(S, S), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                          A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03, p=0.8),
                          A.GaussianBlur(blur_limit=(3, 5), p=0.2)] + _TO01)
    if kind == "train":                       # shared recipe; scale jitter replaces the inert 2.x ElasticTransform
        return A.Compose([_rrc((0.6, 1.0)), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                          A.Rotate(limit=90, border_mode=_REFLECT, p=0.7), A.GridDistortion(border_mode=_REFLECT, p=0.3),
                          A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.HueSaturationValue(10, 15, 10, p=0.5),
                          _coarse_dropout()] + _TO01)
    raise ValueError(kind)

def centred_crop(img: np.ndarray, rng: np.random.Generator, scale=(0.35, 0.8), centre_sd=0.10) -> np.ndarray:
    """Lesion-centred local view. Dermoscopic images are acquired with the lesion in the centre of the field, so the
    crop centre is drawn around the image centre (SD = 10 % of the side) and the crop covers `scale` of the area
    (GLC: 35–80 % of the 256-px image; GLC-v2: CFG.GLC2_SCALE of the 640-px source image); a crop that misses the
    lesion would otherwise be supervised with the lesion's label."""
    H, W = img.shape[:2]
    area = rng.uniform(*scale) * H * W
    ratio = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
    h = int(min(H, round(math.sqrt(area / ratio))))
    w = int(min(W, round(math.sqrt(area * ratio))))
    cy = H / 2 + rng.normal(0, centre_sd * H)
    cx = W / 2 + rng.normal(0, centre_sd * W)
    y0 = int(np.clip(round(cy - h / 2), 0, H - h))
    x0 = int(np.clip(round(cx - w / 2), 0, W - w))
    return img[y0:y0 + h, x0:x0 + w]

def centre_zoom(img: np.ndarray, area: float) -> np.ndarray:
    """Deterministic centred square crop covering `area` of the image (multi-scale inference)."""
    H, W = img.shape[:2]
    s = int(round(min(H, W) * math.sqrt(area)))
    y0, x0 = (H - s) // 2, (W - s) // 2
    return img[y0:y0 + s, x0:x0 + s]

class SkinDataset(torch.utils.data.Dataset):
    """views: 'single' -> (x, y, i)
              'glc'    -> (x_global, x_local, y, i): the shared augmentation and a lesion-centred local crop of the
                          same 256-px image (GLC; the second view never comes from another image)
              'glc2'   -> the same with the local crop cut from the 640-px source image (GLC-v2): the local
                          view carries native-like detail instead of an enlargement of the low-resolution image.
    size: input resolution. zoom (evaluation only): the input is the centred crop covering this area of the image.
    source_hr (evaluation only): read the 640-px source even for a 256-px model (zoom views of 256-px models)."""

    def __init__(self, df: pd.DataFrame, kind: str, views: str = "single", seed: int = 0, size: Optional[int] = None,
                 zoom: Optional[float] = None, source_hr: bool = False):
        assert views in ("single", "glc", "glc2"), views
        self.size = size or CFG.IMG_SIZE
        assert views != "glc2" or self.size > CFG.IMG_SIZE, "GLC-v2 needs the high-resolution source cache"
        assert zoom is None or (kind == "eval" and views == "single")
        assert not source_hr or kind == "eval"
        self.cache = open_cache(HR_SOURCE) if source_hr else cache_for(self.size)
        self.rows = self.cache.rows(df["filepath"])
        self.y = df["target"].values.astype(np.int64)
        self.views = views
        self.zoom = zoom
        self.tf = (A.Compose([A.Resize(self.size, self.size, interpolation=_AREA)] + _TO01) if source_hr
                   else build_transform(kind, self.size))
        self.tf_local = build_transform("local", self.size) if views != "single" else None
        self.local_scale = CFG.GLC2_SCALE if views == "glc2" else (0.35, 0.8)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        for t, off in ((self.tf, 0), (self.tf_local, 7919)):
            if t is not None and hasattr(t, "set_random_seed"):
                t.set_random_seed((seed + off) % 2 ** 32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self.cache.get(self.rows[i])
        if self.zoom is not None:
            img = centre_zoom(img, self.zoom)
        x = self.tf(image=img)["image"]
        y = int(self.y[i])
        if self.views == "single":
            return x, y, i
        x2 = self.tf_local(image=centred_crop(img, self.rng, scale=self.local_scale))["image"]
        return x, x2, y, i

def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32           # differs per worker, reproducible from the loader generator
    np.random.seed(s)
    random.seed(s)
    ds = torch.utils.data.get_worker_info().dataset
    ds.reseed(s)                                  # albumentations 2.x keeps its own RNG: reseed it per worker

def make_loader(df: pd.DataFrame, kind: str, views: str = "single", train: bool = False,
                seed: int = 0, persistent: bool = False, size: Optional[int] = None,
                zoom: Optional[float] = None, source_hr: bool = False) -> torch.utils.data.DataLoader:
    ds = SkinDataset(df, kind, views, seed=seed, size=size, zoom=zoom, source_hr=source_hr)
    nw = CFG.NUM_WORKERS if train else min(CFG.NUM_WORKERS, 6)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=CFG.BATCH_SIZE if train else 128, shuffle=train, drop_last=train,
        num_workers=nw, pin_memory=DEVICE.type == "cuda", persistent_workers=(nw > 0 and (train or persistent)),
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=_worker_init, generator=g,
        timeout=1800 if nw > 0 else 0,
        multiprocessing_context="fork" if (nw > 0 and sys.platform.startswith("linux")) else None)

def dihedral_views(x: torch.Tensor) -> List[torch.Tensor]:
    out = []
    for flip in (False, True):
        v = x.flip(3) if flip else x
        out += [torch.rot90(v, k, dims=(2, 3)) for k in range(4)]
    return out

def flip_views(x: torch.Tensor) -> List[torch.Tensor]:
    """Identity, horizontal, vertical and both flips."""
    return [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]

def to_device(x):
    return x.to(DEVICE, non_blocking=True)

def loader_spec(run: Dict[str, Any]) -> Dict[str, Any]:
    """Input resolution of a run."""
    return {"size": run_size(run)}

print("transforms ready; open caches:", {s: len(c.paths) for s, c in CACHES.items()})
