# ── 3 · Configuration + experiment registry ───────────────────────────────────
@dataclass
class Config:
    SEED: int = 42
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 32                # images per view per optimisation step, identical for all experiments
    NUM_WORKERS: int = max(2, min(12, os.cpu_count() or 4))
    AMP_DTYPE: str = "bfloat16"
    USE_TORCH_COMPILE: bool = True
    # one fixed, fully annealed schedule for every run (no early stopping). Primary model = final-epoch weights;
    # the best-validation checkpoint is predicted as well (sensitivity analysis).
    EPOCHS: int = 30
    WARMUP_EPOCHS: int = 2
    # shared recipe: published fine-tuning defaults; peak LR and layer decay come from the learning-rate grid
    LR: float = 1e-4                    # fallback only; every shared run uses the LR in results/recipe_lock.json
    WEIGHT_DECAY: float = 0.05
    DROPOUT: float = 0.10
    DROP_PATH_RATE: float = 0.10        # tiny-model fine-tuning value of the Swin / ConvNeXt V2 recipes
    LABEL_SMOOTH: float = 0.10          # used by the shared recipe and the grid (not by the plain recipe)
    FOCAL_GAMMA: float = 1.150476730017008     # focal logit-adjusted loss, values from preliminary experiments
    LA_TAU: float = 0.24798240002047486
    GLC_LOCAL_WEIGHT: float = 0.70      # weight of the supervised loss on the lesion-centred crop
    GLC_CONSISTENCY_WEIGHT: float = 0.25   # weight of the global→local distillation term
    EMA_DECAY: float = 0.9995           # ≈ 2,000-step window (≈ 3.6 epochs at 554 steps/epoch)
    EMA_WARMUP: int = 500
    SELECTION_LAST_EPOCHS: int = 5      # grid score = mean validation macro-F1 of the last 5 epochs
    LADDER_LOSS: str = "focal_la"       # loss of the four ablation models and the 384-px network
    # plain recipe of the six baselines
    PLAIN_LR: float = 1e-4
    PLAIN_WD: float = 0.05
    PLAIN_WARMUP_EPOCHS: int = 2
    # learning-rate × layer-decay grid on the P2 validation set (SwinV2-T, CE + LS, one view)
    GRID_LRS: Tuple[float, ...] = (3e-5, 6e-5, 1.2e-4, 2.4e-4)
    GRID_LLRD: Tuple[float, ...] = (0.0, 0.9)          # 0 = one learning rate for all layers
    GRID_SEED: int = 7                                 # not one of the seeds of the reported runs
    LLRD_MIN_GAIN: float = 0.005                       # layer decay adopted only if it gains at least this much
    EXTRA_SEEDS: Tuple[int, ...] = (43, 44)            # lesion-level replicates of the ablation models and R4/R5
    CV_FOLDS: int = 5                                  # plain SwinV2-T, grouped vs stratified folds
    KEEP_CHECKPOINTS: str = "all"                      # final.pt (fp16) kept for "all" | "block_b" | "none"
    SHOW_TEST_DURING_TRAINING: bool = False            # only validation is printed while runs are going
    LOCK_STALE_MINUTES: float = 20.0                   # a claim without heartbeat for this long is taken over
    CKPT_DRIVE_EVERY: int = 0                          # 0: resume checkpoints stay on the local disk
    WORKERS_PER_GPU: int = 1
    BUDGET_HOURS: float = float(os.environ.get("ISICLEAK_BUDGET_HOURS", "1e9"))   # optional limit per VM session
    HR_SOURCE_SIZE: int = 640            # source cache of the 384-px network (INTER_AREA from the original files)
    ZOOM_AREAS: Tuple[float, ...] = (0.55, 0.30)   # optional multi-scale inference (not used by the reported runs)
    GLC2_SCALE: Tuple[float, float] = (0.25, 0.60)  # GLC-v2: area range of the lesion-centred local view, cut from
                                                    # the 640-px source cache

CFG = Config()
if SMOKE:
    CFG.EPOCHS = int(os.environ.get("ISICLEAK_SMOKE_EPOCHS", "2"))
    CFG.WARMUP_EPOCHS = CFG.PLAIN_WARMUP_EPOCHS = 1
    CFG.SELECTION_LAST_EPOCHS = 1
    CFG.EXTRA_SEEDS = (43,)
    CFG.CV_FOLDS = 2
    CFG.GRID_LRS, CFG.GRID_LLRD = (6e-5, 1.2e-4), (0.0, 0.9)
    CFG.CKPT_DRIVE_EVERY = 1
    CFG.USE_TORCH_COMPILE = False
    CFG.NUM_WORKERS = 0 if not IN_COLAB else 2
    CFG.BATCH_SIZE = int(os.environ.get("ISICLEAK_SMOKE_BATCH", "8"))      # CPU smoke tests only
if os.environ.get("ISICLEAK_LOADER_WORKERS"):
    CFG.NUM_WORKERS = int(os.environ["ISICLEAK_LOADER_WORKERS"])

SWIN = "swinv2_tiny_window16_256.ms_in1k"
CNX = "convnextv2_tiny.fcmae_ft_in22k_in1k"
CNX384 = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"      # same network, ImageNet-22k weights fine-tuned at 384 px
BACKBONES = {"swin": (SWIN, 256), "convnext": (CNX, 256), "convnext384": (CNX384, 384)}

PLAIN_MODELS = {   # six baselines, plain recipe
    "R1_ResNet50":         ("resnet50.a1_in1k", {}),
    "R2_EfficientNetV2S":  ("tf_efficientnetv2_s.in21k_ft_in1k", {}),
    "R3_ViTS16":           ("vit_small_patch16_224.augreg_in21k_ft_in1k", {"img_size": 256}),
    "R4_SwinV2T":          (SWIN, {}),
    "R5_ConvNeXtV2T":      (CNX, {}),
    "R6_MaxViTT":          ("maxvit_rmlp_tiny_rw_256.sw_in1k", {}),
}
if os.environ.get("ISICLEAK_PLAIN_SUBSET"):            # CPU smoke tests only: a subset of the six baselines
    PLAIN_MODELS = {k: PLAIN_MODELS[k] for k in os.environ["ISICLEAK_PLAIN_SUBSET"].split(",")}
LADDER = {   # 2 x 2 ablation: two backbones × {one view, GLC}, shared recipe
    "B1_SwinV2T":             dict(arch="swin",     views="single"),
    "B1c_ConvNeXtV2T":        dict(arch="convnext", views="single"),
    "B2_SwinV2T_GLC":         dict(arch="swin",     views="glc"),
    "B3_ConvNeXtV2T_GLC":     dict(arch="convnext", views="glc"),
}
HR_LADDER = {   # exploratory 384-px network, shared recipe with GLC-v2
    "C2_ConvNeXtV2T_384_GLC2": dict(arch="convnext384", views="glc2", img_size=384),
}
ALL_SEEDS = (CFG.SEED,) + tuple(CFG.EXTRA_SEEDS)

def run_size(run: Dict[str, Any]) -> int:
    """Input resolution of a run."""
    return int(run.get("img_size", CFG.IMG_SIZE))

def uses_zoom(run: Dict[str, Any]) -> bool:
    """Multi-scale inference is not used by the reported runs."""
    return False

def train_kind(run: Dict[str, Any]) -> str:
    """Training transforms: the shared augmentation (shared recipe) or flips only (plain recipe)."""
    return "train" if (run["recipe"] != "plain" or run.get("aug") == "shared") else "plain"

def uses_tta(run: Dict[str, Any]) -> bool:
    """Test-time augmentation is not used by the reported runs."""
    return False

def make_run(block: str, name: str, protocol: str, seed: int, **kw) -> Dict[str, Any]:
    run = dict(block=block, name=name, protocol=protocol, seed=seed, epochs=CFG.EPOCHS, **kw)
    run["dir"] = str(RESULTS / "runs" / protocol / name / f"seed{seed}")
    return run

RECIPE_LOCK = RESULTS / "recipe_lock.json"

def current_recipe() -> Optional[Dict[str, float]]:
    """Peak LR and layer decay selected by the learning-rate grid (None until the grid has finished)."""
    if RECIPE_LOCK.exists():
        j = json.loads(RECIPE_LOCK.read_text())
        return {"lr": float(j["lr"]), "llrd": float(j["llrd"])}
    return None

def _shared(**kw) -> Dict[str, Any]:
    rec = current_recipe() or {"lr": CFG.LR, "llrd": 0.0}
    return dict(recipe="shared", lr=rec["lr"], llrd=rec["llrd"], **kw)

def grid_runs(extra_lrs: Sequence[Tuple[float, float]] = ()) -> List[Dict[str, Any]]:
    pts = [(lr, ld) for ld in CFG.GRID_LLRD for lr in CFG.GRID_LRS] + list(extra_lrs)
    return [make_run("G", f"G_lr{lr:g}_llrd{ld:g}", "lesion", CFG.GRID_SEED, arch="swin", views="single",
                     loss="ce", recipe="shared", lr=lr, llrd=ld) for lr, ld in pts]

def block_b_runs(loss: str) -> List[Dict[str, Any]]:
    """All seed and protocol combinations of the ablation models (seed-major, so seed 42 of every model comes first).

    The runner (08_runner.py) trains seed 42 under both protocols and seeds 43 and 44 under P2 only, as in the paper."""
    return [make_run("B", n, proto, s, **_shared(loss=loss, **spec))
            for s in ALL_SEEDS for proto in ("lesion", "image") for n, spec in LADDER.items()]

def block_c_runs(loss: str) -> List[Dict[str, Any]]:
    """The 384-px network: seeds 42–44 under P2 and seed 42 under P1."""
    runs = [make_run("C", n, "lesion", s, **_shared(loss=loss, **spec)) for s in ALL_SEEDS for n, spec in HR_LADDER.items()]
    return runs + [make_run("C", n, "image", CFG.SEED, **_shared(loss=loss, **spec)) for n, spec in HR_LADDER.items()]

def block_r_runs(seeds: Sequence[int] = ()) -> List[Dict[str, Any]]:
    return [make_run("R", n, proto, s, arch="plain", timm_name=m, timm_kwargs=kw, views="single",
                     loss="plain_ce", recipe="plain", lr=CFG.PLAIN_LR, llrd=0.0)
            for s in (seeds or (CFG.SEED,)) for proto in ("lesion", "image") for n, (m, kw) in PLAIN_MODELS.items()]

def cv_runs() -> List[Dict[str, Any]]:
    m, kw = SWIN, {}                                   # plain SwinV2-T
    return [make_run("CV", "CV_R4_SwinV2T", f"{proto}_cv{k}", CFG.SEED, arch="plain", timm_name=m, timm_kwargs=kw,
                     views="single", loss="plain_ce", recipe="plain", lr=CFG.PLAIN_LR, llrd=0.0)
            for k in range(CFG.CV_FOLDS) for proto in ("lesion", "image")]

print(f"Run definitions (08_runner.py selects the reported runs from these): grid {len(grid_runs())} | "
      f"ablation {len(block_b_runs('x'))} | 384-px {len(block_c_runs('x'))} | baselines {len(block_r_runs(ALL_SEEDS))} | "
      f"CV {len(cv_runs())} | seeds {ALL_SEEDS}")
