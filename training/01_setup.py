# ── 1 · Environment, paths and reproducibility ─────────────────────────────────
SMOKE = False          # True → tiny subset, 2 epochs per run, separate results folder (pipeline test only)
PLAN = "final"         # the runs reported in the paper

import os, sys, json, time, math, random, copy, gc, shutil, socket, subprocess, warnings, zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ISICLEAK_COLAB=1 marks the second training process that cell 9 starts on the same Colab VM (it shares the kernel's
# Drive mount and local caches); WORKER_ID 0 is the notebook kernel itself.
IN_COLAB = "google.colab" in sys.modules or os.environ.get("ISICLEAK_COLAB") == "1"
WORKER_ID = int(os.environ.get("ISICLEAK_WORKER_ID", "0"))
if IN_COLAB and not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")

SMOKE = bool(int(os.environ.get("ISICLEAK_SMOKE", int(SMOKE))))
PLAN = os.environ.get("ISICLEAK_PLAN", PLAN)
assert PLAN in ("final",), PLAN
PROJECT = Path(os.environ.get(
    "ISICLEAK_PROJECT",
    "/content/drive/MyDrive/isic2019-lesion-leakage" if IN_COLAB else "."))
DATA_DIR = Path(os.environ.get("ISICLEAK_DATA", "/content/isic2019" if IN_COLAB else "./_data"))
RESOURCES = PROJECT / "resources"
RESULTS = PROJECT / ("results_smoke" if SMOKE else "results")
for p in (RESULTS, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "tools"))   # split construction and audit scripts

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# Software stack of the reported runs (done.json "env"): timm and albumentations are pinned before they are imported,
# so the augmentation and model code match; a different torch build is reported and recorded in every done.json.
PINNED = {"timm": "1.0.30", "albumentations": "2.0.8"}
TORCH_PINNED = "2.11.0"
from importlib import metadata as _md
_missing = []
for _k, _v in PINNED.items():
    try:
        if _md.version(_k) != _v:
            _missing.append(f"{_k}=={_v}")
    except _md.PackageNotFoundError:
        _missing.append(f"{_k}=={_v}")
if _missing and IN_COLAB and not SMOKE:
    print("pinning", _missing)
    _pip(*_missing)
try:
    import timm, albumentations  # noqa: F401
    if tuple(int(x) for x in timm.__version__.split(".")[:2]) < (1, 0):
        raise ImportError
except ImportError:
    _pip("timm>=1.0.9", "albumentations>=1.4")

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# compiled kernels are cached on the local disk and reused by every later run of the same architecture
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/content/inductor_cache" if IN_COLAB else str(Path.home() / ".inductor_cache"))

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# wall-clock start of this Colab VM (kept on its local disk, so a kernel restart does not reset it; a new VM does)
_T0_FILE = Path("/content/.isicleak_t0") if IN_COLAB else DATA_DIR / ".isicleak_t0"
if not _T0_FILE.exists():
    _T0_FILE.write_text(str(time.time()))
SESSION_T0 = float(_T0_FILE.read_text())

def session_hours() -> float:
    return (time.time() - SESSION_T0) / 3600

def seed_everything(seed: int) -> None:
    """Seeds data order, augmentation streams and initialisation. cuDNN autotuning is left on for speed, so GPU
    kernels are not bit-deterministic; run-to-run variation is captured by the replicate seeds instead."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

ENV = {
    "python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
    "albumentations": A.__version__, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
    "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if DEVICE.type == "cuda" else 0,
    "cpus": os.cpu_count(), "smoke": SMOKE,
}
if WORKER_ID == 0:
    save_json(ENV, RESULTS / "environment.json")
print(json.dumps(ENV, indent=2))
if IN_COLAB and not SMOKE and not torch.__version__.startswith(TORCH_PINNED):
    print(f"NOTE: torch {torch.__version__} differs from the {TORCH_PINNED} of the reported runs; the version of "
          "every run is recorded in its done.json")
print("PROJECT :", PROJECT)
print("RESULTS :", RESULTS)
