# ── 5 · Backbones of the shared recipe and plain baselines ──
PRETRAINED = not bool(int(os.environ.get("ISICLEAK_NO_PRETRAINED", "0")))   # offline CPU tests only

class InputNorm(nn.Module):
    """[0,1] RGB -> backbone-specific normalisation (mean/std of the backbone's pre-training)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

def _norm_of(m: nn.Module) -> InputNorm:
    cfg = getattr(m, "pretrained_cfg", None) or {}
    return InputNorm(cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)))

def make_head(d: int, kind: str) -> nn.Module:
    return nn.Linear(d, NC)

class Expert(nn.Module):
    """A pretrained backbone with its OWN pretrained pooling and normalisation (timm forward_head, pre_logits=True).
    `features_only` heads are not used: they drop the final norm, and the raw last-stage activations of SwinV2 /
    ConvNeXtV2 are signed with a few very large channels, which makes generic pooling collapse the embedding."""

    def __init__(self, name: str, head: str):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=PRETRAINED, num_classes=0,
                                          drop_path_rate=CFG.DROP_PATH_RATE)
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        assert cfg.get("architecture", name.split(".")[0]) == name.split(".")[0], f"wanted {name}, got {cfg}"
        print(f"[backbone] {name} (tag {cfg.get('tag', '?')}, pretrained={PRETRAINED})")
        self.norm = _norm_of(self.backbone)
        self.channels = int(self.backbone.num_features)
        self.drop = nn.Dropout(CFG.DROPOUT)
        self.head = make_head(self.channels, head)

    def feature_map(self, x):
        """Last-stage feature map in NCHW (SwinV2 returns NHWC and is permuted)."""
        f = self.backbone.forward_features(self.norm(x))
        if f.ndim == 4 and f.shape[-1] == self.channels and f.shape[1] != self.channels:
            f = f.permute(0, 3, 1, 2)
        return f

    def embed(self, fmap):
        f = fmap.permute(0, 2, 3, 1) if self._nhwc else fmap        # back to the layout forward_head expects
        return self.backbone.forward_head(f, pre_logits=True)

    def forward(self, x):
        fmap = self.feature_map(x)
        e = self.embed(fmap)
        return self.head(self.drop(e)), e, fmap

    @property
    def _nhwc(self) -> bool:
        return "swin" in type(self.backbone).__name__.lower()

class SingleExpertNet(nn.Module):
    def __init__(self, expert: Expert):
        super().__init__()
        self.expert = expert

    def forward(self, x):
        logits, emb, feat = self.expert(x)
        return {"logits": logits, "embedding": emb, "features": feat}

class PlainNet(nn.Module):
    def __init__(self, name: str, kwargs: dict):
        super().__init__()
        self.net = timm.create_model(name, pretrained=PRETRAINED, num_classes=NC, **kwargs)
        print(f"[plain] {name} (pretrained={PRETRAINED})")
        self.norm = _norm_of(self.net)

    def forward(self, x):
        out = self.net.forward_head(self.net.forward_features(self.norm(x)), pre_logits=True)
        return {"logits": self.net.get_classifier()(out), "embedding": out}

def build_model(run: Dict[str, Any]) -> nn.Module:
    head = "linear"
    a = run["arch"]
    if a == "plain":
        return PlainNet(run["timm_name"], run.get("timm_kwargs", {}))
    if a not in BACKBONES:
        raise ValueError(a)
    name, native = BACKBONES[a]
    assert run_size(run) == native, f"{a} expects {native}-px inputs, run asks for {run_size(run)}"
    return SingleExpertNet(Expert(name, head=head))

def _no_decay_names(model: nn.Module) -> set:
    """Parameter names without weight decay: 1-D tensors (norms, biases, scales), timm's no_weight_decay() entries
    (parameter or module names, e.g. SwinV2 relative-position MLPs and logit scales)."""
    skip = set()
    for mname, mod in model.named_modules():
        if hasattr(mod, "no_weight_decay") and callable(mod.no_weight_decay):
            try:
                skip |= {f"{mname}.{n}" if mname else n for n in mod.no_weight_decay()}
            except Exception:
                pass
    return {n for n, p in model.named_parameters()
            if p.ndim <= 1 or n.endswith("logit_scale") or any(n == s or n.startswith(s + ".") for s in skip)}

def param_groups(model: nn.Module, wd: float, llrd: float = 0.0) -> List[Dict[str, Any]]:
    """AdamW groups. With llrd > 0 every backbone gets layer-wise learning-rate decay (timm layer map, one decay per
    block, final layer = 1.0); the classification head always trains at the full learning rate.
    Each group carries 'lr_scale'; the trainer multiplies the peak learning rate by it."""
    nd = _no_decay_names(model)
    groups, seen = [], set()
    if llrd and llrd > 0:
        from timm.optim import param_groups_layer_decay
        for mname, mod in model.named_modules():
            if isinstance(mod, Expert):
                prefix = f"{mname}.backbone."
                local_nd = [n[len(prefix):] for n in nd if n.startswith(prefix)]
                for g in param_groups_layer_decay(mod.backbone, weight_decay=wd, no_weight_decay_list=local_nd,
                                                  layer_decay=llrd):
                    groups.append({"params": g["params"], "weight_decay": g["weight_decay"], "lr_scale": g["lr_scale"]})
                    seen |= {id(p) for p in g["params"]}
    name_of = {id(p): n for n, p in model.named_parameters()}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups.append({"params": [p for p in rest if name_of[id(p)] not in nd], "weight_decay": wd, "lr_scale": 1.0})
    groups.append({"params": [p for p in rest if name_of[id(p)] in nd], "weight_decay": 0.0, "lr_scale": 1.0})
    return [g for g in groups if g["params"]]

def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6

def dummy_input(size: Optional[int] = None, n: int = 1, device=None):
    s = size or CFG.IMG_SIZE
    return torch.zeros(n, 3, s, s, device=device or DEVICE)

def count_gflops(m: nn.Module, size: Optional[int] = None) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = m.eval()
        x = dummy_input(size, device=next(m.parameters()).device)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        return fc.get_total_flops() / 2e9           # multiply-accumulates (fvcore convention, reported as GFLOPs)
    except Exception as e:
        print("[flops] unavailable:", e)
        return None
