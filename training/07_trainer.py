# ── 7 · Trainer ───────────────────────────────────────────────────────────────
AMP = torch.bfloat16 if CFG.AMP_DTYPE == "bfloat16" else torch.float16
def _session_id() -> str:
    """Stable across kernel restarts of the same Colab VM (stored on the local disk), new on a new VM; the second
    training process of the VM (cell 9) is a session of its own (suffix -w1)."""
    f = Path("/content/.isicleak_session") if IN_COLAB else DATA_DIR / ".isicleak_session"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{socket.gethostname()}-{int(time.time())}-{random.randint(0, 10**6)}")
    return f.read_text().strip() + (f"-w{WORKER_ID}" if WORKER_ID else "")

SESSION_ID = _session_id()
PARALLEL_SESSIONS = True      # processes / Colab sessions share the plan: runs are claimed with heartbeat locks
LOCAL_CKPT = Path("/content/ckpt") if IN_COLAB else DATA_DIR / "ckpt"

class WeightEMA:
    def __init__(self, model: nn.Module, decay: float, warmup: int):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.steps = decay, warmup, 0
        self._pairs = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = self.decay * (1.0 - math.exp(-self.steps / max(self.warmup, 1)))
        if self._pairs is None:
            msd, esd = model.state_dict(), self.ema.state_dict()
            fl = [k for k in esd if esd[k].dtype.is_floating_point]
            self._pairs = ([esd[k] for k in fl], [msd[k] for k in fl],
                           [(esd[k], msd[k]) for k in esd if not esd[k].dtype.is_floating_point])
        e_f, m_f, other = self._pairs
        torch._foreach_lerp_(e_f, m_f, 1.0 - d)
        for e, m in other:
            e.copy_(m)

def lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 1e-3):
    """Linear warm-up, then cosine decay to min_ratio × peak (multiplier applied to every group's own peak)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1 + math.cos(math.pi * prog))
    return f

@torch.no_grad()
def predict(model: nn.Module, df: pd.DataFrame, tta: bool = False, want_emb: bool = False, loader=None,
            size: Optional[int] = None, zoom: Optional[float] = None, source_hr: bool = False):
    """Logits of one view. tta: logits averaged over the eight dihedral transforms (256 px) or class probabilities
    averaged over the four flips (384 px; log returned). zoom: the centred crop covering this area of the image.
    source_hr: read the 640-px source. Neither option is used by the reported runs."""
    model.eval()
    loader = loader or make_loader(df, "eval", size=size, zoom=zoom, source_hr=source_hr)
    hr = (size or CFG.IMG_SIZE) > CFG.IMG_SIZE
    logits, embs = [], []
    for x, _, _ in loader:
        x = to_device(x)
        with torch.autocast(DEVICE.type, dtype=AMP, enabled=DEVICE.type == "cuda"):
            if tta and hr:
                p = torch.stack([model(v)["logits"].float().softmax(1) for v in flip_views(x)]).mean(0)
                z = p.clamp_min(1e-12).log()
            elif tta:
                z = torch.stack([model(v)["logits"].float() for v in dihedral_views(x)]).mean(0)
            else:
                out = model(x)
                z = out["logits"].float()
                if want_emb and "embedding" in out:
                    embs.append(out["embedding"].float().cpu().numpy())
        logits.append(z.cpu().numpy())
    return np.concatenate(logits), (np.concatenate(embs) if embs else None)

def macro_f1_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, z.argmax(1), average="macro", labels=list(range(NC)), zero_division=0))

def save_predictions(path: Path, df: pd.DataFrame, z: np.ndarray, group_col: str) -> None:
    out = pd.DataFrame({"image": df["image"].values, "y_true": df["target"].values,
                        "group": df[group_col].astype(str).values})
    for j, c in enumerate(CLASS_NAMES):
        out[f"z_{c}"] = z[:, j].astype(np.float32)
    out.to_csv(path, index=False, float_format="%.5f")

def _atomic_torch_save(obj, path: Path) -> None:
    """Write to a temporary file, then rename: an interrupted write never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = Path(str(dst) + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def _claim(run_dir: Path) -> bool:
    if not PARALLEL_SESSIONS:
        return True
    lock = run_dir / "claim.json"
    if lock.exists():
        try:
            info = json.loads(lock.read_text())
            if info["session"] != SESSION_ID and time.time() - info["heartbeat"] < CFG.LOCK_STALE_MINUTES * 60:
                return False
        except Exception:
            pass
    _heartbeat(run_dir)
    return True

def _heartbeat(run_dir: Path) -> None:
    if PARALLEL_SESSIONS:
        save_json({"session": SESSION_ID, "heartbeat": time.time()}, run_dir / "claim.json")

def _run_key(run: Dict[str, Any]) -> str:
    return f"{run['protocol']}__{run['name']}__s{run['seed']}"

def _pid_alive(pid: int) -> bool:
    if sys.platform.startswith("win"):                          # local tests only (os.kill would terminate)
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                                # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

LOCAL_LOCKS = (Path("/content/locks") if IN_COLAB else DATA_DIR / "locks") / RESULTS.name

def _local_lock(run: Dict[str, Any]) -> Optional[Path]:
    """Atomic lock on the local disk (O_EXCL): two training processes on the same machine never take the same run.
    A lock whose process no longer exists is removed. Returns the lock file, or None if another live process has it."""
    LOCAL_LOCKS.mkdir(parents=True, exist_ok=True)
    f = LOCAL_LOCKS / f"{_run_key(run)}.lock"
    for _ in range(2):
        try:
            fd = os.open(str(f), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return f
        except FileExistsError:
            try:
                pid = int(f.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return None
            f.unlink(missing_ok=True)                           # the owner died: take the run over
    return None

def train_run(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    d = Path(run["dir"])
    if (d / "done.json").exists():
        return json.loads((d / "done.json").read_text())
    lock = _local_lock(run)
    if lock is None:
        print(f"[skip] {_run_key(run)} is being trained by another process on this machine")
        return None
    try:
        if (d / "done.json").exists():                          # finished while this process waited
            return json.loads((d / "done.json").read_text())
        d.mkdir(parents=True, exist_ok=True)
        if not _claim(d):
            print(f"[skip] {_run_key(run)} is being trained by another session")
            return None
        return _train_and_predict(run, d)
    finally:
        lock.unlink(missing_ok=True)

def _log_mean_prob(zs: Sequence[np.ndarray]) -> np.ndarray:
    """log of the mean class probability over several views (scales) of the same images."""
    p = np.mean([np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
                 for z in zs], axis=0)
    return np.log(np.clip(p, 1e-12, 1.0))

def _predict_all(net: nn.Module, d: Path, suffix: str, frames, tta: bool, want_emb: bool,
                 val_loader, spec: Optional[Dict[str, Any]] = None, zoom: bool = False) -> Dict[str, np.ndarray]:
    """Logits of one set of weights on validation, internal test and every scored official image
    -> pred_<split><suffix>.csv. Optional multi-scale (zoom) and test-time augmented (tta) predictions are written
    as pred_<split>_zoom.csv and pred_<split>_tta.csv; the reported runs use neither."""
    out = {}
    spec = spec or {}
    for split, frame, gcol in frames:
        z, e = predict(net, frame, want_emb=want_emb, loader=val_loader if split == "val" else None, **spec)
        save_predictions(d / f"pred_{split}{suffix}.csv", frame, z, gcol)
        out[split] = z
        if want_emb:
            np.save(d / f"emb_{split}.npy", e.astype(np.float16))
        if zoom:
            zz = [z] + [predict(net, frame, zoom=a, **spec)[0] for a in CFG.ZOOM_AREAS]
            save_predictions(d / f"pred_{split}{suffix}_zoom.csv", frame, _log_mean_prob(zz), gcol)
        if tta and split != "val":
            if zoom:
                zt = _log_mean_prob([predict(net, frame, tta=True, zoom=a, **spec)[0]
                                     for a in (None,) + tuple(CFG.ZOOM_AREAS)])
            else:
                zt, _ = predict(net, frame, tta=True, **spec)
            save_predictions(d / f"pred_{split}{suffix}_tta.csv", frame, zt, gcol)
        _heartbeat(d)
    return out

def training_loss(fwd, loss_fn: BaseLoss, batch, views: str) -> torch.Tensor:
    """Loss of one training batch (shared by train_run and the GPU pre-flight, so both measure the same step)."""
    if views == "single":
        x, y, _ = batch
        x, y = to_device(x), y.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, dtype=AMP, enabled=DEVICE.type == "cuda"):
            return supervised_term(loss_fn, fwd(x), y)
    x, x2, y, _ = batch                                         # GLC: full view + lesion-centred crop
    B = x.size(0)
    x = torch.cat([x, x2]).to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    with torch.autocast(DEVICE.type, dtype=AMP, enabled=DEVICE.type == "cuda"):
        z = fwd(x)["logits"]
        return supervised_term(loss_fn, {"logits": z[:B]}, y) + consistency_term(loss_fn, z[:B], z[B:], y)

def _train_and_predict(run: Dict[str, Any], d: Path) -> Dict[str, Any]:
    save_json(run, d / "run.json")
    seed_everything(run["seed"])
    tr, va, te = protocol_frames(run["protocol"])
    counts = np.bincount(tr["target"], minlength=NC).tolist()
    plain = run["recipe"] == "plain"
    model = build_model(run).to(DEVICE)
    loss_fn = make_loss(run["loss"], counts).to(DEVICE)
    ema = None if plain else WeightEMA(model, CFG.EMA_DECAY, CFG.EMA_WARMUP)
    peak = float(run.get("lr", CFG.PLAIN_LR if plain else CFG.LR))
    wd = CFG.PLAIN_WD if plain else CFG.WEIGHT_DECAY
    groups = param_groups(model, wd, float(run.get("llrd", 0.0)))
    for g in groups:
        g["lr"] = peak * g.pop("lr_scale", 1.0)
    opt = torch.optim.AdamW(groups, lr=peak, fused=DEVICE.type == "cuda")
    views = run["views"]
    spec = loader_spec(run)
    train_loader = make_loader(tr, train_kind(run), views, train=True, seed=run["seed"], **spec)
    val_loader = make_loader(va, "eval", persistent=True, **spec)
    spe = len(train_loader)
    epochs = run["epochs"]
    warm = (CFG.PLAIN_WARMUP_EPOCHS if plain else CFG.WARMUP_EPOCHS) * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warm, epochs * spe))

    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "history": [], "train_seconds": 0.0}
    lck = LOCAL_CKPT / RESULTS.name / _run_key(run)
    lck.mkdir(parents=True, exist_ok=True)
    cands = [p for p in (lck / "last.pt", d / "last.pt") if p.exists()]
    if cands:                                  # resume from the most advanced copy (local disk or Drive)
        cks = [(torch.load(p, map_location="cpu", weights_only=False), p) for p in cands]
        ck, src = max(cks, key=lambda t: t[0]["state"]["epoch"])
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.ema.load_state_dict(ck["ema"])
            ema.steps = ck["ema_steps"]
        state = ck["state"]
        best_src = src.parent / "best.pt"
        if best_src.exists() and best_src.parent != lck:
            shutil.copyfile(best_src, lck / "best.pt")
        print(f"[resume] {_run_key(run)} from epoch {state['epoch'] + 1} ({src})")
        del cks, ck
    fwd = torch.compile(model) if (CFG.USE_TORCH_COMPILE and DEVICE.type == "cuda") else model

    t_start = time.time()
    for epoch in range(state["epoch"] + 1, epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch)
        t0, tot = time.time(), 0.0
        for batch in train_loader:
            loss = training_loss(fwd, loss_fn, batch, views)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {run['name']} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot += float(loss.detach())
        eval_model = model if ema is None else ema.ema
        zv, _ = predict(eval_model, va, loader=val_loader)
        f1 = macro_f1_from_logits(zv, va["target"].values)
        state["train_seconds"] += time.time() - t0
        state["history"].append({"epoch": epoch, "train_loss": tot / max(1, spe), "val_macro_f1": f1,
                                 "val_bacc": float(balanced_accuracy_score(va["target"], zv.argmax(1))),
                                 "lr": opt.param_groups[-1]["lr"], "seconds": time.time() - t0})
        improved = f1 > state["best_f1"]
        if improved:
            state.update(best_f1=f1, best_epoch=epoch)
            _atomic_torch_save({k: v.detach().cpu() for k, v in eval_model.state_dict().items()}, lck / "best.pt")
        state["epoch"] = epoch
        print(f"[{_run_key(run)}] ep {epoch:02d}/{epochs} loss {tot / max(1, spe):.4f} "
              f"val-F1 {f1:.4f}{' *' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        _atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ema": ema.ema.state_dict() if ema else None, "ema_steps": ema.steps if ema else 0,
                            "state": state}, lck / "last.pt")
        if CFG.CKPT_DRIVE_EVERY and (epoch % CFG.CKPT_DRIVE_EVERY == 0 or epoch == epochs):   # optional Drive copy
            _copy_atomic(lck / "last.pt", d / "last.pt")
            if (lck / "best.pt").exists():
                _copy_atomic(lck / "best.pt", d / "best.pt")
        _heartbeat(d)
    assert state["epoch"] == epochs, f"{_run_key(run)} stopped at epoch {state['epoch']}"
    pd.DataFrame(state["history"]).to_csv(d / "history.csv", index=False)
    del opt, fwd, train_loader
    gc.collect()

    # ---- primary model: the final-epoch weights (EMA for the shared recipe, raw weights for the plain recipe) ------
    final = model if ema is None else ema.ema
    final.eval()
    keep = CFG.KEEP_CHECKPOINTS == "all" or (CFG.KEEP_CHECKPOINTS == "block_b" and run["block"] == "B")
    if keep:                                              # fp16 halves the Drive space; load_state_dict casts back
        _atomic_torch_save({k: (v.detach().half() if v.is_floating_point() else v.detach()).cpu()
                            for k, v in final.state_dict().items()}, d / "final.pt")
    frames = (("val", va, "group_id"), ("test", te, "group_id"), ("official", off_eval, "lesion_id_archive"))
    want_emb = False
    tta = uses_tta(run)
    preds = _predict_all(final, d, "", frames, tta=tta, want_emb=want_emb, val_loader=val_loader, spec=spec,
                         zoom=uses_zoom(run))
    n_params = count_params(final)
    final_val = macro_f1_from_logits(preds["val"], va["target"].values)
    del model, ema, final
    gc.collect()

    # ---- sensitivity analysis: the best-validation checkpoint ------------------------------------------------------
    best_net = build_model(run).to(DEVICE)
    best_path = lck / "best.pt" if (lck / "best.pt").exists() else d / "best.pt"
    best_net.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds_best = _predict_all(best_net, d, "_best", frames, tta=False, want_emb=False, val_loader=val_loader, spec=spec)
    del best_net

    known = off_eval["known_class"].values
    y_off = off_eval["target"].values[known]
    done = {
        **{k: v for k, v in run.items() if k != "dir"},
        "checkpoint_rule": "final", "peak_lr": peak, "epochs_run": state["epoch"],
        "final_val_macro_f1": final_val,
        "test_macro_f1": macro_f1_from_logits(preds["test"], te["target"].values),
        "official_macro_f1": macro_f1_from_logits(preds["official"][known], y_off),
        "best_epoch": state["best_epoch"], "best_val_macro_f1": state["best_f1"],
        "test_macro_f1_best": macro_f1_from_logits(preds_best["test"], te["target"].values),
        "official_macro_f1_best": macro_f1_from_logits(preds_best["official"][known], y_off),
        "train_minutes": state["train_seconds"] / 60, "wall_minutes": (time.time() - t_start) / 60,
        "params_m": n_params, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "class_counts_train": counts, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session": SESSION_ID, "env": ENV,
    }
    save_json(done, d / "done.json")
    for f in ("last.pt", "best.pt", "failure.json"):          # a successful retry supersedes an earlier failure record
        (d / f).unlink(missing_ok=True)
    shutil.rmtree(lck, ignore_errors=True)
    msg = (f"[done] {_run_key(run)}: final val macro-F1 {final_val:.4f} | best val {state['best_f1']:.4f} "
           f"(epoch {state['best_epoch']})")
    if CFG.SHOW_TEST_DURING_TRAINING:
        msg += f" | test {done['test_macro_f1']:.4f} | official {done['official_macro_f1']:.4f}"
    print(msg + f" ({done['wall_minutes']:.1f} min)", flush=True)
    del val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return done
