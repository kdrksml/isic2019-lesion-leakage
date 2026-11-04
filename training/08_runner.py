# ── 8 · Plan execution ────────────────────────────────────────────────────────
# Order of the stages: seed 42 of the four ablation models and of the six baselines under both splits -> lesion-level
# seeds 43/44 of the ablation models and of the SwinV2-T / ConvNeXtV2-T baselines -> five-fold cross-validation ->
# the 384-px network (seeds 42-44 under the lesion-level split, seed 42 under the image-level split).
# A run whose done.json exists is skipped. The learning-rate grid "G" runs first if results/recipe_lock.json is absent.
PLAN_STAGES = {"final": "B42,R42,Bseeds,Rseeds,CV,C42,Cseeds,Cimage"}
RUN_ORDER = os.environ.get("ISICLEAK_RUN_ORDER", PLAN_STAGES[PLAN]).split(",")
if current_recipe() is None and "G" not in RUN_ORDER:
    RUN_ORDER = ["G"] + RUN_ORDER
print(f"PLAN = {PLAN}: stages {RUN_ORDER}")

def _selection_score(run_dir: Path) -> Optional[float]:
    h = Path(run_dir) / "history.csv"
    if not h.exists():
        return None
    return float(pd.read_csv(h)["val_macro_f1"].tail(CFG.SELECTION_LAST_EPOCHS).mean())

def _run_with_retry(r: Dict[str, Any], max_retries: int = 1) -> None:
    for attempt in range(max_retries + 1):
        try:
            train_run(r)
            return
        except Exception as exc:          # one failure must not stop the plan
            import traceback
            save_json({"error": str(exc), "attempt": attempt, "traceback": traceback.format_exc()},
                      Path(r["dir"]) / "failure.json")
            print(f"[FAIL] {r['protocol']}/{r['name']}/s{r['seed']} (attempt {attempt}): {exc}")
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

# ---- learning-rate grid: learning rate × layer decay on the P2 validation set --------------------
def select_recipe(extra=()):
    """Score = mean P2 validation macro-F1 of the last epochs. Layer decay is adopted only if its best learning rate
    beats the best single-LR run by LLRD_MIN_GAIN; the best LR of the chosen family is locked. If that LR lies on
    the edge of the grid, one more point beyond the edge is requested first. Returns (recipe or None, extra points)."""
    rows = []
    for r in grid_runs(extra):
        sc = _selection_score(r["dir"]) if (Path(r["dir"]) / "done.json").exists() else None
        if sc is None:
            return None, []
        rows.append({"lr": r["lr"], "llrd": r["llrd"], "score": sc})
    tab = pd.DataFrame(rows)
    flat, dec = tab[tab.llrd == 0], tab[tab.llrd > 0]
    fam = dec if (len(dec) and dec.score.max() >= flat.score.max() + CFG.LLRD_MIN_GAIN) else flat
    best = fam.loc[fam.score.idxmax()]
    lrs = sorted(fam.lr.unique())
    if len(lrs) >= 2 and best.lr == lrs[-1] and best.lr < 1e-3:
        return None, [(float(best.lr) * 2, float(best.llrd))]
    if len(lrs) >= 2 and best.lr == lrs[0] and best.lr > 5e-6:
        return None, [(float(best.lr) / 2, float(best.llrd))]
    rec = {"lr": float(best.lr), "llrd": float(best.llrd)}
    save_json({**rec, "score": float(best.score), "rule": "P2 validation, mean of the last epochs",
               "grid": tab.sort_values(["llrd", "lr"]).to_dict("records"), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              RECIPE_LOCK)
    print(tab.sort_values(["llrd", "lr"]).round(4).to_string(index=False))
    print(f"[recipe] locked lr={rec['lr']:g} layer-decay={rec['llrd']:g}")
    return rec, []

def run_grid() -> None:
    extra: List[Tuple[float, float]] = []
    while current_recipe() is None:
        for r in grid_runs(extra):
            if not (Path(r["dir"]) / "done.json").exists():
                _run_with_retry(r)
        rec, more = select_recipe(extra)
        if rec is None and not more:
            print("[recipe] grid incomplete (runs owned by another session or failed) — later stages wait")
            return
        extra += more

# ---- identical configurations are trained once ----------------------------------------------------------------
_ALIAS_KEYS = ("arch", "views", "loss", "recipe", "lr", "llrd", "epochs", "seed", "timm_name", "timm_kwargs")
_CV0_IS_P2 = bool((SPLITS["cv0_lesion"] == SPLITS["p2_lesion_level"]).all())

def alias_source(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The planned run that has exactly the same configuration and data as `r` (or None):
    CV grouped fold 0 ≡ P2 (asserted) ≡ R4 under P2."""
    src = None
    if r["block"] == "CV" and r["protocol"] == "lesion_cv0" and _CV0_IS_P2 and "R4_SwinV2T" in PLAIN_MODELS:
        src = next((x for x in block_r_runs((r["seed"],)) if x["name"] == "R4_SwinV2T" and x["protocol"] == "lesion"),
                   None)
    if src is None or any(src.get(k) != r.get(k) for k in _ALIAS_KEYS):
        return None
    return src

def adopt(r: Dict[str, Any]) -> bool:
    """Copies the predictions of the identical finished run instead of training `r` again (recorded in done.json)."""
    src = alias_source(r)
    if src is None or not (Path(src["dir"]) / "done.json").exists():
        return False
    s, d = Path(src["dir"]), Path(r["dir"])
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted(list(s.glob("*.csv")) + list(s.glob("*.npy"))):
        _copy_atomic(f, d / f.name)
    dj = json.loads((s / "done.json").read_text())
    save_json({**dj, **{k: v for k, v in r.items() if k != "dir"}, "adopted_from": str(s)}, d / "done.json")
    print(f"[adopt] {_run_key(r)} <- {_run_key(src)} (identical configuration and data, not trained twice)")
    return True

def stage_runs(stage: str) -> Optional[List[Dict[str, Any]]]:
    if stage == "R42":
        return block_r_runs((CFG.SEED,))
    if stage == "Rseeds":
        return [r for r in block_r_runs(CFG.EXTRA_SEEDS)
                if r["protocol"] == "lesion" and r["name"] in ("R4_SwinV2T", "R5_ConvNeXtV2T")]
    if stage == "CV":
        return cv_runs()
    if current_recipe() is None:
        print(f"[stage {stage}] waits for the learning-rate grid")
        return None
    if stage == "B42":
        return [r for r in block_b_runs(CFG.LADDER_LOSS) if r["seed"] == CFG.SEED]
    if stage == "Bseeds":
        return [r for r in block_b_runs(CFG.LADDER_LOSS) if r["seed"] != CFG.SEED and r["protocol"] == "lesion"]
    if stage in ("C42", "Cseeds", "Cimage"):
        runs = block_c_runs(CFG.LADDER_LOSS)
        if stage == "C42":
            return [r for r in runs if r["protocol"] == "lesion" and r["seed"] == CFG.SEED]
        if stage == "Cseeds":
            return [r for r in runs if r["protocol"] == "lesion" and r["seed"] != CFG.SEED]
        return [r for r in runs if r["protocol"] == "image"]
    raise ValueError(stage)

def _projected_hours(r: Dict[str, Any]) -> float:
    """Measured duration (GPU pre-flight) or a conservative fallback."""
    meas = json.loads(PREFLIGHT.read_text()).get("models", {}) if PREFLIGHT.exists() else {}
    m = meas.get(_compute_key(r))
    if m:
        return _run_hours(r, m)
    return {"swin:single": 23, "swin:two-view": 39, "convnext:single": 19, "convnext:two-view": 33,
            "convnext384:two-view@384": 80}.get(_compute_key(r), 90) / 60

def fits_budget(r: Dict[str, Any]) -> bool:
    """A run starts only if it is projected to finish before CFG.BUDGET_HOURS after the VM started."""
    if SMOKE or DEVICE.type != "cuda":
        return True
    return session_hours() + _projected_hours(r) <= CFG.BUDGET_HOURS

# ---- GPU pre-flight: real models, real batches, compiled + bf16; measured time per run -> projected plan time ----
PREFLIGHT = RESULTS / "gpu_preflight.json"

def _compute_key(r: Dict[str, Any]) -> str:
    """Runs with the same key cost the same per step (the loss does not change the cost)."""
    if r["arch"] == "plain":
        return f"plain:{r['timm_name']}"
    key = f"{r['arch']}:{'single' if r['views'] == 'single' else 'two-view'}"
    return key if run_size(r) == CFG.IMG_SIZE else f"{key}@{run_size(r)}"

def _all_planned_runs() -> Dict[str, List[Dict[str, Any]]]:
    """Every run of every stage, with a placeholder loss (the cost does not depend on it)."""
    out = {"G": grid_runs()}
    for st in RUN_ORDER:
        if st != "G":
            runs = stage_runs(st)                    # None while the recipe lock is missing (grid first)
            if runs is not None:
                out[st] = runs
    return {k: out[k] for k in RUN_ORDER if k in out}

def _preflight_one(r: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """`steps` optimisation steps exactly as in train_run (same model, loss, views, batch, AMP, compile)."""
    seed_everything(r["seed"])
    tr, va, _ = protocol_frames(r["protocol"])
    counts = np.bincount(tr["target"], minlength=NC).tolist()
    plain = r["recipe"] == "plain"
    model = build_model(r).to(DEVICE)
    loss_fn = make_loss(r["loss"], counts).to(DEVICE)
    ema = None if plain else WeightEMA(model, CFG.EMA_DECAY, CFG.EMA_WARMUP)
    groups = param_groups(model, CFG.PLAIN_WD if plain else CFG.WEIGHT_DECAY, float(r.get("llrd", 0.0)))
    for g in groups:
        g["lr"] = 1e-5 * g.pop("lr_scale", 1.0)
    opt = torch.optim.AdamW(groups, lr=1e-5, fused=DEVICE.type == "cuda")
    views = r["views"]
    spec = loader_spec(r)
    loader = make_loader(tr, train_kind(r), views, train=True, seed=r["seed"], **spec)
    fwd = torch.compile(model) if (CFG.USE_TORCH_COMPILE and DEVICE.type == "cuda") else model
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, t_first, it = [], None, iter(loader)
    model.train()
    for i in range(min(steps, len(loader))):
        batch = next(it)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        loss = training_loss(fwd, loss_fn, batch, views)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"pre-flight: non-finite loss for {_compute_key(r)}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            ema.update(model)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if i == 0:
            t_first = time.time() - t0
        else:
            times.append(time.time() - t0)
    step_s = float(np.median(times[len(times) // 3:])) if times else float(t_first)   # steady state, after compile
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
    eval_model = model if ema is None else ema.ema
    sub = va.head(min(len(va), 512))
    predict(eval_model, sub.head(128), **spec)                                          # warm-up (cuDNN autotune)
    t0 = time.time()
    predict(eval_model, sub, **spec)
    ips = len(sub) / max(time.time() - t0, 1e-6)                                        # eval images per second
    res = {"step_seconds": step_s, "first_step_seconds": float(t_first or 0.0), "steps_per_epoch": len(loader),
           "peak_mem_gb": round(peak, 2), "eval_images_per_s": ips, "loss": float(loss.detach())}
    del model, fwd, opt, ema, loader, it
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return res

def _run_hours(r: Dict[str, Any], m: Dict[str, Any]) -> float:
    n_val, n_test, n_off = 2533, 5067, len(off_eval)
    train_s = r["epochs"] * m["steps_per_epoch"] * m["step_seconds"]
    # per-epoch validation + final weights + best-validation weights
    n_ext = 0
    n_scales = 1 + len(CFG.ZOOM_AREAS) if uses_zoom(r) else 1
    n_tta = (4 * n_scales if uses_zoom(r) else 8) if uses_tta(r) else 0
    eval_imgs = (r["epochs"] * n_val + (n_scales + 1) * n_val + (n_scales + 1 + n_tta) * (n_test + n_off + n_ext))
    # the pre-flight eval rate includes loader start-up; a forward pass costs about a third of a training step, which
    # bounds the rate from below
    ips = max(m["eval_images_per_s"], CFG.BATCH_SIZE / max(m["step_seconds"], 1e-6))
    return (train_s + eval_imgs / ips + 90) / 3600       # + build / compile / file I/O

def gpu_preflight(steps: int = 30, force: bool = False) -> Optional[pd.DataFrame]:
    """Runs once per results folder: trains every distinct model type for a few steps on the GPU (catches OOM,
    compile and bf16 problems before the long plan) and projects the plan duration from the measured speed."""
    plan = _all_planned_runs()
    reps = {}
    for runs in plan.values():
        for r in runs:
            reps.setdefault(_compute_key(r), r)
    prev = json.loads(PREFLIGHT.read_text()) if (PREFLIGHT.exists() and not force) else {}
    meas = dict(prev.get("models", {}))
    todo = {k: r for k, r in reps.items() if k not in meas}
    if todo:
        print(f"=== GPU pre-flight: {len(todo)} model types x {steps} steps (compiled, {CFG.AMP_DTYPE}) ===")
    for k, r in todo.items():
        rr = {**r, "loss": "plain_ce" if r["recipe"] == "plain" else "ce"}
        try:
            meas[k] = m = _preflight_one(rr, steps)
        except Exception as exc:              # a failing model type must not stop the plan before any training
            print(f"  [pre-flight] {k} FAILED: {exc} -> fallback estimate; the run itself reports the error")
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            meas[k] = m = {"step_seconds": _projected_hours(r) * 3600 / (r["epochs"] * 554), "steps_per_epoch": 554,
                           "peak_mem_gb": 0.0, "eval_images_per_s": 1e9, "first_step_seconds": 0.0, "failed": str(exc)}
        print(f"  {k:52s} {m['step_seconds'] * 1000:7.1f} ms/step  {m['peak_mem_gb']:5.1f} GB  "
              f"{m['eval_images_per_s']:6.0f} img/s eval  -> {_run_hours(r, m) * 60:5.1f} min/run", flush=True)
    if todo and DEVICE.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        worst = max(m["peak_mem_gb"] for m in meas.values())
        if worst > 0.92 * total_mem:
            raise RuntimeError(f"pre-flight: peak memory {worst:.1f} GB is too close to the GPU's {total_mem:.1f} GB")
    rows = []
    for stage, runs in plan.items():
        left = [r for r in runs if not (Path(r["dir"]) / "done.json").exists() and alias_source(r) is None]
        rows.append({"stage": stage, "runs": len(runs), "to run": len(left),
                     "hours left": round(sum(_run_hours(r, meas[_compute_key(r)]) for r in left), 2)})
    tab = pd.DataFrame(rows)
    total = float(tab["hours left"].sum())
    print(f"[pre-flight] PLAN = {PLAN} (measured on {prev.get('gpu', ENV.get('gpu')) if not todo else ENV.get('gpu')})")
    print(tab.to_string(index=False))
    print(f"remaining: {total:.1f} GPU-hours; with the {session_hours():.1f} h already used this session the plan "
          f"ends after about {session_hours() + total:.1f} h (budget {CFG.BUDGET_HOURS:.0f} h; runs that would end "
          f"later are not started — the last stage first — and can be completed by running the notebook again)",
          flush=True)
    save_json({"gpu": ENV.get("gpu"), "steps": steps, "batch_per_view": CFG.BATCH_SIZE, "models": meas,
               "plan": PLAN, "stages": rows, "total_hours": total, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              PREFLIGHT)
    return tab

def plan_status() -> pd.DataFrame:
    """One line per stage: finished / total runs (cheap; reads only done.json files)."""
    rows = []
    for stage in RUN_ORDER:
        runs = grid_runs() if stage == "G" else stage_runs(stage)
        if runs is None:
            rows.append({"stage": stage, "done": 0, "total": "waiting"})
            continue
        rows.append({"stage": stage, "done": sum((Path(r["dir"]) / "done.json").exists() for r in runs),
                     "total": len(runs)})
    return pd.DataFrame(rows)

def _plan_pass(order: Sequence[str]) -> None:
    for stage in order:
        if stage == "G":
            print("\n=== learning-rate × layer-decay grid (P2 validation) ===")
            run_grid()
            continue
        runs = stage_runs(stage)
        if runs is None:
            continue
        todo = [r for r in runs if not (Path(r["dir"]) / "done.json").exists()]
        if WORKER_ID % 2:                              # the second process of a VM works from the other end
            todo = todo[::-1]
        print(f"\n=== [process {WORKER_ID}] stage {stage}: {len(runs) - len(todo)}/{len(runs)} done, "
              f"{len(todo)} to run ===", flush=True)
        for r in todo:
            if adopt(r):
                continue
            if not fits_budget(r):
                print(f"[budget] {_run_key(r)} not started: {session_hours():.1f} h used, "
                      f"{_projected_hours(r):.2f} h needed, budget {CFG.BUDGET_HOURS:.0f} h")
                continue
            _run_with_retry(r)

def _n_done() -> int:
    return sum(1 for _ in RESULTS.glob("runs/*/*/seed*/done.json"))

def _held_by_other_sessions() -> List[Path]:
    """Unfinished runs whose claim belongs to another session with a fresh heartbeat."""
    out = []
    for c in RESULTS.glob("runs/*/*/seed*/claim.json"):
        if (c.parent / "done.json").exists():
            continue
        try:
            info = json.loads(c.read_text())
            if info["session"] != SESSION_ID and time.time() - info["heartbeat"] < CFG.LOCK_STALE_MINUTES * 60:
                out.append(c.parent)
        except Exception:
            pass
    return out

def _plan_complete(order: Sequence[str]) -> bool:
    st = plan_status() if list(order) == list(RUN_ORDER) else None
    if st is None:
        return False
    return all(t != "waiting" and int(d) == int(t) for d, t in zip(st["done"], st["total"]))

def execute(order: Sequence[str] = RUN_ORDER, wait_for_others: bool = True, poll_minutes: float = 3.0) -> None:
    """Repeats plan passes until the plan is complete. With two sessions, a session that has nothing left except runs
    held by the other session (or stages waiting for their results) sleeps and continues instead of stopping."""
    t0 = time.time()
    while True:
        before = _n_done()
        _plan_pass(order)
        if _plan_complete(order):
            print("\nplan complete")
            break
        if _n_done() > before:
            continue                                   # progress: new stages may have become available
        held = _held_by_other_sessions()
        if not (wait_for_others and held):
            break                                      # nothing runnable here: failed runs need attention
        print(f"[wait] {len(held)} run(s) in progress in another session; checking again in {poll_minutes:.0f} min")
        time.sleep(poll_minutes * 60)
    print(f"\nexecution finished in {(time.time() - t0) / 3600:.2f} h")
    print(plan_status().to_string(index=False))
    failed = [p.parent for p in RESULTS.glob("runs/*/*/seed*/failure.json") if not (p.parent / "done.json").exists()]
    if failed:
        print(f"{len(failed)} run(s) failed twice; see failure.json, then run this cell again:")
        for d in failed:
            print("   ", d.relative_to(RESULTS))
    locked = [d for d in _held_by_other_sessions()]
    if locked:
        print(f"{len(locked)} unfinished run(s) are still claimed by another session:")
        for d in locked:
            print("   ", d.relative_to(RESULTS))
