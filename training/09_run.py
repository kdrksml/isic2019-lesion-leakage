# ── 9 · Run (re-run until every stage reports "0 to run") ────────────────────
if (DEVICE.type == "cuda" and not SMOKE) or os.environ.get("ISICLEAK_FORCE_PREFLIGHT") == "1":
    gpu_preflight(steps=int(os.environ.get("ISICLEAK_PREFLIGHT_STEPS", "30")))   # once per results folder
execute(RUN_ORDER)
print(plan_status().to_string(index=False))
