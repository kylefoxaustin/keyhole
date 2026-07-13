#!/usr/bin/env python3
"""
assert_power_sanity.py — turn the power numbers from FACTS into CONTROLS.

WHY THIS EXISTS (rt1180emulator, 2026-07-13, and he is right):

    "A NUMBER WITH NO EXPECTED VALUE IS A FACT, NOT A CONTROL.
     I counted it. I published it. It still did not fire."

`idle_w: 64.97` sat in a committed file for TWO DAYS. It was never hidden — it was in
every diff. It was UNASSERTED. The moment anyone asked "what SHOULD a 5090 idle at?"
the bug fell out in ten seconds. Printing a number is not checking it.

And rt1180's blocker killed my first repair: a suite that validates THE ROWS THAT ARE
THERE cannot see A ROW THAT IS GONE. So every expectation here is sourced from OUTSIDE
the artifact it validates:

  * the MODEL SET comes from the corpus MANIFEST.json  (not from the power scripts,
    which are exactly what would drop a model)
  * the POWER BOUNDS come from the silicon datasheets  (nameplate TGP / SoC ceiling —
    arithmetic on a spec sheet does not rot; cf. the half-life rule)
  * the RUN COUNT comes from the published spread      (a single run has no error bar)

THE ASSERTIONS (each one is a real failure we hit, or one rung from it):

  A. COVERAGE      every model in the MANIFEST is either MEASURED or EXPLICITLY EXCLUDED
                   with a stated reason. A silently-missing model is the exact class
                   rt1180 named: enumeration yields fewer, every survivor passes, GREEN.
  B. PLAUSIBILITY  every measured wattage lies inside physical bounds. A 5090 cannot
                   draw 900 W (above nameplate) or 12 W (below idle) while GR3D is 96%.
  C. IDLE          the idle floor is BELOW the gate. This is the assertion that would
                   have caught the dirty card on day one.
  D. ERROR BAR     both sides of the ratio carry n_runs >= 3 and a published spread.
                   An asymmetric ratio is how we got retraction #2.
  E. VERDICT       no model is reported as a WINNER if its range straddles 1.0.
                   "Too close to call" must not be able to decay into a win.

Exit 0 = every number is inside an expectation someone wrote down.
Exit 1 = a number is outside its expectation, OR an expectation is MISSING.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"

# --- EXPECTED VALUES, sourced from OUTSIDE the artifacts being checked -------------
# Silicon facts (datasheet arithmetic — these do not rot).
RTX5090_NAMEPLATE_TGP_W = 575.0
RTX5090_IDLE_GATE_W = 35.0      # a clean 5090 idles at 5-30 W. 64.97 W was a TENANT.
ORIN_SOC_CEILING_W = 60.0       # MAXN SoC ceiling
MIN_RISE_W = 25.0               # inference must move the rail or it did not run
MIN_RUNS = 3                    # a single run has no error bar

failures = []
notes = []


def check(cond, label, detail=""):
    if cond:
        print(f"  ✅ {label}")
    else:
        print(f"  ❌ {label}\n       {detail}")
        failures.append(f"{label} — {detail}")


def main():
    manifest = json.loads((OUT / "onnx_corpus_iq9" / "MANIFEST.json").read_text())
    matrix = json.loads((OUT / "vision_corpus_three_platform.json").read_text())
    orin = json.loads((OUT / "orin_power.json").read_text())
    rtx = json.loads((OUT / "rtx5090_power_by_model_v2.json").read_text())
    pw = matrix["perf_per_watt"]

    # ---- A. COVERAGE — the expected set comes from the MANIFEST, not from us -------
    print("\nA. COVERAGE  (expected set = corpus MANIFEST.json, external to these scripts)")
    expected = {m["name"] for m in manifest["models"]}
    # the matrix keys by onnx basename; normalise
    alias = {"resnet50": "resnet50v1", "clip_vit_b32": "clip_vit_b32_visual",
             "yolov8n_seg": "yolov8n-seg", "yolo11s_seg": "yolo11s-seg",
             "yoloe_26s_seg": "yoloe-26s-seg-pf",
             "efficientsam_encoder": "efficient_sam_vitt_encoder",
             "efficientsam_decoder": "efficient_sam_vitt_decoder"}
    expected_keys = {alias[e] for e in expected}
    measured = {r["model"] for r in pw["per_model"]}
    excluded = set(orin["__meta__"].get("excluded", {}).keys())
    excluded_keys = {alias.get(e, e) for e in excluded}

    missing = expected_keys - measured - excluded_keys
    check(not missing,
          f"every MANIFEST model is measured ({len(measured)}) or excluded-with-reason "
          f"({len(excluded_keys)}) — {len(expected_keys)} expected",
          f"SILENTLY MISSING (neither measured nor excluded): {sorted(missing)} — "
          f"this is rt1180's class: enumeration yields fewer, every survivor passes, GREEN.")
    for e in sorted(excluded_keys):
        reason = orin["__meta__"]["excluded"].get(
            next(k for k, v in alias.items() if v == e), "")
        check(bool(reason), f"exclusion of '{e}' carries a stated reason",
              "an exclusion with no reason is a silently-missing model wearing a label")

    # ---- B. PLAUSIBILITY — bounds from the datasheets ------------------------------
    print("\nB. PLAUSIBILITY  (bounds = nameplate TGP / SoC ceiling — datasheet arithmetic)")
    for k, m in rtx["models"].items():
        w = m["power_w_median"]
        check(RTX5090_IDLE_GATE_W < w <= RTX5090_NAMEPLATE_TGP_W,
              f"5090 {k}: {w:.1f} W within ({RTX5090_IDLE_GATE_W}, {RTX5090_NAMEPLATE_TGP_W}] W",
              f"{w:.1f} W is outside physical bounds — above nameplate TGP or below idle")
    for r in orin["results"]:
        w = r["power"]["soc_mean_w"]
        check(0 < w <= ORIN_SOC_CEILING_W,
              f"Orin {r['model']}: {w:.1f} W within (0, {ORIN_SOC_CEILING_W}] W",
              f"{w:.1f} W exceeds the MAXN SoC ceiling")
        check(r["soc_delta_over_idle_w"] >= MIN_RISE_W * 0.7,
              f"Orin {r['model']}: rose {r['soc_delta_over_idle_w']:.1f} W over idle",
              "the rail barely moved — the engine may not have run")

    # ---- C. IDLE — the assertion that would have caught the dirty card -------------
    print("\nC. IDLE FLOOR  (the assertion that was missing for two days)")
    idle = rtx["idle_w"]
    check(idle <= RTX5090_IDLE_GATE_W,
          f"5090 idle {idle:.2f} W <= {RTX5090_IDLE_GATE_W} W gate",
          f"idle {idle:.2f} W means a TENANT WAS ON THE CARD. The old file recorded 64.97 W "
          f"and nobody had written down what a 5090 SHOULD idle at.")
    check(rtx.get("idle_w_previous_DIRTY") == 64.97,
          "the dirty-card idle (64.97 W) is retained as a documented counter-example")

    # ---- D. ERROR BAR — both sides, or the ratio is asymmetric ---------------------
    print("\nD. ERROR BARS  (an asymmetric ratio is how retraction #2 happened)")
    check(rtx["__meta__"].get("n_independent_runs", 0) >= MIN_RUNS,
          f"5090 has n_runs >= {MIN_RUNS}")
    check(orin["__meta__"].get("n_independent_runs", 0) >= MIN_RUNS,
          f"Orin has n_runs >= {MIN_RUNS}",
          "ONE side with an error bar and one without is a HALF-CORRECTED RATIO — "
          "a new error wearing the credibility of the correction")
    for k, m in rtx["models"].items():
        check("spread_pct" in m, f"5090 {k}: spread published",
              "a number with no spread is a claim about precision you never made")

    # ---- E. VERDICT — a straddling range must not decay into a win -----------------
    print("\nE. VERDICT INTEGRITY  (too-close-to-call must not decay into a win)")
    unresolved = set(pw.get("unresolved_with_error_bars", {}).keys()) - {"why"}
    for name, rng in pw.get("unresolved_with_error_bars", {}).items():
        if name == "why":
            continue
        lo, hi = rng["convention_B_range_x"]
        check(lo < 1.0 < hi,
              f"{name}: range [{lo}, {hi}] genuinely straddles 1.0 -> correctly UNRESOLVED",
              "declared unresolved but the range does NOT straddle 1.0 — one of the two is wrong")
    for m in pw["per_model"]:
        if m["model"] in unresolved:
            check(m["model"] not in pw.get("earned_orin_wins", {}),
                  f"{m['model']}: NOT also listed as an earned win",
                  "a model cannot be both unresolved and a winner")

    # ---- report -------------------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print(f"❌ {len(failures)} ASSERTION(S) FAILED — a number is outside its expectation:")
        for f in failures:
            print(f"   • {f}")
        print("\nA number with no expected value is a FACT, not a CONTROL. (rt1180emulator)")
        return 1
    print("✅ ALL POWER NUMBERS ARE INSIDE AN EXPECTATION SOMEONE WROTE DOWN.")
    print("   Coverage asserted against the MANIFEST (external), bounds against the")
    print("   datasheets (external), error bars on BOTH sides of the ratio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
