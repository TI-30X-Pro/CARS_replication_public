#!/usr/bin/env python3
"""
Aggregate paper metrics from replay_results.csv for one or more runs.

CR computation differs by save semantics:
  - CARS (collision_only): HDF5 saves only collision episodes. CR = N_HDF5_keys /
    total_target_adv_pairs in configs/simulation_config.json. All HDF5 episodes are
    collisions (including those with raw_collision=False, which are coordinate
    transform artifacts — NEVER filter them out).
  - SafeSim (all): HDF5 saves every attempted episode. CR = raw_collision.sum()
    / total_episodes (directly from the replay CSV).
  - STRIVE (collision_only): HDF5 saves optimization successes (all collisions).
    CR = N_HDF5_keys / total_attempts — the denominator must come from STRIVE's
    own logs (not automatically inferrable).

Metrics per run (all from replay_results.csv):
  CR            Collision rate (above)
  Hard%         Fraction of FSM-Hard criticality episodes
  TTC_min       Min time-to-collision (mean / median) from FSM row
  FSM_unprev%   UN R157 Annex 5 unpreventable rate
  CCJP_unprev%  R157 Appendix 3 Careful-Competent Japanese Driver unpreventable
  RSS_unprev%   RSS (Mobileye) unpreventable
  Angle_std     Collision-angle std in degrees (geometric diversity)
  Speed_col     Mean target speed at collision (m/s)
  BD_pos%       Fraction with max_braking_deficit > 0

Usage:
  python -m scripts.evaluate aggregate \\
      --runs <run1> <run2> <run3> \\
      --labels "CARS (Ours)" "STRIVE" "SafeSim" \\
      --save_semantics collision_only collision_only all \\
      --total_attempts 679 309 143 \\
      --output_csv paper_metrics.csv \\
      --latex

Tip — for CARS runs, use `config:configs/simulation_config.json` to auto-count pairs:
  --total_attempts "config:configs/simulation_config.json" auto auto
  (auto = use len(replay_results rows), equivalent to trusting HDF5 count)
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd


def _h_crit_from_criticality(crit_arr) -> float:
    """Normalised Shannon entropy of the Easy/Medium/Hard distribution
    in the given array, divided by log(3). Returns 0.0 if the array is empty."""
    if len(crit_arr) == 0:
        return 0.0
    counts = pd.Series(crit_arr).value_counts()
    p = np.array([counts.get(t, 0) for t in ["Easy", "Medium", "Hard"]],
                 dtype=float)
    if p.sum() == 0:
        return 0.0
    pn = p / p.sum()
    pnz = pn[pn > 0]
    if len(pnz) == 0:
        return 0.0
    return float(-(pnz * np.log(pnz)).sum() / np.log(3))


def bootstrap_h_crit_lcb(crit_arr, B: int = 10000, n_seeds: int = 10,
                         alpha: float = 0.05) -> float:
    """95% lower confidence bound of normalised H_crit via non-parametric
    bootstrap. Median across `n_seeds` independent runs (each B reps) to
    suppress discrete-percentile artefacts at small N."""
    if len(crit_arr) == 0:
        return float("nan")
    crit_arr = np.asarray(crit_arr)
    lcbs = []
    for s in range(1, n_seeds + 1):
        rng = np.random.default_rng(s)
        Hs = np.empty(B)
        n = len(crit_arr)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            Hs[b] = _h_crit_from_criticality(crit_arr[idx])
        lcbs.append(float(np.percentile(Hs, alpha * 100)))
    return float(np.median(lcbs))


def count_pairs_from_config(config_path: str) -> int:
    """Sum len(pairs) across all scenes in simulation_config.target_adv_pairs."""
    with open(config_path) as f:
        cfg = json.load(f)
    tap = cfg.get("target_adv_pairs", {})
    total = 0
    for _, pairs in tap.items():
        total += len(pairs) if isinstance(pairs, (dict, list)) else 0
    return total


def resolve_total_attempts(spec: str, run_path: str, csv_path: str) -> int:
    """Resolve the --total_attempts spec for one run.

    Accepted forms:
      "679"                      → literal integer
      "config:PATH"              → count target_adv_pairs in the given JSON
      "auto"                     → len(fsm rows) in the CSV
    """
    if spec == "auto":
        df = pd.read_csv(csv_path)
        return int((df["controller"] == "FSM").sum())
    if spec.startswith("config:"):
        return count_pairs_from_config(spec[len("config:"):])
    return int(spec)


def compute_cr(csv_path: str, hdf5_path: str, total_attempts: int,
               save_semantics: str) -> tuple[int, int, float]:
    """Return (n_collisions, total_attempts, collision_rate)."""
    df = pd.read_csv(csv_path)
    fsm = df[df["controller"] == "FSM"]

    if save_semantics == "collision_only":
        # Every saved episode IS a collision. Use HDF5 key count if available,
        # otherwise fall back to len(fsm).
        if os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, "r") as f:
                n_col = len(list(f.keys()))
        else:
            n_col = len(fsm)
    elif save_semantics == "all":
        # HDF5 includes non-collisions. Use raw_collision sum.
        n_col = int(fsm["raw_collision"].sum())
    else:
        raise ValueError(f"Unknown save_semantics: {save_semantics}")

    cr = n_col / total_attempts if total_attempts > 0 else float("nan")
    return n_col, total_attempts, cr


def compute_controller_unprev(df: pd.DataFrame, controller: str) -> tuple[int, int, float]:
    sub = df[df["controller"] == controller]
    n = len(sub)
    if n == 0:
        return 0, 0, float("nan")
    unprev = int((sub["preventable"] == False).sum())  # noqa: E712
    return unprev, n, unprev / n


def compute_metrics(csv_path: str, hdf5_path: str, total_attempts: int,
                    save_semantics: str, bootstrap_b: int = 10000,
                    bootstrap_seeds: int = 10) -> dict:
    df = pd.read_csv(csv_path)
    fsm = df[df["controller"] == "FSM"]
    if len(fsm) == 0:
        return {}

    # --- CR (method-dependent) ---
    n_col, n_total, cr = compute_cr(csv_path, hdf5_path, total_attempts, save_semantics)

    # --- Per-controller unpreventable rates ---
    fsm_unprev_n, fsm_n, fsm_unprev = compute_controller_unprev(df, "FSM")
    ccjp_unprev_n, ccjp_n, ccjp_unprev = compute_controller_unprev(df, "CC-JP-LaneAware")
    rss_unprev_n, rss_n, rss_unprev   = compute_controller_unprev(df, "RSS")

    # --- Hard% (FSM criticality) ---
    n_fsm = len(fsm)
    hard_n = int((fsm["criticality"] == "Hard").sum())
    med_n  = int((fsm["criticality"] == "Medium").sum())
    easy_n = int((fsm["criticality"] == "Easy").sum())
    hard_pct = hard_n / n_fsm

    # --- H_crit on FSM-preventable subset (NC paper headline metric) ---
    fsm_prev = fsm[fsm["preventable"] == True]  # noqa: E712
    n_fsm_prev = len(fsm_prev)
    crit_prev = fsm_prev["criticality"].values
    hard_n_prev = int((fsm_prev["criticality"] == "Hard").sum())
    med_n_prev  = int((fsm_prev["criticality"] == "Medium").sum())
    easy_n_prev = int((fsm_prev["criticality"] == "Easy").sum())
    h_crit_raw_prev = _h_crit_from_criticality(crit_prev)
    h_crit_lcb = bootstrap_h_crit_lcb(crit_prev, B=bootstrap_b,
                                       n_seeds=bootstrap_seeds)

    # --- TTC_min (FSM row, finite only) ---
    finite_ttc = fsm["min_ttc"].replace([np.inf, -np.inf], np.nan).dropna()
    ttc_mean   = finite_ttc.mean()
    ttc_median = finite_ttc.median()

    # --- Angle diversity ---
    angles_deg = np.degrees(fsm["collision_angle_rad"].dropna().abs())
    angle_std  = float(angles_deg.std()) if len(angles_deg) else float("nan")

    # --- Speed at collision (mean, m/s) ---
    speed_col = float(fsm["speed_at_collision"].dropna().mean())

    # --- Braking deficit > 0 fraction (any point during scenario, per NC paper
    # §2 line 230: "BD turns positive at any point during the scenario").
    # `bd_at_trigger` is kept in the CSV for IEEE-version reaction-boundary
    # semantics but the NC tables read max-over-episode. ---
    bd_col = "max_braking_deficit" if "max_braking_deficit" in fsm.columns else "bd_at_trigger"
    bd = fsm[bd_col].dropna()
    bd_pos_pct = float((bd > 0).mean()) if len(bd) else float("nan")

    return {
        "N_total_attempts": n_total,
        "N_collisions":     n_col,
        "CR":               cr,
        "N_replay_eps":     n_fsm,
        "Hard%":            hard_pct,
        "Hard_n":           hard_n,
        "Med_n":            med_n,
        "Easy_n":           easy_n,
        "N_FSM_prev":       n_fsm_prev,
        "Hard_n_prev":      hard_n_prev,
        "Med_n_prev":       med_n_prev,
        "Easy_n_prev":      easy_n_prev,
        "H_crit_prev":      h_crit_raw_prev,
        "H_crit_LCB":       h_crit_lcb,
        "TTC_mean":         ttc_mean,
        "TTC_median":       ttc_median,
        "FSM_unprev%":      fsm_unprev,
        "FSM_unprev_n":     fsm_unprev_n,
        "CCJP_unprev%":     ccjp_unprev,
        "CCJP_unprev_n":    ccjp_unprev_n,
        "CCJP_n":           ccjp_n,
        "RSS_unprev%":      rss_unprev,
        "RSS_unprev_n":     rss_unprev_n,
        "RSS_n":            rss_n,
        "Angle_std":        angle_std,
        "Speed_col":        speed_col,
        "BD_pos%":          bd_pos_pct,
    }


def compute_per_tier_metrics(csv_path: str) -> dict:
    """Per-FSM-criticality breakdown (Easy/Medium/Hard) for SI Table S1/S2.

    Returns: {tier: {n, TTC_med, v_col, BD_pos%, d_min}} with an extra
    "Overall" key aggregating across all tiers. d_min is `min_distance` from the
    CSV (smallest OBB-inflated separation observed during replay)."""
    df = pd.read_csv(csv_path)
    fsm = df[df["controller"] == "FSM"]
    bd_col = "max_braking_deficit" if "max_braking_deficit" in fsm.columns else "bd_at_trigger"

    out = {}
    for tier in ["Easy", "Medium", "Hard", "Overall"]:
        sub = fsm if tier == "Overall" else fsm[fsm["criticality"] == tier]
        if len(sub) == 0:
            out[tier] = {"n": 0, "TTC_med": float("nan"), "v_col": float("nan"),
                         "BD_pos%": float("nan"), "d_min": float("nan")}
            continue
        ttc = sub["min_ttc"].replace([np.inf, -np.inf], np.nan).dropna()
        bd  = sub[bd_col].dropna()
        out[tier] = {
            "n":        int(len(sub)),
            "TTC_med":  float(ttc.median()) if len(ttc) else float("nan"),
            "v_col":    float(sub["speed_at_collision"].dropna().mean()),
            "BD_pos%":  float((bd > 0).mean()) if len(bd) else float("nan"),
            "d_min":    float(sub["min_distance"].dropna().mean())
                        if "min_distance" in sub.columns else float("nan"),
        }
    return out


def latex_table(rows: list[dict], labels: list[str]) -> str:
    """LaTeX snippet with the three UN R157 controllers (FSM primary, CC-JP/RSS auxiliary).

    CR is intentionally omitted: different generators save HDF5s under different
    semantics (collision-only vs all), making a single CR column misleading. The
    scenario-validity story is told via FSM/CC-JP/RSS unpreventable rates instead.
    """
    header = (
        r"\begin{table*}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Adversarial scenario generation comparison under the UN R157 "
        r"responsibility-sensitive driver models. FSM (Annex 5) is the primary "
        r"careful and competent driver model; CC-JP (Annex 4 Appendix 3, Japanese "
        r"driver) and RSS (Mobileye) provide complementary lower and upper bounds. "
        r"Hard\% and TTC are computed from FSM. "
        r"$v_\mathrm{col}$ denotes the target speed at collision.}" + "\n"
        r"\label{tab:paper_metrics}" + "\n"
        r"\setlength{\tabcolsep}{4pt}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r"Method & Hard\%$\uparrow$ & TTC$_\mathrm{med}\downarrow$ "
        r"& FSM-UP$\downarrow$ & CC-JP-UP$\downarrow$ & RSS-UP$\downarrow$ & $v_\mathrm{col}$ \\" + "\n"
        r"\midrule"
    )
    body = []
    for label, m in zip(labels, rows):
        if not m:
            body.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        body.append(
            f"{label} & "
            f"{100*m['Hard%']:.1f}\\% & "
            f"{m['TTC_median']:.2f}s & "
            f"{100*m['FSM_unprev%']:.1f}\\% & "
            f"{100*m['CCJP_unprev%']:.1f}\\% & "
            f"{100*m['RSS_unprev%']:.1f}\\% & "
            f"{m['Speed_col']:.2f} m/s \\\\"
        )
    footer = r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table*}"
    return "\n".join([header] + body + [footer])


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to run directories "
                             "(each containing replay_results/replay_results.csv and data.hdf5)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display name for each run (same order as --runs)")
    parser.add_argument("--save_semantics", nargs="+", default=None,
                        choices=None,
                        help="Per-run save semantics: 'collision_only' (CARS/STRIVE) "
                             "or 'all' (SafeSim). Same order as --runs. "
                             "Default: collision_only for every run.")
    parser.add_argument("--total_attempts", nargs="+", default=None,
                        help="Per-run total attempts. Each entry is either an integer "
                             "(e.g. '679'), 'config:PATH.json' (count target_adv_pairs), "
                             "or 'auto' (use len(FSM rows) in CSV). Same order as --runs.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional path to save comparison CSV")
    parser.add_argument("--latex", action="store_true",
                        help="Print a LaTeX table snippet")
    parser.add_argument("--by_criticality", action="store_true",
                        help="Also print per-FSM-tier (Easy/Medium/Hard) metrics "
                             "for SI Tables S1/S2.")
    parser.add_argument("--bootstrap_b", type=int, default=10000,
                        help="Bootstrap resamples for H_crit 95%% LCB (default 10000).")
    parser.add_argument("--bootstrap_seeds", type=int, default=10,
                        help="Independent bootstrap runs (median across seeds) "
                             "to suppress small-N percentile-shelf artefacts (default 10).")


def run(args) -> None:
    n_runs = len(args.runs)
    if len(args.labels) != n_runs:
        raise SystemExit("--runs and --labels must have the same length")

    # Fill save_semantics / total_attempts with defaults if not provided
    sems = args.save_semantics if args.save_semantics else ["collision_only"] * n_runs
    atts = args.total_attempts if args.total_attempts else ["auto"] * n_runs
    if len(sems) != n_runs or len(atts) != n_runs:
        raise SystemExit("--save_semantics and --total_attempts (if given) must match --runs length")

    all_metrics = []
    for run_path, label, sem, att_spec in zip(args.runs, args.labels, sems, atts):
        csv_path  = os.path.join(run_path, "replay_results", "replay_results_laneaware.csv")
        hdf5_path = os.path.join(run_path, "data.hdf5")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.")
            all_metrics.append({})
            continue

        total_attempts = resolve_total_attempts(att_spec, run_path, csv_path)
        m = compute_metrics(csv_path, hdf5_path, total_attempts, sem,
                            bootstrap_b=args.bootstrap_b,
                            bootstrap_seeds=args.bootstrap_seeds)
        all_metrics.append(m)

        print(f"\n=== {label} ({run_path}) ===")
        print(f"  save semantics:     {sem}")
        print(f"  Attempts:           {m['N_total_attempts']}  (bookkeeping only, not in paper table)")
        print(f"  Collisions:         {m['N_collisions']}")
        print(f"  CR:                 {100*m['CR']:.1f}%  (bookkeeping only — not in paper table)")
        print(f"  Replay episodes:    {m['N_replay_eps']}")
        print(f"  Hard%:              {100*m['Hard%']:.1f}%  "
              f"[Hard={m['Hard_n']}, Med={m['Med_n']}, Easy={m['Easy_n']}]")
        print(f"  FSM-preventable:    N={m['N_FSM_prev']}  "
              f"[Hard={m['Hard_n_prev']}, Med={m['Med_n_prev']}, Easy={m['Easy_n_prev']}]")
        print(f"  H_crit (raw, prev): {m['H_crit_prev']:.4f}")
        print(f"  H_crit (95% LCB):   {m['H_crit_LCB']:.4f}  "
              f"(B={args.bootstrap_b}, seeds={args.bootstrap_seeds})")
        print(f"  TTC_min mean/med:   {m['TTC_mean']:.3f}s / {m['TTC_median']:.3f}s")
        print(f"  FSM    unprev:      {m['FSM_unprev_n']}/{m['N_replay_eps']}  ({100*m['FSM_unprev%']:.1f}%)")
        print(f"  CC-JP  unprev:      {m['CCJP_unprev_n']}/{m['CCJP_n']}  ({100*m['CCJP_unprev%']:.1f}%)")
        print(f"  RSS    unprev:      {m['RSS_unprev_n']}/{m['RSS_n']}  ({100*m['RSS_unprev%']:.1f}%)")
        print(f"  Angle std:          {m['Angle_std']:.1f} deg")
        print(f"  Speed@collision:    {m['Speed_col']:.2f} m/s")
        print(f"  Braking deficit >0: {100*m['BD_pos%']:.1f}%")

        if args.by_criticality:
            tier_m = compute_per_tier_metrics(csv_path)
            print(f"\n  --- per-FSM-tier (S1/S2) ---")
            print(f"  {'Tier':<10} {'n':>5} {'TTC_med':>10} {'v_col':>8} "
                  f"{'BD+%':>7} {'d_min':>8}")
            for tier in ["Easy", "Medium", "Hard", "Overall"]:
                t = tier_m[tier]
                ttc_s = f"{t['TTC_med']:.2f}s" if not np.isnan(t['TTC_med']) else "  --"
                vcol_s = f"{t['v_col']:.2f}" if not np.isnan(t['v_col']) else "  --"
                bd_s   = f"{100*t['BD_pos%']:.1f}%" if not np.isnan(t['BD_pos%']) else "  --"
                dmin_s = f"{t['d_min']:.2f}" if not np.isnan(t['d_min']) else "  --"
                print(f"  {tier:<10} {t['n']:>5d} {ttc_s:>10} {vcol_s:>8} "
                      f"{bd_s:>7} {dmin_s:>8}")
            m["per_tier"] = tier_m

    if args.output_csv:
        rows = []
        for label, m in zip(args.labels, all_metrics):
            if m:
                rows.append({"Method": label, **m})
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nComparison table saved → {args.output_csv}")

    if args.latex:
        print("\n" + "=" * 60)
        print("LaTeX table:")
        print("=" * 60)
        print(latex_table(all_metrics, args.labels))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Aggregate paper metrics from replay_results.csv"
    )
    add_args(_parser)
    run(_parser.parse_args())
