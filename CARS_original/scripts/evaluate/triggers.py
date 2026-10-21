#!/usr/bin/env python3
"""Compute the trigger-mechanism breakdown for unpreventable scenarios.

Reads `replay_results_laneaware.csv` for one or more runs and produces the
per-controller mechanism count + percentage table that backs the paper's
"Trigger mechanisms for unpreventable cases" table (tab:up_mechanisms in
paper/root_nc.tex).

For each controller (FSM, CC-JP-LaneAware, RSS), counts how many of its
unpreventable episodes trigger each of:
  - BD+ during scenario   (max_braking_deficit > 0; "BD turns positive at any
                           point during the scenario", per NC paper §2 BD+%
                           definition)
  - CFS >= 0.9            (max_cfs >= 0.9; FSM only)
  - TTC < 1 s             (min_ttc < 1.0)

The "Any" column is the union of unpreventable episode keys across all three
controllers; for each mechanism it counts how many union episodes trigger that
mechanism under at least one controller.

Usage:
  python -m scripts.evaluate triggers \
      --runs <run_dir> \
      --labels "CARS"
"""

import argparse
import os

import pandas as pd


CONTROLLERS = ["FSM", "CC-JP-LaneAware", "RSS"]


def mech_counts(sub: pd.DataFrame, controller: str = None) -> dict:
    """BD+ uses `max_braking_deficit` ("BD positive at any point during the
    scenario", matching the NC paper §2 BD+% definition). The CSV also carries
    `bd_at_trigger` (BD at reaction boundary) for the IEEE-version semantics,
    but the NC tables read max-over-episode.

    `max_cfs` is only computed by FSMController during replay; CC-JP-LaneAware
    and RSS rows have a constant 0 because their controller doesn't write
    state["_cfs"]. Mark CFS as N/A for non-FSM controllers (display "--")
    rather than reporting a misleading zero."""
    bd_col = "max_braking_deficit" if "max_braking_deficit" in sub.columns else "bd_at_trigger"
    bd  = (sub[bd_col] > 0).fillna(False)
    ttc = (sub["min_ttc"] < 1.0).fillna(False)
    if controller == "FSM" or controller == "Any":
        cfs_count = int((sub["max_cfs"] >= 0.9).fillna(False).sum())
    else:
        cfs_count = None    # display as "--"
    return {
        "BD+":   int(bd.sum()),
        "CFS":   cfs_count,
        "TTC":   int(ttc.sum()),
        "n":     int(len(sub)),
    }


def compute_one_run(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    unprev = {c: df[(df.controller == c) & (~df.preventable)] for c in CONTROLLERS}

    rows = {c: mech_counts(unprev[c], controller=c) for c in CONTROLLERS}

    # Union of unpreventable episodes across all 3 controllers. CFS for the
    # union is computed against FSM-only rows (CFS is a scenario-geometric
    # quantity and only FSM writes it).
    union_keys = set()
    for c in CONTROLLERS:
        union_keys |= set(unprev[c].episode.tolist())

    union_rows = df[df.episode.isin(union_keys) & (~df.preventable)]
    fsm_cfs_lookup = (df[df.controller == "FSM"]
                       .drop_duplicates("episode")
                       .set_index("episode")["max_cfs"])
    bd_col = "max_braking_deficit" if "max_braking_deficit" in union_rows.columns else "bd_at_trigger"
    bd_any, cfs_any, ttc_any = 0, 0, 0
    for ep, g in union_rows.groupby("episode"):
        if (g[bd_col] > 0).fillna(False).any():        bd_any += 1
        if float(fsm_cfs_lookup.get(ep, 0.0)) >= 0.9:  cfs_any += 1
        if (g["min_ttc"] < 1.0).fillna(False).any():   ttc_any += 1
    rows["Any"] = {"BD+": bd_any, "CFS": cfs_any, "TTC": ttc_any, "n": len(union_keys)}

    return rows


def print_table(rows: dict, label: str):
    print(f"\n=== Trigger mechanisms — {label} ===")
    cols = ["FSM", "CC-JP-LaneAware", "RSS", "Any"]
    print(f"{'Mechanism':<24}", *(f"{c:>14s}" for c in cols))
    print("-" * 88)
    print(f"{'  N (unprev set)':<24}",
          *(f"{rows[c]['n']:>14d}" for c in cols))
    for mech_key, mech_label in [("BD+",  "BD+ during scenario"),
                                  ("CFS",  "CFS >= 0.9"),
                                  ("TTC",  "TTC < 1 s")]:
        cells = []
        for c in cols:
            v, n = rows[c][mech_key], rows[c]["n"]
            if v is None:
                cells.append("--")
            else:
                pct = 100 * v / n if n else 0.0
                cells.append(f"{v:>3d} ({pct:>4.0f}%)")
        print(f"{mech_label:<24}", *(f"{cell:>14s}" for cell in cells))


def latex_table(rows: dict) -> str:
    n_fsm  = rows["FSM"]["n"]
    n_ccjp = rows["CC-JP-LaneAware"]["n"]
    n_rss  = rows["RSS"]["n"]
    n_any  = rows["Any"]["n"]

    def fmt(v, n):
        if v is None:
            return "--"
        return f"{v}\\,({100*v/n:.0f}\\%)" if n else "--"

    out = []
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\toprule")
    out.append(rf"\textbf{{Mechanism}} & "
               rf"\textbf{{FSM\,({n_fsm})}} & "
               rf"\textbf{{CC-JP\,({n_ccjp})}} & "
               rf"\textbf{{RSS\,({n_rss})}} & "
               rf"\textbf{{Any\,({n_any})}} \\")
    out.append(r"\midrule")
    for mech_key, mech_label in [("BD+",  r"BD$^+$ during scenario"),
                                  ("CFS",  r"CFS\,$\geq$\,0.9"),
                                  ("TTC",  r"TTC\,$<$\,1\,s")]:
        cells = " & ".join(
            fmt(rows[c][mech_key], rows[c]["n"])
            for c in ["FSM", "CC-JP-LaneAware", "RSS", "Any"]
        )
        out.append(f"{mech_label} & {cells} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories containing replay_results/replay_results_laneaware.csv")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display label per run (same order as --runs)")
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX tabular snippet")


def run(args) -> None:
    if len(args.runs) != len(args.labels):
        raise SystemExit("--runs and --labels must have the same length")

    for run_dir, label in zip(args.runs, args.labels):
        csv_path = os.path.join(run_dir, "replay_results", "replay_results_laneaware.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        rows = compute_one_run(csv_path)
        print_table(rows, label)
        if args.latex:
            print(f"\n--- LaTeX ({label}) ---\n")
            print(latex_table(rows))


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Trigger-mechanism breakdown for unpreventable scenarios",
    )
    add_args(_parser)
    run(_parser.parse_args())
