#!/usr/bin/env python3
"""Aggregate the Table-2 steering runs and emit the two LaTeX rows.

    python3 scripts/steer_table.py                      # runs named T2_<prop>_s<seed>
    python3 scripts/steer_table.py --prefix T2_ --std   # also print +/- SD

Reads outputs/steer_<prop>/<prefix><prop>_s*/summary.json for the properties listed
in configs/steer/qm9_table2.tsv, averages over whatever seeds are present, and
prints the multiMFM (final endpoint) and multiMFM-SS (steer-search) rows.

MAE is the held-out-ORACLE mean absolute error over PoseBusters-valid molecules
(the paper's convention); the steering gradient uses the separate guide regressor.
Energy properties are stored in Hartree and converted to meV here.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

HA_TO_MEV = 27211.4
# column order of Table 2, with the display scale and number of decimals
COLUMNS = [
    ("cv", 1.0, 2), ("mu", 1.0, 2), ("alpha", 1.0, 2),
    ("gap", HA_TO_MEV, 0), ("homo", HA_TO_MEV, 0), ("lumo", HA_TO_MEV, 0),
]
REPO = Path(__file__).resolve().parents[1]


def collect(prop: str, scale: float, prefix: str) -> dict | None:
    end, ss, epb, spb, seeds = [], [], [], [], []
    for p in sorted(glob.glob(str(REPO / f"outputs/steer_{prop}/{prefix}{prop}_s*/summary.json"))):
        meta = json.load(open(p))
        args, rows = meta["args"], {r["run"]: r for r in meta["summary"]}
        f, s = rows["final_endpoint"], rows["best_lookahead_pbvalid"]
        end.append(f["oracle_mae_pbvalid"] * scale)
        ss.append(s["oracle_mae_pbvalid"] * scale)
        epb.append(f["pb_pb_valid"])
        spb.append(s["pb_pb_valid"])
        seeds.append(args["seed"])
    if not end:
        return None
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    return dict(n=len(end), seeds=seeds,
                end=st.mean(end), end_sd=sd(end), ss=st.mean(ss), ss_sd=sd(ss),
                epb=st.mean(epb), spb=st.mean(spb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="T2_", help="run-directory prefix (default T2_)")
    ap.add_argument("--std", action="store_true", help="include +/- SD in the LaTeX cells")
    args = ap.parse_args()

    data = {p: collect(p, sc, args.prefix) for p, sc, _ in COLUMNS}
    missing = [p for p, d in data.items() if d is None]
    if missing:
        print(f"% WARNING: no runs found for: {', '.join(missing)} "
              f"(expected outputs/steer_<prop>/{args.prefix}<prop>_s*/). "
              f"Run scripts/reproduce_steer_table.sh first.")

    print("% per-property detail (MAE over PB-valid molecules; PB = PoseBusters validity)")
    print(f"% {'prop':6s} {'n':>2s} {'endMAE':>10s} {'endPB':>6s} {'ssMAE':>10s} {'ssPB':>6s}  seeds")
    for prop, _, dec in COLUMNS:
        d = data[prop]
        if d is None:
            continue
        print(f"% {prop:6s} {d['n']:2d} {d['end']:10.3f} {d['epb']:6.3f} "
              f"{d['ss']:10.3f} {d['spb']:6.3f}  {d['seeds']}")

    def cell(prop: str, key: str, dec: int) -> str:
        d = data[prop]
        if d is None:
            return "---"
        v, s = d[key], d[f"{key}_sd"]
        return f"${v:.{dec}f} \\pm {s:.{dec+2}f}$" if args.std else f"${v:.{dec}f}$"

    print()
    for label, key in [("multiMFM (Ours)", "end"), ("multiMFM-SS (Ours)", "ss")]:
        cells = " \n & ".join(cell(p, key, dec) for p, _, dec in COLUMNS)
        print(f" {label}\n & Multimodal Flow\n & {cells} \\\\")
        print()
    print("% Emphasis (bold = best, underline = second best among non-reference rows) is not")
    print("% applied here: it depends on the baseline rows, so set it by hand in the paper.")


if __name__ == "__main__":
    main()
