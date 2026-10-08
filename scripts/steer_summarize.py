"""Tabulate steer_search summary.json files.

    python3 summarize.py [RUN_DIR ...]      # all runs, every sampler row
    python3 summarize.py --final [RUN_DIR]  # final_endpoint rows only, ranked

Selection target: lowest oracle MAE (over PB-valid molecules) subject to
final-endpoint PB validity in the ~91% band, matching the paper's reported
91.3% for alpha.
"""
import glob
import json
import os
import sys

PB_LO, PB_HI = 0.900, 0.930

args = [a for a in sys.argv[1:] if not a.startswith("--")]
final_only = "--final" in sys.argv
ss_mode = "--ss" in sys.argv
# Default: every run directory under the current working directory.
dirs = args or sorted(d for d in glob.glob("*/") if os.path.exists(f"{d}/summary.json"))
dirs = [d for d in dirs if os.path.exists(f"{d}/summary.json")]

rows = []
for d in dirs:
    meta = json.load(open(f"{d}/summary.json"))
    name = os.path.basename(os.path.normpath(d))
    cfg = meta["args"]
    for e in meta["summary"]:
        rows.append(dict(name=name, run=e["run"], mae=e.get("oracle_mae_pbvalid", float("nan")),
                         mae_all=e["oracle_mae"], pb=e["pb_pb_valid"], n=e.get("n_pbvalid", -1),
                         guide=e["guide_mae"], mu=cfg["mu"], rs=cfg["reward_scale"],
                         gmaxt=cfg["guide_max_t"], gevery=cfg["guide_every"],
                         steps=cfg["sample_steps"], crms=cfg.get("guidance_max_coord_rms", 0.0),
                         smt=cfg.get("select_min_t", 0.3),
                         vs=cfg["value_samples"], nsamp=cfg["num_samples"]))

if ss_mode:
    # Per run: does steer-search beat the final endpoint on EVERY reported number
    # (oracle MAE over PB-valid molecules, and PB validity)? PB is >= by
    # construction; MAE is not, because the two rows average over each
    # selection's own PB-valid population.
    by = {}
    for r in rows:
        by.setdefault(r["name"], {})[r["run"]] = r
    out = []
    for name, d in by.items():
        f, s_ = d.get("final_endpoint"), d.get("best_lookahead_pbvalid")
        if not f or not s_:
            continue
        in_band = PB_LO <= f["pb"] <= PB_HI
        ss_win = s_["mae"] <= f["mae"] and s_["pb"] >= f["pb"]
        out.append((not in_band, not ss_win, s_["mae"] - f["mae"], name, f, s_, in_band, ss_win))
    out.sort()
    print(f"{'run':38s} {'mu':>5s} {'rs':>7s} {'crms':>5s} {'stp':>4s} {'smT':>4s} "
          f"{'endMAE':>9s} {'endPB':>6s} {'ssMAE':>9s} {'ssPB':>6s} {'dMAE':>8s} band ssWin")
    for _, _, dm, name, f, s_, in_band, ss_win in out:
        print(f"{name:38s} {f['mu']:5g} {f['rs']:7g} {f['crms']:5g} {f['steps']:4d} {f['smt']:4g} "
              f"{f['mae']:9.4f} {f['pb']:6.3f} {s_['mae']:9.4f} {s_['pb']:6.3f} {dm:+8.4f} "
              f"{'YES ' if in_band else 'no  '} {'YES' if ss_win else 'no'}")
    sys.exit(0)

hdr = (f"{'run':34s} {'sampler':24s} {'oracleMAE_PB':>12s} {'MAE_all':>8s} {'PB':>6s} "
       f"{'nPB':>5s} {'guideMAE':>8s}")
if final_only:
    sel = [r for r in rows if r["run"] == "final_endpoint"]
    sel.sort(key=lambda r: (not (PB_LO <= r["pb"] <= PB_HI), r["mae"]))
    print(f"ranked by MAE within PB band [{PB_LO}, {PB_HI}]  (* = in band)")
    print(f"{'run':38s} {'mu':>5s} {'rs':>6s} {'gmaxT':>5s} {'ev':>3s} {'stp':>4s} {'crms':>5s} "
          f"{'oracleMAE_PB':>12s} {'PB':>6s} {'nPB':>5s} {'N':>5s}")
    for r in sel:
        mark = "*" if PB_LO <= r["pb"] <= PB_HI else " "
        print(f"{mark}{r['name']:37s} {r['mu']:5g} {r['rs']:6g} {r['gmaxt']:5g} {r['gevery']:3d} "
              f"{r['steps']:4d} {r['crms']:5g} "
              f"{r['mae']:12.3f} {r['pb']:6.3f} {r['n']:5d} {r['nsamp']:5d}")
else:
    print(hdr)
    for r in rows:
        print(f"{r['name']:34s} {r['run']:24s} {r['mae']:12.3f} {r['mae_all']:8.3f} "
              f"{r['pb']:6.3f} {r['n']:5d} {r['guide']:8.3f}")
