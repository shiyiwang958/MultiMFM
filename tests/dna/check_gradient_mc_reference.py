#!/usr/bin/env python
"""GPU check: the ported gradient-MC code reproduces stored paper-run gradients (L=50).

Reruns the first two probe states of three original runs with their recorded arguments
(only ``--n_probes``, ``--n_repeats``, ``--mc_values`` and ``--bootstrap`` are reduced; every
seed is derived per probe, so probes 0-1 get exactly the original noise):

* ``fig5_motif_L50``   Fig 5 / Tables 14-15, motif reward, GLASS-2048 reference on the base DFM
                       (``workdir/rebuttal_gradient_accuracy_motif``)
* ``fig5_gc_L50``      same for the GC reward (``workdir/rebuttal_gradient_accuracy_gc_pairmetrics``)
* ``table18_motif_L50`` Table 18, 4-step dMFM estimates vs GLASS-128 RK4 reference
                       (``workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/motif``)

and compares with ``tests/dna/reference/gradient_mc_L50.json`` (slices of the stored
``gradient_pairs.pt`` / ``metadata.json``). The Fig 5 cases are run with both
``--glass_end_time 1.0`` (pre-flag behaviour) and ``0.999`` (current script default) to settle
which one the published runs used.

Tolerances. Bit-exact agreement is impossible: every script enables TF32 matmuls, and the
originals ran on H100s. How much that matters was measured on 2026-09-29 (A100, job
49272897): switching TF32 off on the same GPU moves the GLASS-2048 reference gradients by
0.2-0.4 % (relative L2), while switching the GLASS end time from 1.0 to 0.999 moves them by
0.5-1.3 %. The stored Fig 5 references are 0.3-0.6 % from the end-time-1.0 recomputation and
1.2-1.3 % from the 0.999 one (for every probe and both rewards), so the runs integrated to 1.0
(``PAPER_FLAGS["fig5"]["glass_end_time"] = 1.0``). Pass criteria: calibration statistics
within 1e-3, reference gradients within 1 %, finite-MC estimates (MC 1-2, more sensitive)
within 5 %, and for Fig 5 the end-time-1.0 run closer to the stored gradients than the 0.999
run. The Table 18 GLASS-128 RK4 reference (integrated to 0.999 with RK4, where the drift
is stiff) is far more precision-sensitive: for probe 0, TF32 on vs off alone changes its norm
by 35 % (0.0125 vs 0.0168; stored 0.0187), so it is reported but only its direction is
checked (cosine >= 0.99); the Table 18 probe states, noise and calibration must match exactly
and the 4-step dMFM estimates within 5 %.

    python tests/dna/check_gradient_mc_reference.py --out_dir outputs/dna/smoke/gradmc_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REF = Path(__file__).resolve().parent / "reference" / "gradient_mc_L50.json"


def rel_err(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_pairs(run_dir: Path):
    pairs = torch.load(run_dir / "L50" / "gradient_pairs.pt", map_location="cpu", weights_only=False)
    meta = json.loads((run_dir / "L50" / "metadata.json").read_text())
    return pairs, meta


def fig5_argv(run_args: dict, reward: str, out_dir: Path, end_time: float) -> list[str]:
    calib = run_args.get("calibration_samples", run_args.get("gc_calibration_samples"))
    argv = [
        "--output_dir", str(out_dir), "--lengths", "50", "--reward", reward,
        "--seed", str(run_args["seed"]), "--t_eval", str(run_args["t_eval"]),
        "--nfe_probe", str(run_args["nfe_probe"]), "--nfe_calibration", str(run_args["nfe_calibration"]),
        "--calibration_samples", str(calib), "--sample_batch_size", str(run_args["sample_batch_size"]),
        "--nfe_value", str(run_args["nfe_value"]), "--glass_end_time", str(end_time),
        "--reference_mc", str(run_args["reference_mc"]), "--mc_chunk", str(run_args["mc_chunk"]),
        "--z_target", str(run_args["z_target"]), "--reward_beta", str(run_args["reward_beta"]),
        "--reward_scale", str(run_args["reward_scale"]),
        "--n_probes", "2", "--n_repeats", "1", "--mc_values", "1", "2", "--bootstrap", "10",
    ]
    for key in ("motif", "motif2", "motif_tau", "conjunction_tau", "target_percentile"):
        if run_args.get(key) is not None:
            argv += [f"--{key}", str(run_args[key])]
    return argv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tol_calib", type=float, default=1e-3)
    p.add_argument("--tol_reference", type=float, default=1e-2)
    p.add_argument("--tol_estimate", type=float, default=5e-2)
    p.add_argument("--cases", nargs="+", default=["fig5_motif_L50", "fig5_gc_L50", "table18_motif_L50"])
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0

    from dmfm.experiments import ablate_dmfm_one_step_gradient_mc as table18
    from dmfm.experiments import ablate_glass_gradient_mc as fig5

    ref = json.loads(REF.read_text())["cases"]
    out = Path(args.out_dir)
    report: dict[str, dict] = {}
    ok = True

    for case in args.cases:
        r = ref[case]
        if case.startswith("fig5"):
            reward = "motif" if "motif" in case else "gc"
            for end_time in (1.0, 0.999):
                run_dir = out / f"{case}_end{end_time}"
                fig5.main(fig5_argv(r["run_args"], reward, run_dir, end_time))
                pairs, meta = load_pairs(run_dir)
                got_ref = pairs["reference_gradients"][:2].numpy()
                res = {
                    "score_center_rel": rel_err(meta["score_center"], r["score_center"]),
                    "score_std_rel": rel_err(meta["score_std"], r["score_std"]),
                    "reference_grad_rel": rel_err(got_ref, r["reference_gradients"]),
                    "estimate_grad_rel": rel_err(
                        pairs["estimate_gradients"][:2, :, 0].numpy(), r["estimate_gradients_repeat0"]
                    ),
                }
                res["match"] = (res["score_center_rel"] <= args.tol_calib and res["score_std_rel"] <= args.tol_calib
                                and res["reference_grad_rel"] <= args.tol_reference
                                and res["estimate_grad_rel"] <= args.tol_estimate)
                res["reference_grad_rel_per_probe"] = [rel_err(got_ref[i], r["reference_gradients"][i]) for i in range(2)]
                report[f"{case} glass_end_time={end_time}"] = res
        else:
            run_dir = out / case
            ra = r["run_args"]
            table18.main([
                "--output_dir", str(run_dir), "--lengths", "50", "--reward", "motif",
                "--reward_objective", ra.get("reward_objective", "target"), "--seed", str(ra["seed"]),
                "--t_eval", str(ra["t_eval"]), "--nfe_calibration", str(ra["nfe_calibration"]),
                "--calibration_samples", str(ra["calibration_samples"]),
                "--sample_batch_size", str(ra["sample_batch_size"]),
                "--reference_mc", str(ra["reference_mc"]), "--reference_sampler", ra["reference_sampler"],
                "--reference_glass_steps", str(ra["reference_glass_steps"]),
                "--reference_glass_solver", ra["reference_glass_solver"],
                "--glass_end_time", str(ra["glass_end_time"]), "--dmfm_sampler", ra["dmfm_sampler"],
                "--dmfm_steps", str(ra["dmfm_steps"]), "--dmfm_end_time", str(ra["dmfm_end_time"]),
                "--mc_values", *[str(m) for m in r["mc_values"]], "--nested_mc_pools",
                # run_metadata.json holds the args of the last array task (L=400, mc_chunk 8);
                # the L=50 task used MC_CHUNK=16 (slurm/evaluate_dmfm4_glass_mae_precision.sbatch).
                "--mc_chunk", "16", "--z_target", str(ra["z_target"]),
                "--reward_beta", str(ra["reward_beta"]), "--reward_scale", str(ra["reward_scale"]),
                "--motif", ra["motif"], "--motif_tau", str(ra["motif_tau"]),
                "--target_percentile", str(ra["target_percentile"]),
                "--n_probes", "2", "--n_repeats", "1", "--bootstrap", "10",
            ])
            pairs, meta = load_pairs(run_dir)
            res = {
                "probe_source_indices_equal": float(meta["probe_source_indices"][:2] != r["probe_source_indices"]),
                "score_center_rel": rel_err(meta["score_center"], r["score_center"]),
                "score_std_rel": rel_err(meta["score_std"], r["score_std"]),
                "reference_grad_rel": rel_err(pairs["reference_gradients"][:2].numpy(), r["reference_gradients"]),
                "estimate_grad_rel": rel_err(pairs["estimate_gradients"][:2, :, 0].numpy(), r["estimate_gradients_repeat0"]),
            }
            got = pairs["reference_gradients"][:2].numpy()
            want = np.asarray(r["reference_gradients"], dtype=np.float64)
            res["reference_cosine_min"] = float(min(
                (got[i] * want[i]).sum() / (np.linalg.norm(got[i]) * np.linalg.norm(want[i])) for i in range(2)))
            res["match"] = (res["probe_source_indices_equal"] == 0.0 and res["score_center_rel"] <= args.tol_calib
                            and res["score_std_rel"] <= args.tol_calib and res["estimate_grad_rel"] <= args.tol_estimate
                            and res["reference_cosine_min"] >= 0.99)
            report[case] = res
            ok &= res["match"]

    print(json.dumps(report, indent=2))
    # Fig 5: the paper setting (end time 1.0) must match, and be closer than 0.999 on every probe.
    for case in [c for c in args.cases if c.startswith("fig5")]:
        one, other = report[f"{case} glass_end_time=1.0"], report[f"{case} glass_end_time=0.999"]
        closer = all(a < b for a, b in zip(one["reference_grad_rel_per_probe"], other["reference_grad_rel_per_probe"]))
        print(f"{case}: end 1.0 matches: {one['match']}; closer to the stored gradients than 0.999 on every probe: {closer}")
        ok &= one["match"] and closer
    (out / "gradient_mc_reference_check.json").write_text(json.dumps(report, indent=2))
    print("GRADMC_CHECK_OK" if ok else "GRADMC_CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
