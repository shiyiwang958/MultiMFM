#!/usr/bin/env python
"""(steps, MC) operating-point grids for the C0-guided DNA demos, at matched NFE.

The value gradient costs ``steps x mc`` network calls, so one NFE budget can be spent many
ways.  ``docs/dmfm_deficit_rescue.md`` established on the Table 1 targets (y* = +-1) that
dMFM prefers one composed flow-map step with MC 32 and GLASS two Euler steps with MC 16, both
at NFE 32.  This module runs the same symmetric grid on *any* C0 target, so the other C0
demos (Fig 7's tilt ladder at target 0.30, Fig 8's single trajectory) can be reported at the
operating point each arm actually wants instead of at 4 steps x MC 8.

It is a thin wrapper around ``dmfm.experiments.rescue_dmfm_glass_c0``: the harness, the
noise/pool sharing, the scale rules and the output format are that module's, unchanged.  The
only addition is the GLASS one-Euler-step configuration, which the matched-NFE grid needs at
the 1 x 32 corner and which no earlier sweep defined.  Nothing on disk is edited: the config
table is extended in this process only.

    python -m dmfm.experiments.operating_point_c0 --targets 0.30 --n_samples 64 \
        --out_dir results/dna/operating_point/fig7_grid --configs \
        diag_fm1@norm@mc=32 dmfm4_fm2@norm@mc=16 dmfm4_fm4@norm@mc=8 dmfm4_fm8@norm@mc=4 \
        glass_euler1@norm@mc=32 glass_euler2@norm@mc=16 glass_euler4@norm@mc=8 glass_euler8@norm@mc=4
"""
from __future__ import annotations

from dmfm.experiments import rescue_dmfm_glass_c0 as rescue

#: Configurations the matched-NFE grid needs and the earlier sweeps did not define.
EXTRA_CONFIGS: dict[str, dict] = {
    # The 1 x MC corner on the GLASS side: one Euler step of the GLASS posterior velocity.
    "glass_euler1": dict(kind="glass", student=None, n_steps=1, end_time=1.0,
                         note="GLASS, 1 Euler step (the 1 x MC corner of the matched-NFE grid)"),
    # Convention 1 (REPRODUCIBILITY_PLAN.md S7 item 8) selects the diagonal student at one
    # composed step and the 4-step ESD student at any other step count; the diagonal student at
    # 2 and 8 steps is therefore *not* convention-compliant and is here only as a control.
    "diag_fm2": dict(kind="dmfm", student="dmfm", n_steps=2, end_time=1.0,
                     note="diagonal student at 2 steps (control, not convention-compliant)"),
    "diag_fm8": dict(kind="dmfm", student="dmfm", n_steps=8, end_time=1.0,
                     note="diagonal student at 8 steps (control, not convention-compliant)"),
}

for _name, _spec in EXTRA_CONFIGS.items():
    rescue.CONFIGS.setdefault(_name, _spec)


def main(argv=None) -> None:
    rescue.main(argv)


if __name__ == "__main__":
    main()
