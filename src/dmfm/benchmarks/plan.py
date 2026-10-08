"""The budget sweeps and hyperparameter pilots behind Figure 4 (right).

Every method gets its *own* budget knob and its own hyperparameter pilot, so that
no baseline is tuned less carefully than the dMFM:

=========  ==========================  =====================================
method     budget knob (swept)         tuned on the pilot (then frozen)
=========  ==========================  =====================================
dMFM       MC samples per guided step  nothing (the paper's own settings:
                                       guidance_frac 8, cap 10, clip 10,
                                       t in [0.01, 0.95], NFE_value 1)
best-of-N  N                           nothing
FK         particles K                 beta, resample interval
beam       checkpoint interval K       beam width W, branching L
MCTS       iterations                  exploration C, children, interval K
=========  ==========================  =====================================

The pilot runs at ``n_outputs = 32`` with a *different* seed (12345) from the
final sweep (0, 1, 2), so the reported curves are not selected on their own data.

``eta`` (the stochasticity level of the base sampler, see ``core``) is not a tuning
knob: it is fixed by ``calibrate``, which checks that the SDE sampler reproduces
the deterministic sampler's unguided C0 distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Stochasticity level of the base sampler for the branching methods. eta = 0 is the
# deterministic interpolant sampler (best-of-N and dMFM); the value below is
# confirmed against the deterministic sampler's unguided C0 distribution by
# `run_steering.py --mode calibrate` (results/dna/fig4/calibration.json).
DEFAULT_ETA = 1.0


@dataclass(frozen=True)
class Job:
    method: str
    label: str
    config: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.method}__{self.label}"


def _dmfm(mc: int) -> Job:
    return Job("dmfm", f"MC={mc}", {"n_mc": mc, "nfe_value": 1})


def _bon(n: int, eta: float = 0.0) -> Job:
    return Job("best_of_n", f"N={n}", {"n_candidates": n, "eta": eta})


def _fk(k: int, *, beta: float, every: int, eta: float = DEFAULT_ETA) -> Job:
    return Job("fk", f"K={k}", {"n_particles": k, "beta": beta, "resample_every": every, "eta": eta})


def _beam(w: int, l: int, k: int, eta: float = DEFAULT_ETA) -> Job:
    return Job("beam", f"W={w},L={l},K={k}", {"width": w, "branch": l, "checkpoint_every": k, "eta": eta})


def _mcts(it: int, *, children: int, k: int, c: float, eta: float = DEFAULT_ETA) -> Job:
    return Job(
        "mcts",
        f"I={it},children={children},K={k},C={c}",
        {"iterations": it, "children": children, "checkpoint_every": k, "c_uct": c, "eta": eta},
    )


# --------------------------------------------------------------------------- pilot

def _relabel(job: Job, extra: str) -> Job:
    """Pilot labels must be unique: the shard file name is derived from them."""
    return Job(job.method, f"{job.label},{extra}", job.config)


PILOT: list[Job] = (
    [_relabel(_bon(n, eta=e), f"eta={e}") for n in (4, 16) for e in (0.0, DEFAULT_ETA)]
    # FK: every configuration has K = 8 particles, i.e. exactly 768 NFE per output.
    + [
        _relabel(_fk(8, beta=b, every=e), f"beta={b},R={e}")
        for b in (1.0, 5.0, 20.0, 50.0)
        for e in (8, 16, 32)
    ]
    # beam / MCTS: the configurations cost different NFE, so each family is run at
    # three budgets and families are compared at matched NFE (see frozen_from_pilot).
    + [_beam(w, l, k) for (w, l) in ((1, 2), (2, 2), (1, 4)) for k in (48, 16, 8)]
    + [
        _mcts(it, children=c, k=k, c=u)
        for c in (2, 4)
        for k in (16, 24)
        for u in (0.1, 0.5, 1.0)
        for it in (4, 16)
    ]
)

# --------------------------------------------------------------------------- final sweep
# Fallback hyperparameters, used only if no pilot shards are on disk. The reported
# sweep uses :func:`frozen_from_pilot`, i.e. the pilot winners, and
# ``scripts/dna/fig4_freeze_pilot.py`` records the whole grid and the choice in
# ``results/dna/fig4/pilot_choice.json``.
FALLBACK_FROZEN = {
    "fk": {"beta": 20.0, "resample_every": 16},
    "beam": {"width": 1, "branch": 2},
    "mcts": {"children": 4, "checkpoint_every": 16, "c_uct": 0.5},
}

# which config keys are frozen from the pilot; the remaining key is the budget knob
FREEZE_KEYS = {
    "fk": ("beta", "resample_every"),
    "beam": ("width", "branch"),
    "mcts": ("children", "checkpoint_every", "c_uct"),
}


def pilot_dirs() -> list:
    """Every directory a pilot shard may live in.

    The GPU arrays write to ``outputs/dna/fig4/panelB/pilot``. If the pilot is run on
    CPU (``serial_requeue``) it writes to a *separate* tree so provenance stays obvious,
    and both are read here so the sweep still sees one pilot. The pilot only *ranks*
    baseline hyperparameter families and never contributes a plotted number, so which
    device produced it does not enter the figure; each shard records its device,
    account and partition regardless.
    """
    from dmfm import paths

    return [
        paths.OUTPUTS / "fig4" / "panelB" / "pilot",
        paths.OUTPUTS / "fig4_cpu" / "panelB" / "pilot",
    ]


def pilot_scores(shard_dir=None) -> dict:
    """Budget-fair score of every pilot family (lower is better).

    A *family* is one setting of the frozen keys. If every family of a method was run
    at a single budget with the same NFE (FK), the score is simply its MAE. Otherwise
    each family's MAE is interpolated (linear in log NFE) onto 5 log-spaced NFE values
    spanning the range all families share, and the score is the mean of those: a
    family cannot win just by spending more. Returns
    ``{method: {family_tuple: {"score", "points": [(nfe, mae), ...]}}}``.
    """
    import json
    from pathlib import Path

    import numpy as np

    from dmfm import paths

    if shard_dir is None:
        dirs = pilot_dirs()
    elif isinstance(shard_dir, (str, Path)):
        dirs = [Path(shard_dir)]
    else:
        dirs = [Path(d) for d in shard_dir]
    # All-or-nothing across trees: a selection is made from ONE tree, so the comparison
    # between hyperparameter families is never half CPU (true fp32, CPU RNG) and half GPU
    # (TF32, CUDA RNG). Prefer the first complete tree; else the fullest one.
    counts = [(d, len(list(d.glob("*.json"))) if d.is_dir() else 0) for d in dirs]
    n_expected = len(PILOT)
    chosen = next((d for d, n in counts if n >= n_expected), None)
    if chosen is None:
        chosen, best = None, 0
        for d, n in counts:
            if n > best:
                chosen, best = d, n
    fams: dict[str, dict[tuple, list]] = {}
    if chosen is not None and chosen.is_dir():
        for path in sorted(chosen.glob("*.json")):
            try:
                s = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            m = s["method"]
            if m not in FREEZE_KEYS or s.get("meta", {}).get("plan", "pilot") != "pilot":
                continue
            fam = tuple(s["config_params"].get(k) for k in FREEZE_KEYS[m])
            fams.setdefault(m, {}).setdefault(fam, []).append(
                (float(s["nfe"]["nfe_gen_per_output"]), float(s["metrics"]["mae"]))
            )
    out: dict = {}
    for m, by_fam in fams.items():
        pts = {f: sorted(v) for f, v in by_fam.items()}
        lo = max(v[0][0] for v in pts.values())
        hi = min(v[-1][0] for v in pts.values())
        single = all(len(v) == 1 for v in pts.values())
        out[m] = {}
        for f, v in pts.items():
            x = np.log([p[0] for p in v])
            y = np.array([p[1] for p in v])
            if single or len(v) == 1 or lo > hi:
                score = float(np.interp(np.log(lo), x, y)) if len(v) > 1 else float(y[0])
            else:
                grid = np.linspace(np.log(lo), np.log(hi), 5)
                score = float(np.interp(grid, x, y).mean())
            out[m][f] = {"score": score, "points": v, "matched_nfe_range": [lo, hi]}
    return out


def frozen_choice_file():
    """Where the pilot's winners are pinned once a complete pilot exists."""
    from dmfm import paths

    return paths.OUTPUTS / "fig4" / "frozen_hyperparameters.json"


def frozen_from_pilot(shard_dir=None) -> dict:
    """The best family per method by :func:`pilot_scores`.

    Computed in process from the pilot shards, so every array task of the sweep
    agrees without a shared file to race on. Falls back to :data:`FALLBACK_FROZEN`
    for a method with no pilot shards.

    **Pinning.** The first caller that sees a *complete* pilot writes the winners to
    :func:`frozen_choice_file` (atomically), and every later caller reads that file.
    Without this, a sweep that started against one pilot tree could silently switch
    hyperparameters mid-run if a second, later-finishing pilot tree (e.g. a GPU rerun
    of a pilot first done on CPU) ranked a different family top -- which would leave two
    different configurations plotted for one method. The pin also records which tree and
    device the choice came from.
    """
    import json

    pin = frozen_choice_file()
    if shard_dir is None and pin.exists():
        try:
            return json.loads(pin.read_text())["chosen"]
        except (json.JSONDecodeError, KeyError):
            pass

    out = {k: dict(v) for k, v in FALLBACK_FROZEN.items()}
    scores = pilot_scores(shard_dir)
    for m, by_fam in scores.items():
        fam = min(by_fam, key=lambda f: (by_fam[f]["score"], min(p[0] for p in by_fam[f]["points"])))
        out[m] = dict(zip(FREEZE_KEYS[m], fam))

    complete = all(m in scores for m in FREEZE_KEYS)
    if shard_dir is None and complete and not pin.exists():
        pin.parent.mkdir(parents=True, exist_ok=True)
        tmp = pin.with_name(f".{pin.name}.{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "_what": "Baseline hyperparameters pinned from the first complete pilot, so a "
                    "sweep cannot switch configurations mid-run. Delete this file to re-derive.",
                    "chosen": out,
                    "source_dirs": [str(d) for d in (pilot_dirs() if shard_dir is None else [shard_dir])],
                },
                indent=2,
                default=float,
            )
            + "\n"
        )
        try:
            os.replace(tmp, pin)  # atomic; a racing writer simply wins
        except OSError:
            tmp.unlink(missing_ok=True)
    return out


def sweep(frozen: dict | None = None) -> list[Job]:
    f = frozen_from_pilot() if frozen is None else frozen
    jobs: list[Job] = [_dmfm(mc) for mc in (1, 2, 4, 8, 16)]
    jobs += [_bon(n) for n in (2, 3, 5, 9, 16, 32)]
    jobs += [_fk(k, beta=f["fk"]["beta"], every=f["fk"]["resample_every"]) for k in (2, 3, 5, 9, 16, 32)]
    jobs += [_beam(f["beam"]["width"], f["beam"]["branch"], k) for k in (48, 32, 24, 16, 12, 8, 6, 4)]
    jobs += [
        _mcts(it, children=f["mcts"]["children"], k=f["mcts"]["checkpoint_every"], c=f["mcts"]["c_uct"])
        for it in (2, 3, 5, 9, 16, 32, 64)
    ]
    return jobs


class _Plans(dict):
    """``PLANS['sweep']`` is rebuilt on each access so a sweep task launched after the
    pilot picks up the pilot's winners without any file being rewritten."""

    def __getitem__(self, key):
        return sweep() if key == "sweep" else super().__getitem__(key)

    def __iter__(self):
        return iter(("pilot", "sweep"))

    def keys(self):  # noqa: D102
        return ("pilot", "sweep")


PLANS = _Plans({"pilot": PILOT, "sweep": []})

# Shard file names are derived from Job.key, so the keys of a plan must be unique.
for _plan_name in PLANS:
    _keys = [j.key for j in PLANS[_plan_name]]
    if len(set(_keys)) != len(_keys):
        _dupes = sorted({k for k in _keys if _keys.count(k) > 1})
        raise RuntimeError(f"duplicate job keys in plan {_plan_name!r}: {_dupes}")
