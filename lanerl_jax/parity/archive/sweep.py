"""Run a perturbation-response sweep across seeds and perturbations.

WHY A SWEEP AND NOT A SINGLE PAIR
---------------------------------
One perturbation at one seed told us call-for-help moves the sim's response
toward the server on most metrics -- and also that one of those metrics was
closer numerically while being wrong in character (an rms of 0.00 against the
server's 301.75, because our wave never reached the turret at all). A single
pair cannot distinguish "closer" from "differently wrong", and it cannot say
whether an effect survives a change of seed.

A sweep can. Each cell is a self-contained differential -- perturbed minus
baseline, same engine, same seed -- so the comparison across cells is between
RESPONSES, and the chaotic divergence that dominates raw trajectories cancels
inside each cell before anything is compared.

The null control is not optional. It is measured at every seed, and it is the
floor every other response has to clear. On the pilot it was exactly zero, on
both engines, at all 300 grid points -- which is the only reason the other
numbers mean anything.

BUDGET
------
Measured: ~18 s per 10-minute sim episode, ~36 s per server episode, on one
core. A cell is two episodes (baseline + perturbed). `desktop` has 16 cores, so
8 concurrent server episodes is the sensible ceiling -- that is also roughly
the point where a server stops getting a core to itself.

    perturbations x seeds x 2 engines x 2 (baseline, perturbed)

so 3 x 4 x 2 x 2 = 48 episodes, about half of them server ones. Serialised that
is ~25 minutes; at 8-way it is a few.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .perturbation import (
    KillMinions,
    NullControl,
    StandInWave,
    curve_to_dict,
    response,
    run_server_episode,
    run_sim_episode,
    summarize_response,
)

PERTURBATIONS = {"null": NullControl, "stand": StandInWave, "kill": KillMinions}

#: Distinct EXPERIMENTS, not distinct random seeds.
#:
#: The first version of this swept `seed=0,1,2,3` and every cell came back
#: bit-identical on both engines -- 14.3267 four times over. Neither engine has
#: any RNG on this path: the sim's tick is deterministic and `init_lane(seed=)`
#: never reaches anything that varies, and the server's determinism comes from
#: `bot_seed` and the fixed config, not from this argument (its own docstring
#: says so). So a "4-seed sweep" was n=1 replicated four times, which looks
#: like evidence and is not.
#:
#: What actually varies the experiment is the perturbation itself -- WHEN the
#: champion steps into the wave and for HOW LONG. Different trigger times land
#: on different phases of the wave cycle (waves spawn every 30 s and clash
#: around 120 s), so these are genuinely different lane states, which is the
#: robustness the seed axis was supposed to provide and could not.
VARIANTS = {
    "stand": [
        {"trigger_ms": 150_000.0, "hold_s": 5.0},
        {"trigger_ms": 180_000.0, "hold_s": 5.0},
        {"trigger_ms": 210_000.0, "hold_s": 5.0},
        {"trigger_ms": 180_000.0, "hold_s": 2.0},
        {"trigger_ms": 180_000.0, "hold_s": 10.0},
    ],
    "null": [{}],
    "kill": [{}],
}


def run_cell(engine: str, name: str, seed: int, decisions: int,
             port_base: int, variant: Optional[Dict] = None) -> Dict:
    """One cell: baseline and perturbed, same engine and variant, differenced."""
    pert = PERTURBATIONS[name](**(variant or {}))
    if engine == "sim":
        base = run_sim_episode(pert, perturbed=False, seed=seed,
                               decisions=decisions)
        pert_run = run_sim_episode(pert, perturbed=True, seed=seed,
                                   decisions=decisions)
    else:
        base = run_server_episode(pert, perturbed=False, seed=seed,
                                  decisions=decisions, port_base=port_base)
        pert_run = run_server_episode(pert, perturbed=True, seed=seed,
                                      decisions=decisions,
                                      port_base=port_base + 200)
    return {
        "engine": engine, "perturbation": name, "seed": seed,
        "variant": variant or {},
        "summary": summarize_response(response(base, pert_run)),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["sim", "server", "both"],
                    default="both")
    ap.add_argument("--perturbations", default="null,stand")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--port-base", type=int, default=52000)
    ap.add_argument("--out", default="lanerl_jax/runs/sweep.json")
    a = ap.parse_args(argv)

    decisions = int(a.minutes * 60 * 30)
    seeds = [int(s) for s in a.seeds.split(",")]
    names = a.perturbations.split(",")
    engines = ["sim", "server"] if a.engine == "both" else [a.engine]

    rows: List[Dict] = []
    port = a.port_base
    for engine in engines:
        for name in names:
            for variant in VARIANTS.get(name, [{}]):
                tag = ",".join(f"{k}={v}" for k, v in variant.items()) or "default"
                print(f"[{engine}] {name} {tag} ...", flush=True)
                rows.append(run_cell(engine, name, seeds[0], decisions, port,
                                     variant))
                port += 400
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # --- the null floor first, because nothing else means anything without it
    print("\n=== NULL CONTROL (the floor every response must clear) ===")
    for r in rows:
        if r["perturbation"] != "null":
            continue
        worst = max((v["rms"] for v in r["summary"].values()
                     if not np.isnan(v["rms"])), default=float("nan"))
        print(f"  {r['engine']:<7} worst rms {worst:.4f}")

    print("\n=== RESPONSES, mean over seeds ===")
    keys = sorted({k for r in rows for k in r["summary"]})
    for name in names:
        if name == "null":
            continue
        print(f"\n  perturbation: {name}")
        print(f"    {'metric':<22}" + "".join(f"{e:>12}" for e in engines))
        for k in keys:
            cells = []
            for e in engines:
                vals = [r["summary"][k]["rms"] for r in rows
                        if r["engine"] == e and r["perturbation"] == name
                        and not np.isnan(r["summary"][k]["rms"])]
                cells.append(vals)
            # mean and spread across VARIANTS, so a number that is identical
            # in every cell is visibly identical rather than hidden by a mean
            txt = ""
            for vals in cells:
                if not vals:
                    txt += f"{'nan':>20}"
                else:
                    txt += f"{np.mean(vals):>11.3f}+-{np.std(vals):<8.3f}"
            print(f"    {k:<22}{txt}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
