#!/usr/bin/env python
"""Is this PPO run learning, or destroying its prior?

Written because a run was reported as healthy on the evidence that it had
STARTED and had loaded its BC checkpoint. It had, and it was simultaneously
tearing the prior apart: kl_ref 0.55 away from it, clip_frac 0.75, approx_kl
0.343 against a target of 0.02, the KL early-stop firing on all 150 updates,
42% of rollouts rejected, and cs_at_10 None on every episode. None of that is visible in "started 3 actors".

Each check prints PASS/FAIL with the number and, on failure, what it means.

  python lanerl/rl_health.py runs/<run-name>/metrics.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not path.exists():
        print(f"no metrics at {path}")
        return 1
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print("metrics file is empty")
        return 1

    loss = [r for r in rows if "loss/loss" in r]
    stale = [r for r in rows if "reject_rate" in r]
    eps = [r for r in rows if r.get("kind") == "episode"]
    updates = max((r.get("update", 0) for r in rows), default=0)
    print(f"run={path.parent.name}  updates={updates}  "
          f"loss_rows={len(loss)}  episodes={len(eps)}\n")

    fails = 0

    def check(name, ok, detail, why=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {detail}")
        if not ok:
            fails += 1
            if why:
                print(f"         -> {why}")

    if len(loss) >= 4:
        # Entropy must not RISE. A prior worth having is confident; an entropy
        # bonus sized for a random init pays the policy to discard it.
        # Compare against the FIRST update, not an average of the first 20%.
        # The prior's entropy is whatever it is at init; a bad run blows past it
        # within a handful of updates, so a 20% window is already contaminated.
        # On the run this was written for, first-vs-last is 1.64 -> 3.96 (+141%)
        # while window-vs-window was 3.51 -> 3.84 (+9%), which reads as fine.
        head = loss[0]["loss/entropy"]
        tail = _mean([r["loss/entropy"] for r in loss[-max(2, len(loss) // 5):]])
        klr = [r["loss/kl_ref"] for r in loss if r.get("loss/kl_ref") is not None]

        # Entropy alone does NOT measure drift from the prior, and reading it
        # that way is a mistake worth not repeating. It is state-dependent: as
        # the agent reaches lane, more minions are in range, so more target
        # slots are legal and the target head's entropy rises on its own. A rise
        # can mean the policy decayed OR that it simply got somewhere.
        #
        # When a frozen reference exists, kl_ref is the direct measurement and
        # entropy is context. Measured: the run that really was destroying its
        # prior sat at kl_ref 0.55; the healthy one at 0.03, ~18x closer, with
        # entropy rising in BOTH.
        if klr:
            m = _mean(klr)
            check("close to the BC prior", m < 0.25,
                  f"kl_ref mean {m:.3f} (raw KL, not weighted)",
                  "the policy has walked away from the prior it was initialised "
                  "from: raise --kl-ref-coef or lower --lr")
            print(f"  [info] entropy                {head:.3f} (at init) -> {tail:.3f} "
                  f"-- state-dependent, judge drift by kl_ref above")
        else:
            check("entropy not rising", tail <= head * 1.15,
                  f"{head:.3f} (at init) -> {tail:.3f}",
                  "the policy is becoming MORE random: lower --entropy-coef")

        # Prefer the EXCESS over the staleness baseline. Raw approx_kl now
        # includes the drift an off-policy rollout already carries before any
        # gradient step, so judging the absolute value flags a healthy run.
        exc = _mean([r.get("loss/approx_kl_excess") for r in loss])
        if exc is not None:
            check("approx_kl excess small", exc < 0.05,
                  f"mean {exc:.4f} over the staleness baseline (target_kl 0.02)",
                  "each update moves the policy far too far: lower --lr")
            stale = _mean([r.get("loss/approx_kl_staleness") for r in loss])
            if stale is not None:
                print(f"  [info] staleness drift        {stale:.4f} "
                      f"(present before any gradient step)")
        else:
            akl = _mean([r["loss/approx_kl"] for r in loss])
            check("approx_kl near target", akl is not None and akl < 0.05,
                  f"mean {akl:.4f} (target_kl 0.02)",
                  "each update moves the policy far too far: lower --lr")

        cf = _mean([r["loss/clip_frac"] for r in loss])
        check("clip_frac sane", cf is not None and cf < 0.35,
              f"mean {cf:.3f}",
              "most of the batch is being clipped, so most of the gradient is "
              "discarded: lower --lr")

        es = _mean([r["loss/kl_early_stop"] for r in loss])
        check("early-stop rare", es is not None and es < 0.5,
              f"fires on {es:.0%} of updates",
              "the KL guard is saving the run from every single update; it is a "
              "backstop, not a scheduler")

        ep_run = _mean([r.get("loss/epochs_run") for r in loss])
        if ep_run is not None:
            check("epochs actually run", ep_run > 1.5, f"mean {ep_run:.2f}",
                  "the run is throwing away most of each batch's epochs")

        # EXPLAINED VARIANCE, not value_loss. A critic that predicts a
        # constant still has non-zero value_loss, so the old check passed on
        # exactly the failure it was meant to catch: rl-bc4-0912's critic went
        # 0.047 -> 0.55 while never tracking its target, and nothing said so.
        ev = _mean([r.get("loss/explained_variance")
                    for r in loss[-max(2, len(loss) // 5):]])
        if ev is not None:
            check("critic explains the returns", ev > 0.1,
                  f"explained_variance {ev:+.3f}",
                  "the value function is not tracking the returns; advantages "
                  "are mostly noise. Raise --critic-lr or --critic-warmup-updates")
        else:
            vl = _mean([r["loss/value_loss"] for r in loss[-max(2, len(loss) // 5):]])
            check("critic alive", vl is not None and vl > 1e-6, f"value_loss {vl:.2e}",
                  "no explained_variance logged; falling back to a weak check")

    if stale:
        rr = _mean([r["reject_rate"] for r in stale])
        check("rollouts accepted", rr is not None and rr < 0.15,
              f"reject_rate mean {rr:.1%}",
              "server time is the bottleneck and this fraction of it is thrown "
              "away: raise --max-staleness or lower --num-actors")

    if not eps and loss:
        # "episodes=0" is ambiguous: it can mean episodes are broken, or simply
        # that none has finished yet. Say which. Updates are produced by ONE
        # actor at a time, so each instance advances at updates/num_actors --
        # with full 10-minute episodes (--no-end-on-death) the first one lands
        # about 3x later than a naive updates*rollout_steps estimate suggests.
        import collections as _c
        actors = len({r.get("actor") for r in loss if r.get("actor") is not None}) or 1
        rollout = 256  # matches --rollout-steps; only used for this estimate
        per_instance = (len(loss) / actors) * rollout
        due = 18000 / rollout * actors
        print(f"  [info] no episode yet         each instance has played "
              f"{per_instance / 30.0:.0f} s of a 600 s episode; "
              f"first one due around update {due:.0f}")

    if eps:
        cs = [r["cs_at_10"] for r in eps if r.get("cs_at_10") is not None]
        check("cs_at_10 reachable", bool(cs),
              f"{len(cs)}/{len(eps)} episodes reported one"
              + (f", mean {_mean(cs):.1f}" if cs else ""),
              "cs_at_10 is an ABSOLUTE 10-minute metric, so an episode that ends "
              "early can never produce one -- check --no-end-on-death")
        lens = [r["length_steps"] for r in eps if r.get("length_steps") is not None]
        if lens:
            print(f"  [info] episode length         mean {_mean(lens):.0f} steps "
                  f"({_mean(lens) / 30.0:.0f} s of game)")
        rets = [r["ep_return"] for r in eps if r.get("ep_return") is not None]
        if rets:
            print(f"  [info] ep_return              mean {_mean(rets):+.3f}")
        from collections import Counter
        print(f"  [info] end reasons            {dict(Counter(r.get('reason') for r in eps))}")

    print(f"\n{'ALL CHECKS PASS' if not fails else f'{fails} CHECK(S) FAILED'}")
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
