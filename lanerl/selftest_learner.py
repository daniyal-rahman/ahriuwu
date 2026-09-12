#!/usr/bin/env python
"""Can the learner learn AT ALL, on a task where we know the answer?

Every failure this project has spent a day on was in the plumbing around the
learner -- spell ranks stuck at 0, an unsupervised target head, a prior nothing
loaded, an argmax evaluator that never moved, an observation that understated
the agent's own damage by 21%. Not one of them was PPO being wrong. But we only
ever found that out by running the whole stack, twelve servers and all, for two
hours, and then arguing about a CS number.

This runs the REAL ``LanePolicy`` and the REAL ``DualClipPPO`` on synthetic
tasks whose optimal policy is known by construction, with no server, no game,
and no observation pipeline. It answers one question per layer:

    button_bandit   can it learn a direct action->reward mapping at all?
                    (forward pass, log_prob, ratio, advantage, optimiser)
    target_bandit   can it learn to pick the right ENTITY SLOT?
                    (the entity encoder and the target head -- the exact head
                    that was silently untrained for weeks)
    memory          can it carry a cue across time? (the GRU, burn-in, BPTT)
    delayed         can it assign credit to an action rewarded 4 steps later?
                    (GAE)

If these pass, the learner is fine and a bad CS number is the environment's
fault. If one fails, stop looking at the game.

    python lanerl/selftest_learner.py            # all layers
    python lanerl/selftest_learner.py --task memory --updates 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanerl_rl import constants as C
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import ACTION_KEYS, DualClipPPO, PPOConfig, RecurrentRolloutBuffer

TASKS = ("button_bandit", "target_bandit", "memory", "delayed")


def _blank_obs(mc: ModelConfig, B: int) -> Dict[str, torch.Tensor]:
    return {
        "entities": torch.zeros(B, mc.n_slots, mc.entity_dim),
        "entity_pad_mask": torch.zeros(B, mc.n_slots, dtype=torch.bool),
        "self_vec": torch.zeros(B, mc.self_dim),
        "global_vec": torch.zeros(B, mc.global_dim),
        "priv_entities": torch.zeros(B, mc.n_slots, mc.entity_dim),
        "priv_pad_mask": torch.zeros(B, mc.n_slots, dtype=torch.bool),
        "priv_vec": torch.zeros(B, mc.priv_dim),
    }


def _all_legal(mc: ModelConfig, B: int) -> Dict[str, torch.Tensor]:
    return {
        "button": torch.ones(B, mc.n_buttons, dtype=torch.bool),
        "move_x": torch.ones(B, mc.n_move_bins, dtype=torch.bool),
        "move_z": torch.ones(B, mc.n_move_bins, dtype=torch.bool),
        "target": torch.ones(B, mc.n_slots, dtype=torch.bool),
    }


class SyntheticTask:
    """A cue in the observation, an action, and a reward with a known optimum.

    ``n_options`` is the size of the correct head, so a uniform-random policy
    scores ``1 / n_options`` and the optimum is 1.0. Those two numbers are what
    make the result readable without a baseline run.
    """

    def __init__(self, kind: str, mc: ModelConfig, B: int, seed: int = 0):
        self.kind = kind
        self.mc = mc
        self.B = B
        self.g = torch.Generator().manual_seed(seed)
        self.head = "target" if kind == "target_bandit" else "button"
        self.n_options = mc.n_slots if self.head == "target" else mc.n_buttons
        # keep it well inside the real space so nothing depends on edge slots
        self.n_options = min(self.n_options, 6)
        self.delay = 4 if kind == "delayed" else 0
        self.cue = torch.zeros(B, dtype=torch.long)
        self.t = 0
        self._pending: list = []

    def reset_cue(self) -> None:
        self.cue = torch.randint(0, self.n_options, (self.B,), generator=self.g)

    def observe(self) -> Dict[str, torch.Tensor]:
        obs = _blank_obs(self.mc, self.B)
        show = True
        if self.kind == "memory":
            # the cue exists ONLY on the first step of each episode; after that
            # the network must be remembering it
            show = self.t % 8 == 0
            if show:
                self.reset_cue()
        else:
            self.reset_cue()
        if show:
            if self.head == "target":
                # flag one entity slot; the answer is "attack that slot"
                obs["entities"][torch.arange(self.B), self.cue, 0] = 1.0
            else:
                obs["self_vec"][torch.arange(self.B), self.cue] = 1.0
        return obs

    def reward(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        hit = (action[self.head].reshape(-1) == self.cue).float()
        if self.delay == 0:
            return hit
        self._pending.append(hit)
        if len(self._pending) > self.delay:
            return self._pending.pop(0)
        return torch.zeros(self.B)


def run_task(kind: str, updates: int, steps: int, envs: int, lr: float,
             seed: int, verbose: bool) -> Tuple[float, float, float]:
    torch.manual_seed(seed)
    mc = ModelConfig()
    policy = LanePolicy(mc)
    learner = DualClipPPO(policy, PPOConfig(lr=lr, entropy_coef=0.003))
    task = SyntheticTask(kind, mc, envs, seed=seed)

    state = policy.initial_state(envs, device="cpu")
    first = last = None
    for u in range(updates):
        buf = RecurrentRolloutBuffer(steps, envs, mc, device="cpu")
        total = 0.0
        for t in range(steps):
            task.t = t
            obs = task.observe()
            masks = _all_legal(mc, envs)
            batch = {k: v.unsqueeze(1) for k, v in obs.items()}
            batch["action_masks"] = {k: v.unsqueeze(1) for k, v in masks.items()}
            resets = torch.zeros(envs, 1)
            if kind == "memory" and t % 8 == 0:
                resets = torch.ones(envs, 1)
            batch["resets"] = resets
            entering = state
            with torch.no_grad():
                action, logp, value, state = policy.act(batch, entering)
            flat = {k: action[k].reshape(-1) for k in ACTION_KEYS}
            r = task.reward(flat)
            total += float(r.mean())
            buf.add(
                obs=obs, masks=masks, action=flat,
                log_prob=logp.reshape(-1), value=value.reshape(-1),
                reward=r, done=torch.zeros(envs),
                reset=resets.reshape(-1), state=entering,
            )
        with torch.no_grad():
            buf.finish(torch.zeros(envs), learner.cfg.gamma, learner.cfg.gae_lambda)
        learner.update(buf)
        mean_r = total / steps
        if first is None:
            first = mean_r
        last = mean_r
        if verbose and (u % max(1, updates // 8) == 0 or u == updates - 1):
            print(f"    update {u:4d}   mean reward {mean_r:.3f}")
    return first, last, 1.0 / task.n_options


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ("all",), default="all")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else (args.task,)
    rows = []
    for kind in tasks:
        print(f"\n=== {kind} ===")
        first, last, chance = run_task(kind, args.updates, args.steps, args.envs,
                                       args.lr, args.seed, not args.quiet)
        # "learned" = closed at least half the gap from chance to optimal
        target = chance + 0.5 * (1.0 - chance)
        ok = last >= target
        rows.append((kind, chance, first, last, target, ok))
        print(f"  chance {chance:.3f} -> start {first:.3f} -> end {last:.3f}"
              f"   (needs >= {target:.3f})   {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 66)
    for kind, chance, first, last, target, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {kind:14s} chance {chance:.3f} "
              f"end {last:.3f} (needs {target:.3f})")
    bad = [r[0] for r in rows if not r[5]]
    if bad:
        print(f"\n{len(bad)} LAYER(S) FAILED: {', '.join(bad)}")
        print("The learner cannot solve a task whose answer is written in the "
              "observation. Stop debugging the game.")
        return 2
    print("\nALL LAYERS PASS -- the learner works on known-solvable tasks, so a "
          "bad CS number is the environment's fault, not PPO's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
