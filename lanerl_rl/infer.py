"""Batched inference across parallel environments.

The measurement this module exists for
--------------------------------------
The reference measurement on the dev machine: **1.70 ms/decision** unbatched
against **0.058 ms/decision** at batch 24 -- 29x, almost all of it fixed
per-call overhead (kernel launches, the transformer's small matmuls, python).
It shows up directly as wall-clock simulator throughput.  **The budget is
33.3 ms**: ``constants.DECISION_HZ`` is 30 (``STEP_TICKS = 2``).  Every
percentage below used to be quoted against 66.7 ms, i.e. against a 15 Hz rate
this stack left behind, so they were all a factor of two too flattering.

1.70 ms of policy per agent against 24 parallel envs is 40.8 ms of serialised
inference -- **122% of a 33.3 ms budget**, spent before the simulator does
anything.  Batched, the same 24 agents cost 1.4 ms: 4% of the budget, 96% left
for the game, and that one holds at either rate.

The "41% of the budget spent" that stood here for a while was arithmetic from
a different env count: 16 x 1.70 = 27.2 ms is 41% of 66.7 ms.  40.8 ms of
66.7 ms is 61%.  One sentence, two env counts and a superseded decision rate.

Reproduced here on a loaded 6-core login node, with the larger post-review
observation (32 slots x 40 fields, up from 20 x 32), single threaded.  Budget
column recomputed at 30 Hz; the 15 Hz figure this table used to print is in
brackets::

    batch  1:  11.11 ms/decision   ->  266.7 ms per 24-agent tick   800% [400%]
    batch  8:   2.39 ms/decision   ->   57.5 ms                     173% [ 86%]
    batch 24:   1.51 ms/decision   ->   36.3 ms                     109% [ 54%]
    batch 48:   1.36 ms/decision   ->   32.7 ms                      98% [ 49%]

The absolute numbers are ~6x worse than the reference because the box is busy
and the model grew; the ratio is 7.3x rather than 29x for the same reason.

The conclusion DOES move, and this is the part the old 15 Hz arithmetic hid: on
this box at 30 Hz, batch 24 is still *over* budget (109%) and only batch 48
gets under it, where the old table read a comfortable 54%.  Batching is still
worth 7-8x and is still mandatory; what is no longer true is that it "brings it
back under half".  A GPU actor, or 15 Hz, is what buys the headroom back.

So the rollout loop must hand the policy a BATCH of observations, not make one
round trip per agent per tick.  ``LanePolicy.act`` is still there for a single
step, but it is not the fast path.

Use
---
::

    actor = BatchedActor(policy, n_agents=len(envs) * 2)
    obs = [env.reset()[team] for env in envs for team in (BLUE, RED)]
    while True:
        actions, extras = actor.act(obs)          # ONE forward for all agents
        obs = [step(env, a) for env, a in ...]

``act`` returns plain python action dicts, ready for ``env.decode``.  The
per-agent core state is carried inside the actor; :meth:`reset` clears the rows
whose episode ended.

Run ``python -m lanerl_rl.infer`` to reproduce the benchmark on this machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from . import constants as C
from .model import LanePolicy, ModelConfig, RecurrentState
from .obs import AgentObservation

__all__ = ["collate_observations", "BatchedActor", "benchmark"]

_FLOAT_KEYS = ("entities", "self_vec", "global_vec", "priv_entities", "priv_vec")
_BOOL_KEYS = ("entity_pad_mask", "priv_pad_mask")
_MASK_KEYS = ("button", "screen_x", "screen_y", "target")
_ACTION_KEYS = ("button", "screen_x", "screen_y", "target")


def collate_observations(
    observations: Sequence[AgentObservation],
    device: torch.device | str = "cpu",
    with_masks: bool = True,
) -> Dict[str, object]:
    """Stack ``N`` observations into ``(N, 1, ...)`` tensors for one forward pass.

    The ``1`` is the time axis: the model is written for ``(B, T, ...)`` and a
    rollout step is ``T = 1``.  Doing the stacking once here, rather than once
    per agent, is most of what makes batched inference fast.
    """
    out: Dict[str, object] = {}
    for k in _FLOAT_KEYS:
        arr = np.stack([getattr(o, k) for o in observations]).astype(np.float32, copy=False)
        out[k] = torch.from_numpy(arr).unsqueeze(1).to(device)
    for k in _BOOL_KEYS:
        arr = np.stack([getattr(o, k) for o in observations])
        out[k] = torch.from_numpy(arr).unsqueeze(1).to(device)
    if with_masks:
        out["action_masks"] = {
            k: torch.from_numpy(np.stack([getattr(o.action_mask, k) for o in observations]))
            .unsqueeze(1)
            .to(device)
            for k in _MASK_KEYS
        }
    return out


@dataclass
class BatchedActorOutput:
    """Everything a PPO rollout buffer needs, for the whole batch at once."""

    actions: List[Dict[str, int]]
    action_tensors: Dict[str, torch.Tensor]  # (N,) long
    log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    state_in: RecurrentState  # the state ENTERING this step
    obs: Dict[str, object]


class BatchedActor:
    """Holds the per-agent core state and runs one forward pass per tick.

    ``n_agents`` is the number of *agents*, not environments: a 1v1 env has two.
    Ordering is fixed by the caller and must stay stable across ticks, because
    row ``i`` of the core state belongs to agent ``i``.
    """

    def __init__(
        self,
        policy: LanePolicy,
        n_agents: int,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
    ):
        self.policy = policy
        self.n_agents = int(n_agents)
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.state = policy.initial_state(self.n_agents, device=self.device)

    # -- state -------------------------------------------------------------

    def reset(self, agent_indices: Optional[Iterable[int]] = None) -> None:
        """Zero the core state for the given agents (default: all of them)."""
        fresh = self.policy.initial_state(self.n_agents, device=self.device)
        if agent_indices is None:
            self.state = fresh
            return
        idx = torch.as_tensor(list(agent_indices), dtype=torch.long, device=self.device)
        actor = self.state.actor.clone()
        critic = self.state.critic.clone()
        actor[:, idx] = 0.0
        critic[:, idx] = 0.0
        self.state = RecurrentState(actor=actor, critic=critic)

    # -- the fast path -----------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        observations: Sequence[AgentObservation],
        resets: Optional[Sequence[bool]] = None,
    ) -> BatchedActorOutput:
        """One forward pass for every agent in the batch."""
        if len(observations) != self.n_agents:
            raise ValueError(
                f"BatchedActor was built for {self.n_agents} agents but got "
                f"{len(observations)} observations; the row order must be stable "
                f"because the core state is indexed by it"
            )
        batch = collate_observations(observations, device=self.device)
        reset_t = None
        if resets is not None:
            reset_t = torch.as_tensor(
                np.asarray(resets, dtype=np.float32), device=self.device
            ).reshape(self.n_agents, 1)
        state_in = self.state
        dist, value, new_state = self.policy(
            entities=batch["entities"],
            entity_pad_mask=batch["entity_pad_mask"],
            self_vec=batch["self_vec"],
            global_vec=batch["global_vec"],
            priv_entities=batch["priv_entities"],
            priv_pad_mask=batch["priv_pad_mask"],
            priv_vec=batch["priv_vec"],
            state=state_in,
            resets=reset_t,
            action_masks=batch["action_masks"],
        )
        action = dist.mode() if self.deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        self.state = new_state

        flat = {k: action[k][:, 0] for k in _ACTION_KEYS}
        actions = [
            {k: int(flat[k][i]) for k in _ACTION_KEYS} for i in range(self.n_agents)
        ]
        return BatchedActorOutput(
            actions=actions,
            action_tensors=flat,
            log_probs=log_prob[:, 0],
            values=value[:, 0],
            state_in=state_in,
            obs=batch,
        )


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def _synthetic_observations(n: int, seed: int = 0) -> List[AgentObservation]:
    from .obs import ActionMask

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ent = rng.standard_normal((C.N_SLOTS, C.ENTITY_DIM)).astype(np.float32)
        ent[:, C.E_TYPE_ONEHOT] = 0.0
        ent[np.arange(C.N_SLOTS), 3 + rng.integers(0, C.N_ENTITY_TYPES, C.N_SLOTS)] = 1.0
        pad = rng.random(C.N_SLOTS) < 0.3
        pad[0] = False
        priv = rng.standard_normal((C.N_SLOTS, C.ENTITY_DIM)).astype(np.float32)
        priv[:, C.E_TYPE_ONEHOT] = ent[:, C.E_TYPE_ONEHOT]
        out.append(
            AgentObservation(
                entities=ent,
                entity_pad_mask=pad,
                self_vec=rng.standard_normal(C.SELF_DIM).astype(np.float32),
                global_vec=rng.standard_normal(C.GLOBAL_DIM).astype(np.float32),
                priv_entities=priv,
                priv_pad_mask=pad.copy(),
                priv_vec=rng.standard_normal(C.PRIV_DIM).astype(np.float32),
                action_mask=ActionMask(
                    button=np.ones(C.N_BUTTONS, dtype=bool),
                    screen_x=np.ones(C.N_SCREEN_X, dtype=bool),
                    screen_y=np.ones(C.N_SCREEN_Y, dtype=bool),
                    target=~pad,
                ),
                t_ms=0,
                fog_source="approx",
            )
        )
    return out


def benchmark(
    policy: Optional[LanePolicy] = None,
    batch_sizes: Sequence[int] = (1, 4, 8, 16, 24, 48),
    iters: int = 30,
    device: str = "cpu",
) -> Dict[int, float]:
    """Milliseconds per *decision* at each batch size.  Lower is better."""
    policy = policy or LanePolicy(ModelConfig())
    policy.eval()
    results: Dict[int, float] = {}
    for n in batch_sizes:
        obs = _synthetic_observations(n, seed=n)
        actor = BatchedActor(policy, n, device=device)
        actor.act(obs)  # warm up
        t0 = time.perf_counter()
        for _ in range(iters):
            actor.act(obs)
        dt = time.perf_counter() - t0
        results[n] = 1000.0 * dt / (iters * n)
    return results


def main() -> int:  # pragma: no cover - CLI
    res = benchmark()
    base = res[min(res)]
    budget_ms = 1000.0 / C.DECISION_HZ
    print(f"batched inference, {C.DECISION_HZ:g} Hz decision budget = {budget_ms:.1f} ms")
    print(f"{'batch':>6} {'ms/decision':>12} {'speedup':>8} {'24-agent tick':>14}")
    for n, ms in sorted(res.items()):
        print(f"{n:>6} {ms:>12.4f} {base / ms:>7.1f}x {ms * 24:>13.2f} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
