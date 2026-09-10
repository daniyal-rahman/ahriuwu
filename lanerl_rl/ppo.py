"""Dual-clip recurrent PPO.

Dual clip
---------
Standard PPO's clipped surrogate is unbounded below when the importance ratio
explodes on a *negative* advantage: ``min(rA, clip(r)A)`` equals ``rA`` there,
and ``r`` can be arbitrarily large.  One bad minibatch then produces an
enormous policy gradient.  Ye et al. (2020), "Mastering Complex Control in MOBA
Games with Deep Reinforcement Learning" (JueWu), add a second clip for that
case::

    A >= 0:  L = min(rA, clip(r, 1-e, 1+e) A)            # standard
    A <  0:  L = max( min(rA, clip(r, 1-e, 1+e) A), cA ) # c = 3.0

so the objective can never be worse than ``c * A``.  ``c > 1`` is required for
the bound to be a relaxation rather than a constraint.

Recurrence
----------
The rollout stores the core state entering *every* timestep, so a minibatch can
start at an arbitrary offset.  Each training chunk is preceded by ``burn_in``
steps replayed under ``no_grad`` to re-warm the hidden state under the *current*
parameters (R2D2-style), which removes most of the staleness that using the
stored state directly would introduce.  Gradients flow only through the scored
chunk.

With ``ModelConfig.core = "mlp"`` the stored "core state" is just the previous
``frame_stack - 1`` core inputs, so all of this machinery still runs but the
burn-in becomes exact rather than approximate (a frame stack has no staleness
to re-warm away).  That is one of the reasons to want the MLP ablation.

Discount
--------
``PPOConfig`` takes a HORIZON IN SECONDS, not a raw gamma; see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import constants as C
from .model import LaneActionDist, LanePolicy, ModelConfig, RecurrentState

__all__ = ["PPOConfig", "RecurrentRolloutBuffer", "DualClipPPO", "compute_gae"]

OBS_KEYS = (
    "entities",
    "entity_pad_mask",
    "self_vec",
    "global_vec",
    "priv_entities",
    "priv_pad_mask",
    "priv_vec",
)
MASK_KEYS = ("button", "move_x", "move_z", "target")
ACTION_KEYS = ("button", "move_x", "move_z", "target")


@dataclass
class PPOConfig:
    """PPO hyper-parameters.

    The discount is configured as a **horizon in seconds**, not as a raw gamma.
    A raw gamma means a different amount of game time at every decision rate,
    so copying one across a rate change silently changes the objective.  At the
    15 Hz decision rate this stack uses::

        horizon_s = 30  ->  gamma = 0.997778
        horizon_s = 45  ->  gamma = 0.998519

    Set ``gamma`` explicitly only to override; ``horizon_s`` is then ignored and
    :meth:`effective_horizon_s` reports what you actually asked for.
    """

    horizon_s: float = C.DEFAULT_HORIZON_S
    decision_hz: float = C.DECISION_HZ
    gamma: Optional[float] = None
    gae_lambda: float = 0.99
    clip_eps: float = 0.2
    dual_clip: float = 3.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    lr: float = 3e-4
    epochs: int = 4
    chunk_len: int = 16
    burn_in: int = 8
    minibatch_chunks: int = 8
    normalize_advantage: bool = True
    clip_value_loss: bool = True
    value_clip_eps: float = 0.2

    def __post_init__(self) -> None:
        if self.dual_clip <= 1.0:
            raise ValueError("dual_clip must be > 1.0 to be a relaxation")
        # Reject a decision rate the 60 Hz server cannot express (50 Hz is 1.2
        # ticks per decision and does not exist).
        self.step_ticks = C.legal_step_ticks(self.decision_hz)
        if self.gamma is None:
            self.gamma = C.gamma_for_horizon(self.horizon_s, self.decision_hz)
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")

    def effective_horizon_s(self) -> float:
        """``1 / ((1 - gamma) * decision_hz)`` -- the horizon actually in force."""
        return C.horizon_for_gamma(self.gamma, self.decision_hz)


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over (T, B) tensors.

    ``dones[t]`` is 1 when step ``t`` is terminal (so the value of ``t+1`` must
    not be bootstrapped through it).  Returns ``(advantages, returns)``.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


# --------------------------------------------------------------------------
# Buffer
# --------------------------------------------------------------------------


class RecurrentRolloutBuffer:
    """Fixed-size (T, B) rollout storage with per-step GRU states."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        cfg: Optional[ModelConfig] = None,
        device: torch.device | str = "cpu",
    ):
        self.T = int(num_steps)
        self.B = int(num_envs)
        self.cfg = cfg or ModelConfig()
        self.device = torch.device(device)
        c = self.cfg
        T, B = self.T, self.B
        f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=self.device)  # noqa: E731
        b = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=self.device)  # noqa: E731
        i = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=self.device)  # noqa: E731

        self.obs: Dict[str, torch.Tensor] = {
            "entities": f(T, B, c.n_slots, c.entity_dim),
            "entity_pad_mask": b(T, B, c.n_slots),
            "self_vec": f(T, B, c.self_dim),
            "global_vec": f(T, B, c.global_dim),
            "priv_entities": f(T, B, c.n_slots, c.entity_dim),
            "priv_pad_mask": b(T, B, c.n_slots),
            "priv_vec": f(T, B, c.priv_dim),
        }
        self.masks: Dict[str, torch.Tensor] = {
            "button": b(T, B, c.n_buttons),
            "move_x": b(T, B, c.n_move_bins),
            "move_z": b(T, B, c.n_move_bins),
            "target": b(T, B, c.n_slots),
        }
        self.actions: Dict[str, torch.Tensor] = {k: i(T, B) for k in ACTION_KEYS}
        self.log_probs = f(T, B)
        self.values = f(T, B)
        self.rewards = f(T, B)
        self.dones = f(T, B)
        self.resets = f(T, B)
        # The core state is whatever the configured core carries: a GRU hidden
        # vector, or the previous (frame_stack - 1) core inputs for the MLP core.
        self.h_actor = f(T, B, c.actor_state_dim)
        self.h_critic = f(T, B, c.critic_state_dim)
        self.advantages = f(T, B)
        self.returns = f(T, B)
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    @property
    def full(self) -> bool:
        return self.step >= self.T

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        reset: torch.Tensor,
        state: RecurrentState,
    ) -> None:
        """Append one (B,) transition.  ``state`` is the state *entering* the step."""
        if self.full:
            raise RuntimeError("rollout buffer is full")
        t = self.step
        for k in OBS_KEYS:
            self.obs[k][t] = obs[k].to(self.obs[k].dtype)
        for k in MASK_KEYS:
            self.masks[k][t] = masks[k].to(torch.bool)
        for k in ACTION_KEYS:
            self.actions[k][t] = action[k].to(torch.long)
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done.to(torch.float32)
        self.resets[t] = reset.to(torch.float32)
        self.h_actor[t] = state.actor[0]
        self.h_critic[t] = state.critic[0]
        self.step += 1

    def finish(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        adv, ret = compute_gae(
            self.rewards[: self.step],
            self.values[: self.step],
            self.dones[: self.step],
            last_value,
            gamma,
            lam,
        )
        self.advantages[: self.step] = adv
        self.returns[: self.step] = ret

    # -- minibatching ------------------------------------------------------

    def chunk_starts(self, chunk_len: int, burn_in: int) -> Dict[int, List[Tuple[int, int]]]:
        """Group ``(env, start)`` chunk descriptors by their available burn-in."""
        groups: Dict[int, List[Tuple[int, int]]] = {}
        for t0 in range(0, self.step, chunk_len):
            if t0 + chunk_len > self.step:
                break  # keep every chunk rectangular
            bi = min(burn_in, t0)
            groups.setdefault(bi, []).extend((e, t0) for e in range(self.B))
        return groups

    def gather(self, items: List[Tuple[int, int]], bi: int, chunk_len: int) -> Dict[str, object]:
        """Slice ``[t0-bi, t0+chunk_len)`` for each ``(env, t0)`` -> (N, L, ...)."""
        envs = torch.tensor([e for e, _ in items], dtype=torch.long, device=self.device)
        t0s = torch.tensor([t for _, t in items], dtype=torch.long, device=self.device)
        L = bi + chunk_len
        offs = torch.arange(L, device=self.device)
        tidx = (t0s - bi).unsqueeze(1) + offs.unsqueeze(0)  # (N, L)
        eidx = envs.unsqueeze(1).expand(-1, L)

        out: Dict[str, object] = {
            "obs": {k: self.obs[k][tidx, eidx] for k in OBS_KEYS},
            "masks": {k: self.masks[k][tidx, eidx] for k in MASK_KEYS},
            "actions": {k: self.actions[k][tidx, eidx] for k in ACTION_KEYS},
            "log_probs": self.log_probs[tidx, eidx],
            "values": self.values[tidx, eidx],
            "advantages": self.advantages[tidx, eidx],
            "returns": self.returns[tidx, eidx],
            "resets": self.resets[tidx, eidx],
            "burn_in": bi,
            "chunk_len": chunk_len,
        }
        start = t0s - bi
        out["state"] = RecurrentState(
            actor=self.h_actor[start, envs].unsqueeze(0).contiguous(),
            critic=self.h_critic[start, envs].unsqueeze(0).contiguous(),
        )
        return out

    def iter_minibatches(
        self, chunk_len: int, burn_in: int, minibatch_chunks: int, generator=None
    ) -> Iterator[Dict[str, object]]:
        groups = self.chunk_starts(chunk_len, burn_in)
        for bi, items in groups.items():
            order = torch.randperm(len(items), generator=generator).tolist()
            for i in range(0, len(order), minibatch_chunks):
                sel = [items[j] for j in order[i : i + minibatch_chunks]]
                yield self.gather(sel, bi, chunk_len)


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class DualClipPPO:
    """PPO with the JueWu dual-clip term, recurrent minibatching and burn-in."""

    def __init__(self, policy: LanePolicy, cfg: Optional[PPOConfig] = None, optimizer=None):
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        self.optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=self.cfg.lr, eps=1e-5)

    # -- losses ------------------------------------------------------------

    def policy_loss(
        self, log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        inner = torch.min(surr1, surr2)
        # Dual clip: bound the objective from below by c*A when A < 0.
        dual = torch.max(inner, cfg.dual_clip * adv)
        obj = torch.where(adv < 0.0, dual, inner)
        loss = -obj.mean()

        with torch.no_grad():
            logr = log_prob - old_log_prob
            approx_kl = ((torch.exp(logr) - 1.0) - logr).mean().item()
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
            dual_frac = ((adv < 0.0) & (cfg.dual_clip * adv > inner)).float().mean().item()
        return loss, {"approx_kl": approx_kl, "clip_frac": clip_frac, "dual_clip_frac": dual_frac}

    def value_loss(
        self, value: torch.Tensor, old_value: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        unclipped = (value - returns) ** 2
        if not cfg.clip_value_loss:
            return 0.5 * unclipped.mean()
        clipped_v = old_value + torch.clamp(value - old_value, -cfg.value_clip_eps, cfg.value_clip_eps)
        clipped = (clipped_v - returns) ** 2
        return 0.5 * torch.max(unclipped, clipped).mean()

    # -- one minibatch -----------------------------------------------------

    def _forward_chunk(self, batch: Dict[str, object]) -> Tuple[LaneActionDist, torch.Tensor]:
        bi = int(batch["burn_in"])
        obs = batch["obs"]
        masks = batch["masks"]
        resets = batch["resets"]
        state: RecurrentState = batch["state"]

        if bi > 0:
            with torch.no_grad():
                _, _, state = self.policy.forward(
                    entities=obs["entities"][:, :bi],
                    entity_pad_mask=obs["entity_pad_mask"][:, :bi],
                    self_vec=obs["self_vec"][:, :bi],
                    global_vec=obs["global_vec"][:, :bi],
                    priv_entities=obs["priv_entities"][:, :bi],
                    priv_pad_mask=obs["priv_pad_mask"][:, :bi],
                    priv_vec=obs["priv_vec"][:, :bi],
                    state=state,
                    resets=resets[:, :bi],
                )
            state = state.detach()

        dist, value, _ = self.policy.forward(
            entities=obs["entities"][:, bi:],
            entity_pad_mask=obs["entity_pad_mask"][:, bi:],
            self_vec=obs["self_vec"][:, bi:],
            global_vec=obs["global_vec"][:, bi:],
            priv_entities=obs["priv_entities"][:, bi:],
            priv_pad_mask=obs["priv_pad_mask"][:, bi:],
            priv_vec=obs["priv_vec"][:, bi:],
            state=state,
            resets=resets[:, bi:],
            action_masks={k: masks[k][:, bi:] for k in MASK_KEYS},
        )
        return dist, value

    def update_minibatch(self, batch: Dict[str, object]) -> Dict[str, float]:
        cfg = self.cfg
        bi = int(batch["burn_in"])
        dist, value = self._forward_chunk(batch)

        actions = {k: batch["actions"][k][:, bi:] for k in ACTION_KEYS}
        old_log_prob = batch["log_probs"][:, bi:]
        old_value = batch["values"][:, bi:]
        adv = batch["advantages"][:, bi:]
        returns = batch["returns"][:, bi:]

        if cfg.normalize_advantage and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        pi_loss, stats = self.policy_loss(log_prob, old_log_prob, adv)
        v_loss = self.value_loss(value, old_value, returns)
        loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        stats.update(
            {
                "loss": float(loss.detach()),
                "policy_loss": float(pi_loss.detach()),
                "value_loss": float(v_loss.detach()),
                "entropy": float(entropy.detach()),
                "grad_norm": float(grad_norm),
            }
        )
        return stats

    # -- one full update ---------------------------------------------------

    def update(self, buffer: RecurrentRolloutBuffer, generator=None) -> Dict[str, float]:
        """Run ``cfg.epochs`` passes, stopping early on KL divergence."""
        cfg = self.cfg
        agg: Dict[str, List[float]] = {}
        n_batches = 0
        stopped = False
        for epoch in range(cfg.epochs):
            epoch_kls = []
            for batch in buffer.iter_minibatches(
                cfg.chunk_len, cfg.burn_in, cfg.minibatch_chunks, generator=generator
            ):
                stats = self.update_minibatch(batch)
                for k, v in stats.items():
                    agg.setdefault(k, []).append(v)
                epoch_kls.append(stats["approx_kl"])
                n_batches += 1
            if epoch_kls and float(np.mean(epoch_kls)) > cfg.target_kl:
                stopped = True
                break
        out = {k: float(np.mean(v)) for k, v in agg.items()}
        out["n_minibatches"] = float(n_batches)
        out["epochs_run"] = float(epoch + 1)
        out["kl_early_stop"] = float(stopped)
        return out
