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

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

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
MASK_KEYS = ("button", "screen_x", "screen_y", "target")
ACTION_KEYS = ("button", "screen_x", "screen_y", "target")


@dataclass
class PPOConfig:
    """PPO hyper-parameters.

    The discount is configured as a **horizon in seconds**, not as a raw gamma.
    A raw gamma means a different amount of game time at every decision rate,
    so copying one across a rate change silently changes the objective.  At the
    30 Hz decision rate this stack uses (``constants.STEP_TICKS = 2``)::

        horizon_s = 30  ->  gamma = 0.998889
        horizon_s = 45  ->  gamma = 0.999259

    (At 15 Hz the same horizons are 0.997778 and 0.998519.  This docstring said
    15 Hz long after the stack moved to 30, so the worked example disagreed with
    what ``PPOConfig()`` actually produces -- exactly the silent objective change
    the paragraph above is warning about.)

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
    #: Weight on KL(reference || policy) toward a frozen behaviour-cloning
    #: prior. This is the AlphaStar shape: a supervised prior gets the agent
    #: into the part of the state space where reward exists, and the KL term
    #: keeps it there while PPO improves on it. Without an anchor the policy
    #: drifts straight back to uniform -- measured on runs/rl-overnight-0911-0608:
    #: factored entropy ended at 8.876 of a 9.940 maximum (89%) after 16,209
    #: updates, and every one of the 298 recorded cs_at_10 readings was 0.0.  (The
    #: "88% over 13,475 updates" this used to say matches no point in that run:
    #: at update 13,475 entropy was 8.365, i.e. 84%.)
    #: 0.0 disables it, which is the behaviour when no reference is supplied.
    kl_ref_coef: float = 0.0
    #: Decay ``kl_ref_coef`` linearly to ZERO over this many training steps.
    #: 0 means never decay, which is what it did before and is almost certainly
    #: wrong for a prior you intend to EXCEED.
    #:
    #: A BC prior is a floor, not a target. Measured on run rl-bc4-0912 with a
    #: flat coefficient: the KL penalty was 0.0062 against a mean |policy_loss|
    #: of 0.0090 -- 69% of the entire learning signal -- and kl_ref climbed
    #: 0.098 -> 0.138 over the run, i.e. the policy pushed and the leash held.
    #: Tethered permanently to a heuristic bot that caps out at 49 CS, the
    #: agent cannot do better than the thing it was cloned from.
    #:
    #: UNITS: whatever ``DualClipPPO``'s ``train_step_source`` counts, which is
    #: NOT training steps.  ``lanerl_train.__main__`` hands it
    #: ``TrainingLoop.train_steps``, whose clock is chosen by ``--anneal-clock``
    #: and defaults to ``env_steps`` -- environment ROWS (rollout_steps x envs
    #: per update, e.g. 2,040 per update on rl-bc4-0912).  So 100,000 here is
    #: about 49 updates at that shape, not 100,000 updates.  ``--anneal-clock
    #: updates`` is what makes the number mean updates.  (The docstring said
    #: "training steps" for as long as the field existed, a factor of ~2,000.)
    kl_ref_anneal_steps: int = 0
    max_grad_norm: float = 1.0
    #: Stop after an epoch whose mean approx_kl exceeds EPOCH 0's by this much.
    #: Not the absolute KL: an off-policy rollout already carries staleness
    #: drift before a single gradient has been taken, and stopping on the
    #: absolute value stops on that.  See :meth:`DualClipPPO.update`.
    target_kl: float = 0.02
    #: Learning rate for the ACTOR (and the default for anything not in a
    #: named group).  Fine-tuning a behaviour-cloned actor wants this small.
    lr: float = 3e-4
    #: Learning rate for the CRITIC, which is a DIFFERENT problem: the BC
    #: checkpoint carries no value function (``lanerl_train.bc`` discards the
    #: value output and saves a randomly-initialised critic -- 2.6M of the
    #: model's 4.6M parameters), so the critic trains from scratch while the
    #: actor fine-tunes.  One shared Adam at the actor's rate starves it:
    #: measured on rl-bc4-0912 at lr 1e-5, ``loss/value_loss`` went 0.047 ->
    #: 0.55 (max 34.3) over 2,690 updates, i.e. the advantages PPO was
    #: learning from came from a value function that never caught up.
    #: None means "use ``lr``".
    critic_lr: Optional[float] = 3e-4
    #: Train ONLY the critic for this many updates before unfreezing the
    #: actor, so the first policy gradients are taken against a value function
    #: that has seen the state distribution at least once.
    critic_warmup_updates: int = 0
    epochs: int = 4
    chunk_len: int = 16
    burn_in: int = 8
    #: Chunks per minibatch.  Mostly a THROUGHPUT knob: the model is small
    #: enough that a learner step is bound by kernel-launch latency, not by
    #: arithmetic, so quartering the number of minibatches is close to
    #: quartering the update time.  Measured on a 5080 over a 255x8 rollout,
    #: 4 epochs, with a KL reference, before the rest of this file's speed
    #: work: 8 chunks 0.630 s, 16 -> 0.350 s, 32 -> 0.244 s, 64 -> 0.182 s.
    #:
    #: It is NOT free, though: it also divides the number of Adam steps taken
    #: per rollout (at 255x8 and 4 epochs, 60 steps at 8 chunks against 20 at
    #: 32), so a run that was learning at the edge of its learning rate may
    #: need a larger one to keep the same progress per rollout.  The KL
    #: early-stop fix pushes the other way -- rl-bc4-0912 ran a mean of 2.91
    #: epochs of its 4 because staleness tripped the stop, and it no longer
    #: does.
    minibatch_chunks: int = 32
    #: Normalise advantages over the WHOLE ROLLOUT once, in
    #: :meth:`DualClipPPO.update`, not per minibatch.  Per-minibatch
    #: normalisation over 128 samples estimates the mean and the standard
    #: deviation from the minibatch itself, which both adds noise and
    #: re-centres every minibatch on its own mean -- a chunk that happens to
    #: contain only good steps has its best step pushed DOWN.
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

    def critic_learning_rate(self) -> float:
        return self.lr if self.critic_lr is None else self.critic_lr


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

    def kl_ref_at(self, train_step: int) -> float:
        """``kl_ref_coef`` decayed toward 0, linearly, over ``kl_ref_anneal_steps``.

        ``train_step`` is read from ``DualClipPPO``'s ``train_step_source``, so
        both are in units of the ``--anneal-clock`` -- environment ROWS by
        default, not updates.  See ``kl_ref_anneal_steps``.
        """
        if self.kl_ref_anneal_steps <= 0:
            return self.kl_ref_coef
        t = min(max(float(train_step) / float(self.kl_ref_anneal_steps), 0.0), 1.0)
        return self.kl_ref_coef * (1.0 - t)


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
            "screen_x": b(T, B, c.n_screen_x),
            "screen_y": b(T, B, c.n_screen_y),
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
        #: Frozen-reference logits for the whole rollout, filled once per
        #: update by :meth:`DualClipPPO.update` and sliced by :meth:`gather`
        #: exactly like an observation.  Empty when there is no KL anchor.
        self.ref_logits: Dict[str, torch.Tensor] = {}
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

    def to(self, device: torch.device | str) -> "RecurrentRolloutBuffer":
        """Move every tensor onto ``device``, in place, and return self.

        Needed because a rollout now crosses a PROCESS boundary: the actor
        collects on its own CUDA context and the learner updates on the
        parent's, and a CUDA tensor cannot simply be pickled between the two.
        The actor calls ``.to("cpu")`` before queueing -- ~15 MB at T=128,
        B=24, so a couple of milliseconds over PCIe -- and the learner calls
        ``.to(its device)`` on receipt.

        Truncated to ``self.step`` nowhere: the shapes stay fixed so ``gather``
        and ``chunk_starts`` keep indexing the same way on both sides.
        """
        dev = torch.device(device)
        for d in (self.obs, self.masks, self.actions, self.ref_logits):
            for k, v in d.items():
                d[k] = v.to(dev)
        for name in (
            "log_probs", "values", "rewards", "dones", "resets",
            "h_actor", "h_critic", "advantages", "returns",
        ):
            setattr(self, name, getattr(self, name).to(dev))
        self.device = dev
        return self

    def normalize_advantages(self, eps: float = 1e-8) -> None:
        """Whiten the advantages over the WHOLE rollout, in place.

        PPO normalises advantages to keep the policy-gradient scale
        independent of the reward scale.  Doing it per minibatch -- 8 chunks x
        16 steps = 128 samples -- estimates both moments from 128 correlated
        samples of one rollout, and re-centres each minibatch on its own mean,
        so a minibatch whose steps were all good has its best step pushed
        negative.  The rollout is 2,040 samples at the shape rl-bc4-0912 ran,
        which is a far better estimator of the same two numbers.
        """
        a = self.advantages[: self.step]
        if a.numel() > 1:
            self.advantages[: self.step] = (a - a.mean()) / (a.std(unbiased=False) + eps)

    def explained_variance(self) -> torch.Tensor:
        """``1 - Var(returns - values) / Var(returns)`` over the rollout.

        The single number that says whether the critic is doing anything: 1 is
        a perfect value function, 0 is "no better than predicting the mean",
        and negative is worse than that.  Nobody logged it, which is how
        rl-bc4-0912 ran 2,690 updates with a critic that was never trained at
        a usable learning rate (``loss/value_loss`` 0.047 -> 0.55) without
        anyone seeing it.  Computed against the values the BEHAVIOUR policy
        produced during the rollout, which is the standard definition.
        """
        v = self.values[: self.step]
        r = self.returns[: self.step]
        var_r = r.var(unbiased=False)
        return torch.where(
            var_r > 0, 1.0 - (r - v).var(unbiased=False) / var_r, torch.full_like(var_r, float("nan"))
        )

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
            "ref_logits": {k: v[tidx, eidx] for k, v in self.ref_logits.items()},
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

    def __init__(self, policy: LanePolicy, cfg: Optional[PPOConfig] = None, optimizer=None,
                 reference: Optional[LanePolicy] = None,
                 train_step_source=None):
        #: Frozen behaviour-cloning prior for the KL anchor, or None. Kept in
        #: eval mode and never optimised -- it is a fixed target, not a second
        #: learner.
        self.reference = reference
        #: Where the KL anneal reads 'how far in are we'. Defaults to a
        #: clock stuck at 0, which makes kl_ref_at return the flat
        #: coefficient -- the old behaviour, for callers that pass nothing.
        self._train_step_source = train_step_source or (lambda: 0)
        if reference is not None:
            reference.eval()
            for prm in reference.parameters():
                prm.requires_grad_(False)
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        #: Completed calls to :meth:`update`.  Drives ``critic_warmup_updates``
        #: and survives a resume through :meth:`state_payload`.
        self._updates_done = 0
        # Where this learner's parameters live. Read by the training loop to
        # put a rollout collected in ANOTHER PROCESS (and therefore handed over
        # on CPU) back on the right device before the update. Derived from the
        # policy rather than stored, so it cannot drift from where the weights
        # actually are after a .to() somewhere else.
        self._device_probe = policy
        self.optimizer = optimizer or self._build_optimizer()

    # -- optimiser ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Where this learner's weights are.

        Raises rather than guessing ``cpu`` for a parameterless policy: the one
        caller that needs this is moving a rollout collected in another process
        onto the learner's device, and a wrong answer there is a silent
        CPU-speed run or a device-mismatch traceback pointing at the PPO step
        instead of at the handover.
        """
        for prm in self._device_probe.parameters():
            return prm.device
        raise RuntimeError(
            "the policy has no parameters, so its device cannot be determined"
        )

    def _split_parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``(actor_params, critic_params)``, in ``policy.parameters()`` order.

        The order is load-bearing: it is what lets a one-group Adam state from
        an older checkpoint be split across the two groups in
        :meth:`_load_optimizer_state`.  ``LanePolicy`` registers ``self.critic``
        last, so actor-then-critic IS the flat order -- asserted rather than
        assumed, because a reordering of ``__init__`` would silently hand
        every parameter the wrong Adam moments on the next resume.
        """
        critic_names = self.policy.critic_parameter_names()
        actor, critic = [], []
        for n, p in self.policy.named_parameters():
            (critic if n in critic_names else actor).append(p)
        flat = list(self.policy.parameters())
        if [id(p) for p in actor + critic] != [id(p) for p in flat]:
            raise AssertionError(
                "LanePolicy no longer yields every actor parameter before every "
                "critic parameter; two-group optimiser state would be mis-assigned"
            )
        return actor, critic

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """One Adam, two param groups: the actor and the critic learn at
        DIFFERENT rates.

        They are different problems.  ``--init-from`` fine-tunes a
        behaviour-cloned actor, which wants a small step; the critic inside
        that same checkpoint is random initialisation (``lanerl_train.bc``
        never trains it) and has to learn a value function from scratch, which
        does not.  One Adam at the actor's rate is the slower of the two.
        """
        actor, critic = self._split_parameters()
        return torch.optim.Adam(
            [
                {"params": actor, "lr": self.cfg.lr, "name": "actor"},
                {"params": critic, "lr": self.cfg.critic_learning_rate(), "name": "critic"},
            ],
            lr=self.cfg.lr,
            eps=1e-5,
        )

    def _apply_configured_lrs(self) -> None:
        """The CONFIG is the authority on learning rate, not the checkpoint.

        ``Adam.load_state_dict`` copies the saved ``param_groups`` hyper-
        parameters over the live ones, so without this a resume would silently
        restore the learning rates the run was launched with the first time and
        ignore the ones it was relaunched with -- including the whole point of
        ``critic_lr``.
        """
        for g in self.optimizer.param_groups:
            if g.get("name") == "critic":
                g["lr"] = self.cfg.critic_learning_rate()
            elif g.get("name") == "actor":
                g["lr"] = self.cfg.lr

    # -- lanerl_train.protocols.Learner -------------------------------------
    #
    # TrainingLoop needs these three.  No longer untested: the "resume that
    # 'worked' was only exercised against FakeInstance's fake learner" note here
    # is stale.  tests/test_ppo.py::
    # test_policy_and_optimizer_state_round_trip_through_the_learner_payloads
    # round-trips all three through the real DualClipPPO, Adam state tensors
    # included, and runs/rl-overnight-0911-0608 resumed from real checkpoints 7
    # times (kind:"resume" at updates 51, 800, 1275, 1750, 2200, 9075, 13350),
    # each continuing forward.

    def policy_payload(self) -> Dict[str, object]:
        """Just the weights an actor needs to act -- no optimiser state, so an
        actor process never needs to import ``torch.optim`` to load one."""
        return {"policy": {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}}

    def state_payload(self) -> Dict[str, object]:
        """Everything needed to resume: weights, optimiser state, PPO config."""
        payload = dict(self.policy_payload())
        payload["optimizer"] = self.optimizer.state_dict()
        payload["cfg"] = asdict(self.cfg)
        payload["updates_done"] = int(self._updates_done)
        return payload

    def load_payload(self, payload: Dict[str, object]) -> None:
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            self._load_optimizer_state(payload["optimizer"])
        self._updates_done = int(payload.get("updates_done", 0))

    def _load_optimizer_state(self, saved: Dict[str, object]) -> None:
        """Load Adam state, adapting a pre-two-group checkpoint if necessary.

        Every checkpoint written before the actor/critic split has ONE param
        group.  ``Adam.load_state_dict`` refuses a group-count mismatch
        outright, which would turn "resume rl-bc4-0912" into a crash; the
        per-parameter moments are keyed by position in the flat parameter
        list, and :meth:`_split_parameters` guarantees the two groups
        concatenate back to exactly that list, so splitting the saved group's
        index list at the actor/critic boundary recovers them intact.
        """
        groups = list(saved.get("param_groups", []))
        mine = self.optimizer.param_groups
        if len(groups) == 1 and len(mine) == 2:
            n_actor = len(mine[0]["params"])
            idx = list(groups[0]["params"])
            saved = dict(saved)
            a, c = dict(groups[0]), dict(groups[0])
            a["params"], c["params"] = idx[:n_actor], idx[n_actor:]
            a["name"], c["name"] = "actor", "critic"
            saved["param_groups"] = [a, c]
        self.optimizer.load_state_dict(saved)
        self._apply_configured_lrs()

    # -- losses ------------------------------------------------------------

    def policy_loss(
        self, log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``(loss, diagnostics)``.  The diagnostics are 0-dim tensors ON THE
        DEVICE, not floats: a ``.item()`` is a synchronisation, and there were
        eight of them per minibatch (times up to 60 minibatches per update) in
        service of numbers nothing reads until the end of the update."""
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
            approx_kl = ((torch.exp(logr) - 1.0) - logr).mean()
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            dual_frac = ((adv < 0.0) & (cfg.dual_clip * adv > inner)).float().mean()
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

    def _forward_chunk(self, batch: Dict[str, object],
                       policy: Optional[LanePolicy] = None) -> Tuple[LaneActionDist, torch.Tensor]:
        net = policy if policy is not None else self.policy
        bi = int(batch["burn_in"])
        obs = batch["obs"]
        masks = batch["masks"]
        resets = batch["resets"]
        state: RecurrentState = batch["state"]

        if bi > 0:
            # carry_state, not forward: the burn-in wants the re-warmed core
            # state and nothing else, and forward would build four action
            # heads (32-slot attention included) and a value head per burn-in
            # step only to discard them.
            state = net.carry_state(
                entities=obs["entities"][:, :bi],
                entity_pad_mask=obs["entity_pad_mask"][:, :bi],
                self_vec=obs["self_vec"][:, :bi],
                global_vec=obs["global_vec"][:, :bi],
                priv_entities=obs["priv_entities"][:, :bi],
                priv_pad_mask=obs["priv_pad_mask"][:, :bi],
                priv_vec=obs["priv_vec"][:, :bi],
                state=state,
                resets=resets[:, :bi],
            ).detach()

        dist, value, _ = net.forward(
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

    def _actor_logits_for_chunk(
        self, batch: Dict[str, object], net: LanePolicy
    ) -> Dict[str, torch.Tensor]:
        """Chunk logits from ``net``'s ACTOR ONLY -- no critic anywhere.

        The KL reference is consumed through ``dist.kl_to(ref.logits)``; its
        value output was computed and discarded, and the critic is 2.6M of the
        model's 4.6M parameters.  Only used when :meth:`update_minibatch` is
        called outside :meth:`update` (which precomputes the whole rollout's
        reference logits in one pass instead -- see
        :meth:`_reference_logits_for_rollout`).
        """
        bi = int(batch["burn_in"])
        obs, masks, resets = batch["obs"], batch["masks"], batch["resets"]
        h = batch["state"].actor
        if bi > 0:
            _d, _t, h = net.actor_forward(
                obs["entities"][:, :bi], obs["entity_pad_mask"][:, :bi],
                obs["self_vec"][:, :bi], obs["global_vec"][:, :bi], h, resets[:, :bi],
            )
        dist, _t, _h = net.actor_forward(
            obs["entities"][:, bi:], obs["entity_pad_mask"][:, bi:],
            obs["self_vec"][:, bi:], obs["global_vec"][:, bi:], h, resets[:, bi:],
            action_masks={k: masks[k][:, bi:] for k in MASK_KEYS},
        )
        return dist.logits

    @torch.no_grad()
    def _reference_logits_for_rollout(
        self, buffer: RecurrentRolloutBuffer, time_chunk: int = 64
    ) -> Dict[str, torch.Tensor]:
        """Every reference logit in the rollout, in one sweep, once per update.

        The reference is FROZEN and the rollout is fixed, so its logits are
        the same on every epoch and in every minibatch that happens to contain
        a given step.  Recomputing them per minibatch per epoch was 4x60 full
        forward passes (critic included) per update for a tensor that fits in
        half a megabyte.

        It is also more faithful.  Per-chunk, the reference was burned in from
        ``buffer.h_actor``, which is the state the BEHAVIOUR POLICY was in --
        another network's hidden vector, injected into the reference every
        ``chunk_len`` steps and only partly washed out by ``burn_in`` steps of
        replay.  The KL anchor wants ``KL(pi_ref(.|h_ref) || pi_theta(.|h))``,
        and ``h_ref`` is whatever the REFERENCE would have carried through this
        history -- which is what rolling it forward produces.  The first chunk
        is bit-identical to the old behaviour (it started from
        ``h_actor[0]`` there too); after that this is the better number.  The
        reference's memory still restarts from the stored policy state once per
        ROLLOUT rather than being carried across rollouts, which is the one
        approximation left.
        """
        net = self.reference
        T = buffer.step
        state = buffer.h_actor[0].unsqueeze(0).contiguous()
        out: Dict[str, List[torch.Tensor]] = {k: [] for k in MASK_KEYS}
        for t0 in range(0, T, time_chunk):
            t1 = min(t0 + time_chunk, T)
            sl = lambda x: x[t0:t1].transpose(0, 1)  # noqa: E731  (T,B,..) -> (B,T,..)
            dist, _tokens, state = net.actor_forward(
                sl(buffer.obs["entities"]),
                sl(buffer.obs["entity_pad_mask"]),
                sl(buffer.obs["self_vec"]),
                sl(buffer.obs["global_vec"]),
                state,
                sl(buffer.resets),
                action_masks={k: sl(buffer.masks[k]) for k in MASK_KEYS},
            )
            for k, v in dist.logits.items():
                out[k].append(v.transpose(0, 1))
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    def update_minibatch(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """One gradient step.  Returns 0-dim tensors ON THE DEVICE.

        :meth:`update` reduces them with a single host synchronisation at the
        end; a caller wanting python floats should say ``float(...)``.
        """
        cfg = self.cfg
        bi = int(batch["burn_in"])
        # Critic-only warm-up: the actor is frozen by omitting its terms from
        # the loss entirely, so no actor gradient exists to be stepped.
        actor_frozen = self._updates_done < cfg.critic_warmup_updates
        dist, value = self._forward_chunk(batch)

        actions = {k: batch["actions"][k][:, bi:] for k in ACTION_KEYS}
        old_log_prob = batch["log_probs"][:, bi:]
        old_value = batch["values"][:, bi:]
        adv = batch["advantages"][:, bi:]
        returns = batch["returns"][:, bi:]

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        pi_loss, stats = self.policy_loss(log_prob, old_log_prob, adv)
        v_loss = self.value_loss(value, old_value, returns)
        loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

        # Anchor to the behaviour-cloning prior, if one was supplied. Computed
        # against the SAME steps, so the reference sees exactly the inputs the
        # policy just saw; a reference evaluated on different inputs would
        # regularise toward the wrong thing.
        kl_ref = None
        kl_coef = cfg.kl_ref_at(int(self._train_step_source()))
        if self.reference is not None and kl_coef > 0.0:
            cached = batch.get("ref_logits") or {}
            if cached:
                ref_logits = {k: v[:, bi:] for k, v in cached.items()}
            else:
                with torch.no_grad():
                    ref_logits = self._actor_logits_for_chunk(batch, self.reference)
            kl_ref = dist.kl_to(ref_logits).mean()
            loss = loss + kl_coef * kl_ref

        total = cfg.value_coef * v_loss if actor_frozen else loss
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        if kl_ref is not None:
            stats["kl_ref"] = kl_ref.detach()
        stats.update(
            {
                "loss": loss.detach(),
                "policy_loss": pi_loss.detach(),
                "value_loss": v_loss.detach(),
                "entropy": entropy.detach(),
                "grad_norm": grad_norm.detach(),
            }
        )
        return stats

    # -- one full update ---------------------------------------------------

    def update(self, buffer: RecurrentRolloutBuffer, generator=None) -> Dict[str, float]:
        """Run ``cfg.epochs`` passes, stopping early on KL divergence.

        The early stop is on the KL this UPDATE has ADDED, not on the absolute
        approx_kl.  A rollout arrives up to ``max_staleness`` parameter
        versions old, so the importance ratio is already off 1 before a single
        gradient is taken, and the absolute test was mostly measuring that:
        on rl-bc4-0912, ``loss/epochs_run`` was bimodal -- 4 epochs 1,684
        times and ONE epoch 948 times (35%), almost never 2 or 3 -- because
        the staleness drift tripped the 0.02 target inside the first epoch and
        threw three epochs of work away.

        The baseline is EPOCH 0's mean, not the first minibatch's KL.  The
        first minibatch is the only measurement taken before any gradient
        step, so it is the cleanest estimate of the drift and is logged as
        ``approx_kl_staleness`` -- but it is one minibatch out of ~15, and its
        distance from the epoch mean is composition noise of the same order as
        ``target_kl`` itself, which is how you get spurious stops back.  An
        epoch mean over every chunk in the rollout does not have that problem,
        at the cost of folding epoch 0's own learning into the baseline (which
        only ever makes the stop harder to trip).  A consequence worth stating:
        epoch 0 can no longer be cut short, so every rollout now gets at least
        one full pass -- which is the standard arrangement (CleanRL, SB3 both
        test at epoch boundaries).
        """
        cfg = self.cfg
        if cfg.normalize_advantage:
            buffer.normalize_advantages()
        if self.reference is not None and cfg.kl_ref_at(int(self._train_step_source())) > 0.0:
            buffer.ref_logits = self._reference_logits_for_rollout(buffer)

        agg: Dict[str, List[torch.Tensor]] = {}
        n_batches = 0
        stopped = False
        epoch = -1
        first_kl: Optional[torch.Tensor] = None
        baseline: Optional[torch.Tensor] = None
        excess = None
        for epoch in range(cfg.epochs):
            epoch_kls: List[torch.Tensor] = []
            for batch in buffer.iter_minibatches(
                cfg.chunk_len, cfg.burn_in, cfg.minibatch_chunks, generator=generator
            ):
                stats = self.update_minibatch(batch)
                for k, v in stats.items():
                    agg.setdefault(k, []).append(v)
                if first_kl is None:
                    first_kl = stats["approx_kl"]
                epoch_kls.append(stats["approx_kl"])
                n_batches += 1
            if not epoch_kls:
                continue
            epoch_mean = torch.stack(epoch_kls).mean()
            if baseline is None:
                baseline = epoch_mean
                continue
            excess = epoch_mean - baseline
            if float(excess) > cfg.target_kl:  # one sync per epoch, at most
                stopped = True
                break

        keys = list(agg)
        nan = torch.tensor(float("nan"))
        extra = {
            "approx_kl_staleness": first_kl if first_kl is not None else nan,
            "approx_kl_baseline": baseline if baseline is not None else nan,
            "approx_kl_excess": excess if excess is not None else nan,
            "explained_variance": buffer.explained_variance(),
        }
        # ONE host synchronisation for every scalar the update produced.
        parts = [torch.stack(agg[k]).mean() for k in keys] + list(extra.values())
        dev = parts[0].device
        flat = torch.stack([p.to(device=dev, dtype=torch.float32).reshape(()) for p in parts]).cpu()
        out = {k: float(v) for k, v in zip(keys + list(extra), flat)}
        out["n_minibatches"] = float(n_batches)
        out["epochs_run"] = float(epoch + 1)
        out["kl_early_stop"] = float(stopped)
        # Log the LIVE KL coefficient, not the configured one: with an anneal
        # they differ, and a run whose KL term has decayed to nothing should
        # say so.
        out["kl_ref_coef"] = float(cfg.kl_ref_at(int(self._train_step_source())))
        out["actor_frozen"] = float(self._updates_done < cfg.critic_warmup_updates)
        out["lr_actor"] = float(cfg.lr)
        out["lr_critic"] = float(cfg.critic_learning_rate())
        self._updates_done += 1
        buffer.ref_logits = {}
        return out
