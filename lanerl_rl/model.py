"""Entity-attention policy with an asymmetric, history-conditioned critic.

Architecture
------------
Actor::

    entities (B,T,32,40) --Linear(40->128)--+
                                            |--> +type_embedding(6->128)
                                            v
                       TransformerEncoder(depth 2, heads 4, d=128, ffn 256,
                                          key_padding_mask=~valid)
                                            |
                                    tokens (B,T,32,128)   <-- target head reads THIS
                                            |
                            masked max ++ masked mean -> (B,T,256)
                                            |
    self (B,T,64) ++ global (B,T,48) --MLP--> (B,T,256)
                                            |
                                    concat -> (B,T,512) --> CORE --> h (B,T,512)
                                            |
      heads: button (8) | move_x (9) | move_z (9) | target = softmax(FC(h) . tokens^T)

Critic (separate trunk, **different inputs**, but the SAME history)::

    priv_entities (B,T,32,40) unfogged --MLP--> masked pool -> (B,T,256)
    priv_vec (B,T,96) ++ self ++ global --MLP--> (B,T,256)
    h_actor (B,T,512), detached                                  <-- see below
                            concat -> CORE -> Linear(512,1)

The core is swappable
---------------------
``ModelConfig.core`` is ``"gru"`` or ``"mlp"``.

``"gru"``  a single-layer GRU, hidden state ``(1, B, core_dim)``.

``"mlp"``  a 4-frame stack of the core input followed by a
           ``mlp_layers``-deep MLP.  The "recurrent state" is then simply the
           previous ``frame_stack - 1`` core inputs, carried in the same
           ``RecurrentState`` slot, so resets, burn-in and chunked minibatching
           all keep working unchanged.

The point of offering both: GT Sophy (Wurman et al. 2022, Nature 602) reached
superhuman Gran Turismo with **no recurrence at all** -- a 4x2048 MLP over
hand-derived features.  Once the long-horizon memory lives in the observation
builder (enemy cooldown estimates, last-seen positions, reachability radii,
attack-cycle phase -- see ``obs.py``), there may be nothing left for a GRU to
do, and dropping it removes BPTT, burn-in staleness, stored hidden state and
sequence minibatching in one go.  Both paths train; the ablation decides.

Why the critic gets the actor's history
---------------------------------------
An asymmetric critic that sees only the privileged *state* while the actor is a
function of the *history* gives a **biased** policy gradient: the state-value
baseline is not the expected return of the actor's own conditional
distribution, so the residual is not a valid advantage.  Baisero & Amato,
"Unbiased Asymmetric Actor-Critic for Partially Observable Reinforcement
Learning" (arXiv:2105.11674), show the fix is a history-*and*-state value
``V(h, s)``.  So the actor's core output is concatenated into the critic's
input.  It is detached by default (``detach_actor_core_for_critic``) so the
value loss shapes the value function, not the policy encoder.

The upside of a privileged critic is large: AlphaStar's ablation
(Vinyals et al. 2019, Extended Data Fig. 3) moves from 22% to 82% win rate when
the value function is given the opponent's observation.

The asymmetry itself is enforced by the API, not by convention:
:meth:`LanePolicy.forward` takes ``actor``/``critic`` argument groups, there is
no code path from ``priv_*`` into the policy trunk, and
:func:`assert_actor_critic_disjoint` checks that no parameter tensor is shared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C

__all__ = [
    "ModelConfig",
    "RecurrentState",
    "LaneActionDist",
    "LanePolicy",
    "assert_actor_critic_disjoint",
    "apply_action_mask",
    "masked_pool",
    "gru_with_resets",
    "frame_stack_with_resets",
]


@dataclass
class ModelConfig:
    n_slots: int = C.N_SLOTS
    entity_dim: int = C.ENTITY_DIM
    self_dim: int = C.SELF_DIM
    global_dim: int = C.GLOBAL_DIM
    priv_dim: int = C.PRIV_DIM
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 256
    ctx_dim: int = 256
    core_dim: int = 512
    n_buttons: int = C.N_BUTTONS
    n_move_bins: int = C.N_MOVE_BINS
    n_entity_types: int = C.N_ENTITY_TYPES
    #: ``"gru"`` or ``"mlp"``.
    core: str = "gru"
    #: Frames in the stack for the ``"mlp"`` core.  Ignored by ``"gru"``.
    frame_stack: int = 4
    #: Depth of the ``"mlp"`` core (GT Sophy used 4 layers).
    mlp_layers: int = 4
    mlp_hidden: int = 1024
    #: Feed the actor's core output into the critic -- V(h, s), not V(s).
    critic_sees_actor_core: bool = True
    #: Stop the value loss from back-propagating into the actor trunk.
    detach_actor_core_for_critic: bool = True

    def __post_init__(self) -> None:
        if self.core not in ("gru", "mlp"):
            raise ValueError(f"core must be 'gru' or 'mlp', got {self.core!r}")
        if self.core == "mlp" and self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")

    # -- derived shapes ----------------------------------------------------

    @property
    def gru_dim(self) -> int:
        """Backwards-compatible alias for :attr:`core_dim`."""
        return self.core_dim

    @property
    def actor_core_input_dim(self) -> int:
        return 2 * self.d_model + self.ctx_dim

    @property
    def critic_core_input_dim(self) -> int:
        extra = self.core_dim if self.critic_sees_actor_core else 0
        return 2 * self.d_model + self.ctx_dim + extra

    def _state_dim(self, core_input_dim: int) -> int:
        if self.core == "gru":
            return self.core_dim
        return (self.frame_stack - 1) * core_input_dim

    @property
    def actor_state_dim(self) -> int:
        return self._state_dim(self.actor_core_input_dim)

    @property
    def critic_state_dim(self) -> int:
        return self._state_dim(self.critic_core_input_dim)


class RecurrentState(NamedTuple):
    """Core states.  Actor and critic keep *separate* memory.

    For ``core="gru"`` these are GRU hidden states ``(1, B, core_dim)``.
    For ``core="mlp"`` they are the flattened previous ``frame_stack - 1`` core
    inputs, ``(1, B, (k-1) * core_input_dim)``.  Same slot, same lifecycle, so
    nothing downstream has to know which core is in use.
    """

    actor: torch.Tensor
    critic: torch.Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(self.actor.detach(), self.critic.detach())

    def index(self, idx) -> "RecurrentState":
        return RecurrentState(self.actor[:, idx], self.critic[:, idx])


# --------------------------------------------------------------------------
# Action distribution
# --------------------------------------------------------------------------

_NEG = -1e9


def apply_action_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Set masked logits to a large negative value.

    A fully-masked row would produce a uniform-over-nothing softmax and NaN
    gradients, so a row with no legal action is silently re-opened on index 0.
    Callers should not rely on that: build masks that always leave a no-op
    legal (:class:`~lanerl_rl.obs.ActionMask` does).
    """
    if mask is None:
        return logits
    mask = mask.to(dtype=torch.bool)
    any_legal = mask.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(mask)
    fallback[..., 0] = True
    mask = torch.where(any_legal, mask, fallback)
    return logits.masked_fill(~mask, _NEG)


class LaneActionDist:
    """Factored action: (button, move_x, move_z, target).

    ``move_x`` / ``move_z`` are the LANE-LOCAL (s, n) components; see
    ``constants.MOVE_AXIS_NAMES``.  They and ``target`` are always sampled --
    the environment decides which of them a given button actually consumes (see
    ``constants.MOVE_BUTTONS`` / ``constants.TARGETED_BUTTONS``).  Sampling all
    four unconditionally keeps the log-probability well defined and the gradient
    dense; the alternative (conditional heads) makes the PPO ratio depend on the
    sampled button, which is a well known source of silent bias.
    """

    HEADS = ("button", "move_x", "move_z", "target")

    def __init__(self, logits: Dict[str, torch.Tensor]):
        self.logits = logits
        self.dists = {k: torch.distributions.Categorical(logits=v) for k, v in logits.items()}

    def sample(self) -> Dict[str, torch.Tensor]:
        return {k: d.sample() for k, d in self.dists.items()}

    def mode(self) -> Dict[str, torch.Tensor]:
        return {k: v.argmax(dim=-1) for k, v in self.logits.items()}

    def log_prob(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.dists[k].log_prob(action[k]) for k in self.HEADS)

    def entropy(self) -> torch.Tensor:
        return sum(d.entropy() for d in self.dists.values())

    def kl_to(self, other_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``KL(other || self)``, summed over heads, computed in log space.

        This used to call ``torch.distributions.kl_divergence``, which is
        defined for Categorical as::

            t = p.probs * (p.logits - q.logits)
            t[q.probs == 0] = inf          # <-- here
            t[p.probs == 0] = 0

        ``q`` is *this* distribution, the policy. As the policy sharpens, a
        softmax entry underflows to exactly 0 while the reference still has
        support there, and the whole term becomes ``inf``. Measured on run
        rl-bc2-0912: 7 of 56 updates reported an infinite total loss. The
        gradients happened to stay finite, so it did not blow the run up -- it
        just silently removed the KL anchor on 12.5% of updates and poisoned
        the logged loss.

        In log space there is no such cliff. ``log_softmax`` is
        ``logit - logsumexp``, which stays finite even where ``softmax``
        underflows: a masked logit of -1e9 gives a log-prob of about -1e9, and
        the reference's probability there is exactly 0, so the product is 0
        rather than ``0 * -inf = nan``. This relies on masked logits being a
        large finite negative (-1e9), never ``-inf``.
        """
        total = 0.0
        for k in self.HEADS:
            q_logp = torch.log_softmax(self.dists[k].logits, dim=-1)
            p_logp = torch.log_softmax(other_logits[k], dim=-1)
            p_prob = p_logp.exp()
            total = total + (p_prob * (p_logp - q_logp)).sum(dim=-1)
        return total


# --------------------------------------------------------------------------
# Pooling / sequence helpers
# --------------------------------------------------------------------------


def masked_pool(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked max ++ masked mean over the slot axis.

    ``tokens``: (N, S, D); ``valid``: (N, S) bool.  Rows with no valid slot
    pool to zeros rather than to -inf / NaN.
    """
    vf = valid.unsqueeze(-1).to(tokens.dtype)
    masked = tokens * vf
    count = vf.sum(dim=1).clamp(min=1.0)
    mean = masked.sum(dim=1) / count

    neg = torch.finfo(tokens.dtype).min
    filled = torch.where(valid.unsqueeze(-1), tokens, torch.full_like(tokens, neg))
    mx = filled.max(dim=1).values
    any_valid = valid.any(dim=1, keepdim=True)
    mx = torch.where(any_valid, mx, torch.zeros_like(mx))
    return torch.cat([mx, mean], dim=-1)


def gru_with_resets(
    gru: nn.GRU, x: torch.Tensor, h: torch.Tensor, resets: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a GRU over (B, T, D), zeroing the hidden state where ``resets`` is 1.

    ``resets[b, t] == 1`` means "step t begins a new episode", so the hidden
    state carried *into* step t is zeroed.  Without resets the whole sequence
    goes through cuDNN in one call.
    """
    if resets is None or not bool(resets.any()):
        out, hn = gru(x, h)
        return out, hn
    B, T, _ = x.shape
    outs = []
    for t in range(T):
        m = (1.0 - resets[:, t]).view(1, B, 1).to(h.dtype)
        h = h * m
        o, h = gru(x[:, t : t + 1], h)
        outs.append(o)
    return torch.cat(outs, dim=1), h


def frame_stack_with_resets(
    x: torch.Tensor, h: torch.Tensor, resets: Optional[torch.Tensor], k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sliding window of the last ``k`` frames of ``x``, honouring episode resets.

    ``x`` is (B, T, D), ``h`` is (1, B, (k-1)*D) holding the frames that precede
    ``x[:, 0]``, and ``resets`` is (B, T) with 1 at the first step of an episode.
    Returns ``(stacked (B, T, k*D), new_h (1, B, (k-1)*D))``, oldest frame first.

    A frame ``x[t-j]`` may be used at step ``t`` only if no reset occurred in
    ``(t-j, t]``.  With ``c = cumsum(resets)`` that is exactly
    ``c[t] - c[t-j] == 0``, which makes the whole thing one gather plus one
    comparison -- no python loop over time, unlike the GRU path.  Frames that
    fail the test are zeroed, and the carried-out state inherits that zeroing so
    a reset near the end of a chunk does not leak into the next chunk.
    """
    B, T, D = x.shape
    if k < 1:
        raise ValueError("frame_stack must be >= 1")
    if k == 1:
        return x, h
    hist = h.reshape(B, k - 1, D)
    seq = torch.cat([hist, x], dim=1)  # (B, T + k - 1, D)
    base = torch.arange(T, device=x.device).unsqueeze(1)
    off = torch.arange(k, device=x.device).unsqueeze(0)
    idx = base + off  # (T, k); seq[t + k - 1] is x[t]
    win = seq[:, idx]  # (B, T, k, D)

    if resets is not None and bool(resets.any()):
        c = torch.cumsum(resets.to(torch.long), dim=1)  # (B, T)
        pad = torch.zeros(B, k - 1, dtype=c.dtype, device=c.device)
        cpad = torch.cat([pad, c], dim=1)  # (B, T + k - 1)
        cwin = cpad[:, idx]  # (B, T, k)
        ok = (cwin[:, :, -1:] - cwin) == 0
        win = win * ok.unsqueeze(-1).to(win.dtype)

    stacked = win.reshape(B, T, k * D)
    hn = win[:, -1, 1:, :].reshape(1, B, (k - 1) * D).contiguous()
    return stacked, hn


# --------------------------------------------------------------------------
# Cores
# --------------------------------------------------------------------------


class _GRUCore(nn.Module):
    """Single-layer GRU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim, batch_first=True)
        self.state_dim = out_dim
        self.out_dim = out_dim

    def forward(self, x, h, resets):
        return gru_with_resets(self.gru, x, h, resets)


class _MLPCore(nn.Module):
    """Frame stack + MLP.  No recurrence, no BPTT."""

    def __init__(self, in_dim: int, out_dim: int, frame_stack: int, layers: int, hidden: int):
        super().__init__()
        self.k = int(frame_stack)
        self.in_dim = in_dim
        self.state_dim = (self.k - 1) * in_dim
        self.out_dim = out_dim
        dims = [self.k * in_dim] + [hidden] * max(0, layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.GELU())
        self.net = nn.Sequential(*mods)

    def forward(self, x, h, resets):
        stacked, hn = frame_stack_with_resets(x, h, resets, self.k)
        return self.net(stacked), hn


def _make_core(cfg: ModelConfig, in_dim: int) -> nn.Module:
    if cfg.core == "gru":
        return _GRUCore(in_dim, cfg.core_dim)
    return _MLPCore(in_dim, cfg.core_dim, cfg.frame_stack, cfg.mlp_layers, cfg.mlp_hidden)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


class _EntityEncoder(nn.Module):
    """Linear(entity_dim -> d) + type embedding + TransformerEncoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.entity_dim, cfg.d_model)
        self.type_emb = nn.Embedding(cfg.n_entity_types, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would only
        # emit a warning before falling back to the dense path anyway.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

    def forward(self, entities: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``entities``: (N, S, E); ``valid``: (N, S) bool -> tokens (N, S, D)."""
        type_onehot = entities[..., C.E_TYPE_ONEHOT]
        type_idx = type_onehot.argmax(dim=-1)
        x = self.proj(entities) + self.type_emb(type_idx)

        pad = ~valid
        # nn.TransformerEncoder produces NaNs for a fully-padded row; re-open
        # such rows and drop their output with the real mask afterwards.
        all_pad = pad.all(dim=-1, keepdim=True)
        pad_safe = pad & ~all_pad
        x = self.encoder(x, src_key_padding_mask=pad_safe)
        return x * valid.unsqueeze(-1).to(x.dtype)


class _PrivilegedCritic(nn.Module):
    """Value trunk.  Consumes the *unfogged* state plus the actor's history.

    Never shares a parameter with the actor; the actor's core output arrives as
    a (detached, by default) input tensor.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.entity_mlp = nn.Sequential(
            nn.Linear(cfg.entity_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ctx = nn.Sequential(
            nn.Linear(cfg.priv_dim + cfg.self_dim + cfg.global_dim, cfg.ctx_dim),
            nn.GELU(),
            nn.Linear(cfg.ctx_dim, cfg.ctx_dim),
            nn.GELU(),
        )
        self.core = _make_core(cfg, cfg.critic_core_input_dim)
        self.value = nn.Sequential(nn.Linear(cfg.core_dim, cfg.ctx_dim), nn.GELU(), nn.Linear(cfg.ctx_dim, 1))

    def forward(
        self,
        priv_entities: torch.Tensor,
        priv_pad_mask: torch.Tensor,
        priv_vec: torch.Tensor,
        self_vec: torch.Tensor,
        global_vec: torch.Tensor,
        h: torch.Tensor,
        resets: Optional[torch.Tensor],
        actor_core: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, S, _ = priv_entities.shape
        flat = priv_entities.reshape(B * T, S, -1)
        valid = (~priv_pad_mask.reshape(B * T, S)).to(torch.bool)
        tok = self.entity_mlp(flat) * valid.unsqueeze(-1).to(flat.dtype)
        pooled = masked_pool(tok, valid).reshape(B, T, -1)
        ctx = self.ctx(torch.cat([priv_vec, self_vec, global_vec], dim=-1))
        parts = [pooled, ctx]
        if self.cfg.critic_sees_actor_core:
            if actor_core is None:
                raise ValueError(
                    "critic_sees_actor_core=True but no actor core output was supplied; "
                    "a state-only critic paired with a history-dependent actor biases the "
                    "policy gradient (Baisero & Amato, arXiv:2105.11674)"
                )
            parts.append(
                actor_core.detach() if self.cfg.detach_actor_core_for_critic else actor_core
            )
        out, hn = self.core(torch.cat(parts, dim=-1), h, resets)
        return self.value(out).squeeze(-1), hn


class LanePolicy(nn.Module):
    """The full actor-critic.

    ``forward`` deliberately takes two disjoint argument groups.  Actor inputs
    are fogged and screenshot-recoverable; critic inputs are the unfogged
    server state plus the actor's own history summary.  Nothing routes the
    privileged group into the policy trunk.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.entity_encoder = _EntityEncoder(c)
        self.ctx = nn.Sequential(
            nn.Linear(c.self_dim + c.global_dim, c.ctx_dim),
            nn.GELU(),
            nn.Linear(c.ctx_dim, c.ctx_dim),
            nn.GELU(),
        )
        self.core = _make_core(c, c.actor_core_input_dim)
        self.head_button = nn.Linear(c.core_dim, c.n_buttons)
        self.head_move_x = nn.Linear(c.core_dim, c.n_move_bins)
        self.head_move_z = nn.Linear(c.core_dim, c.n_move_bins)
        self.head_target_q = nn.Linear(c.core_dim, c.d_model)
        self.critic = _PrivilegedCritic(c)
        self._init_heads()

    def _init_heads(self) -> None:
        for head in (self.head_button, self.head_move_x, self.head_move_z):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    # -- state -------------------------------------------------------------

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> RecurrentState:
        device = device or next(self.parameters()).device
        c = self.cfg
        return RecurrentState(
            actor=torch.zeros(1, batch, c.actor_state_dim, device=device, dtype=dtype),
            critic=torch.zeros(1, batch, c.critic_state_dim, device=device, dtype=dtype),
        )

    # -- actor -------------------------------------------------------------

    def _actor_trunk(
        self,
        entities: torch.Tensor,
        entity_pad_mask: torch.Tensor,
        self_vec: torch.Tensor,
        global_vec: torch.Tensor,
        h: torch.Tensor,
        resets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> ``(core_out (B,T,core_dim), tokens (B*T,S,D), valid (B*T,S), hn)``."""
        B, T, S, _ = entities.shape
        flat = entities.reshape(B * T, S, -1)
        valid = (~entity_pad_mask.reshape(B * T, S)).to(torch.bool)
        tokens = self.entity_encoder(flat, valid)
        pooled = masked_pool(tokens, valid).reshape(B, T, -1)
        ctx = self.ctx(torch.cat([self_vec, global_vec], dim=-1))
        core_out, hn = self.core(torch.cat([pooled, ctx], dim=-1), h, resets)
        return core_out, tokens, valid, hn

    def _actor_heads(
        self,
        core_out: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        shape: Tuple[int, int, int],
        action_masks: Optional[Dict[str, torch.Tensor]],
    ) -> LaneActionDist:
        B, T, S = shape
        # Target head reads the PRE-POOL tokens.
        q = self.head_target_q(core_out).reshape(B * T, 1, self.cfg.d_model)
        target_logits = torch.bmm(q, tokens.transpose(1, 2)).squeeze(1) / math.sqrt(self.cfg.d_model)
        target_logits = target_logits.reshape(B, T, S)

        logits = {
            "button": self.head_button(core_out),
            "move_x": self.head_move_x(core_out),
            "move_z": self.head_move_z(core_out),
            "target": target_logits,
        }
        if action_masks is not None:
            logits = {k: apply_action_mask(v, action_masks.get(k)) for k, v in logits.items()}
        else:
            # Even without an explicit mask, never let the policy target an
            # empty slot.
            logits["target"] = apply_action_mask(logits["target"], valid.reshape(B, T, S))
        return LaneActionDist(logits)

    def actor_forward(
        self,
        entities: torch.Tensor,
        entity_pad_mask: torch.Tensor,
        self_vec: torch.Tensor,
        global_vec: torch.Tensor,
        h: torch.Tensor,
        resets: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[LaneActionDist, torch.Tensor, torch.Tensor]:
        """Returns ``(dist, tokens, new_state)``.

        Shapes: ``entities`` (B,T,32,40), ``entity_pad_mask`` (B,T,32) bool,
        ``self_vec`` (B,T,64), ``global_vec`` (B,T,48),
        ``h`` (1,B,``cfg.actor_state_dim``).
        """
        B, T, S, _ = entities.shape
        core_out, tokens, valid, hn = self._actor_trunk(
            entities, entity_pad_mask, self_vec, global_vec, h, resets
        )
        dist = self._actor_heads(core_out, tokens, valid, (B, T, S), action_masks)
        return dist, tokens.reshape(B, T, S, -1), hn

    # -- combined ----------------------------------------------------------

    def forward(
        self,
        *,
        entities: torch.Tensor,
        entity_pad_mask: torch.Tensor,
        self_vec: torch.Tensor,
        global_vec: torch.Tensor,
        priv_entities: torch.Tensor,
        priv_pad_mask: torch.Tensor,
        priv_vec: torch.Tensor,
        state: RecurrentState,
        resets: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[LaneActionDist, torch.Tensor, RecurrentState]:
        B, T, S, _ = entities.shape
        core_out, tokens, valid, h_actor = self._actor_trunk(
            entities, entity_pad_mask, self_vec, global_vec, state.actor, resets
        )
        dist = self._actor_heads(core_out, tokens, valid, (B, T, S), action_masks)
        value, h_critic = self.critic(
            priv_entities,
            priv_pad_mask,
            priv_vec,
            self_vec,
            global_vec,
            state.critic,
            resets,
            actor_core=core_out,
        )
        return dist, value, RecurrentState(actor=h_actor, critic=h_critic)

    # -- convenience -------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs_batch: Dict[str, torch.Tensor],
        state: RecurrentState,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, RecurrentState]:
        """One-step rollout helper.  All tensors are (B, 1, ...).

        Prefer :class:`lanerl_rl.infer.BatchedActor` for the actual rollout
        loop: it batches every parallel environment into one forward pass, which
        is worth 29x per decision (see that module's benchmark).
        """
        masks = obs_batch.get("action_masks")
        dist, value, new_state = self.forward(
            entities=obs_batch["entities"],
            entity_pad_mask=obs_batch["entity_pad_mask"],
            self_vec=obs_batch["self_vec"],
            global_vec=obs_batch["global_vec"],
            priv_entities=obs_batch["priv_entities"],
            priv_pad_mask=obs_batch["priv_pad_mask"],
            priv_vec=obs_batch["priv_vec"],
            state=state,
            resets=obs_batch.get("resets"),
            action_masks=masks,
        )
        action = dist.mode() if deterministic else dist.sample()
        return action, dist.log_prob(action), value, new_state

    def actor_parameter_names(self) -> set:
        return {n for n, _ in self.named_parameters() if not n.startswith("critic.")}

    def critic_parameter_names(self) -> set:
        return {n for n, _ in self.named_parameters() if n.startswith("critic.")}


def assert_actor_critic_disjoint(policy: LanePolicy) -> None:
    """Fail if any parameter tensor is shared between the actor and the critic.

    The asymmetric-critic trick only works if the privileged information stays
    on the critic side.  Weight tying between the two trunks would leak it.
    Note that the critic *consuming* the actor's core output is not sharing:
    the tensor flows one way, detached, and no parameter is common.
    """
    actor_ids = {
        id(p) for n, p in policy.named_parameters() if not n.startswith("critic.")
    }
    shared = [n for n, p in policy.named_parameters() if n.startswith("critic.") and id(p) in actor_ids]
    if shared:
        raise AssertionError(f"actor/critic share parameters: {shared}")
