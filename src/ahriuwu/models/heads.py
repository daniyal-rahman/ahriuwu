"""Prediction heads for agent training.

Phase 2 (Agent Finetuning): RewardHead, PolicyHead
Phase 3 (Imagination Training): ValueHead (initialized later)

Reference: DreamerV4 Section 3.3
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateHead(nn.Module):
    """Aux game-state regression head (offset n=0 only).

    Predicts label-known scalar state — own HP fraction, own level/18, lane-
    opponent HP fraction, opponent visibility — from the agent token. The point
    is the GRADIENT, not the readout: the v7 tokenizer preserves HUD detail too
    weakly for probes (cross-game HP R2~0.16), so this forces game semantics
    into the trainable agent blocks straight from replay labels. Targets live
    in [0,1]; missing labels / unseen opponents are handled by the caller's
    masked MSE (multiply squared error by the per-target validity mask).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_targets: int = 4):
        super().__init__()
        self.num_targets = num_targets
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, num_targets)
        # Zero-init like the other heads: predictions start at 0, so
        # d(loss)/d(agent_out) through this head is 0 until step 2.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, agent_out: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, num_targets), unbounded (targets are in [0,1])."""
        return self.out(self.mlp(agent_out))


class RewardHead(nn.Module):
    """Reward prediction head with symexp twohot output.

    Predicts rewards for current and future timesteps using MTP (Multi-Token Prediction).
    Uses symexp twohot representation for robust learning across reward scales.

    Reference: DreamerV4 Section 3.3 "Behavior cloning and reward model"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_buckets: int = 255,
        mtp_length: int = 9,
        bucket_low: float = -3.0,
        bucket_high: float = 3.0,
    ):
        """Initialize reward head.

        Args:
            input_dim: Dimension of agent token features
            hidden_dim: Hidden layer dimension
            num_buckets: Number of twohot buckets (paper uses 255)
            mtp_length: Multi-token prediction length (paper Eq 9: n=0..L with L=8 = 9 predictions)
            bucket_low: Lower bound for symlog value buckets. Default -3.0 =
                symlog(~-19). Solo-gold reward (Δ own gold_total) gives tiny
                per-frame values and O(0.5-1) discounted returns; ±3 symlog
                leaves headroom for gold_scale tuning + kill/streak spikes
                without saturating. TUNE once real return magnitudes are seen.
            bucket_high: Upper bound for symlog value buckets (3.0 = symlog(~19)).
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.mtp_length = mtp_length
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # MTP heads: predict reward for t, t+1, ..., t+L (paper Eq 9: n=0..L, L=8, so 9 heads)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_buckets) for _ in range(mtp_length)
        ])

        # Zero-init output heads: initial predictions are zero/uniform,
        # which is a good starting point. Hidden layers keep default init.
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Register bucket centers as buffer
        self.register_buffer(
            "bucket_centers",
            torch.linspace(bucket_low, bucket_high, num_buckets)
        )

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Predict reward distributions.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, L, num_buckets) logits for each timestep and MTP offset
        """
        x = self.mlp(agent_tokens)  # (B, T, hidden_dim)

        # Predict for each MTP offset
        logits = torch.stack([head(x) for head in self.heads], dim=2)
        return logits  # (B, T, L, num_buckets)

    def predict(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Get expected reward values.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, L) predicted reward values (in original scale via symexp)
        """
        from .returns import twohot_decode, symexp

        logits = self.forward(agent_tokens)
        symlog_values = twohot_decode(logits, self.bucket_centers)
        return symexp(symlog_values)


class PolicyHead(nn.Module):
    """Policy head for action prediction (vectorized binary + binned movement).

    Phase 2: Trained with behavioral cloning
    Phase 3: Trained with PMPO on imagined trajectories

    Predicts a FACTORIZED action distribution:
    - ``num_abilities`` independent binary ability actions (vectorized binary,
      per DreamerV4 paper). Each ability (Q/W/E/R/Flash/Ignite/AA/Recall/Stride)
      is an independent Bernoulli. Exponentially cheaper than one big categorical.
    - Movement (x, y) ∈ [0, 1] as TWO independent per-axis categoricals over
      ``movement_bins`` bins each (paper-style discretized/foveated mouse). This
      replaces the old sigmoid+MSE head, which had no ``log_prob`` and therefore
      received ZERO gradient under the PMPO (likelihood-ratio) objective. With
      categorical movement, both BC (max-likelihood of the demonstrated bin) and
      PMPO (advantage-weighted log-prob of the sampled bin) train movement.

    The full action log-prob is the sum of the per-ability Bernoulli log-probs
    and the two per-axis categorical log-probs (a product of independent factors).

    Uses MTP to predict actions for multiple future timesteps.

    Reference: DreamerV4 Section 3.3
    """

    def __init__(
        self,
        input_dim: int,
        num_abilities: int = 9,  # = len(constants.ABILITY_KEYS); pass explicitly to track
        hidden_dim: int = 256,
        mtp_length: int = 9,
        movement_dim: int = 2,
        movement_bins: int = 21,
        movement_gate: bool = False,
        movement_mode: str = "axis",
    ):
        """Initialize policy head.

        Args:
            input_dim: Dimension of agent token features
            num_abilities: Number of independent binary abilities (default 9 =
                len(ABILITY_KEYS): Q/W/E/R/Flash/Ignite/AA/Recall/Stride)
            hidden_dim: Hidden layer dimension
            mtp_length: Multi-token prediction length (paper Eq 9: n=0..L with L=8 = 9 predictions)
            movement_dim: Movement axes (default 2 for x, y)
            movement_bins: Number of discrete bins PER AXIS for movement. Each
                axis ∈ [0, 1] is split into this many equal-width bins whose
                centers tile the interval; bin ``i`` center = i/(bins-1). 21 bins
                ≈ 5% screen resolution per step, a sane foveated-grid default.
        """
        super().__init__()
        if movement_mode not in ("axis", "joint_noop"):
            raise ValueError(f"movement_mode={movement_mode!r} not in ('axis','joint_noop')")
        if movement_mode == "joint_noop" and movement_gate:
            raise ValueError(
                "movement_gate and movement_mode='joint_noop' are alternative "
                "solutions to the SAME problem (most frames carry no new order) "
                "and must not be combined: joint_noop already spends a class on "
                "'no new order', so a gate on top would model it twice.")
        self.num_abilities = num_abilities
        self.mtp_length = mtp_length
        self.movement_dim = movement_dim
        self.movement_bins = movement_bins
        self.movement_mode = movement_mode
        # joint_noop: ONE categorical over the flattened bins x bins grid plus a
        # trailing NO_OP class meaning "the previous order is still executing".
        # This works because League orders persist -- a no-op frame is a real,
        # correct action, not a missing label. Two consequences vs 'axis':
        #   * P(x,y) is joint, so it can express "top-left OR bottom-right";
        #     the per-axis product cannot (it would also put mass on the other
        #     two corners).
        #   * no previous action is needed to score a frame, so PMPO/Phase 3
        #     works with the plain categorical log_prob.
        # Cost: 441 classes see ~21x fewer examples each than 21 per-axis bins.
        self.movement_classes = movement_bins * movement_bins + 1 if movement_mode == "joint_noop" else None
        self.NO_OP = (self.movement_classes - 1) if self.movement_classes else None

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # MTP heads for vectorized binary abilities: each head predicts
        # num_abilities independent logits (one per ability key)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_abilities) for _ in range(mtp_length)
        ])

        # MTP heads for binned movement. 'axis': movement_dim * movement_bins
        # logits = one categorical per axis. 'joint_noop': one categorical over
        # bins**2 grid cells + NO_OP.
        _mv_out = (self.movement_classes if movement_mode == "joint_noop"
                   else movement_dim * movement_bins)
        self.movement_heads = nn.ModuleList([
            nn.Linear(hidden_dim, _mv_out) for _ in range(mtp_length)
        ])

        # Optional STICKY-CATEGORICAL movement (the action-model rewrite):
        # humans issue ~2-5 discrete movement commands/s but the data is 20fps
        # per-frame held targets, so ~77% of frames are "repeat the previous
        # action" and a plain categorical mostly learns to copy. With the gate,
        # movement is a MIXTURE: with prob (1-g) repeat the previous bin, with
        # prob g draw a fresh bin from the categorical. One gate logit per MTP
        # offset (the cursor moves as a unit; a per-axis gate would let x jump
        # while y holds, which humans don't do).
        self.movement_gate = movement_gate
        if movement_gate:
            self.gate_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(mtp_length)
            ])
            for head in self.gate_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        # Zero-init output heads: initial predictions are zero/uniform
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.movement_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Bin centers tile [0, 1]: center[i] = i / (bins - 1). Registered as a
        # buffer so .to(device)/checkpointing move it with the module.
        self.register_buffer(
            "movement_bin_centers",
            torch.linspace(0.0, 1.0, movement_bins),
        )

    def discretize_movement(self, movement: torch.Tensor) -> torch.Tensor:
        """Map continuous movement (..., 2) ∈ [0, 1] to nearest bin indices (..., 2).

        Out-of-range values are clamped into [0, 1] first. Uses round-to-nearest
        against the (bins-1)-spaced grid so the inverse (bin center) is the
        closest representable point.
        """
        m = movement.clamp(0.0, 1.0)
        idx = (m * (self.movement_bins - 1)).round().long()
        return idx.clamp(0, self.movement_bins - 1)

    def forward(self, agent_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ability logits and per-axis movement logits.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            tuple of:
                ability_logits: (B, T, L, num_abilities) independent binary logits
                movement_logits: (B, T, L, movement_dim, movement_bins) per-axis
                    categorical logits over movement bins
        """
        x = self.mlp(agent_tokens)  # (B, T, hidden_dim)
        B, T = x.shape[0], x.shape[1]

        # Vectorized binary ability predictions (independent Bernoulli per ability)
        ability_logits = torch.stack([head(x) for head in self.heads], dim=2)

        # Movement logits. Shape depends on movement_mode:
        #   'axis'       -> (B, T, L, movement_dim, movement_bins)
        #   'joint_noop' -> (B, T, L, bins**2 + 1)
        movement_logits = torch.stack(
            [mhead(x) for mhead in self.movement_heads], dim=2
        )
        if self.movement_mode == "axis":
            movement_logits = movement_logits.view(
                B, T, self.mtp_length, self.movement_dim, self.movement_bins
            )

        return ability_logits, movement_logits

    def gate_logits(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Movement-command gate logits, (B, T, L). Requires ``movement_gate``.

        sigmoid(logit) = P(issue a NEW movement command at this offset) — the
        mixture weight of the fresh-categorical branch vs repeat-previous-bin.
        (Recomputes the tiny shared trunk; negligible next to the backbone.)
        """
        assert self.movement_gate, "PolicyHead was built without movement_gate"
        x = self.mlp(agent_tokens)
        return torch.stack([h(x).squeeze(-1) for h in self.gate_heads], dim=2)

    # ---- joint_noop encode/decode. Single source of truth for the class<->grid
    # mapping so the dataset, trainer and inference decoder cannot disagree.
    def joint_encode(self, xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        """(x_bin, y_bin) long tensors -> flat grid class. Row-major: y*bins+x."""
        return yi * self.movement_bins + xi

    def joint_decode(self, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """flat class -> (x_bin, y_bin, is_noop). x/y are 0 where is_noop."""
        noop = cls == self.NO_OP
        safe = torch.where(noop, torch.zeros_like(cls), cls)
        return safe % self.movement_bins, safe // self.movement_bins, noop

    def joint_to_unit(self, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """flat class -> (xy in [0,1] shape (...,2), is_noop). Bin i center =
        i/(bins-1), matching the 'axis' decoding so the two modes agree."""
        xi, yi, noop = self.joint_decode(cls)
        d = max(self.movement_bins - 1, 1)
        return torch.stack([xi.float() / d, yi.float() / d], dim=-1), noop

    def gated_movement_log_prob(
        self,
        movement_logits: torch.Tensor,   # (B, N, L, movement_dim, bins) — pre-sliced
        gate_logits: torch.Tensor,       # (B, N, L) — same slicing
        target_idx: torch.Tensor,        # (B, N, L, movement_dim) long
        prev_idx: torch.Tensor,          # (B, N, L, movement_dim) long
    ) -> torch.Tensor:
        """Mixture log-prob of the sticky categorical, (B, N, L).

        transition (target != prev on any axis): log g + log p_cat(target)
        hold: log((1-g) + g * p_cat(target))  [the categorical may also land on
        the previous bin — both branches explain a hold], computed as
        logaddexp(log(1-g), log g + log p_cat(target)) for stability.
        """
        lsm = F.log_softmax(movement_logits, dim=-1)
        cat_lp = lsm.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # (B,N,L)
        log_g = F.logsigmoid(gate_logits)
        log_1mg = F.logsigmoid(-gate_logits)
        transition = (target_idx != prev_idx).any(dim=-1)                          # (B,N,L)
        return torch.where(transition, log_g + cat_lp,
                           torch.logaddexp(log_1mg, log_g + cat_lp))

    def sample(
        self, agent_tokens: torch.Tensor, temperature: float = 1.0,
        prev_movement_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the factorized policy.

        Args:
            agent_tokens: (B, T, D) agent token features
            temperature: Sampling temperature (1.0 = standard, 0 = greedy/argmax)

        Returns:
            tuple of:
                abilities: (B, T, L, num_abilities) binary samples {0, 1}
                movement: (B, T, L, movement_dim) continuous (x, y) ∈ [0, 1],
                    decoded from the sampled bin centers (ready to feed back to
                    the action-conditioned dynamics, which expects continuous xy)
                movement_idx: (B, T, L, movement_dim) sampled bin indices (long),
                    kept so log_prob can be computed on the exact sampled bins
        """
        ability_logits, movement_logits = self.forward(agent_tokens)

        if temperature == 0:
            abilities = (ability_logits > 0).float()
            movement_idx = movement_logits.argmax(dim=-1)
        else:
            probs = torch.sigmoid(ability_logits / temperature)
            abilities = torch.bernoulli(probs)
            mp = F.softmax(movement_logits / temperature, dim=-1)
            # multinomial needs 2D (N, K); flatten the leading dims
            flat = mp.reshape(-1, mp.shape[-1])
            sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
            movement_idx = sampled.view(*movement_logits.shape[:-1])

        if self.movement_mode == "joint_noop":
            # movement_idx is (B,T,L) flat classes. Decode to xy; on NO_OP repeat
            # the previous action if the caller supplied one (that is what the
            # game does anyway -- the standing order keeps executing), else hold
            # screen centre.
            movement, noop = self.joint_to_unit(movement_idx)      # (B,T,L,2), (B,T,L)
            if prev_movement_idx is not None:
                prev = prev_movement_idx
                if prev.dim() == movement_idx.dim() + 1:           # (B,T,L,1)
                    prev = prev.squeeze(-1)
                prev_xy, _ = self.joint_to_unit(prev.expand_as(movement_idx))
                movement = torch.where(noop.unsqueeze(-1), prev_xy, movement)
            else:
                movement = torch.where(noop.unsqueeze(-1),
                                       torch.full_like(movement, 0.5), movement)
            return abilities, movement, movement_idx

        # Sticky-categorical decode: sample the gate; where it says "hold",
        # repeat the caller-supplied previous bin instead of the fresh sample.
        # prev_movement_idx broadcasts against (B, T, L, movement_dim) — at
        # inference the caller passes the last executed action for every offset
        # (only the decoded offset is consumed).
        if self.movement_gate and prev_movement_idx is not None:
            gl = self.gate_logits(agent_tokens)                    # (B, T, L)
            if temperature == 0:
                fire = (gl > 0).unsqueeze(-1)
            else:
                fire = torch.bernoulli(torch.sigmoid(gl / temperature)).bool().unsqueeze(-1)
            prev = prev_movement_idx.expand_as(movement_idx)
            movement_idx = torch.where(fire, movement_idx, prev)

        movement = self.movement_bin_centers[movement_idx]  # (B, T, L, movement_dim)
        return abilities, movement, movement_idx

    def log_prob(
        self,
        agent_tokens: torch.Tensor,
        ability_actions: torch.Tensor,
        movement_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Log probability of factorized actions (abilities + binned movement).

        Args:
            agent_tokens: (B, T, D) agent token features
            ability_actions: (B, T, L, num_abilities) binary targets {0, 1}
            movement_actions: movement targets, either continuous (B, T, L, 2) in
                [0, 1] (discretized internally to the nearest bin) OR already-long
                bin indices (B, T, L, 2). Long dtype is treated as indices.

        Returns:
            (B, T, L) sum of per-ability Bernoulli log-probs and the two per-axis
            categorical movement log-probs. ``L`` matches the action tensors'
            MTP axis: the first ``L`` MTP heads are scored (so L=1 scores offset
            n=0 only — the on-policy case for PMPO).
        """
        if self.movement_gate:
            raise ValueError(
                "log_prob() is not gate-aware: a movement_gate PolicyHead's "
                "movement likelihood is the sticky-categorical mixture, which "
                "needs the previous movement bin (see gated_movement_log_prob). "
                "BC uses the gated loss in the trainer; wire prev-action plumbing "
                "into the imagination path before running PMPO on a gated head, "
                "or use movement_mode='joint_noop', whose NO_OP class makes the "
                "previous action unnecessary."
            )
        ability_logits, movement_logits = self.forward(agent_tokens)
        # Score only as many MTP offsets as the caller supplied targets for.
        L = ability_actions.shape[2]
        ability_logits = ability_logits[:, :, :L, :]

        # Per-ability Bernoulli log-prob, summed over abilities.
        ability_lp = -F.binary_cross_entropy_with_logits(
            ability_logits, ability_actions, reduction='none'
        ).sum(dim=-1)  # (B, T, L)

        if self.movement_mode == "joint_noop":
            # ONE categorical over grid cells + NO_OP. No previous action needed:
            # "no new order" is an explicit class, not something inferred by
            # comparing to the last frame. This is what lets PMPO score a
            # dreamed trajectory.
            movement_logits = movement_logits[:, :, :L, :]          # (B,T,L,classes)
            move_idx = movement_actions
            if move_idx.dtype not in (torch.long, torch.int32, torch.int64):
                raise TypeError(
                    "joint_noop movement targets must be LONG class indices "
                    f"(got {move_idx.dtype}); a continuous (x,y) cannot express NO_OP.")
            if move_idx.dim() == movement_logits.dim():               # (B,T,L,1)
                move_idx = move_idx.squeeze(-1)
            move_lp = F.log_softmax(movement_logits, dim=-1).gather(
                -1, move_idx.unsqueeze(-1)).squeeze(-1)               # (B,T,L)
            return ability_lp + move_lp

        # 'axis': per-axis categorical log-prob, summed over the 2 axes.
        movement_logits = movement_logits[:, :, :L, :, :]
        if movement_actions.dtype in (torch.long, torch.int32, torch.int64):
            move_idx = movement_actions
        else:
            move_idx = self.discretize_movement(movement_actions)
        move_log_softmax = F.log_softmax(movement_logits, dim=-1)  # (B,T,L,move_dim,bins)
        move_lp = move_log_softmax.gather(
            -1, move_idx.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)  # gather selected bin, sum over axes -> (B,T,L)

        return ability_lp + move_lp


class ValueHead(nn.Module):
    """Value head for imagination training (Phase 3 only).

    Initialized at start of Phase 3, not during Phase 2.
    Predicts discounted sum of future rewards using symexp twohot.

    Reference: DreamerV4 Section 3.3 "Reinforcement learning"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_buckets: int = 255,
        bucket_low: float = -3.0,
        bucket_high: float = 3.0,
    ):
        """Initialize value head.

        Args:
            input_dim: Dimension of agent token features
            hidden_dim: Hidden layer dimension
            num_buckets: Number of twohot buckets
            bucket_low: Lower bound for symlog value buckets. Default -3.0 =
                symlog(~-19). Solo-gold reward (Δ own gold_total) gives tiny
                per-frame values and O(0.5-1) discounted returns; ±3 symlog
                leaves headroom for gold_scale tuning + kill/streak spikes
                without saturating. TUNE once real return magnitudes are seen.
            bucket_high: Upper bound for symlog value buckets (3.0 = symlog(~19)).
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high

        # MLP for value prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_buckets),
        )

        # Zero-init output layer: initial predictions are zero/uniform,
        # which is a good starting point. Hidden layers keep default init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Register bucket centers as buffer
        self.register_buffer(
            "bucket_centers",
            torch.linspace(bucket_low, bucket_high, num_buckets)
        )

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Predict value distribution logits.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T, num_buckets) logits
        """
        return self.mlp(agent_tokens)

    def predict(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        """Get expected value estimates.

        Args:
            agent_tokens: (B, T, D) agent token features

        Returns:
            (B, T) predicted values (in original scale via symexp)
        """
        from .returns import twohot_decode, symexp

        logits = self.forward(agent_tokens)
        symlog_values = twohot_decode(logits, self.bucket_centers)
        return symexp(symlog_values)


# ---------------------------------------------------------------------------
# Frozen behavioral prior (Fix 3)
# ---------------------------------------------------------------------------

def create_behavioral_prior(policy_head: PolicyHead) -> PolicyHead:
    """Create a frozen deep copy of a policy head to serve as behavioral prior.

    The prior is used for KL regularization during imagination training (Phase 3)
    to prevent the policy from diverging too far from the behavior-cloned policy.

    Args:
        policy_head: Trained policy head (typically after Phase 2)

    Returns:
        Frozen deep copy of the policy head (all requires_grad=False)
    """
    prior = copy.deepcopy(policy_head)
    prior.requires_grad_(False)
    return prior


def kl_to_prior(
    policy_logits: torch.Tensor,
    prior_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence from current policy to frozen prior (categorical).

    KL(policy || prior) = sum(policy * (log policy - log prior))

    Args:
        policy_logits: (..., num_actions) logits from current policy
        prior_logits: (..., num_actions) logits from frozen behavioral prior

    Returns:
        (...,) KL divergence per state (non-negative)
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    prior_log_probs = F.log_softmax(prior_logits, dim=-1)
    policy_probs = policy_log_probs.exp()

    kl = (policy_probs * (policy_log_probs - prior_log_probs)).sum(dim=-1)
    return kl


def kl_to_prior_continuous(
    policy_mean: torch.Tensor,
    policy_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence for continuous (Gaussian) distributions.

    KL(N(mu1,s1) || N(mu2,s2)) = log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5

    Summed over the last dimension (e.g., 2 for x,y movement).

    Args:
        policy_mean: (..., D) mean of current policy
        policy_std: (..., D) std of current policy
        prior_mean: (..., D) mean of frozen prior
        prior_std: (..., D) std of frozen prior

    Returns:
        (...,) KL divergence per state
    """
    var_ratio = (policy_std / prior_std).pow(2)
    diff_sq = ((policy_mean - prior_mean) / prior_std).pow(2)
    kl = 0.5 * (var_ratio + diff_sq - 1 - var_ratio.log())
    return kl.sum(dim=-1)


# ---------------------------------------------------------------------------
# Freeze/unfreeze utilities for imagination training (Fix 4)
# ---------------------------------------------------------------------------

def freeze_for_imagination(
    model: nn.Module,
    dynamics_attr: str = "dynamics",
    freeze_reward: bool = False,
) -> nn.Module:
    """Freeze the dynamics transformer for imagination training.

    During imagination (Phase 3), gradients flow through the frozen dynamics
    model to train the policy and value heads. The dynamics weights themselves
    are not updated.

    Args:
        model: Agent model containing dynamics, policy, value, reward submodules
        dynamics_attr: Attribute name for the dynamics transformer
        freeze_reward: If False (default), reward head stays unfrozen
            (paper trains reward head during imagination too)

    Returns:
        The model (modified in-place)
    """
    # Freeze dynamics transformer
    dynamics = getattr(model, dynamics_attr, None)
    if dynamics is None:
        raise ValueError(f"Model has no attribute '{dynamics_attr}'. Cannot freeze dynamics.")
    dynamics.requires_grad_(False)

    # Optionally freeze reward head
    if freeze_reward:
        reward_head = getattr(model, "reward_head", None)
        if reward_head is not None:
            reward_head.requires_grad_(False)

    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters in the model.

    Call this to restore full training after imagination phase, or when
    switching between training phases.

    Args:
        model: Agent model

    Returns:
        The model (modified in-place)
    """
    model.requires_grad_(True)
    return model
