#!/usr/bin/env python3
"""Phase 3: Imagination Training - RL inside the frozen world model.

DreamerV4 Section 3.3 "Reinforcement learning". With the dynamics backbone AND
the agent tokens / reward head FROZEN (Phase 2 output), we:

1. Roll out IMAGINED trajectories from real context latents, *feeding the
   policy's own sampled actions back into the dynamics* so the dream is
   on-policy (the archived trainer never did this — its dream was
   policy-independent).
2. Generate each dreamed frame with the correct denoiser: the KV-cached
   ``DynamicsTransformer.rollout()`` (right tau direction, integer shortcut step
   d). The archived loop denoised tau BACKWARDS and passed a float step_size, so
   every "dreamed" latent was noise.
3. Compute lambda-returns (gamma=0.997, lambda=0.95) on the imagined
   rewards/values (Eq 10).
4. Train ``ValueHead`` (twohot regression to the returns) and ``PolicyHead``
   (PMPO, Eq 11) with the KL regularizer matched to the FACTORIZED policy —
   per-ability Bernoulli KL + per-axis movement categorical KL, to a frozen
   behavioral prior (a copy of the Phase 2 policy).

Gradients flow only into the value + policy heads; the imagined states
(agent tokens) are treated as fixed data, which is exactly what PMPO's
sign-of-advantage update and the twohot value regression need.

Usage (real run):
    PYTHONPATH=src python scripts/train_imagination.py \
        --agent-checkpoint checkpoints/agent_finetune_latest.pt \
        --latents-dir /opt/ahriuwu/latents_pt \
        --labels-root /mnt/storage/ahriuwu-data/replays \
        --epochs 1

Smoke test (CPU, synthetic, no checkpoint/data needed):
    PYTHONPATH=src python scripts/train_imagination.py --smoke-test

Reference: DreamerV4 Section 3.3 "Reinforcement learning".
"""

import argparse
import copy
import glob
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM

# WHICH MTP HEAD PHASE 3 MAY USE.
# Phase-2 BC runs `for n in range(1, mtp_length)` -- offset 0 is deliberately
# dropped because the action token for frame t is an INPUT to frame t's tokens,
# so predicting a_t from h_t is a label leak. Consequence: head 0 is zero-init
# and receives no gradient, ever. Verified on a 99,421-step checkpoint:
#   movement_heads.0.weight norm = 0.0   vs  offsets 1..8 norm ~ 23
# Phase 3 previously sampled and scored at offset 0, so its "behavior-cloned"
# policy was Bernoulli(0.5) per ability and uniform over movement, and the
# behavioral prior (a copy of the same zeros) made the PMPO KL vacuous.
# Offset 1 is the first TRAINED head and is what inference uses (agent_infer).
MTP_OFFSET = 1
from ahriuwu.models import (
    create_dynamics,
    RewardHead,
    PolicyHead,
    ValueHead,
    DiffusionSchedule,
    symlog,
    twohot_loss,
    compute_lambda_returns,
    compute_pmpo_loss,
    factorized_policy_kl,
    bernoulli_kl_logits,
    RunningRMS,
)
from ahriuwu.data.dataset import VideoGroupedSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
    load_state_dict_guarded,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Imagination Training (PMPO + Value)")
    add_training_args(parser)
    parser.add_argument("--agent-checkpoint", type=str, default=None,
                        help="Phase 2 checkpoint (dynamics + reward + policy). "
                             "Omit only with --smoke-test.")
    parser.add_argument("--latents-dir", type=str, default=None,
                        help="Dir of packed per-match latents for the rollout context.")
    parser.add_argument("--labels-root", type=str, default=None,
                        help="Dir of <match>/labels.json (used to build the context dataset).")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Optional outcomes manifest; dummy outcomes used if omitted.")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16, help="Context window length (frames).")
    parser.add_argument("--degenerate-advantage-patience", type=int, default=5,
                    help="Abort if pos_advantage_frac stays >=99.5%% or <=0.5%% for this "
                         "many consecutive logged steps. PMPO uses only sign(A), so a "
                         "one-sided split reduces the loss to likelihood maximisation of "
                         "the policy's own samples -- mode collapse with the reward "
                         "contributing nothing. Observed at 100%% on every step of the "
                         "first real runs (zero-init critic, near-zero imagined rewards). "
                         "0 disables.")
    parser.add_argument("--horizon", type=int, default=8, help="Imagination rollout length.")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--mtp-length", type=int, default=9)
    parser.add_argument("--num-buckets", type=int, default=255)
    parser.add_argument("--movement-bins", type=int, default=21)
    parser.add_argument("--hidden-dim", type=int, default=256)
    # RL hyperparameters (paper defaults)
    parser.add_argument("--gamma", type=float, default=0.997, help="Discount factor.")
    parser.add_argument("--lambda_", type=float, default=0.95, help="TD(lambda) parameter.")
    parser.add_argument("--pmpo-alpha", type=float, default=0.5, help="PMPO D+/D- balance.")
    parser.add_argument("--pmpo-beta", type=float, default=0.3, help="PMPO prior-KL weight.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Policy sampling temperature.")
    # Generation (shortcut denoiser) config
    parser.add_argument("--gen-steps", type=int, default=4,
                        help="Denoising steps K per dreamed frame (shortcut). d = k_max // K.")
    parser.add_argument("--k-max", type=int, default=64, help="Shortcut grid size.")
    parser.add_argument("--tau-ctx", type=float, default=0.1,
                        help="Context corruption WIDTH for rollout: context tau ~ U(1-tau_ctx, 1).")
    # Dynamics arch flags (must match the checkpoint)
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--agent-layers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny synthetic CPU step + assert movement_heads get PMPO gradient.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N optimizer steps (0 = no limit). For a real-checkpoint "
                             "smoke run: proves a step happens on real weights without "
                             "starting a training run.")
    parser.set_defaults(num_workers=0, wandb=False)
    add_wandb_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def build_dynamics(args, *, use_actions, device):
    return create_dynamics(
        size=args.model_size, latent_dim=args.latent_dim,
        use_agent_tokens=True, use_actions=use_actions, num_tasks=1,
        agent_layers=args.agent_layers, use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens, num_kv_heads=args.num_kv_heads,
        gradient_checkpointing=False,
    ).to(device)


def load_phase2(args, device):
    """Load Phase 2 dynamics (frozen) + reward head (frozen) + policy head (trainable)."""
    ckpt = torch.load(args.agent_checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    cfg = ckpt.get("dynamics_config") or {}
    args.latent_dim = cfg.get("latent_dim", saved.get("latent_dim", args.latent_dim))
    args.mtp_length = saved.get("mtp_length", args.mtp_length)
    args.num_buckets = saved.get("num_buckets", args.num_buckets)
    args.movement_bins = saved.get("movement_bins", args.movement_bins)
    args.hidden_dim = saved.get("hidden_dim", args.hidden_dim)

    # The Phase-2 policy may carry a sticky movement gate. Rebuilding without it
    # makes the strict load below reject every gate tensor, and rebuilding WITH it
    # against an ungated checkpoint leaves the gate at random init -- so read the
    # flag from the checkpoint rather than assuming, exactly as agent_infer does.
    args.movement_gate = saved.get("movement_gate", False)
    # Restore EVERY arch flag the checkpoint recorded. Restoring only a subset built
    # a 512-dim MHA model for a 768-dim GQA checkpoint and failed on register_tokens;
    # omitting movement_mode built a 42-logit head for a 442-class checkpoint.
    args.movement_mode = saved.get("movement_mode", "axis")
    for _k in ("model_size", "num_kv_heads", "agent_layers", "num_register_tokens",
               "soft_cap", "no_qk_norm"):
        if _k in saved:
            setattr(args, _k, saved[_k])
    _cfg_size = cfg.get("size_preset")
    if _cfg_size:
        args.model_size = _cfg_size

    dyn_state = ckpt["dynamics_state_dict"]
    if any(k.startswith("_orig_mod.") for k in dyn_state):
        dyn_state = {k.replace("_orig_mod.", ""): v for k, v in dyn_state.items()}
    use_actions = cfg.get("use_actions", any("action_embed" in k for k in dyn_state))
    dynamics = build_dynamics(args, use_actions=use_actions, device=device)
    load_state_dict_guarded(dynamics, dyn_state, what="Phase-3 dynamics backbone")
    dynamics.eval()
    dynamics.requires_grad_(False)
    model_dim = dynamics.model_dim

    reward_head = RewardHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets, mtp_length=args.mtp_length).to(device)
    reward_head.load_state_dict(ckpt["reward_head_state_dict"])
    reward_head.eval()
    reward_head.requires_grad_(False)

    policy_head = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                             hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins,
                             movement_gate=args.movement_gate,
                             movement_mode=args.movement_mode).to(device)
    policy_head.load_state_dict(ckpt["policy_head_state_dict"])

    # NOTE: this used to `raise SystemExit` on a gated checkpoint. The guard was
    # protecting a REAL incompatibility, not a hypothetical one -- PolicyHead.
    # log_prob() is mathematically undefined for a sticky-categorical head,
    # because the movement likelihood is a mixture whose "hold" branch is a point
    # mass on the PREVIOUS executed bin, and imagination did not carry that bin.
    # Deleting the guard would only have moved the crash deeper. So the guard is
    # gone because the incompatibility is gone: imagine() now threads the
    # previous executed order through the dream (it needs it for joint_noop's
    # NO_OP anyway), run_step scores movement with the head's own
    # gated_movement_log_prob -- the same function BC trains against
    # (train_agent_finetune.py:914) -- and gated_movement_kl below computes the
    # mixture KL exactly. This matters because the DEPLOYED checkpoint
    # (phase2_bc_clicks, DEMO_RUNBOOK 1a) is a gated one: the guard was locking
    # Phase 3 out of the live lineage.
    print(f"Loaded Phase 2 from {args.agent_checkpoint}")
    print(f"  dynamics use_actions={use_actions} (frozen), reward head frozen, policy head trainable")
    print(f"  movement_gate={args.movement_gate}")
    return dynamics, reward_head, policy_head, model_dim, use_actions


def build_context_dataset(args):
    """ReplayLatentSequenceDataset used purely for its latent context windows.

    We need sequence_length >= seq_len + horizon so the same window can also
    supply the real future actions for the CONTEXT region only (the dreamed
    region uses sampled actions). Imported lazily (circular-import dodge).
    """
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
    if not args.latents_dir or not args.labels_root:
        raise SystemExit("Real run needs --latents-dir and --labels-root (or --smoke-test).")
    outcomes = None
    if not args.manifest:
        mids = [Path(p).stem for p in glob.glob(str(Path(args.latents_dir) / "*.pt"))
                if Path(p).stem != "index"]
        outcomes = {m: False for m in mids}
    return ReplayLatentSequenceDataset(
        latents_dir=args.latents_dir, labels_root=args.labels_root,
        outcomes=outcomes, manifest_path=args.manifest,
        sequence_length=args.seq_len, stride=args.stride,
    )


# ---------------------------------------------------------------------------
# Movement bookkeeping: the standing order, and mode-aware KL
# ---------------------------------------------------------------------------
# Three movement modes reach Phase 3 and they disagree about what "an action" is:
#   axis (plain) : two independent per-axis categoricals; action = (bx, by).
#   axis + gate  : a MIXTURE -- hold the previous cell w.p. (1-g), else draw a
#                  fresh cell. Action = the executed (bx, by); scoring needs the
#                  previous cell, which is why it used to be rejected outright.
#   joint_noop   : ONE categorical over bins**2 cells + a NO_OP class. Action =
#                  the sampled class, NO_OP included -- "no new order" is a real
#                  action here, not a missing one.
# Everything below keeps those three straight in one place instead of scattering
# `if movement_mode ==` through the rollout and the loss.


def _seed_prev_movement(policy_head, last_xy, B):
    """Standing order at the start of the dream, from the last context action.

    Returns (prev_idx, prev_arg): the bookkeeping index and the same value shaped
    to broadcast into PolicyHead.sample()'s (B, T, L[, movement_dim]) expectation.
    """
    axis_idx = policy_head.discretize_movement(last_xy)          # (B, movement_dim)
    if policy_head.movement_mode == "joint_noop":
        prev_idx = policy_head.joint_encode(axis_idx[..., 0], axis_idx[..., 1])  # (B,)
        return prev_idx, prev_idx.view(B, 1, 1)
    return axis_idx, axis_idx.view(B, 1, 1, policy_head.movement_dim)


def _advance_prev_movement(policy_head, prev_idx, sampled_idx, B):
    """Standing order after one dreamed step."""
    if policy_head.movement_mode == "joint_noop":
        # NO_OP issues no new order: the previous one keeps executing.
        nxt = torch.where(sampled_idx == policy_head.NO_OP, prev_idx, sampled_idx)
        return nxt, nxt.view(B, 1, 1)
    # sample() already resolved the gate's hold branch, so this is the executed bin.
    return sampled_idx, sampled_idx.view(B, 1, 1, policy_head.movement_dim)


def movement_kl_factors(policy_head, m_logits, m_prior):
    """Slice movement logits at MTP_OFFSET into the (..., factors, classes) layout
    ``factorized_policy_kl`` reduces over.

    This is the whole of blocker 3. ``factorized_policy_kl`` sums a per-axis
    categorical KL over a ``movement_dim`` axis, which an 'axis' head has and a
    'joint_noop' head does not -- its logits are (B, T, L, classes), so indexing
    ``[:, :, off, :, :]`` raised ``IndexError: too many indices for tensor of
    dimension 4``. A joint head is not a different kind of object, just a
    distribution with ONE factor instead of two, so give it a singleton factor
    axis and the shared reducer is exactly right.
    """
    if policy_head.movement_mode == "joint_noop":
        return (m_logits[:, :, MTP_OFFSET, :].unsqueeze(-2),      # (B, H, 1, classes)
                m_prior[:, :, MTP_OFFSET, :].unsqueeze(-2))
    return (m_logits[:, :, MTP_OFFSET, :, :],                     # (B, H, move_dim, bins)
            m_prior[:, :, MTP_OFFSET, :, :])


def gated_movement_kl(policy_logits, prior_logits, policy_gate, prior_gate,
                      prev_idx, bins):
    """Exact KL between two sticky-categorical movement mixtures, (B, H), in nats.

    The executed action is a CELL, and the head's law over cells is
        P(cell) = (1 - g) * 1[cell == prev] + g * p_x(bx) * p_y(by)
    -- a mixture, which has no closed-form KL in general. It does not need one:
    written out over the bins**2 cells it is a plain categorical on a small
    discrete support, so enumerating it is EXACT. 441 cells is nothing next to a
    dynamics forward.

    Both arguments share ``prev_idx`` (same dreamed trajectory), so the point
    masses land on the same cell and the KL is finite.

    Args:
        policy_logits/prior_logits: (B, H, movement_dim, bins) per-axis logits.
        policy_gate/prior_gate: (B, H) gate logits.
        prev_idx: (B, H, movement_dim) previous executed bins.
        bins: bins per axis.
    """
    def _log_cells(axis_logits, gate_logits):
        lsm = F.log_softmax(axis_logits.float(), dim=-1)              # (B,H,2,bins)
        # Independent axes -> log p(bx,by) = log p_x(bx) + log p_y(by). Cell index
        # is bx * bins + by, matching the unsqueeze order below.
        log_cells = (lsm[..., 0, :].unsqueeze(-1)
                     + lsm[..., 1, :].unsqueeze(-2)).flatten(-2)      # (B,H,bins*bins)
        log_g = F.logsigmoid(gate_logits.float()).unsqueeze(-1)
        log_1mg = F.logsigmoid(-gate_logits.float()).unsqueeze(-1)
        mixed = log_g + log_cells
        # Add the (1-g) point mass on the previous cell, in log space.
        hold = torch.full_like(mixed, float("-inf"))
        flat_prev = (prev_idx[..., 0] * bins + prev_idx[..., 1]).unsqueeze(-1)
        hold.scatter_(-1, flat_prev, log_1mg)
        return torch.logaddexp(mixed, hold)                           # (B,H,cells)

    log_p = _log_cells(policy_logits, policy_gate)
    log_q = _log_cells(prior_logits, prior_gate)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)                # (B, H)


# ---------------------------------------------------------------------------
# Imagined rollout (on-policy: policy actions fed back into the dynamics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine(dynamics, policy_head, reward_head, value_head, z_context, args, device,
            actions_context=None):
    """Roll out ``args.horizon`` imagined steps on-policy.

    Per step t:
      * forward the FROZEN dynamics on the current latent window (context +
        dreamed-so-far) at near-clean tau, with the actions taken so far, to get
        the agent token h_t for the LAST frame in the window;
      * sample a_t ~ policy(h_t), and read r_t (frozen reward head) and v_t
        (current value head) off h_t;
      * dream the next frame z_{t+1} via DynamicsTransformer.rollout() conditioned
        on a_t (predict_frames=1, correct tau direction + integer step d), and
        append it to the window; append a_t to the action history.

    Everything here is no_grad (dynamics + reward frozen; value/policy are
    re-forwarded WITH grad in the training step on the returned agent tokens).

    Args:
      actions_context: optional real action dict for the CONTEXT frames
        ({movement: (B, Ctx, 2), <ability>: (B, Ctx) long}) so the context is
        conditioned on its true actions (in-distribution for the dynamics). The
        DREAMED frames always use the policy's sampled actions. None -> neutral
        context (center movement, no abilities).

    Returns dict of:
      agent_outs:    (B, H, D)
      ability_acts:  (B, H, num_abilities) float {0,1}
      movement_acts: (B, H, 2) continuous xy actually EXECUTED (bin centers)
      movement_idx:  sampled movement index -- (B, H) flat class under
                     'joint_noop', (B, H, movement_dim) per-axis bins under
                     'axis'. This, not movement_acts, is what PMPO scores.
      movement_prev: the standing order each step was sampled under, same shape
                     as movement_idx (the gated mixture NLL needs it).
      rewards:       (B, H) original scale
      values:        (B, H) original scale
    """
    was_training = dynamics.training
    dynamics.eval()
    B, Ctx, C, Hh, Ww = z_context.shape
    use_actions = dynamics.use_actions

    z_window = z_context  # grows by one frame each step
    # Action history aligned with z_window frames. Seed from the real context
    # actions when given (keeps the context in-distribution for the dynamics);
    # otherwise neutral (center movement, no abilities). Dreamed frames extend
    # this history with the policy's sampled actions.
    if actions_context is not None:
        move_hist = actions_context["movement"].to(device).float().clone()
        abil_hist = torch.stack(
            [actions_context[k].to(device).float() for k in ABILITY_KEYS], dim=-1
        )
    else:
        move_hist = torch.full((B, Ctx, MOVEMENT_DIM), 0.5, device=device)
        abil_hist = torch.zeros((B, Ctx, len(ABILITY_KEYS)), device=device)

    # The PREVIOUS EXECUTED movement order. Both non-plain movement modes need it:
    # joint_noop's NO_OP class means "the standing order is still executing", and
    # the gated axis head's hold branch repeats the previous bin. Seed it from the
    # last real context action (screen centre when the context is neutral) and
    # carry it forward across dreamed steps. Without this, sample() decoded every
    # NO_OP to screen centre -- injecting a fake click-the-middle into ~28% of
    # dreamed frames instead of letting the standing order run (MTP_INVESTIGATION 4).
    prev_idx, prev_arg = _seed_prev_movement(policy_head, move_hist[:, -1, :], B)

    agent_outs, ability_acts, movement_acts = [], [], []
    movement_idx_acts, movement_prevs, rewards, values = [], [], [], []
    schedule = DiffusionSchedule(device=device)

    for _ in range(args.horizon):
        Tw = z_window.shape[1]
        tau = args.tau_ctx_forward + torch.rand(B, Tw, device=device) * (1.0 - args.tau_ctx_forward)
        z_noisy, _ = schedule.add_noise(z_window, tau)
        actions_win = None
        if use_actions:
            actions_win = {"movement": move_hist}
            for i, k in enumerate(ABILITY_KEYS):
                actions_win[k] = abil_hist[..., i].long()
        d_one = torch.ones(B, dtype=torch.long, device=device)
        _, agent_out = dynamics(z_noisy, tau, step_size=d_one, actions=actions_win)
        h_t = agent_out[:, -1:, :]  # (B, 1, D) — token for the last/newest frame

        # Sample on-policy action at the first TRAINED MTP head (see MTP_OFFSET).
        abilities, movement, move_idx = policy_head.sample(
            h_t, temperature=args.temperature, prev_movement_idx=prev_arg)
        a_abil = abilities[:, 0, MTP_OFFSET, :]        # (B, num_abilities)
        a_move = movement[:, 0, MTP_OFFSET, :]         # (B, 2) EXECUTED xy
        # The SAMPLED index -- what PMPO must score. joint_noop: (B,) flat class,
        # possibly NO_OP. axis: (B, 2) per-axis bins (post-gate, i.e. executed).
        # The decoded xy above cannot stand in for it: under joint_noop a NO_OP
        # decodes to the PREVIOUS order's xy, so re-discretizing a_move would
        # score the wrong class. That lossy round-trip is exactly what
        # log_prob()'s LONG-dtype check exists to reject.
        a_move_idx = move_idx[:, 0, MTP_OFFSET]

        # reward head DOES train offset 0 (reward is a target, never an input),
        # so it keeps 0 -- verified: reward_head.heads.0 norm 2.86e+01.
        r_t = reward_head.predict(h_t)[:, 0, 0]   # (B,) original scale
        v_t = value_head.predict(h_t)[:, 0]       # (B,) original scale

        agent_outs.append(h_t[:, 0, :])
        ability_acts.append(a_abil)
        movement_acts.append(a_move)
        movement_idx_acts.append(a_move_idx)
        movement_prevs.append(prev_idx)   # the prev the gated mixture was sampled under
        rewards.append(r_t)
        values.append(v_t)

        # Advance the standing order. Under joint_noop a sampled NO_OP issues no
        # new order, so the previous one persists; the gated axis head's sample()
        # already applied its hold branch, so its index IS the executed order.
        prev_idx, prev_arg = _advance_prev_movement(policy_head, prev_idx, a_move_idx, B)

        # Dream the next frame conditioned on the SAMPLED action (fed back).
        roll_future = None
        roll_ctx = None
        if use_actions:
            roll_future = {"movement": a_move.unsqueeze(1)}  # (B, 1, 2)
            for i, k in enumerate(ABILITY_KEYS):
                roll_future[k] = a_abil[:, i].long().unsqueeze(1)  # (B, 1)
            roll_ctx = {"movement": move_hist}
            for i, k in enumerate(ABILITY_KEYS):
                roll_ctx[k] = abil_hist[..., i].long()
        z_next = dynamics.rollout(
            context=z_window, predict_frames=1,
            num_steps=args.gen_steps, k_max=args.k_max, tau_ctx=args.tau_ctx,
            actions_context=roll_ctx, actions_future=roll_future,
            device=device,
        )  # (B, 1, C, H, W)

        z_window = torch.cat([z_window, z_next], dim=1)
        move_hist = torch.cat([move_hist, a_move.unsqueeze(1)], dim=1)
        abil_hist = torch.cat([abil_hist, a_abil.unsqueeze(1)], dim=1)

    if was_training:
        dynamics.train()
    return {
        "agent_outs": torch.stack(agent_outs, dim=1),       # (B, H, D)
        "ability_acts": torch.stack(ability_acts, dim=1),   # (B, H, A)
        "movement_acts": torch.stack(movement_acts, dim=1), # (B, H, 2) executed xy
        # The sampled movement INDEX and the standing order it was sampled under.
        # joint_noop -> (B, H); axis -> (B, H, movement_dim). These are what PMPO
        # scores; movement_acts is what the dynamics was fed.
        "movement_idx": torch.stack(movement_idx_acts, dim=1),
        "movement_prev": torch.stack(movement_prevs, dim=1),
        "rewards": torch.stack(rewards, dim=1),             # (B, H)
        "values": torch.stack(values, dim=1),               # (B, H)
    }


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def run_step(roll, policy_head, policy_prior, value_head, args, device, amp_dtype, rms):
    """Value (twohot regression to lambda-returns) + policy (PMPO with factorized
    KL) losses, from a precomputed imagined rollout.

    The policy/value heads are re-forwarded WITH grad on the (fixed) agent tokens.
    """
    agent_outs = roll["agent_outs"]                 # (B, H, D), no grad
    ability_acts = roll["ability_acts"]             # (B, H, A)
    movement_idx = roll["movement_idx"]             # joint:(B,H)  axis:(B,H,move_dim)
    movement_prev = roll["movement_prev"]           # same shape
    rewards = roll["rewards"]                        # (B, H)
    values = roll["values"].detach()                # (B, H)
    B, H = rewards.shape

    with autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                  enabled=(amp_dtype in (torch.bfloat16, torch.float16)) and device != "cpu"):
        continues = torch.ones_like(rewards)  # no terminals inside imagination
        returns = compute_lambda_returns(rewards, values, continues,
                                         gamma=args.gamma, lambda_=args.lambda_)  # (B,H)
        advantages = returns - values         # raw; PMPO uses sign only

        # --- Value loss: twohot regression to lambda-returns (Eq 10) ---
        value_logits = value_head(agent_outs)                 # (B, H, num_buckets)
        value_targets = symlog(returns.detach())
        value_loss = twohot_loss(value_logits, value_targets, value_head.bucket_centers)

        # --- Policy loss: PMPO with factorized KL to the frozen prior (Eq 11) ---
        # log_prob scores the FIRST L heads, so to score offset MTP_OFFSET we pass
        # MTP_OFFSET+1 targets and read the last one. Padding with the same action
        # is harmless: only index MTP_OFFSET is read out.
        _L = MTP_OFFSET + 1
        abil_mtp = ability_acts.unsqueeze(2).expand(-1, -1, _L, -1)    # (B,H,L,A)
        # Score the SAMPLED movement INDEX, never the decoded xy: under joint_noop
        # NO_OP decodes to the previous order's xy, so a float round-trip would
        # score a real click the policy never chose.
        if policy_head.movement_mode == "joint_noop":
            move_mtp = movement_idx.unsqueeze(2).expand(-1, -1, _L)         # (B,H,L)
        else:
            move_mtp = movement_idx.unsqueeze(2).expand(-1, -1, _L, -1)     # (B,H,L,2)

        # Factorized KL needs both heads' logits at MTP_OFFSET, for policy and prior.
        a_logits, m_logits = policy_head(agent_outs)          # (B,H,L,A), (B,H,L,2,bins)
        with torch.no_grad():
            a_prior, m_prior = policy_prior(agent_outs)

        if policy_head.movement_gate:
            # PolicyHead.log_prob() is undefined for a gated head (the movement
            # law is the sticky mixture), so score it the way BC does: ability
            # BCE + the head's own gated_movement_log_prob, with the standing
            # order imagine() carried alongside the action.
            prev_mtp = movement_prev.unsqueeze(2).expand(-1, -1, _L, -1)
            g_logits = policy_head.gate_logits(agent_outs)                  # (B,H,L)
            ability_lp = -F.binary_cross_entropy_with_logits(
                a_logits[:, :, :_L, :], abil_mtp, reduction="none").sum(dim=-1)
            move_lp = policy_head.gated_movement_log_prob(
                m_logits[:, :, :_L], g_logits[:, :, :_L], move_mtp, prev_mtp)
            log_probs = (ability_lp + move_lp)[:, :, MTP_OFFSET]
        else:
            log_probs = policy_head.log_prob(
                agent_outs, abil_mtp, move_mtp)[:, :, MTP_OFFSET]
        # The prior MUST be sliced at the SAME offset as the policy. It is a
        # deepcopy of the policy, so at step 0 this KL is exactly 0 by
        # construction -- that identity is the test, and it failed: sliced at 0
        # the measured KL was 7.287 nats (ability 5.578 + movement 1.709).
        #
        # Offset 0 is never trained by BC (it is dropped as an action-conditioning
        # leak), so its logits are still at zero-init = exactly uniform. That makes
        # KL(pi || uniform) = logK - H, i.e. with --pmpo-beta > 0 the term became an
        # ENTROPY BONUS that actively erases the behaviour-cloned policy, rather
        # than an anchor holding the policy near it. Phase 3 was not merely
        # ignoring Phase 2, it was pushing away from it.
        #
        # The MTP_OFFSET = 1 fix (4c93083) reached sampling and log_prob and
        # stopped here.
        if policy_head.movement_gate:
            # The gated policy's movement law is the mixture, so anchor THAT, not
            # the bare categorical underneath it -- otherwise the KL would leave
            # the gate (the "should I issue an order at all" decision) completely
            # unregularised, free to drift to always-fire while the term reads low.
            with torch.no_grad():
                g_prior = policy_prior.gate_logits(agent_outs)
            kl = (bernoulli_kl_logits(a_logits[:, :, MTP_OFFSET, :],
                                      a_prior[:, :, MTP_OFFSET, :]).sum(dim=-1)
                  + gated_movement_kl(
                      m_logits[:, :, MTP_OFFSET], m_prior[:, :, MTP_OFFSET],
                      g_logits[:, :, MTP_OFFSET], g_prior[:, :, MTP_OFFSET],
                      movement_prev, policy_head.movement_bins))
        else:
            m_pol_kl, m_pri_kl = movement_kl_factors(policy_head, m_logits, m_prior)
            kl = factorized_policy_kl(
                a_logits[:, :, MTP_OFFSET, :], a_prior[:, :, MTP_OFFSET, :],
                m_pol_kl, m_pri_kl,
            )  # (B, H)

        policy_loss = compute_pmpo_loss(
            log_probs=log_probs.reshape(-1),
            advantages=advantages.reshape(-1),
            kl=kl.reshape(-1),
            alpha=args.pmpo_alpha, beta=args.pmpo_beta,
        )

        value_n = rms["value"].update(value_loss)
        policy_n = rms["policy"].update(policy_loss)
        total = value_n + policy_n

    info = {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.mean().detach(),
        "mean_reward": rewards.mean().detach(),
        "mean_value": values.mean().detach(),
        "mean_return": returns.mean().detach(),
        "pos_frac": (advantages >= 0).float().mean().detach(),
    }
    return total, info


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_one_mode(args, movement_mode, movement_gate):
    """One synthetic rollout + train step for a single movement mode.

    Asserts, for this mode:
      1. the on-policy dream runs via rollout() with sampled actions fed back;
      2. lambda-returns / PMPO / value losses compute and backprop;
      3. the step-0 prior KL is EXACTLY 0 (prior is a deepcopy of the policy);
      4. after moving the policy, the KL is > 0 (the term is live, not hardcoded);
      5. PMPO's gradient lands on MTP offset MTP_OFFSET and NOWHERE ELSE.
    """
    label = f"{movement_mode}{'+gate' if movement_gate else ''}"
    print(f"\n--- movement_mode={label} ---")
    torch.manual_seed(0)
    device = "cpu"

    B, Ctx, C, S = 2, 2, args.latent_dim, 16
    dynamics = build_dynamics(args, use_actions=True, device=device)
    dynamics.eval(); dynamics.requires_grad_(False)
    model_dim = dynamics.model_dim

    reward_head = RewardHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets, mtp_length=args.mtp_length).to(device)
    reward_head.eval(); reward_head.requires_grad_(False)
    policy_head = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                             hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins,
                             movement_gate=movement_gate,
                             movement_mode=movement_mode).to(device)
    # Put the head in the state a REAL Phase-2 checkpoint is in: MTP offset 0 at
    # exact zero-init (BC runs `for n in range(1, mtp_length)`, so it is never
    # trained), offsets >= 1 trained.
    #
    # DO NOT PERTURB HEAD 0. The previous version of this test did, with the
    # comment "so the prior KL is non-degenerate" -- and head 0 sitting at exact
    # zero-init is precisely the condition that makes a mis-sliced prior visible.
    # Perturbing it manufactured a plausible 1.9e-3 KL and hid a 7.29-nat bug on
    # real weights. The degeneracy was the signal, not the nuisance.
    with torch.no_grad():
        for n in range(1, args.mtp_length):
            policy_head.heads[n].weight.normal_(0, 0.02)
            policy_head.movement_heads[n].weight.normal_(0, 0.02)
            if movement_gate:
                policy_head.gate_heads[n].weight.normal_(0, 0.02)
    policy_prior = copy.deepcopy(policy_head)
    policy_prior.eval(); policy_prior.requires_grad_(False)
    value_head = ValueHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                           num_buckets=args.num_buckets).to(device)

    z_context = torch.randn(B, Ctx, C, S, S)
    roll = imagine(dynamics, policy_head, reward_head, value_head, z_context, args, device)
    assert roll["agent_outs"].shape == (B, args.horizon, model_dim)
    assert roll["movement_acts"].shape == (B, args.horizon, MOVEMENT_DIM)
    # The sampled movement INDEX must be long class indices, in the mode's own
    # layout -- this is what PMPO scores, and storing continuous xy here is what
    # made joint_noop raise "movement targets must be LONG class indices".
    exp_idx = ((B, args.horizon) if movement_mode == "joint_noop"
               else (B, args.horizon, MOVEMENT_DIM))
    assert roll["movement_idx"].shape == exp_idx, \
        f"movement_idx {tuple(roll['movement_idx'].shape)} != {exp_idx}"
    assert roll["movement_idx"].dtype == torch.long, \
        f"movement_idx must be LONG class indices, got {roll['movement_idx'].dtype}"
    assert roll["movement_prev"].shape == exp_idx

    rms = {"value": RunningRMS(), "policy": RunningRMS()}
    params = list(policy_head.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # --- INVARIANT: prior is a deepcopy of the policy, so step-0 KL is EXACTLY 0 ---
    _, info0 = run_step(roll, policy_head, policy_prior, value_head, args,
                        device, torch.float32, rms)
    kl0 = info0["kl"].item()
    assert kl0 == 0.0, (
        f"[{label}] the prior is a deepcopy of the policy, so the step-0 KL must "
        f"be EXACTLY 0.0, got {kl0:.6e}. Nonzero means policy and prior are read "
        f"at different MTP offsets (that bug measured 7.29 nats on a real "
        f"checkpoint, where it made --pmpo-beta an entropy bonus that erased the "
        f"BC policy).")
    print(f"  step-0 KL           = {kl0:.1f} exactly  (prior IS the policy)")

    # --- Now move the policy off the prior; the KL must become positive. ---
    # Without this the invariant above would also be satisfied by a KL hardcoded
    # to zero. Perturb the offset PMPO actually reads.
    with torch.no_grad():
        policy_head.movement_heads[MTP_OFFSET].weight.add_(
            torch.randn_like(policy_head.movement_heads[MTP_OFFSET].weight) * 0.05)
    rms = {"value": RunningRMS(), "policy": RunningRMS()}
    optimizer.zero_grad()
    total, info = run_step(roll, policy_head, policy_prior, value_head, args,
                           device, torch.float32, rms)
    assert info["kl"].item() > 0, f"[{label}] KL stayed 0 after the policy moved"
    total.backward()

    # --- GRAD-FLOW PROOF, PER OFFSET ---
    # The old assertion was `sum(h.weight.grad.norm() for h in movement_heads) > 0`,
    # which is true whenever ANY offset trains -- so a dead offset could never show.
    # PMPO scores exactly one offset, so assert exactly that: MTP_OFFSET gets
    # gradient and every other offset gets none.
    per_offset = []
    for n in range(args.mtp_length):
        g = policy_head.movement_heads[n].weight.grad
        per_offset.append(0.0 if g is None else g.norm().item())
    assert per_offset[MTP_OFFSET] > 0, (
        f"[{label}] movement_heads[{MTP_OFFSET}] -- the offset PMPO samples and "
        f"scores -- got ZERO gradient. Per-offset norms: {per_offset}")
    for n, gn in enumerate(per_offset):
        if n != MTP_OFFSET:
            assert gn == 0.0, (
                f"[{label}] movement_heads[{n}] got gradient {gn:.3e} but PMPO "
                f"only reads offset {MTP_OFFSET}; the slice is leaking.")
    ability_grad_norm = sum(h.weight.grad.norm().item() for h in policy_head.heads
                            if h.weight.grad is not None)
    value_grad_norm = sum(p.grad.norm().item() for p in value_head.parameters()
                          if p.grad is not None)
    # Frozen modules must have no grads.
    for name, mod in [("dynamics", dynamics), ("reward_head", reward_head),
                      ("policy_prior", policy_prior)]:
        with_grad = [n for n, p in mod.named_parameters() if p.grad is not None]
        assert not with_grad, f"[{label}] frozen {name} got gradients: {with_grad[:3]}"

    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    print(f"  total_loss          = {info['loss'].item():.4f}")
    print(f"  policy_loss (PMPO)  = {info['policy_loss'].item():.4f}")
    print(f"  value_loss          = {info['value_loss'].item():.4f}")
    print(f"  KL after perturb    = {info['kl'].item():.6e}  (> 0: policy != prior)")
    print(f"  mean_return         = {info['mean_return'].item():.4f}")
    print(f"  pos_advantage_frac  = {info['pos_frac'].item():.2%}")
    print(f"  GRAD movement_heads per offset = "
          f"[{', '.join(f'{g:.2e}' for g in per_offset)}]")
    print(f"    ^ only offset {MTP_OFFSET} may be nonzero (PMPO scores one offset)")
    print(f"  GRAD ability heads  = {ability_grad_norm:.6e}")
    print(f"  GRAD value head     = {value_grad_norm:.6e}")
    print("  optimizer.step() OK")


def smoke_test(args):
    """Tiny synthetic CPU rollout + train step, for EVERY movement mode.

    The previous version built only an 'axis' head, so the two joint_noop crashes
    (continuous-xy targets, and the movement_dim axis a joint head does not have)
    were structurally invisible to it, and it perturbed MTP head 0 -- hiding the
    mis-sliced prior it should have caught. It now covers all three modes that
    reach Phase 3 and asserts the step-0 KL invariant instead of working around it.
    """
    print("=" * 60)
    print("PHASE 3 IMAGINATION SMOKE TEST (synthetic, CPU)")
    print("=" * 60)
    args.model_size = "tiny"
    args.latent_dim = 16
    args.mtp_length = 4
    args.num_buckets = 41
    args.movement_bins = 11
    args.hidden_dim = 32
    args.num_register_tokens = 2
    # Kept tiny so the CPU smoke test finishes fast: the 16x16 spatial attention
    # over a tiny model is the cost driver, so we minimize the number of dynamics
    # forward passes (short horizon, single denoise step).
    args.horizon = 2
    args.gen_steps = 1
    args.k_max = 2
    args.tau_ctx = 0.1
    args.tau_ctx_forward = 0.9
    args.temperature = 1.0

    for movement_mode, movement_gate in (("axis", False), ("axis", True),
                                         ("joint_noop", False)):
        _smoke_one_mode(args, movement_mode, movement_gate)

    print("\nSMOKE TEST PASSED (axis, axis+gate, joint_noop)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # tau used by the agent-token forward pass inside imagine() (near-clean ctx).
    args.tau_ctx_forward = 1.0 - args.tau_ctx if args.tau_ctx < 0.5 else 0.9
    if args.smoke_test:
        smoke_test(args)
        return

    print("=" * 60)
    print("Phase 3: Imagination Training (PMPO + Value)")
    print("=" * 60)
    device = args.device
    print(f"Device: {device} | horizon={args.horizon} | gamma={args.gamma} lambda={args.lambda_}")
    print(f"PMPO alpha={args.pmpo_alpha} beta={args.pmpo_beta} | gen K={args.gen_steps} k_max={args.k_max}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.agent_checkpoint:
        raise SystemExit("--agent-checkpoint is required (or use --smoke-test).")
    dynamics, reward_head, policy_head, model_dim, use_actions = load_phase2(args, device)

    print("Creating frozen behavioral prior (copy of Phase 2 policy)...")
    policy_prior = copy.deepcopy(policy_head)
    policy_prior.eval(); policy_prior.requires_grad_(False)

    print("Creating value head (fresh for Phase 3)...")
    value_head = ValueHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                           num_buckets=args.num_buckets).to(device)

    dataset = build_context_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("No context sequences found.")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=VideoGroupedSampler(dataset),
        num_workers=args.num_workers, pin_memory=(device != "cpu"), drop_last=True,
    )

    params = list(policy_head.parameters()) + list(value_head.parameters())
    optimizer = create_optimizer(params, args.lr, args.weight_decay,
                                 use_8bit=args.use_8bit_adam, betas=tuple(args.adam_betas))
    total_steps = args.epochs * max(1, len(dataloader))
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)
    if device == "mps":
        amp_dtype = torch.float16
    elif device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float32  # pre-Ampere (e.g. 1060): autocast off via enabled= check
    else:
        amp_dtype = torch.bfloat16
    scaler = GradScaler(device.split(":")[0], enabled=(amp_dtype == torch.float16))
    rms = {"value": RunningRMS(), "policy": RunningRMS()}

    init_wandb(args, job_type="imagination", extra_config={
        "horizon": args.horizon, "gamma": args.gamma, "lambda": args.lambda_,
        "pmpo_alpha": args.pmpo_alpha, "pmpo_beta": args.pmpo_beta,
    })

    print("\n" + "=" * 60)
    print("Starting imagination training...")
    print("=" * 60)
    global_step = 0
    _degen_steps = 0   # consecutive one-sided-advantage logged steps
    for epoch in range(args.epochs):
        policy_head.train(); value_head.train()
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            z_context = batch["latents"].to(device).float()  # stored fp16; rollout() needs fp32
            # Condition the context window on its real recorded actions so the
            # frozen dynamics sees in-distribution inputs; dreamed frames use the
            # policy's sampled actions.
            ctx_actions = batch.get("actions") if dynamics.use_actions else None
            roll = imagine(dynamics, policy_head, reward_head, value_head,
                           z_context, args, device, actions_context=ctx_actions)
            optimizer.zero_grad()
            total, info = run_step(roll, policy_head, policy_prior, value_head,
                                   args, device, amp_dtype, rms)
            if global_step == 0:
                # PREFLIGHT THAT CAN FAIL, on the real weights. policy_prior is
                # copy.deepcopy(policy_head) a few lines above, so before the
                # first optimizer step the two are the same module and the KL is
                # EXACTLY 0 -- any other value means policy and prior are being
                # read at different MTP offsets. That bug shipped and measured
                # 7.29 nats here, turning --pmpo-beta into an entropy bonus
                # pointed at the BC policy. Checked against real weights because
                # it only shows when MTP head 0 is at its true zero-init state.
                _kl0 = info["kl"].item()
                if _kl0 != 0.0:
                    raise SystemExit(
                        f"Phase-3 step-0 prior KL is {_kl0:.6e}, must be EXACTLY 0: "
                        f"the behavioural prior is a deepcopy of the policy, so "
                        f"before any update the two distributions are identical. "
                        f"A nonzero value means the prior is sliced at a different "
                        f"MTP offset than the policy (MTP_OFFSET={MTP_OFFSET}).")
                print(f"[preflight] step-0 prior KL = {_kl0:.1f} exactly "
                      f"(prior is the policy; the KL anchor is wired correctly)")
            if not torch.isfinite(total):
                print(f"[WARN] non-finite loss at step {global_step}; skipping.")
                continue
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                sps = (batch_idx + 1) * args.batch_size / max(time.time() - t0, 1e-6)
                log_step({
                    "train/loss": info["loss"].item(),
                    "train/policy_loss": info["policy_loss"].item(),
                    "train/value_loss": info["value_loss"].item(),
                    "train/kl": info["kl"].item(),
                    "train/mean_reward": info["mean_reward"].item(),
                    "train/mean_value": info["mean_value"].item(),
                    "train/mean_return": info["mean_return"].item(),
                    "train/pos_advantage_frac": info["pos_frac"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={info['loss'].item():.4f} pi={info['policy_loss'].item():.4f} "
                      f"V={info['value_loss'].item():.4f} KL={info['kl'].item():.3e} "
                      f"R={info['mean_reward'].item():.3f} A+={info['pos_frac'].item():.0%} "
                      f"({sps:.1f} samp/s)")

            # --- degenerate advantage guard -----------------------------
            # PMPO uses only sign(A). If every sample is positive (or every
            # sample negative) the D+/D- split carries no information and
            # compute_pmpo_loss collapses to alpha * (-mean log pi): pure
            # likelihood maximisation of whatever the policy just sampled, i.e.
            # self-reinforcing mode collapse with the reward playing no part.
            #
            # This is not hypothetical. A fresh zero-init value head against
            # near-zero imagined rewards gives advantages ~0, and `advantages
            # >= 0` then makes EVERY sample positive: pos_frac was 100% on every
            # step of the first two real runs. The run looks healthy -- loss
            # falls, KL rises off zero -- while learning nothing from reward.
            #
            # Warming the critic on real trajectories first is the fix; this
            # guard exists so the failure is loud instead of silent.
            _pf = info["pos_frac"].item()
            _degen_steps = (_degen_steps + 1) if (_pf >= 0.995 or _pf <= 0.005) else 0
            if args.degenerate_advantage_patience > 0 and \
                    _degen_steps >= args.degenerate_advantage_patience:
                raise SystemExit(
                    f"\nABORT: pos_advantage_frac has been {_pf:.1%} for "
                    f"{_degen_steps} consecutive logged steps.\n"
                    "PMPO uses only sign(advantage), so a one-sided split makes the "
                    "loss pure likelihood maximisation of the policy's own samples "
                    "-- mode collapse, with the reward contributing nothing.\n"
                    "Usual cause: the value head is untrained, so advantages are ~0 "
                    "and every sample lands on one side. Warm the critic on real "
                    "trajectories before imagination training, or raise --horizon so "
                    "rollouts actually contain reward events (95% of H=8 rollouts "
                    "contain none).\n"
                    "Set --degenerate-advantage-patience 0 to disable this check."
                )

            if args.max_steps and global_step >= args.max_steps:
                # A smoke run is not a training run: stop without writing a
                # part-epoch checkpoint that later reads as a real one.
                print(f"\n[--max-steps {args.max_steps}] reached after "
                      f"{global_step} optimizer step(s); stopping without saving.")
                finish_wandb()
                return

        save_phase3_checkpoint(checkpoint_dir / f"imagination_epoch_{epoch + 1:03d}.pt",
                               policy_head, value_head, policy_prior, optimizer,
                               scheduler, rms, epoch + 1, global_step, args)
        save_phase3_checkpoint(checkpoint_dir / "imagination_latest.pt",
                               policy_head, value_head, policy_prior, optimizer,
                               scheduler, rms, epoch + 1, global_step, args)

    print("\nPhase 3 training complete.")
    finish_wandb()


def save_phase3_checkpoint(path, policy_head, value_head, policy_prior, optimizer,
                           scheduler, rms, epoch, global_step, args):
    ckpt = {
        "policy_head_state_dict": policy_head.state_dict(),
        "value_head_state_dict": value_head.state_dict(),
        "policy_prior_state_dict": policy_prior.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms.items()},
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "phase": "imagination",
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    import os
    os.replace(tmp, path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
