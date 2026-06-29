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
from torch.amp import GradScaler, autocast

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM
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
    RunningRMS,
)
from ahriuwu.data.dataset import VideoGroupedSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
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

    dyn_state = ckpt["dynamics_state_dict"]
    if any(k.startswith("_orig_mod.") for k in dyn_state):
        dyn_state = {k.replace("_orig_mod.", ""): v for k, v in dyn_state.items()}
    use_actions = cfg.get("use_actions", any("action_embed" in k for k in dyn_state))
    dynamics = build_dynamics(args, use_actions=use_actions, device=device)
    dynamics.load_state_dict(dyn_state, strict=False)
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
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins).to(device)
    policy_head.load_state_dict(ckpt["policy_head_state_dict"])

    print(f"Loaded Phase 2 from {args.agent_checkpoint}")
    print(f"  dynamics use_actions={use_actions} (frozen), reward head frozen, policy head trainable")
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
      movement_acts: (B, H, 2) continuous xy (bin centers)
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

    agent_outs, ability_acts, movement_acts, rewards, values = [], [], [], [], []
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

        # Sample on-policy action at MTP offset n=0 (no leak: a_t did not exist
        # when h_t was built). sample() returns continuous movement + bin idx.
        abilities, movement, _ = policy_head.sample(h_t, temperature=args.temperature)
        a_abil = abilities[:, 0, 0, :]        # (B, num_abilities)
        a_move = movement[:, 0, 0, :]         # (B, 2)

        r_t = reward_head.predict(h_t)[:, 0, 0]   # (B,) MTP offset 0, original scale
        v_t = value_head.predict(h_t)[:, 0]       # (B,) original scale

        agent_outs.append(h_t[:, 0, :])
        ability_acts.append(a_abil)
        movement_acts.append(a_move)
        rewards.append(r_t)
        values.append(v_t)

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
        "movement_acts": torch.stack(movement_acts, dim=1), # (B, H, 2)
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
    movement_acts = roll["movement_acts"]           # (B, H, 2)
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
        # Add a singleton MTP axis so log_prob/forward see (B, H, 1, ...). We use
        # MTP offset n=0: log pi(a_t | h_t) for the on-policy sampled a_t.
        abil_mtp = ability_acts.unsqueeze(2)      # (B, H, 1, A)
        move_mtp = movement_acts.unsqueeze(2)     # (B, H, 1, 2)
        log_probs = policy_head.log_prob(agent_outs, abil_mtp, move_mtp)[:, :, 0]  # (B,H)

        # Factorized KL needs both heads' logits at offset 0, for policy and prior.
        a_logits, m_logits = policy_head(agent_outs)          # (B,H,L,A), (B,H,L,2,bins)
        with torch.no_grad():
            a_prior, m_prior = policy_prior(agent_outs)
        kl = factorized_policy_kl(
            a_logits[:, :, 0, :], a_prior[:, :, 0, :],
            m_logits[:, :, 0, :, :], m_prior[:, :, 0, :, :],
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

def smoke_test(args):
    """Tiny synthetic CPU rollout + train step. Proves: (1) on-policy dream runs
    via rollout() with sampled actions fed back, (2) lambda-returns/PMPO/value
    losses compute, (3) forward+backward+optimizer step works, and (4)
    movement_heads receive a nonzero PMPO gradient (today they would get zero)."""
    print("=" * 60)
    print("PHASE 3 IMAGINATION SMOKE TEST (synthetic, CPU)")
    print("=" * 60)
    torch.manual_seed(0)
    device = "cpu"
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

    B, Ctx, C, S = 2, 2, args.latent_dim, 16
    dynamics = build_dynamics(args, use_actions=True, device=device)
    dynamics.eval(); dynamics.requires_grad_(False)
    model_dim = dynamics.model_dim

    reward_head = RewardHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets, mtp_length=args.mtp_length).to(device)
    reward_head.eval(); reward_head.requires_grad_(False)
    policy_head = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                             hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins).to(device)
    # Perturb the policy off zero-init so the prior KL and movement logits are
    # non-degenerate (zero-init would make policy==prior, KL==0 trivially).
    with torch.no_grad():
        for h in policy_head.movement_heads:
            h.weight.normal_(0, 0.02)
        for h in policy_head.heads:
            h.weight.normal_(0, 0.02)
    policy_prior = copy.deepcopy(policy_head)
    policy_prior.eval(); policy_prior.requires_grad_(False)
    # Now nudge the trainable policy so policy != prior (nonzero KL).
    with torch.no_grad():
        for h in policy_head.movement_heads:
            h.weight.add_(torch.randn_like(h.weight) * 0.05)
    value_head = ValueHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                           num_buckets=args.num_buckets).to(device)

    z_context = torch.randn(B, Ctx, C, S, S)
    roll = imagine(dynamics, policy_head, reward_head, value_head, z_context, args, device)
    assert roll["agent_outs"].shape == (B, args.horizon, model_dim)
    assert roll["movement_acts"].shape == (B, args.horizon, MOVEMENT_DIM)

    rms = {"value": RunningRMS(), "policy": RunningRMS()}
    params = list(policy_head.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimizer.zero_grad()
    total, info = run_step(roll, policy_head, policy_prior, value_head, args,
                           device, torch.float32, rms)
    total.backward()

    # --- GRAD-FLOW PROOF: movement_heads get a nonzero PMPO gradient ---
    move_grads = [h.weight.grad for h in policy_head.movement_heads if h.weight.grad is not None]
    assert move_grads, "movement_heads received NO gradient (None) under PMPO!"
    move_grad_norm = sum(g.norm().item() for g in move_grads)
    assert move_grad_norm > 0, f"movement_heads PMPO gradient is exactly zero ({move_grad_norm})!"
    ability_grad_norm = sum(h.weight.grad.norm().item() for h in policy_head.heads
                            if h.weight.grad is not None)
    value_grad_norm = sum(p.grad.norm().item() for p in value_head.parameters()
                          if p.grad is not None)
    # Frozen modules must have no grads.
    for name, mod in [("dynamics", dynamics), ("reward_head", reward_head),
                      ("policy_prior", policy_prior)]:
        with_grad = [n for n, p in mod.named_parameters() if p.grad is not None]
        assert not with_grad, f"frozen {name} got gradients: {with_grad[:3]}"

    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    print(f"  horizon             = {args.horizon} (rollout via dynamics.rollout, actions fed back)")
    print(f"  total_loss          = {info['loss'].item():.4f}")
    print(f"  policy_loss (PMPO)  = {info['policy_loss'].item():.4f}")
    print(f"  value_loss          = {info['value_loss'].item():.4f}")
    print(f"  factorized KL       = {info['kl'].item():.6e}  (> 0: policy != prior)")
    print(f"  mean_return         = {info['mean_return'].item():.4f}")
    print(f"  pos_advantage_frac  = {info['pos_frac'].item():.2%}")
    print(f"  GRAD movement_heads = {move_grad_norm:.6e}  (PROOF: > 0 under PMPO)")
    print(f"  GRAD ability heads  = {ability_grad_norm:.6e}")
    print(f"  GRAD value head     = {value_grad_norm:.6e}")
    print("  optimizer.step() OK")
    print("SMOKE TEST PASSED")
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
    amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
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
    for epoch in range(args.epochs):
        policy_head.train(); value_head.train()
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            z_context = batch["latents"].to(device)
            # Condition the context window on its real recorded actions so the
            # frozen dynamics sees in-distribution inputs; dreamed frames use the
            # policy's sampled actions.
            ctx_actions = batch.get("actions") if dynamics.use_actions else None
            roll = imagine(dynamics, policy_head, reward_head, value_head,
                           z_context, args, device, actions_context=ctx_actions)
            optimizer.zero_grad()
            total, info = run_step(roll, policy_head, policy_prior, value_head,
                                   args, device, amp_dtype, rms)
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
