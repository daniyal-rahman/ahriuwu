#!/usr/bin/env python3
"""Phase 2: Agent Finetuning - Behavior Cloning + Reward Modeling.

DreamerV4 Section 3.3 "Behavior cloning and reward model". The dynamics model
(Phase 1) is the FROZEN backbone here: we run it (with agent tokens + action
conditioning) on real replay latents to get one agent token per frame, then
train two small heads on those tokens:

- ``PolicyHead`` via behavioral cloning on the NEXT-frame action — both the 9
  independent ability Bernoullis AND the binned (x, y) movement categoricals,
  by log-probability. (See the label-leakage note below for why NEXT-frame.)
- ``RewardHead`` via twohot multi-token-prediction (MTP) of the solo-gold
  reward at offsets n = 0..L-1.

Loss = bc_loss + reward_loss (each normalized by its running RMS, then summed).

LABEL-LEAKAGE FIX (vs the archived trainer): the dynamics is *action-conditioned*
— the agent token at frame t is built from a window whose frame-t input already
contains action a_t. So predicting a_t from agent_out[:, t] (the old n=0 BC term)
trivially leaks. BC here predicts the NEXT actions: MTP head n (n >= 1) predicts
a_{t+n}, and the n=0 term is dropped. The reward target is never a model input,
so the reward MTP keeps the full n = 0..L-1.

Usage (real run on the GPU node):
    PYTHONPATH=src python scripts/train_agent_finetune.py \
        --dynamics-checkpoint checkpoints/dynamics_best.pt \
        --latents-dir /opt/ahriuwu/latents_pt \
        --labels-root /mnt/storage/ahriuwu-data/replays \
        --epochs 1

Smoke test (CPU, synthetic, no checkpoint/data needed):
    PYTHONPATH=src python scripts/train_agent_finetune.py --smoke-test

Reference: DreamerV4 Section 3.3 "Behavior cloning and reward model".
"""

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM
from ahriuwu.models import (
    create_dynamics,
    RewardHead,
    StateHead,
    PolicyHead,
    DiffusionSchedule,
    symlog,
    twohot_loss,
    RunningRMS,
    x_prediction_loss,
)
from ahriuwu.data.replay_dataset import STATE_TARGETS
from ahriuwu.data.dataset import VideoGroupedSampler
from ahriuwu.utils.logging import add_wandb_args, init_wandb, log_step, finish_wandb
from ahriuwu.utils.training import (
    add_training_args, create_optimizer, create_wsd_schedule,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Agent Finetuning (BC + Reward)")
    add_training_args(parser)
    parser.add_argument(
        "--dynamics-checkpoint", type=str, default=None,
        help="Path to the Phase 1 dynamics checkpoint (frozen backbone). "
             "Omit only with --init-dynamics / --smoke-test.",
    )
    parser.add_argument(
        "--init-dynamics", action="store_true",
        help="Build a fresh (untrained) dynamics backbone instead of loading a "
             "checkpoint. For wiring/smoke tests only — produces garbage tokens.",
    )
    parser.add_argument(
        "--latents-dir", type=str, default=None,
        help="Dir of packed per-match latents (<match>.pt) — same format Phase 1 uses.",
    )
    parser.add_argument(
        "--labels-root", type=str, default=None,
        help="Dir of <match>/labels.json + clicks.json (replay action + reward labels).",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Optional outcomes manifest (garen_win per match). If omitted, dummy "
             "outcomes (all False) are used — fine for solo-gold reward, which "
             "ignores win/loss.",
    )
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Latent dim per token (must match the tokenizer/dynamics ckpt).")
    parser.add_argument("--seq-len", type=int, default=32, help="Frames per sequence.")
    parser.add_argument("--stride", type=int, default=8, help="Stride between windows.")
    parser.add_argument("--mtp-length", type=int, default=9,
                        help="MTP heads (paper Eq 9: n=0..L with L=8 -> 9).")
    parser.add_argument("--num-buckets", type=int, default=255,
                        help="Twohot buckets for the reward head.")
    parser.add_argument("--movement-bins", type=int, default=21,
                        help="Per-axis movement bins in the policy head.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Head MLP width.")
    # Dynamics architecture flags (must match the checkpoint's build).
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--soft-cap", type=float, default=50.0)
    parser.add_argument("--num-register-tokens", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--agent-layers", type=int, default=4)
    parser.add_argument("--tau-ctx", type=float, default=0.9,
                        help="Near-clean context corruption: per-frame tau ~ U(tau_ctx, 1).")
    # --resume and --checkpoint-minutes come from add_training_args. Pass --resume
    # 'auto' to use <checkpoint-dir>/agent_finetune_latest.pt. The main loop wires
    # checkpoint-minutes into a mid-epoch save (the trainer only saved per-epoch).
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate gradients over N micro-batches before stepping. "
                             "EFFECTIVE batch = --batch-size * N. Needed for paper parity: "
                             "unfreezing the backbone only fits at batch 2 on a 16GB card "
                             "(measured 12.84 GiB; batch 4 OOMs), and batch 2 gives very "
                             "noisy gradients on 148M params. --batch-size 2 --grad-accum 8 "
                             "reproduces the frozen run's effective batch of 16.")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="PAPER PARITY (Algorithm 1 Phase 2: 'finetune world model with "
                             "task inputs for policy and reward heads using (7) and (9)'). "
                             "Trains the diffusion backbone too, not just the agent blocks and "
                             "heads. REQUIRED for --video-loss-weight to do anything: Eq (7) "
                             "predicts latents from the backbone, so with it frozen the video "
                             "gradient has nowhere to go. Forces gradient checkpointing (OOMs "
                             "otherwise: measured 12.16 GiB with, OOM without, on a 16GB card) "
                             "and costs ~3x per step.")
    parser.add_argument("--video-loss-weight", type=float, default=0.0,
                        help="Weight on Eq (7), the x-prediction video loss, added to the BC "
                             "loss and RMS-normalized like every other term. The paper runs it "
                             "throughout Phase 2; we dropped it, which is the leading suspect "
                             "for BC overfitting 119 games (train kept falling while val "
                             "flattened). 0 = off (legacy).")
    parser.add_argument("--movement-mode", choices=["axis", "joint_noop"], default="axis",
                        help="'axis' (legacy): two independent per-axis categoricals, "
                             "which cannot express x-y correlation and needs --movement-gate "
                             "to handle frames with no new order. 'joint_noop': ONE "
                             "categorical over the bins^2 grid plus a NO_OP class meaning "
                             "'previous order still executing'. joint_noop needs no previous "
                             "action to score a frame, so PMPO/Phase 3 works with the plain "
                             "log_prob. Mutually exclusive with --movement-gate.")
    parser.add_argument("--movement-gate", action="store_true",
                        help="Sticky-categorical movement: a per-offset gate predicts P(new "
                             "movement command); the bin categorical only explains transitions. "
                             "Fixes the copy-shortcut (77%% of frames are held actions).")
    parser.add_argument("--action-dropout", type=float, default=0.0,
                        help="Per-frame prob of masking the movement action-history INPUT to "
                             "no_action_embed (cursor_valid=False) during training. Breaks the "
                             "learned copy-of-own-history shortcut so self-fed inference doesn't "
                             "collapse. Targets are unchanged. Ability history is NOT dropped.")
    parser.add_argument("--aux-state-weight", type=float, default=0.0,
                        help="Weight of the aux game-state loss (own/enemy HP, level, "
                             "visibility regressed from agent tokens; targets from replay "
                             "labels). 0 = head not built. The loss is RMS-normalized like "
                             "bc/reward, so ~0.5 is a meaningful-but-not-dominant prior.")
    parser.add_argument("--ability-pos-weight", type=float, default=5.0,
                        help="Positive-class weight for the ability BCE. Casts are sparse, so "
                             "unweighted BCE collapses to 'never press'. >1 makes the agent cast.")
    parser.add_argument("--movement-source", type=str, default="clicks",
                        choices=["clicks", "cursor"],
                        help="'clicks' (default): movement target = the screen-space "
                             "target of each real click event from clicks.json, held "
                             "forward, with a movement_event flag supervising the gate. "
                             "'cursor': the legacy label.cursor.screen target, 43%% of "
                             "whose transitions are camera drift — kept only for A/B.")
    # --- held-out validation (see FIX 2 / audit finding 3) ---
    parser.add_argument("--val-matches", type=str, default=None,
                        help="Comma-separated match ids, or a path to a JSON file "
                             "(list, or {'val': [...]}), to hold out. Overrides "
                             "--val-games.")
    parser.add_argument("--val-games", type=int, default=6,
                        help="If --val-matches is not given, hold out this many WHOLE "
                             "games (never frames — adjacent frames leak). 0 disables "
                             "validation and must be passed explicitly.")
    parser.add_argument("--val-interval", type=int, default=1000,
                        help="Run the held-out eval every N optimizer steps.")
    parser.add_argument("--val-batches", type=int, default=20,
                        help="Batches per held-out eval pass.")
    parser.add_argument("--dataset-cache", type=str, default=None,
                        help="Cache the (slow) dataset index — label/reward parse + per-.pt "
                             "frame_indices reads — to this file. Present -> load; absent -> "
                             "build then save. Delete the file to force a rebuild.")
    # Smoke test
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny synthetic CPU train step end-to-end + assert "
                             "movement_heads receive BC gradient. No data/ckpt needed.")
    parser.set_defaults(num_workers=0, wandb=False)
    add_wandb_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_state_dict_guarded(module, state, *, what: str, allow_missing=()):
    """``load_state_dict(strict=False)`` that RAISES on anything unexplained.

    A bare ``strict=False`` has silently detonated twice in this repo: it
    dropped the trained ``action_embed`` weights in ``agent_infer`` (leaving the
    agent blocks reading out-of-distribution activations) and dropped 36 tensors
    while flipping the attention scale in every dynamics eval for five months.
    Both were invisible because nothing stopped.

    ``allow_missing`` lists the prefixes that are legitimately absent for THIS
    load (e.g. the Phase-2 agent blocks, which a Phase-1 checkpoint never
    trained). Everything else — and every unexpected key, always — aborts.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith(tuple(allow_missing))]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{what}: state-dict mismatch — the checkpoint does not match this "
            f"model build, and loading it non-strictly would silently leave "
            f"tensors at random init.\n"
            f"  UNEXPECTED in checkpoint ({len(unexpected)}): {unexpected[:10]}\n"
            f"  MISSING, not explained by {list(allow_missing) or 'anything'} "
            f"({len(bad_missing)}): {bad_missing[:10]}\n"
            f"  (expected-missing, ignored: {len(missing) - len(bad_missing)})")
    return missing, unexpected


def build_dynamics(args, *, use_actions: bool, device: str):
    """Create a dynamics backbone (agent tokens ON) matching the CLI arch flags."""
    return create_dynamics(
        size=args.model_size,
        latent_dim=args.latent_dim,
        use_agent_tokens=True,
        use_actions=use_actions,
        num_tasks=1,
        agent_layers=args.agent_layers,
        use_qk_norm=not args.no_qk_norm,
        soft_cap=args.soft_cap if args.soft_cap > 0 else None,
        num_register_tokens=args.num_register_tokens,
        num_kv_heads=args.num_kv_heads,
        # Unfreezing the backbone OOMs on a 16GB card without this (measured:
        # 12.16 GiB with checkpointing, OOM without, at batch 16 / seq 16).
        gradient_checkpointing=getattr(args, "unfreeze_backbone", False),
    ).to(device)


# Agent-token params are NEW in Phase 2. Phase-1 pretraining never trains them:
# agent_out is a side readout (dynamics.forward returns it alongside z_0_pred) and
# is absent from the denoising loss, so the agent blocks only ever get the DDP
# zero-grad tap in Phase 1 — i.e. they stay at random init. Phase 2 must therefore
# TRAIN them (along with the heads); only the pretrained diffusion backbone freezes.
AGENT_PARAM_PREFIXES = ("agent_token", "agent_temporal_pos", "agent_blocks",
                        "agent_norm_out")

# Keys that may be missing from a PHASE-1 checkpoint. Superset of the trainable
# agent prefixes because `task_embed` is created alongside the agent-token
# machinery but is DEAD: nothing in this repo ever passes `task_id`, so
# dynamics.forward's `if task_id is not None` branch never runs, and num_tasks=1
# makes multi-task conditioning meaningless anyway. It is therefore correctly
# absent from Phase 1 and correctly left out of AGENT_PARAM_PREFIXES (adding it
# would put a never-used tensor in the optimizer). Allowed to be missing, not
# allowed to be trained.
AGENT_ABSENT_PREFIXES = AGENT_PARAM_PREFIXES + ("task_embed",)


def freeze_backbone_train_agent(dyn, unfreeze: bool = False):
    """Freeze the pretrained diffusion backbone; keep the agent-token blocks
    trainable. Returns the list of trainable agent params (for the optimizer).

    The backbone stays in eval() (LayerNorm-only, so eval/train are identical and
    there's no dropout/BN state). Because every backbone param has requires_grad
    False and the noised-latent input carries no grad, running the backbone WITHOUT
    a no_grad wrapper builds no backbone autograd graph — grad flows only into the
    agent blocks that read the (constant) z-token features. So keep run_step's
    forward OUT of no_grad and do NOT detach agent_out.
    """
    # .eval() everywhere was hiding a no-op: gradient checkpointing is guarded by
    # `self.gradient_checkpointing and self.training`, so with the module in eval
    # mode it NEVER activated. That made --unfreeze-backbone store full fp32-era
    # activations and produced a fake batch-size ceiling (2 on a 16GB card, 4 on
    # 31GB). There is no dropout/BN in the backbone, so train() is safe: the only
    # behavioural difference is that checkpointing now actually runs.
    dyn.train(unfreeze)
    agent_params = []
    for name, p in dyn.named_parameters():
        if unfreeze or name.startswith(AGENT_PARAM_PREFIXES):
            p.requires_grad_(True)
            agent_params.append(p)
        else:
            p.requires_grad_(False)
    return agent_params


def load_frozen_dynamics(args, device: str):
    """Load the Phase 1 dynamics: FROZEN diffusion backbone + TRAINABLE agent
    tokens (see :func:`freeze_backbone_train_agent`).

    The Phase 1 checkpoint has no agent-token / reward-head weights (those are
    trained here), so non-matching keys are loaded non-strictly: the agent blocks
    start from their init and get trained, the diffusion backbone is the pretrained
    one. Returns (dynamics, use_actions, agent_params).
    """
    if args.init_dynamics or args.dynamics_checkpoint is None:
        # Fresh backbone (smoke / wiring). Enable actions so the action path is exercised.
        dyn = build_dynamics(args, use_actions=True, device=device)
        print("  [init] fresh untrained dynamics backbone (no checkpoint loaded)")
    else:
        ckpt = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        # Honor the checkpoint's own resolved config where available.
        cfg = ckpt.get("model_config") or {}
        latent_dim = cfg.get("latent_dim", args.latent_dim)
        if latent_dim != args.latent_dim:
            print(f"  [ckpt] overriding --latent-dim {args.latent_dim} -> {latent_dim} (from ckpt config)")
            args.latent_dim = latent_dim
        use_actions = cfg.get("use_actions", any("action_embed" in k for k in state))
        dyn = build_dynamics(args, use_actions=use_actions, device=device)
        # Phase 1 never trains the agent-token machinery (it is a side readout,
        # absent from the denoising loss), so those keys are legitimately absent
        # from a Phase-1 checkpoint. NOTHING ELSE is.
        missing, unexpected = load_state_dict_guarded(
            dyn, state, what=f"Phase-1 backbone {args.dynamics_checkpoint}",
            allow_missing=AGENT_ABSENT_PREFIXES)
        print(f"  [ckpt] loaded {args.dynamics_checkpoint}")
        print(f"  [ckpt] use_actions={use_actions}; {len(missing)} missing "
              f"(all agent-token keys, trained here) / 0 unexpected")

    agent_params = freeze_backbone_train_agent(
        dyn, unfreeze=getattr(args, 'unfreeze_backbone', False))
    n_agent = sum(p.numel() for p in agent_params)
    if getattr(args, "unfreeze_backbone", False):
        print(f"  [freeze] backbone UNFROZEN (paper parity): {len(agent_params)} tensors "
              f"({n_agent:,} params) TRAINABLE, gradient checkpointing ON")
    else:
        print(f"  [freeze] diffusion backbone FROZEN; {len(agent_params)} agent tensors "
              f"({n_agent:,} params) TRAINABLE")
    return dyn, dyn.use_actions, agent_params


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(args):
    """ReplayLatentSequenceDataset over real replay latents (Phase-1 style).

    Imported lazily to dodge the circular import that keeps it out of
    ahriuwu.data's __init__.
    """
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset

    if not args.latents_dir or not args.labels_root:
        raise SystemExit("Real run needs --latents-dir and --labels-root "
                         "(or use --smoke-test).")
    outcomes = None
    if args.manifest:
        outcomes = None  # let the dataset read it
    else:
        mids = [Path(p).stem for p in glob.glob(str(Path(args.latents_dir) / "*.pt"))
                if Path(p).stem != "index"]
        outcomes = {m: False for m in mids}  # solo-gold ignores win/loss
        print(f"  [data] {len(mids)} matches; dummy outcomes (solo-gold reward ignores them)")
    return ReplayLatentSequenceDataset(
        latents_dir=args.latents_dir,
        labels_root=args.labels_root,
        outcomes=outcomes,
        manifest_path=args.manifest,
        sequence_length=args.seq_len,
        stride=args.stride,
        cache_path=getattr(args, "dataset_cache", None),
        movement_source=args.movement_source,
    )


def select_val_matches(dataset, args) -> set[str]:
    """WHOLE games reserved for validation — never a frame-level split.

    Adjacent frames within a game are near-duplicates (audit finding 17: 11% of
    consecutive pairs differ by <1/255), so any frame- or window-level split
    leaks the answer. Selection is deterministic: evenly-spaced picks over the
    sorted match ids, which spreads the holdout across the corpus (ids are
    roughly chronological) instead of taking a contiguous tail.

    Prefers matches whose movement target came from real clicks, so val metrics
    are measured on the same target definition the model is trained for.
    """
    vids = sorted({s["video_id"] for s in dataset.sequences})
    if args.val_matches:
        p = Path(args.val_matches)
        if p.exists():
            doc = json.loads(p.read_text())
            want = doc.get("val", doc) if isinstance(doc, dict) else doc
        else:
            want = [m.strip() for m in args.val_matches.split(",") if m.strip()]
        want = set(want)
        missing = want - set(vids)
        if missing:
            raise SystemExit(
                f"--val-matches names {len(missing)} match(es) not in the dataset: "
                f"{sorted(missing)[:5]}")
        return want

    n = args.val_games
    if n <= 0:
        return set()
    eligible = [v for v in vids
                if dataset.match_data.get(v, {}).get("movement_from_clicks", True)]
    if len(eligible) < n:  # not enough click-backed games; fall back to all
        eligible = vids
    if len(eligible) <= n:
        raise SystemExit(
            f"--val-games {n} but only {len(eligible)} games available — "
            "that would leave no training data.")
    step = len(eligible) / n
    return {eligible[min(int(i * step), len(eligible) - 1)] for i in range(n)}


def build_val_order(dataset, val_vids, batch_size, n_batches):
    """Fixed val sequence order: an equal, evenly-spaced slice of every val game.

    Two constraints pull against each other. Latent packs are ~210 MB behind a
    2-deep LRU, so an order that hops between videos thrashes the cache; but a
    plain sequential walk of ``dataset.sequences`` would spend the whole eval
    inside the FIRST val game's opening minutes. Resolution: give each val game
    the same number of batches (ceil(n_batches / n_games)) and keep them
    contiguous, then always run the loader to the end. Every game is covered,
    each batch touches exactly one pack, and consecutive batches usually reuse
    it. Within a game the windows are evenly spaced, so the eval sees the whole
    match rather than its opening.

    The order is a fixed list, so every eval scores the identical sequences.
    """
    by_vid: dict[str, list[int]] = {}
    for i, s in enumerate(dataset.sequences):
        if s["video_id"] in val_vids:
            by_vid.setdefault(s["video_id"], []).append(i)
    vids = sorted(by_vid)
    if not vids:
        return []
    per_vid = max(1, -(-n_batches // len(vids))) * batch_size  # ceil-div batches
    order: list[int] = []
    for v in vids:
        idxs = by_vid[v]
        if len(idxs) > per_vid:  # evenly spaced across the game
            st = len(idxs) / per_vid
            idxs = [idxs[min(int(k * st), len(idxs) - 1)] for k in range(per_vid)]
        keep = (len(idxs) // batch_size) * batch_size  # whole batches only
        order.extend(idxs[:keep])
    return order


def actions_to_device(actions: dict, device: str) -> dict:
    """Move the dataset action dict (movement + per-ability) to ``device``."""
    return {k: v.to(device) for k, v in actions.items()}


def stack_ability_targets(actions: dict, device: str) -> torch.Tensor:
    """(B, T, num_abilities) float {0,1} from the per-ability action tensors."""
    return torch.stack(
        [actions[k].to(device).float() for k in ABILITY_KEYS], dim=-1
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def reward_mtp_loss(reward_logits, rewards, bucket_centers, mtp_length):
    """Twohot MTP reward loss over offsets n = 0..L-1 (Eq 9).

    reward_logits: (B, T, L, num_buckets); rewards: (B, T) raw scale.
    Reward is a TARGET (never a model input) so n=0 is legitimate here.
    """
    B, T = rewards.shape
    targets = symlog(rewards)  # (B, T)
    loss = torch.zeros((), device=rewards.device)
    n_terms = 0
    for n in range(mtp_length):
        if T - n <= 0:
            break
        pred = reward_logits[:, :T - n, n, :]      # predict reward at t+n from token t
        tgt = targets[:, n:]                        # (B, T-n)
        loss = loss + twohot_loss(pred, tgt, bucket_centers)
        n_terms += 1
    return loss / max(n_terms, 1)


def bc_next_action_loss(policy_head, agent_out, ability_targets, movement_targets,
                        mtp_length, ability_pos_weight=None, movement_event=None):
    """Behavior-cloning negative log-likelihood of the NEXT actions.

    MTP head n (n >= 1) predicts the action at t+n from the token at t; n=0 is
    dropped to avoid the action-conditioning label leak. Returns the mean NLL of
    the factorized policy (abilities + binned movement) and a (split) breakdown.

    ability_targets: (B, T, num_abilities) {0,1}; movement_targets: (B, T, 2) xy.
    movement_event: (B, T) bool — True on the frames a NEW movement command was
        issued. This is the gate's supervision. Without it the gate falls back
        to "did the 21-bin cell change", which on the legacy target is 43%
        camera drift AND misses the 37.7% of real commands that quantize into
        the same cell (audit findings 1 and 5).
    """
    ability_logits, movement_logits = policy_head(agent_out)
    # ability_logits:  (B, T, L, A)
    # movement_logits: (B, T, L, move_dim, bins)
    B, T = ability_targets.shape[0], ability_targets.shape[1]

    import torch.nn.functional as F
    move_idx_full = policy_head.discretize_movement(movement_targets)  # (B, T, 2)
    # Up-weight the (rare) positive casts so BCE doesn't collapse to "never press".
    pos_w = (torch.tensor(ability_pos_weight, device=agent_out.device)
             if ability_pos_weight and ability_pos_weight != 1.0 else None)

    gated = getattr(policy_head, "movement_gate", False)
    joint_noop = getattr(policy_head, "movement_mode", "axis") == "joint_noop"
    gate_logits_full = policy_head.gate_logits(agent_out) if gated else None

    ability_nll = torch.zeros((), device=agent_out.device)
    move_nll = torch.zeros((), device=agent_out.device)
    gate_fire_t = gate_fire_h = trans_frac = torch.zeros((), device=agent_out.device)
    n_terms = 0
    for n in range(1, mtp_length):  # n >= 1: predict the NEXT actions only
        if T - n <= 0:
            break
        # token positions 0..T-1-n predict action at +n
        a_logits = ability_logits[:, :T - n, n, :]          # (B, T-n, A)
        a_tgt = ability_targets[:, n:, :]                    # (B, T-n, A)
        ability_nll = ability_nll + F.binary_cross_entropy_with_logits(
            a_logits, a_tgt, pos_weight=pos_w)

        if joint_noop:
            # ONE categorical over grid cells + NO_OP. The target is the grid
            # cell on frames where a command actually fired (movement_event) and
            # NO_OP otherwise -- read straight off the event stream, so the 18.6%
            # of commands that land in the PREVIOUS cell stay real commands
            # instead of being mistaken for holds by a bin comparison.
            m_logits = movement_logits[:, :T - n, n, :]      # (B, T-n, classes)
            xy = move_idx_full[:, n:, :]                     # (B, T-n, 2) per-axis bins
            cls = policy_head.joint_encode(xy[..., 0], xy[..., 1])
            if movement_event is not None:
                cls = torch.where(movement_event[:, n:], cls,
                                  torch.full_like(cls, policy_head.NO_OP))
            move_nll = move_nll + F.cross_entropy(
                m_logits.reshape(-1, m_logits.shape[-1]), cls.reshape(-1))
            if n == 1:                                        # diagnostics
                with torch.no_grad():
                    pred = m_logits.argmax(-1)
                    is_noop = cls == policy_head.NO_OP
                    trans_frac = (~is_noop).float().mean()
                    # reuse the gate slots: "fires on a real command" vs "on a hold"
                    gate_fire_t = (pred != policy_head.NO_OP)[~is_noop].float().mean() \
                        if (~is_noop).any() else torch.zeros((), device=cls.device)
                    gate_fire_h = (pred != policy_head.NO_OP)[is_noop].float().mean() \
                        if is_noop.any() else torch.zeros((), device=cls.device)
            n_terms += 1
            continue

        m_logits = movement_logits[:, :T - n, n, :, :]       # (B, T-n, move_dim, bins)
        m_idx = move_idx_full[:, n:, :]                      # (B, T-n, move_dim)
        if gated:
            # Sticky-categorical NLL: prev bin for target a_{t+n} is a_{t+n-1}
            # (n>=1, so the previous target is always in-window).
            p_idx = move_idx_full[:, n - 1:T - 1, :]         # (B, T-n, move_dim)
            if movement_event is not None:
                # gated_movement_log_prob keys the transition branch off
                # (target_idx != prev_idx). Encode the TRUE event mask through
                # prev_idx so the gate is supervised by "a command was issued",
                # not by "the bin happened to change":
                #   event & same-bin -> force a difference  (transition branch)
                #   no event         -> force equality      (hold branch)
                # With the click-event target `movement` is piecewise-constant,
                # so bin-change ⊆ event already; this only ADDS the commands the
                # 21-bin grid quantizes away. No change to heads.py needed.
                ev = movement_event[:, n:].unsqueeze(-1)     # (B, T-n, 1)
                other = (m_idx + 1) % policy_head.movement_bins
                p_idx = torch.where(ev, other, m_idx)
            lp = policy_head.gated_movement_log_prob(
                m_logits.unsqueeze(2), gate_logits_full[:, :T - n, n].unsqueeze(2),
                m_idx.unsqueeze(2), p_idx.unsqueeze(2))      # (B, T-n, 1)
            move_nll = move_nll - lp.mean()
            if n == 1:  # diagnostics at the primary offset
                with torch.no_grad():
                    g = torch.sigmoid(gate_logits_full[:, :T - 1, 1])
                    trans = (m_idx != p_idx).any(-1)
                    trans_frac = trans.float().mean()
                    gate_fire_t = g[trans].mean() if trans.any() else g.mean()
                    gate_fire_h = g[~trans].mean() if (~trans).any() else g.mean()
        else:
            # cross-entropy per axis (flatten axes into the batch dim of CE)
            move_nll = move_nll + F.cross_entropy(
                m_logits.reshape(-1, m_logits.shape[-1]),
                m_idx.reshape(-1),
            )
        n_terms += 1

    n_terms = max(n_terms, 1)
    ability_nll = ability_nll / n_terms
    move_nll = move_nll / n_terms
    return ability_nll + move_nll, {"bc_ability": ability_nll, "bc_movement": move_nll,
                                    "gate_on_trans": gate_fire_t, "gate_on_hold": gate_fire_h,
                                    "trans_frac": trans_frac}


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def _rms_normalize(tracker, value):
    """RunningRMS division WITHOUT updating the tracker (validation path)."""
    if tracker.rms is None:
        return value
    r = tracker.rms
    if r.device != value.device:
        r = r.to(value.device)
    # Mirrors RunningRMS.update's tail exactly (it clamps rms, not sqrt(rms)).
    return value / (torch.sqrt(torch.clamp(r, min=tracker.MIN_RMS ** 2)) + 1e-8)


def run_step(batch, dynamics, reward_head, policy_head, schedule, args, device,
             amp_dtype, rms, state_head=None, update_rms=True):
    """Forward + loss for one batch. Returns (total_loss, info_dict).

    The frozen dynamics runs under no_grad (it's the backbone); gradients flow
    only into the heads via ``agent_out``. ``state_head`` (optional) adds the
    masked aux game-state MSE, weighted by ``args.aux_state_weight``.

    ``update_rms=False`` (validation) normalizes with the CURRENT running RMS
    without folding the val losses into it — otherwise eval batches would
    perturb the training loss weighting.
    """
    z0 = batch["latents"].to(device)                  # (B, T, C, H, W)
    rewards = batch["rewards"].to(device)             # (B, T)
    actions = actions_to_device(batch["actions"], device)
    # Action-history dropout: hide the movement input on a random subset of
    # frames (no_action_embed) so the policy can't lean on copying its own
    # history. Targets are untouched — only the dynamics INPUT is masked.
    p_drop = getattr(args, "action_dropout", 0.0)
    if p_drop > 0 and dynamics.use_actions and "cursor_valid" in actions:
        keep = torch.rand_like(actions["cursor_valid"], dtype=torch.float32) >= p_drop
        actions["cursor_valid"] = actions["cursor_valid"] & keep
    ability_targets = stack_ability_targets(batch["actions"], device)  # (B,T,A)
    movement_targets = actions["movement"]            # (B, T, 2)
    movement_event = actions.get("movement_event")    # (B, T) bool or None
    B, T = rewards.shape

    actions_dict = actions if dynamics.use_actions else None

    # Near-clean context corruption so the frozen denoiser sees in-distribution
    # inputs (it was trained on noised latents): per-frame tau ~ U(tau_ctx, 1).
    tau = args.tau_ctx + torch.rand(B, T, device=device) * (1.0 - args.tau_ctx)
    with torch.no_grad():
        z_noisy, _ = schedule.add_noise(z0, tau)
    d_one = torch.ones(B, dtype=torch.long, device=device)
    _amp = dict(device_type=device.split(":")[0], dtype=amp_dtype,
                enabled=(amp_dtype == torch.bfloat16 or amp_dtype == torch.float16)
                        and device != "cpu")
    # NOT under no_grad and NOT detached: with a frozen backbone this builds no
    # backbone graph, but the agent blocks are trained here — grad must flow from
    # the heads through agent_out into them.
    #
    # The backbone forward MUST be inside autocast. It used to sit outside, which
    # was harmless while the backbone was frozen (no activations retained) but
    # costs ~5x VRAM the moment --unfreeze-backbone stores 18 layers of fp32
    # activations for the backward: measured 14.8 GiB (OOM at batch 4) outside
    # vs 2.8 GiB at batch 2 inside.
    with autocast(**_amp):
        z_pred, agent_out = dynamics(z_noisy, tau, step_size=d_one, actions=actions_dict)

    with autocast(**_amp):
        reward_logits = reward_head(agent_out)
        reward_loss = reward_mtp_loss(
            reward_logits, rewards, reward_head.bucket_centers, args.mtp_length
        )
        bc_loss, bc_info = bc_next_action_loss(
            policy_head, agent_out, ability_targets, movement_targets, args.mtp_length,
            ability_pos_weight=getattr(args, "ability_pos_weight", None),
            movement_event=movement_event,
        )

        aux_loss = torch.zeros((), device=agent_out.device)
        if state_head is not None:
            state = batch["state"].to(device)             # (B, T, S) in [0,1]
            state_mask = batch["state_mask"].to(device)   # (B, T, S) validity
            state_pred = state_head(agent_out)
            aux_loss = ((state_pred - state) ** 2 * state_mask).sum() \
                / state_mask.sum().clamp_min(1.0)

        # Eq (7): the video-prediction loss the paper runs THROUGHOUT Phase 2
        # ("finetune world model ... using (7) and (9)"). Only meaningful with
        # --unfreeze-backbone: z_pred comes from the diffusion backbone, so with
        # it frozen this gradient reaches nothing. Guarded at parse time.
        video_loss = torch.zeros((), device=agent_out.device)
        vw = getattr(args, "video_loss_weight", 0.0)
        if vw > 0:
            video_loss = x_prediction_loss(z_pred, z0, tau, use_ramp_weight=True)

        norm = (lambda k, v: rms[k].update(v)) if update_rms else \
               (lambda k, v: _rms_normalize(rms[k], v))
        bc_n = norm("bc", bc_loss)
        rew_n = norm("reward", reward_loss)
        total = bc_n + rew_n
        if vw > 0:
            total = total + vw * norm("video", video_loss)
        if state_head is not None:
            total = total + args.aux_state_weight * norm("aux", aux_loss)

    info = {
        "loss": total.detach(),
        "bc_loss": bc_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "bc_ability": bc_info["bc_ability"].detach(),
        "bc_movement": bc_info["bc_movement"].detach(),
        "aux_state": aux_loss.detach(),
        **{k: bc_info[k].detach() for k in ("gate_on_trans", "gate_on_hold", "trans_frac")
           if k in bc_info},
    }
    return total, info


@torch.no_grad()
def evaluate(val_loader, dynamics, reward_head, policy_head, schedule, args, device,
             amp_dtype, rms, state_head=None, max_batches=20):
    """Mean losses over held-out GAMES. Never updates the RMS trackers.

    Action-dropout is forced off and the noise RNG is fixed, so successive
    evals differ only because the model changed.
    """
    was_training = reward_head.training
    reward_head.eval(); policy_head.eval()
    if state_head is not None:
        state_head.eval()
    p_drop, args.action_dropout = getattr(args, "action_dropout", 0.0), 0.0
    keys = ("loss", "bc_loss", "bc_ability", "bc_movement", "reward_loss",
            "aux_state", "gate_on_trans", "gate_on_hold", "trans_frac")
    acc = {k: 0.0 for k in keys}
    n = 0
    cpu_rng = torch.get_rng_state()
    dev_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    try:
        torch.manual_seed(1234)  # same tau draw every eval -> comparable numbers
        for batch in val_loader:
            _, info = run_step(batch, dynamics, reward_head, policy_head, schedule,
                               args, device, amp_dtype, rms, state_head=state_head,
                               update_rms=False)
            if not torch.isfinite(info["loss"]):
                continue
            for k in keys:
                if k in info:
                    acc[k] += info[k].item()
            n += 1
            if n >= max_batches:
                break
    finally:
        torch.set_rng_state(cpu_rng)  # don't perturb the training noise stream
        if dev_rng is not None:
            torch.cuda.set_rng_state(dev_rng, device)
        args.action_dropout = p_drop
        if was_training:
            reward_head.train(); policy_head.train()
            if state_head is not None:
                state_head.train()
    if n == 0:
        return None
    return {k: v / n for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(args):
    """Tiny synthetic CPU run: forward+backward+optimizer step end-to-end, and
    PROVE the movement_heads receive a real BC gradient (the headline fix)."""
    print("=" * 60)
    print("PHASE 2 BC SMOKE TEST (synthetic, CPU)")
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
    args.tau_ctx = 0.9
    args.aux_state_weight = 0.5

    B, T, C, S = 2, 6, args.latent_dim, 16
    dynamics = build_dynamics(args, use_actions=True, device=device)
    agent_params = freeze_backbone_train_agent(dynamics)
    model_dim = dynamics.model_dim

    args.movement_gate = True
    args.action_dropout = 0.5
    reward_head = RewardHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets, mtp_length=args.mtp_length).to(device)
    policy_head = PolicyHead(input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
                             hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
                             movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins,
                             movement_gate=True).to(device)
    state_head = StateHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                           num_targets=len(STATE_TARGETS)).to(device)

    # Synthetic batch matching the dataset contract (incl. cursor_valid so the
    # action-dropout path is exercised, and movement_event so the gate is
    # supervised the way the click-event target supervises it).
    # movement is piecewise-constant between events, exactly like the real
    # click-event target — so the "bin change => event" invariant holds here too.
    move_ev = torch.zeros(B, T, dtype=torch.bool)
    move_ev[:, ::3] = True
    mv = torch.rand(B, T, MOVEMENT_DIM)
    for b in range(B):  # hold forward between events
        for t in range(1, T):
            if not move_ev[b, t]:
                mv[b, t] = mv[b, t - 1]
    batch = {
        "latents": torch.randn(B, T, C, S, S),
        "rewards": torch.randn(B, T) * 0.01,
        "actions": {
            "movement": mv,
            "movement_event": move_ev,
            **{k: torch.randint(0, 2, (B, T)) for k in ABILITY_KEYS},
            "cursor_valid": torch.ones(B, T, dtype=torch.bool),
        },
        "state": torch.rand(B, T, len(STATE_TARGETS)),
        "state_mask": (torch.rand(B, T, len(STATE_TARGETS)) > 0.3).float(),
    }

    schedule = DiffusionSchedule(device=device)
    rms = {"bc": RunningRMS(), "reward": RunningRMS(), "aux": RunningRMS(),
           "video": RunningRMS()}
    params = agent_params + list(reward_head.parameters()) \
        + list(policy_head.parameters()) + list(state_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    def agent_backbone_grads():
        """(agent_grad_norm, [backbone params that wrongly got grad])."""
        ag, bad = 0.0, []
        for name, p in dynamics.named_parameters():
            if name.startswith(AGENT_PARAM_PREFIXES):
                if p.grad is not None:
                    ag += p.grad.norm().item()
            elif p.grad is not None:
                bad.append(name)
        return ag, bad

    # === STEP 1: heads must receive BC/reward gradient (headline fix) ===
    optimizer.zero_grad()
    total, info = run_step(batch, dynamics, reward_head, policy_head, schedule,
                           args, device, torch.float32, rms, state_head=state_head)
    total.backward()

    move_grads = [h.weight.grad for h in policy_head.movement_heads if h.weight.grad is not None]
    assert move_grads, "movement_heads received NO gradient (grad is None) under BC!"
    move_grad_norm = sum(g.norm().item() for g in move_grads)
    assert move_grad_norm > 0, f"movement_heads gradient is exactly zero ({move_grad_norm})!"

    ability_grads = [h.weight.grad for h in policy_head.heads if h.weight.grad is not None]
    ability_grad_norm = sum(g.norm().item() for g in ability_grads)
    reward_grad_norm = sum(
        p.grad.norm().item() for p in reward_head.parameters() if p.grad is not None
    )
    # Backbone must NEVER get grad. Agent blocks legitimately get ZERO grad on
    # step 1 — the output heads are zero-init, so d(loss)/d(agent_out) = 0 until
    # the heads move off zero. We assert agent grad on step 2 instead.
    _, backbone_bad = agent_backbone_grads()
    assert not backbone_bad, f"frozen backbone got gradients: {backbone_bad[:3]}"

    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()  # heads move off zero-init

    # === STEP 2: agent blocks must now train (Phase-2 backbone-freeze contract) ===
    optimizer.zero_grad()
    total2, _ = run_step(batch, dynamics, reward_head, policy_head, schedule,
                         args, device, torch.float32, rms, state_head=state_head)
    total2.backward()
    state_grad_norm = sum(p.grad.norm().item() for p in state_head.parameters()
                          if p.grad is not None)
    assert state_grad_norm > 0, "state head received NO gradient under the aux loss!"
    gate_grad_norm = sum(h.weight.grad.norm().item() for h in policy_head.gate_heads
                         if h.weight.grad is not None)
    assert gate_grad_norm > 0, "movement gate heads received NO gradient under the sticky-categorical BC loss!"

    # === The gate must be driven by movement_event, NOT by "did the bin change".
    # Build a target where every event lands in the SAME bin (so the bin-change
    # signal is empty) and assert the reported transition rate still equals the
    # event rate — i.e. the gate is supervised by real commands.
    flat = dict(batch)
    flat["actions"] = dict(batch["actions"])
    flat["actions"]["movement"] = torch.full_like(batch["actions"]["movement"], 0.5)
    _, info_flat = run_step(flat, dynamics, reward_head, policy_head, schedule,
                            args, device, torch.float32, rms, state_head=state_head)
    ev_rate = move_ev[:, 1:].float().mean().item()
    assert abs(info_flat["trans_frac"].item() - ev_rate) < 1e-5, (
        f"gate transition rate {info_flat['trans_frac'].item():.4f} != movement_event "
        f"rate {ev_rate:.4f} — the gate is still keyed off bin changes, not commands!")
    # And with movement_event absent (legacy caches) it must fall back cleanly.
    legacy = dict(batch)
    legacy["actions"] = {k: v for k, v in batch["actions"].items() if k != "movement_event"}
    _, info_legacy = run_step(legacy, dynamics, reward_head, policy_head, schedule,
                              args, device, torch.float32, rms, state_head=state_head)
    assert torch.isfinite(info_legacy["loss"]), "legacy (no movement_event) path broke"
    agent_grad_norm, backbone_bad2 = agent_backbone_grads()
    assert agent_grad_norm > 0, (
        "agent blocks received NO gradient on step 2 — Phase 2 must TRAIN the "
        "agent-token machinery, not just the heads!")
    assert not backbone_bad2, f"frozen backbone got gradients: {backbone_bad2[:3]}"
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    backbone_with_grad = backbone_bad2

    print(f"  total_loss          = {info['loss'].item():.4f}")
    print(f"  bc_loss             = {info['bc_loss'].item():.4f} "
          f"(ability={info['bc_ability'].item():.4f}, movement={info['bc_movement'].item():.4f})")
    print(f"  reward_loss         = {info['reward_loss'].item():.4f}")
    print(f"  GRAD movement_heads = {move_grad_norm:.6e}  (PROOF: > 0 under BC)")
    print(f"  GRAD ability heads  = {ability_grad_norm:.6e}")
    print(f"  GRAD reward head    = {reward_grad_norm:.6e}")
    print(f"  GRAD agent blocks   = {agent_grad_norm:.6e}  (PROOF: > 0, trained in Phase 2)")
    print(f"  GRAD state head     = {state_grad_norm:.6e}  (aux state loss "
          f"{info['aux_state'].item():.4f}, weight {args.aux_state_weight})")
    print(f"  GRAD gate heads     = {gate_grad_norm:.6e}")
    print(f"  gate target rate    = {info_flat['trans_frac'].item():.4f} "
          f"(== movement_event rate {ev_rate:.4f} even with an all-same-bin target: "
          f"PROOF the gate follows COMMANDS, not bin changes)")
    print(f"  frozen backbone grads: {len(backbone_with_grad)} (must be 0)")
    print("  optimizer.step() OK")
    print("SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.video_loss_weight > 0 and not args.unfreeze_backbone:
        raise SystemExit(
            "--video-loss-weight > 0 with a FROZEN backbone is a silent no-op: Eq (7) "
            "predicts latents from the diffusion backbone, so with every backbone param "
            "at requires_grad=False the video gradient reaches nothing and only burns "
            "compute. Add --unfreeze-backbone (paper parity), or set the weight to 0.")
    if args.movement_gate and args.movement_mode == "joint_noop":
        raise SystemExit(
            "--movement-gate and --movement-mode joint_noop both model 'no new "
            "order' and must not be combined; pick one.")
    if args.smoke_test:
        smoke_test(args)
        return

    print("=" * 60)
    print("Phase 2: Agent Finetuning (BC + Reward)")
    print("=" * 60)
    device = args.device
    print(f"Device: {device} | model_size={args.model_size} | latent_dim={args.latent_dim}")
    print(f"seq_len={args.seq_len} mtp={args.mtp_length} movement_bins={args.movement_bins}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading frozen dynamics backbone...")
    dynamics, use_actions, agent_params = load_frozen_dynamics(args, device)
    model_dim = dynamics.model_dim
    print(f"  dynamics params: {sum(p.numel() for p in dynamics.parameters()):,} (frozen)")
    print(f"  action conditioning: {use_actions}")

    reward_head = RewardHead(
        input_dim=model_dim, hidden_dim=args.hidden_dim,
        num_buckets=args.num_buckets, mtp_length=args.mtp_length,
    ).to(device)
    policy_head = PolicyHead(
        input_dim=model_dim, num_abilities=len(ABILITY_KEYS),
        hidden_dim=args.hidden_dim, mtp_length=args.mtp_length,
        movement_dim=MOVEMENT_DIM, movement_bins=args.movement_bins,
        movement_gate=args.movement_gate,
        movement_mode=args.movement_mode,
    ).to(device)
    if args.movement_gate:
        print(f"  movement gate: ON (sticky categorical), action-dropout={args.action_dropout}")
    if args.movement_mode == "joint_noop":
        print(f"  movement: JOINT {args.movement_bins}x{args.movement_bins} grid + NO_OP "
              f"= {policy_head.movement_classes} classes (no gate, PMPO-compatible), "
              f"action-dropout={args.action_dropout}")
    print(f"  reward head: {sum(p.numel() for p in reward_head.parameters()):,}")
    print(f"  policy head: {sum(p.numel() for p in policy_head.parameters()):,}")
    state_head = None
    if args.aux_state_weight > 0:
        state_head = StateHead(input_dim=model_dim, hidden_dim=args.hidden_dim,
                               num_targets=len(STATE_TARGETS)).to(device)
        print(f"  state head:  {sum(p.numel() for p in state_head.parameters()):,} "
              f"(aux weight {args.aux_state_weight}, targets {STATE_TARGETS})")

    print(f"\nLoading replay data from {args.latents_dir}...")
    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("No sequences found. Check --latents-dir / --labels-root / --seq-len.")

    # ── Held-out split (whole games). Before this, BC trained on 125/125 games
    # and reported only training loss (audit finding 3). ─────────────────────
    val_vids = select_val_matches(dataset, args)
    train_vids = {s["video_id"] for s in dataset.sequences} - val_vids
    # Persist the resolved split into the checkpoint's args blob so a future
    # eval can prove which games a checkpoint never saw.
    args.val_matches_resolved = sorted(val_vids)
    val_idx = [i for i, s in enumerate(dataset.sequences) if s["video_id"] in val_vids]
    if val_vids:
        overlap = train_vids & val_vids
        if overlap:  # structurally impossible; assert anyway
            raise SystemExit(f"train/val game overlap: {sorted(overlap)}")
        if not val_idx:
            raise SystemExit(f"val games {sorted(val_vids)} produced 0 sequences")
        print(f"  [split] train {len(train_vids)} games / val {len(val_vids)} games "
              f"({len(dataset) - len(val_idx)} / {len(val_idx)} sequences), disjoint")
        print(f"  [split] val games: {sorted(val_vids)}")
    else:
        print("  [split] !!! NO VALIDATION SET (--val-games 0): every metric this run "
              "reports is IN-SAMPLE and cannot show generalization.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=VideoGroupedSampler(dataset, exclude_videos=val_vids),
        num_workers=args.num_workers, pin_memory=(device != "cpu"), drop_last=True,
    )
    val_loader = None
    if val_idx:
        order = build_val_order(dataset, val_vids, args.batch_size, args.val_batches)
        if not order:
            raise SystemExit(
                f"val split has {len(val_idx)} sequences but batch_size="
                f"{args.batch_size} leaves 0 full batches in any val game")
        assert set(order) <= set(val_idx), "val order leaked a training sequence"
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, order),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device != "cpu"), drop_last=True,
        )
        print(f"  [split] val loader: {len(val_loader)} batches "
              f"({len(order)} sequences, an equal slice of each of "
              f"{len(val_vids)} held-out games)")

    schedule = DiffusionSchedule(device=device)
    rms = {"bc": RunningRMS(), "reward": RunningRMS(), "aux": RunningRMS(),
           "video": RunningRMS()}
    # Trainable set: agent-token blocks (frozen backbone excluded) + all heads.
    params = agent_params + list(reward_head.parameters()) + list(policy_head.parameters())
    if state_head is not None:
        params = params + list(state_head.parameters())
    optimizer = create_optimizer(params, args.lr, args.weight_decay,
                                 use_8bit=args.use_8bit_adam, betas=tuple(args.adam_betas))
    total_steps = args.epochs * max(1, len(dataloader))
    scheduler = create_wsd_schedule(optimizer, total_steps, args.warmup_steps, args.decay_steps)
    amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
    scaler = GradScaler(device.split(":")[0], enabled=(amp_dtype == torch.float16))

    # --- Resume (crash recovery): restore agent blocks + heads + optim + step ---
    start_epoch, global_step = 0, 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(checkpoint_dir / "agent_finetune_latest.pt")
    if resume_path and Path(resume_path).exists():
        rc = torch.load(resume_path, map_location=device, weights_only=False)
        # A Phase-2 checkpoint is a FULL dynamics state_dict written by this same
        # build, so nothing may be missing or unexpected. If it is, the arch
        # flags drifted since the checkpoint and a loose load would resume with
        # randomly-initialised tensors.
        load_state_dict_guarded(
            getattr(dynamics, "_orig_mod", dynamics), rc["dynamics_state_dict"],
            what=f"Phase-2 resume {resume_path}")
        reward_head.load_state_dict(rc["reward_head_state_dict"])
        # The movement head's OUTPUT LAYER changes shape between movement_modes
        # (axis: move_dim*bins = 42 logits; joint_noop: bins**2+1 = 442), and
        # joint_noop has no gate_heads at all. Those specific tensors are
        # deliberately rebuilt; EVERYTHING else must still match exactly, so we
        # drop only the known-incompatible keys rather than loosening the load.
        _ph = dict(rc["policy_head_state_dict"])
        _cur = policy_head.state_dict()
        _rebuilt = [k for k, v in _ph.items()
                    if k not in _cur or _cur[k].shape != v.shape]
        for k in _rebuilt:
            _ph.pop(k)
        _missing, _unexpected = policy_head.load_state_dict(_ph, strict=False)
        _bad = [k for k in _missing if k not in _rebuilt]
        if _bad or _unexpected:
            raise SystemExit(
                f"policy head resume mismatch beyond the movement/gate rebuild:\n"
                f"  unexpected: {_unexpected[:8]}\n  missing: {_bad[:8]}")
        if _rebuilt:
            print(f"  [resume] policy head: rebuilt {len(_rebuilt)} tensor(s) at fresh "
                  f"init (movement_mode change): {_rebuilt[:4]}"
                  f"{' ...' if len(_rebuilt) > 4 else ''}")
        if state_head is not None and "state_head_state_dict" in rc:
            saved_targets = rc.get("state_targets")
            if saved_targets is not None and list(saved_targets) != list(STATE_TARGETS):
                raise SystemExit(
                    f"aux-state targets changed since this checkpoint: it was trained on "
                    f"{list(saved_targets)} but this build uses {list(STATE_TARGETS)}. The "
                    f"head's output columns are positional, so resuming would keep weights "
                    f"trained for different quantities. Re-run with --aux-state-weight 0, or "
                    f"start the head fresh.")
            if saved_targets is None:
                print("WARNING: checkpoint predates state-target recording; assuming its "
                      f"aux head matches {list(STATE_TARGETS)}. If it was trained before "
                      "2026-08-12 the enemy_visible column means something different.")
            state_head.load_state_dict(rc["state_head_state_dict"])
        try:
            optimizer.load_state_dict(rc["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            # Param set changed since the checkpoint (e.g. aux head newly
            # enabled) — fresh optimizer; Adam moments re-estimate in ~1k steps.
            print(f"  [resume] optimizer state incompatible ({e}); starting fresh optimizer")
        scheduler.load_state_dict(rc["scheduler_state_dict"])
        for k, st in rc.get("rms_state", {}).items():
            if k in rms:
                rms[k].load_state_dict(st)
        start_epoch = rc.get("epoch", 0)
        global_step = rc.get("global_step", 0)
        print(f"  [resume] {resume_path}: epoch {start_epoch}, global_step {global_step}")

    init_wandb(args, job_type="agent_finetune", extra_config={
        "reward_head_params": sum(p.numel() for p in reward_head.parameters()),
        "policy_head_params": sum(p.numel() for p in policy_head.parameters()),
        "val_matches": sorted(val_vids),
        "n_train_games": len(train_vids),
        "movement_source": args.movement_source,
    })

    print("\n" + "=" * 60)
    print("Starting BC + reward training...")
    print("=" * 60)
    last_ckpt_t = time.time()
    for epoch in range(start_epoch, args.epochs):
        reward_head.train()
        policy_head.train()
        if state_head is not None:
            state_head.train()
        # Advance the sampler seed with TRAINING PROGRESS, not just epoch. A
        # resume re-enters this loop at batch 0, so seeding by epoch alone would
        # replay the exact batches already trained on this epoch -- which is what
        # was happening (measured: ~2/3 of epoch 1 never reached across three
        # restarts). global_step differs after every resume, so the order does too.
        _sampler = getattr(dataloader, "sampler", None)
        if hasattr(_sampler, "set_epoch"):
            _sampler.set_epoch(epoch * 1_000_003 + global_step)
            print(f"  [sampler] epoch {epoch} order seeded from global_step {global_step}")
        t0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            accum = max(getattr(args, "grad_accum", 1), 1)
            if batch_idx % accum == 0:
                optimizer.zero_grad()
            total, info = run_step(batch, dynamics, reward_head, policy_head,
                                   schedule, args, device, amp_dtype, rms,
                                   state_head=state_head)
            if not torch.isfinite(total):
                print(f"[WARN] non-finite loss at step {global_step}; skipping.")
                continue
            # Divide by accum so the accumulated gradient equals the mean over
            # the effective batch, not its sum (otherwise the effective LR scales
            # with --grad-accum and the run is not comparable to the baseline).
            scaler.scale(total / accum).backward()
            if (batch_idx + 1) % accum != 0:
                continue                      # keep accumulating; do NOT step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # Time-based checkpoint: a crash loses <= checkpoint-minutes, not a
            # whole (~19h on the 1060) epoch. Stores the CURRENT epoch so resume
            # re-enters it with weights/optimizer/step intact.
            if (time.time() - last_ckpt_t) >= args.checkpoint_minutes * 60:
                save_phase2_checkpoint(checkpoint_dir / "agent_finetune_latest.pt", dynamics,
                                       reward_head, policy_head, optimizer, scheduler, rms,
                                       epoch, global_step, args, state_head=state_head)
                last_ckpt_t = time.time()

            if val_loader is not None and global_step % args.val_interval == 0:
                v = evaluate(val_loader, dynamics, reward_head, policy_head, schedule,
                             args, device, amp_dtype, rms, state_head=state_head,
                             max_batches=len(val_loader))
                if v is not None:
                    log_step({f"val/{k}": x for k, x in v.items()}, step=global_step)
                    print(f"  [VAL @ step {global_step}] loss={v['loss']:.4f} "
                          f"bc={v['bc_loss']:.4f} (abil={v['bc_ability']:.3f} "
                          f"move={v['bc_movement']:.3f}) rew={v['reward_loss']:.4f} "
                          f"aux={v['aux_state']:.4f}  [{len(val_vids)} held-out games]")

            # keyed on OPTIMIZER steps, not micro-batches: with --grad-accum the
            # non-step micro-batches `continue` before this point, so a batch_idx
            # condition can land only on step boundaries and may never fire at all
            # (accum=8 puts every boundary on an odd batch_idx -> log_interval=2 never matched).
            if global_step % args.log_interval == 0:
                sps = (batch_idx + 1) * args.batch_size / max(time.time() - t0, 1e-6)  # micro-batches consumed
                log_step({
                    "train/loss": info["loss"].item(),
                    "train/bc_loss": info["bc_loss"].item(),
                    "train/bc_ability": info["bc_ability"].item(),
                    "train/bc_movement": info["bc_movement"].item(),
                    "train/reward_loss": info["reward_loss"].item(),
                    "train/aux_state": info["aux_state"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=global_step)
                aux_s = f" aux={info['aux_state'].item():.4f}" if state_head is not None else ""
                gate_s = ""
                if args.movement_gate and "gate_on_trans" in info:
                    gate_s = (f" gate[t={info['gate_on_trans'].item():.2f}"
                              f" h={info['gate_on_hold'].item():.2f}"
                              f" base={info['trans_frac'].item():.2f}]")
                _pk = (f" vram={torch.cuda.max_memory_allocated()/2**30:.2f}G"
                       if device.startswith("cuda") else "")
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={info['loss'].item():.4f} "
                      f"bc={info['bc_loss'].item():.4f} "
                      f"(abil={info['bc_ability'].item():.3f} move={info['bc_movement'].item():.3f}) "
                      f"rew={info['reward_loss'].item():.4f}{aux_s}{gate_s} ({sps:.1f} samp/s{_pk})")

        if val_loader is not None:  # end-of-epoch held-out eval
            v = evaluate(val_loader, dynamics, reward_head, policy_head, schedule,
                         args, device, amp_dtype, rms, state_head=state_head,
                         max_batches=len(val_loader))
            if v is not None:
                log_step({f"val/{k}": x for k, x in v.items()}, step=global_step)
                print(f"[EPOCH {epoch} VAL] loss={v['loss']:.4f} bc={v['bc_loss']:.4f} "
                      f"(abil={v['bc_ability']:.3f} move={v['bc_movement']:.3f}) "
                      f"rew={v['reward_loss']:.4f} aux={v['aux_state']:.4f}")

        ckpt_path = checkpoint_dir / f"agent_finetune_epoch_{epoch + 1:03d}.pt"
        save_phase2_checkpoint(ckpt_path, dynamics, reward_head, policy_head,
                               optimizer, scheduler, rms, epoch + 1, global_step, args,
                               state_head=state_head)
        save_phase2_checkpoint(checkpoint_dir / "agent_finetune_latest.pt", dynamics,
                               reward_head, policy_head, optimizer, scheduler, rms,
                               epoch + 1, global_step, args, state_head=state_head)

    print("\nPhase 2 training complete.")
    finish_wandb()


def save_phase2_checkpoint(path, dynamics, reward_head, policy_head, optimizer,
                           scheduler, rms, epoch, global_step, args, state_head=None):
    inner = getattr(dynamics, "_orig_mod", dynamics)
    ckpt = {
        "dynamics_state_dict": inner.state_dict(),
        "dynamics_config": getattr(inner, "config", None),
        "reward_head_state_dict": reward_head.state_dict(),
        "state_head_state_dict": state_head.state_dict() if state_head is not None else None,
        "policy_head_state_dict": policy_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rms_state": {k: v.state_dict() for k, v in rms.items()},
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        # The aux head's output columns are positional. Shape alone cannot catch
        # a REDEFINED target (enemy_visible changed meaning on 2026-08-12 when it
        # stopped counting frustum-membership as visible) or a reordered tuple —
        # same width, different semantics, loads clean, silently wrong. Record
        # the names so a resume can refuse.
        "state_targets": list(STATE_TARGETS),
        "phase": "agent_finetune",
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    import os
    os.replace(tmp, path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
