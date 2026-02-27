#!/usr/bin/env python3
"""Phase 2 Validation - Sanity checks before Phase 3.

Validates:
1. Reward model: gold vs non-gold sequence separation
2. Action prediction: GT vs top-3 predicted spot check
3. Checkpoint structure verification
"""

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ahriuwu.models import create_dynamics, RewardHead, PolicyHead, DiffusionSchedule
from ahriuwu.data.dataset import PackedLatentSequenceDataset
from ahriuwu.data.actions import encode_action, decode_action


def load_checkpoint(checkpoint_path: str, device: str):
    """Load Phase 2 checkpoint and reconstruct models."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})

    # Get model config from checkpoint args
    model_size = args.get("model_size", "small")
    latent_dim = args.get("latent_dim", 32)
    mtp_length = args.get("mtp_length", 8)
    num_buckets = args.get("num_buckets", 255)
    action_dim = args.get("action_dim", 128)

    print(f"Checkpoint config:")
    print(f"  Model size: {model_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  MTP length: {mtp_length}")
    print(f"  Num buckets: {num_buckets}")
    print(f"  Action dim: {action_dim}")

    # Create dynamics with agent tokens
    dynamics = create_dynamics(
        size=model_size,
        latent_dim=latent_dim,
        use_agent_tokens=True,
        num_tasks=1,
        agent_layers=4,
        use_qk_norm=not args.get("no_qk_norm", False),
        soft_cap=args.get("soft_cap", 50.0),
        num_register_tokens=args.get("num_register_tokens", 8),
        gradient_checkpointing=False,  # Not needed for inference
    )
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    dynamics = dynamics.to(device).float().eval()

    model_dim = dynamics.model_dim

    # Create heads
    reward_head = RewardHead(
        input_dim=model_dim,
        hidden_dim=256,
        num_buckets=num_buckets,
        mtp_length=mtp_length,
    )
    reward_head.load_state_dict(checkpoint["reward_head_state_dict"])
    reward_head = reward_head.to(device).float().eval()

    policy_head = PolicyHead(
        input_dim=model_dim,
        action_dim=action_dim,
        hidden_dim=256,
        mtp_length=mtp_length,
    )
    policy_head.load_state_dict(checkpoint["policy_head_state_dict"])
    policy_head = policy_head.to(device).float().eval()

    return dynamics, reward_head, policy_head, checkpoint


def load_dataset(latents_dir: str, features_dir: str, seq_len: int = 32):
    """Load dataset with reward indices."""
    dataset = PackedLatentSequenceDataset(
        latents_dir=latents_dir,
        sequence_length=seq_len,
        stride=seq_len // 2,
        load_actions=True,
        features_dir=features_dir,
    )
    return dataset


def get_gold_sequences(dataset, features_dir: str, n_samples: int = 100):
    """Get sequences with gold events (reward_indices) and without."""
    features_dir = Path(features_dir)

    # Get reward indices from dataset
    reward_set = set(dataset.reward_indices)
    all_indices = set(range(len(dataset.sequences)))
    non_reward_indices = list(all_indices - reward_set)
    reward_indices = list(reward_set)

    print(f"Total sequences: {len(dataset.sequences)}")
    print(f"Sequences with rewards: {len(reward_indices)}")
    print(f"Sequences without rewards: {len(non_reward_indices)}")

    # Sample
    random.seed(42)
    gold_samples = random.sample(reward_indices, min(n_samples, len(reward_indices)))
    non_gold_samples = random.sample(non_reward_indices, min(n_samples, len(non_reward_indices)))

    return gold_samples, non_gold_samples


def validate_reward_model(dynamics, reward_head, dataset, gold_indices, non_gold_indices, device):
    """Check reward predictions for gold vs non-gold sequences."""
    print("\n" + "=" * 60)
    print("REWARD MODEL VALIDATION")
    print("=" * 60)

    diffusion = DiffusionSchedule(device=device)

    def get_mean_reward_pred(indices, desc):
        """Get mean predicted reward for a set of sequence indices."""
        all_preds = []
        for idx in indices:
            data = dataset[idx]
            latents = data["latents"].unsqueeze(0).to(device).float()  # (1, T, C, H, W)

            B, T = 1, latents.shape[1]

            with torch.no_grad():
                # Use tau=0 (no noise) for clean predictions
                tau = torch.zeros(B, T, device=device)

                # Forward through dynamics (float32 for inference stability)
                _, agent_out = dynamics(latents, tau)

                # Get reward predictions
                reward_pred = reward_head.predict(agent_out)  # (B, T, L)
                mean_pred = reward_pred.mean().item()
                all_preds.append(mean_pred)

        return all_preds

    print(f"\nProcessing {len(gold_indices)} gold sequences...")
    gold_preds = get_mean_reward_pred(gold_indices, "gold")

    print(f"Processing {len(non_gold_indices)} non-gold sequences...")
    non_gold_preds = get_mean_reward_pred(non_gold_indices, "non-gold")

    gold_mean = sum(gold_preds) / len(gold_preds)
    non_gold_mean = sum(non_gold_preds) / len(non_gold_preds)

    print(f"\nResults:")
    print(f"  Gold sequences mean predicted reward:     {gold_mean:+.4f}")
    print(f"  Non-gold sequences mean predicted reward: {non_gold_mean:+.4f}")
    print(f"  Separation (gold - non-gold):             {gold_mean - non_gold_mean:+.4f}")

    if gold_mean > non_gold_mean:
        print("\n  [PASS] Gold sequences have higher predicted rewards")
    else:
        print("\n  [WARN] Gold sequences do NOT have higher predicted rewards")

    return gold_mean, non_gold_mean


def validate_action_predictions(dynamics, policy_head, dataset, device, n_samples: int = 5):
    """Spot check action predictions vs ground truth."""
    print("\n" + "=" * 60)
    print("ACTION PREDICTION SPOT CHECK")
    print("=" * 60)

    # Sample random sequences
    random.seed(123)
    indices = random.sample(range(len(dataset.sequences)), min(n_samples, len(dataset.sequences)))

    for i, idx in enumerate(indices):
        data = dataset[idx]
        latents = data["latents"].unsqueeze(0).to(device).float()  # (1, T, C, H, W)
        actions_dict = data.get("actions")

        if actions_dict is None:
            print(f"\nSequence {i+1}: No actions available")
            continue

        # Encode ground truth actions
        seq_len = latents.shape[1]
        gt_actions = []
        for t in range(seq_len):
            action = encode_action(
                movement=actions_dict['movement'][t].item(),
                abilities={
                    'Q': bool(actions_dict['Q'][t].item()),
                    'W': bool(actions_dict['W'][t].item()),
                    'E': bool(actions_dict['E'][t].item()),
                    'R': bool(actions_dict['R'][t].item()),
                    'D': bool(actions_dict['D'][t].item()),
                    'F': bool(actions_dict['F'][t].item()),
                    'item': bool(actions_dict['item'][t].item()),
                    'B': bool(actions_dict['B'][t].item()),
                }
            )
            gt_actions.append(action)

        with torch.no_grad():
            tau = torch.zeros(1, seq_len, device=device)
            _, agent_out = dynamics(latents, tau)
            action_logits = policy_head(agent_out)  # (1, T, L, action_dim)

            # Get top-3 predictions for next-step (offset=0)
            logits_t0 = action_logits[0, :-1, 0, :]  # (T-1, action_dim)
            probs = F.softmax(logits_t0, dim=-1)
            top3_probs, top3_indices = probs.topk(3, dim=-1)

        seq_info = dataset.sequences[idx]
        print(f"\nSequence {i+1}: video={seq_info['video_id']}, start={seq_info['start_idx']}")
        print(f"  t | GT Action        | Top-3 Predicted (prob)")
        print(f"  --|------------------|----------------------------------")

        # Show first 5 timesteps
        for t in range(min(5, seq_len - 1)):
            gt = gt_actions[t + 1]  # Next action (what we predict)
            gt_move, gt_ability = decode_action(gt)
            gt_str = f"mv={gt_move:2d}"
            if gt_ability:
                gt_str += f" +{gt_ability}"
            else:
                gt_str += "      "

            pred_strs = []
            for k in range(3):
                pred_idx = top3_indices[t, k].item()
                pred_prob = top3_probs[t, k].item()
                pred_move, pred_ability = decode_action(pred_idx)
                pred_str = f"mv={pred_move:2d}"
                if pred_ability:
                    pred_str += f"+{pred_ability}"
                pred_strs.append(f"{pred_str}({pred_prob:.2f})")

            match = "*" if gt == top3_indices[t, 0].item() else " "
            print(f"  {t:2d}| {gt_str:16s} | {', '.join(pred_strs)} {match}")


def validate_checkpoint_structure(checkpoint_path: str):
    """Verify checkpoint has all expected keys and non-zero weights."""
    print("\n" + "=" * 60)
    print("CHECKPOINT STRUCTURE VERIFICATION")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Phase: {checkpoint.get('phase', 'N/A')}")

    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"\nFinal metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Check state dicts
    for key in ["dynamics_state_dict", "reward_head_state_dict", "policy_head_state_dict"]:
        if key not in checkpoint:
            print(f"\n[FAIL] Missing key: {key}")
            continue

        state_dict = checkpoint[key]
        total_params = 0
        non_zero_params = 0

        for name, param in state_dict.items():
            total_params += param.numel()
            non_zero_params += (param != 0).sum().item()

        pct = 100 * non_zero_params / total_params if total_params > 0 else 0
        print(f"\n{key}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero: {non_zero_params:,} ({pct:.1f}%)")

        if pct > 99:
            print(f"  [PASS] Weights are properly initialized")
        elif pct > 50:
            print(f"  [WARN] Some weights may be zero-initialized")
        else:
            print(f"  [FAIL] Too many zero weights - model may not have trained properly")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/agent_finetune_latest.pt")
    parser.add_argument("--latents-dir", default="data/latents_v2")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2 VALIDATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # 1. Checkpoint structure
    validate_checkpoint_structure(args.checkpoint)

    # 2. Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    dynamics, reward_head, policy_head, checkpoint = load_checkpoint(
        args.checkpoint, args.device
    )
    print("Models loaded successfully")

    # 3. Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.latents_dir, args.features_dir)

    # 4. Get gold vs non-gold sequences
    gold_indices, non_gold_indices = get_gold_sequences(
        dataset, args.features_dir, n_samples=args.n_samples
    )

    # 5. Validate reward model
    validate_reward_model(
        dynamics, reward_head, dataset,
        gold_indices, non_gold_indices, args.device
    )

    # 6. Validate action predictions
    validate_action_predictions(dynamics, policy_head, dataset, args.device)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
