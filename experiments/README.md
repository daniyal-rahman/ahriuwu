# Experiments Registry

This directory tracks different experiment configurations to avoid confusion.

## Directory Structure

```
experiments/
├── README.md                 # This file
├── registry.json             # Machine-readable experiment registry
└── <experiment_id>/          # Per-experiment configs and notes
    ├── config.json
    └── notes.md
```

## Experiment Naming Convention

Format: `{date}_{description}`

Example: `20260302_dynamics_small_baseline`

## Current Experiments

| ID | Date | Tokenizer | Dynamics | Latent Dir | Checkpoint Dir | Status |
|----|------|-----------|----------|------------|----------------|--------|
| `trans_small_20260107` | 2026-01-07 | transformer_small (40M) | small | `data/processed/latents` | `checkpoints/dynamics_*` | Complete |

## Tokenizer Checkpoints

| Name | Path | Params | PSNR | Latent Dim |
|------|------|--------|------|------------|
| Transformer (best) | `checkpoints/run_20260115_195915/transformer_tokenizer_best.pt` | 40M | 27.0 dB | 32 |
| Transformer (latest) | `checkpoints/run_20260117_165401/transformer_tokenizer_latest.pt` | 40M | 27.5 dB | 32 |

## Dynamics Checkpoints

| Name | Path | Tokenizer Used | Notes |
|------|------|----------------|-------|
| Trans dynamics | `checkpoints/dynamics_step_*.pt` | transformer_small | Original training |

## Key Experimental Variables

### Phase 0: Tokenizer (Transformer)
- Model sizes: tiny/small/medium/large (embed_dim: 256/512/768/1024)
- LPIPS weight: 0.2 (paper default)
- MAE mask curriculum: warmup 50k steps (prevents mode collapse)
- Image size: 352x352, patch size: 16x16 = 484 spatial tokens

### Phase 1: Dynamics
- Model sizes: tiny/small/medium/large (model_dim: 256/512/768/1024)
- Latent dim: 32 (from transformer tokenizer small)
- Spatial size: 22x22 (352/16 = 22)
- Diffusion: x-prediction with ramp weight, per-timestep tau (diffusion forcing)
- Shortcut forcing: optional, k_max=64 for 4-step inference
- Independent frame ratio: 0.3 (30% batches disable temporal attention)
- Alternating lengths: T_short=32, T_long=64

### Phase 2: Agent Finetuning
- MTP length: 8 (multi-token prediction horizon)
- Action dim: 128 discrete + continuous (x,y) movement
- Reward: symlog twohot with 255 buckets
- Loss normalization: RunningRMS per component (dynamics, reward, BC)

### Stability Features (DreamerV4)
- QKNorm (normalize Q/K before RoPE)
- Attention soft capping: 50.0
- Register tokens: 8
- Gradient checkpointing: enabled by default
