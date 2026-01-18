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

Format: `{date}_{tokenizer}_{dynamics}_{description}`

Example: `20260118_cnn_small_baseline`

## Current Experiments

| ID | Date | Tokenizer | Dynamics | Latent Dir | Checkpoint Dir | Status |
|----|------|-----------|----------|------------|----------------|--------|
| `trans_small_20260107` | 2026-01-07 | transformer_small | small | `data/processed/latents` | `checkpoints/dynamics_*` | Complete |
| `cnn_small_20260118` | 2026-01-18 | cnn (tokenizer_epoch_010) | small | `data/processed/latents_cnn` | `checkpoints/cnn_dynamics_*` | In Progress |

## Tokenizer Checkpoints

| Name | Path | Params | PSNR | Latent Dim |
|------|------|--------|------|------------|
| CNN (epoch 10) | `checkpoints/tokenizer_epoch_010.pt` | 13M | 32.6 dB | 256 |
| Transformer (best) | `checkpoints/run_20260115_195915/transformer_tokenizer_best.pt` | 40M | 27.0 dB | 32 |

## Dynamics Checkpoints

| Name | Path | Tokenizer Used | Notes |
|------|------|----------------|-------|
| Trans dynamics | `checkpoints/dynamics_step_*.pt` | transformer_small | Original training |
| CNN dynamics | `checkpoints/cnn_dynamics_*.pt` | cnn_epoch_010 | New experiment |
