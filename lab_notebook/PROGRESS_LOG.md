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
- [ ] SIGTERM handler: save checkpoint on signal before exit (Slurm preemption)
- [ ] Checkpoint saving: periodic (every N steps) + best (by validation loss)
- [ ] Logging: track attention entropy per layer during early training — if any layer's entropy drops toward zero, attention is collapsing (the old failure mode)
- [ ] VRAM should be ~7-9 GB at batch=4 T=16. If it's under 3 GB, sequences aren't working. If it's over 12 GB, attention isn't factored.

## Verification Before Training

- [ ] Print total param count — should be ~130M
- [ ] Print input shape entering encoder — should be (B, T, 484, 768) or equivalent
- [ ] Print attention matrix shapes — space layers should be (B×T, heads, 484, 484), time layers should be (B×484, heads, 16, 16) or similar
- [ ] Run 10 steps, check `torch.cuda.max_memory_allocated()` — should be 7-9 GB
- [ ] Run 100 steps, verify all attention entropy values are well above zero
- [ ] Verify decoded output visually looks like a blurry version of input after ~1000 steps (not uniform gray)

---

## Dynamics Model

---

## Architecture

- [ ] Block-causal transformer
- [ ] Total layers: 24
- [ ] Model dim: 768
- [ ] Attention heads: 12 query heads (head_dim=64)
- [ ] GQA: 4 KV heads, 12 query heads (3:1 ratio)
- [ ] FFN: SwiGLU, hidden_dim = 2048 (768 × 8/3)
- [ ] Total params should be ~170M — verify with parameter count print at init
- [ ] Pre-layer RMSNorm (not LayerNorm)

## Attention — Factored Space-Time

- [ ] Space-only layers: 18 out of 24 (every layer that isn't a temporal layer)
- [ ] Temporal layers: every 4th layer (layers 4, 8, 12, 16, 20, 24) = 6 temporal layers
- [ ] Space layers: reshape (B, T×S, D) → (B×T, S, D), attend over S tokens per frame (Nz + registers + signal token + action token)
- [ ] Time layers: reshape (B, T×S, D) → (B×S, T, D), attend over T frames per spatial position
- [ ] Block-causal masking on time layers: frame t attends to frames ≤ t only
- [ ] Space layers: full (non-causal) attention within each frame
- [ ] GQA applied to ALL attention layers (both space and time)
- [ ] Should NEVER materialize a (T×S) × (T×S) attention matrix

## Attention Stability

- [ ] QKNorm on Q and K before attention scores
- [ ] Attention logit soft capping: cap = 50.0, applied as `cap * tanh(logits / cap)` before softmax
- [ ] Both applied in every layer (space and time)

## Positional Encoding

- [ ] RoPE for both spatial and temporal positions
- [ ] Applied to Q and K only
- [ ] Spatial RoPE for space layers, temporal RoPE for time layers

## Input Sequence — Interleaved Per Frame

Each frame t in the sequence contains these tokens in order:

- [ ] Action tokens: Sa=1 token. Multiple action components (movement direction, ability keys, mouse) each embedded separately and summed together with a learned base embedding. For unlabeled data (no actions available): use only the learned base embedding.
- [ ] Signal token: 1 token. Concatenation of discrete embedding for τ (signal level) and discrete embedding for d (step size). τ and d are discrete values, use embedding lookup not linear projection.
- [ ] Latent tokens: Nz=256 tokens from frozen tokenizer output, linearly projected to model dim (768)
- [ ] Register tokens: Sr=8 learned tokens (your current default, fine to keep)
- [ ] **Total tokens per frame: 1 + 1 + 256 + 8 = 266**
- [ ] **Total sequence at T=64: 266 × 64 = 17,024 tokens**

## Shortcut Forcing Objective

- [ ] X-prediction: network predicts clean latents ẑ₁ directly, NOT velocity v
- [ ] Input: corrupted latents z̃ = (1-τ)z₀ + τz₁ where z₀ ~ N(0,1), z₁ from data
- [ ] Kmax = 64 (finest step size dmin = 1/64)
- [ ] Step size d sampled as 1/U({1, 2, 4, 8, ..., 64})
- [ ] Signal level τ sampled uniformly over grid reachable by current d: τ ~ U({0, d, 2d, ..., 1-d})
- [ ] Each frame in the sequence gets independently sampled τ and d (diffusion forcing — different noise per timestep)

## Loss Computation

- [ ] Flow loss (when d = dmin): L = ||ẑ₁ - z₁||²
- [ ] Bootstrap loss (when d > dmin):
  - Teacher call 1 (no grad): b' = (f(z̃, τ, d/2, a) - z_τ) / (1-τ)
  - Intermediate: z' = z̃ + b' × d/2
  - Teacher call 2 (no grad): b'' = (f(z', τ+d/2, d/2, a) - z') / (1-(τ+d/2))
  - Target: v_target = sg(b' + b'') / 2
  - Loss: (1-τ)² × ||(ẑ₁ - z̃)/(1-τ) - v_target||²
  - The (1-τ)² multiplier converts from v-space back to x-space loss scale
- [ ] Teacher calls use `torch.no_grad()` and stop gradient (sg)
- [ ] Gradient checkpointing disabled during shortcut forcing (the autocast crash fix), re-enabled after
- [ ] Ramp loss weight: w(τ) = 0.9τ + 0.1 (upweights high signal levels, downweights noisy ones)
- [ ] RunningRMS loss normalization across all loss terms
- [ ] Velocity MSE clamping (your previous fix for bootstrap loss explosions) — keep this

## Context Corruption

- [ ] Past context frames corrupted to τ_ctx = 0.1 at inference time (slight noise on history to make model robust to its own imperfect generations)
- [ ] During training: the diffusion forcing objective already handles this naturally since each frame gets a random τ

## Alternating Batch Lengths

- [ ] Short batches: T_short=32, batch=2 (80% of training)
- [ ] Long batches: T_long=64, batch=1 (20% of training)
- [ ] Finetune on long batches only at the end of training
- [ ] Batch lengths must be LONGER than the effective context length to prevent the model overfitting to always seeing a start frame at position 0
- [ ] Gradient accumulation: accum=32 for short (effective=64), accum=64 for long (effective=64). Keep effective batch constant.

## Frozen Tokenizer Integration

- [ ] Tokenizer weights are FROZEN during dynamics training. No gradients flow into tokenizer.
- [ ] Latent tokens are pre-computed and stored on disk (your pretokenization pipeline)
- [ ] Linear projection from tokenizer bottleneck dim (Db=32) to dynamics model dim (768) IS trainable — this is part of the dynamics model, not the tokenizer
- [ ] At inference: tokenizer runs live to encode current frame → project → feed to dynamics

## Start Frame Generation

- [ ] 30% of training sequences treated as independent images (no temporal context) — teaches the model to generate plausible start frames without conditioning on history
- [ ] Implementation: for 30% of batch, zero out or mask the temporal attention so each frame is generated independently

## Training Config

- [ ] Optimizer: AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
- [ ] LR schedule: linear warmup 2000 steps → cosine decay
- [ ] Precision: bf16 autocast (NOT fp16)
- [ ] Gradient checkpointing: every 2 layers (disabled during shortcut forcing loss computation)
- [ ] Gradient clipping: global norm 1.0
- [ ] torch.compile if stable

## Infrastructure

- [ ] SIGTERM handler: save checkpoint + optimizer state + step count on signal
- [ ] Checkpoint saving: periodic (every N steps) + best (by validation loss)
- [ ] Separate checkpoints for short-batch phase and long-finetune phase
- [ ] Logging: track flow loss and bootstrap loss separately (if bootstrap diverges while flow is stable, the velocity clamping needs adjustment)
- [ ] Log per-step-size losses: track loss for each d value (1/4, 1/8, 1/16, etc.) separately to see if specific step sizes are struggling

## Verification Before Training

- [ ] Print total param count — should be ~170M
- [ ] Print input token breakdown: "266 tokens/frame × T frames = N total per sequence"
- [ ] Print attention shapes — space layers: (B×T, heads, 266, 266), time layers: (B×266, heads, T, T)
- [ ] Verify GQA: KV projection dim should be 4 heads × 64 = 256, not 12 × 64 = 768
- [ ] Run 10 steps at T_short=32 B=2, check VRAM — should be ~8-9 GB
- [ ] Run 10 steps at T_long=64 B=1, check VRAM — should be ~8-10 GB
- [ ] Verify tokenizer is frozen: `assert all(not p.requires_grad for p in tokenizer.parameters())`
- [ ] Verify bootstrap loss is finite after 100 steps (no explosions to 1e14)
- [ ] Decode a few 1-step denoised latents through the frozen tokenizer decoder — should be recognizable game frames, not noise

## Evaluation Gates

- [ ] 1-step denoising PSNR > 15 dB (your 62M hit 13.7, this should beat it)
- [ ] 16-frame rollout with τ_ctx=0.1 context: stays recognizable, PSNR > 8 dB
- [ ] 32-frame rollout with context: should stay above 6 dB (your 62M died at 12 frames)
- [ ] 64-frame rollout from noise: should produce recognizable LoL game states, not cyan blobs
- [ ] Qualitative: decode 3-second rollout, watch it. Champions should move, minions should exist, terrain should be stable.
- [ ] **Gate: if 32-frame contextual rollout stays above 8 dB PSNR, proceed to agent finetuning. If not, diagnose.**

---
