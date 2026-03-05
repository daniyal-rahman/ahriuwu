# Pre-Flight Checklist

## Tokenizer (Transformer)

### Architecture
- [ ] Block-causal transformer encoder/decoder with factored space-time attention
- [ ] Model size: **medium** (dim=768, 8+8 layers, 12 heads, ~130M params)
- [ ] Pre-layer RMSNorm (not LayerNorm)
- [ ] Soft capping: 50.0 on attention logits
- [ ] QKNorm on Q and K before attention

### Attention — Factored Space-Time
- [ ] Space-only layers: reshape (B, T, N, D) → (B×T, N, D), attend over N=740 tokens per frame
- [ ] Time layers: reshape (B, T, N, D) → (B×N, T, D), attend over T=16 frames per position
- [ ] Time layers use 1D RoPE (rotary embeddings for temporal positions)
- [ ] Space layers use 2D RoPE (rotary embeddings for spatial patch positions)
- [ ] Block-causal masking on time layers ONLY: frame t attends to frames ≤ t
- [ ] Space layers: ASYMMETRIC masks per encoder/decoder
  - Encoder: patches→patches only; latents→all
  - Decoder: patches→all; latents→latents only
- [ ] Should NEVER materialize a 7,744×7,744 attention matrix (was the OOM bug)

### Data & Training
- [ ] Dataset: **FrameSequenceDataset**, NOT SingleFrameDataset
- [ ] Sequence length: **T=16** frames (0.8s at 20fps)
- [ ] Batch size: **2** (reduced from 8 to accommodate sequences)
- [ ] Gradient accumulation: **16** (effective batch = 32)
- [ ] Model size arg: `--model-size medium` (not "small")
- [ ] MAE mask ratio: 10% p=0 (full recon), else U(0.1, 0.9)
- [ ] Mask warmup: 50K steps (curriculum learning)
- [ ] Loss: MSE + 0.2 × LPIPS (paper default)

### Verification Before Training
- [ ] Print total param count — should be ~130M
- [ ] Print input shape to encoder — should be (B, T, 484, 768) after patch embed
- [ ] Print attention matrix shapes:
  - Space layers: (B×T, heads, 740, 740)
  - Time layers: (B×740, heads, 16, 16)
- [ ] Run 10 steps, check `torch.cuda.max_memory_allocated()` — should be 8-10 GB
- [ ] Run 100 steps, verify attention entropy > 0.1 in all layers
- [ ] Run 1000 steps, verify reconstruction is blurry image (not uniform gray)

### Evaluation Gate
- [ ] PSNR on test set > 27 dB (you had 27.5 on old 40M small model)
- [ ] Reconstruction visually looks like input (not mode-collapsed)

---

## Dynamics Model

### Architecture
- [ ] Block-causal transformer with **factored space-time attention**
- [ ] Total layers: **24**
- [ ] Model dim: **768**
- [ ] Attention heads: **12 query heads** (head_dim=64)
- [ ] GQA: **4 KV heads** (3:1 ratio) — saves KV cache
- [ ] FFN: **SwiGLU**, hidden_dim = 2048
- [ ] Total params: **~170M**
- [ ] Pre-layer RMSNorm
- [ ] Soft capping: 50.0
- [ ] QKNorm on Q and K

### Attention — Factored Space-Time
- [ ] Space-only layers: 18 out of 24
- [ ] Temporal layers: every 2nd layer (better for tokenizer than dynamics' every 4th)
  - Layers 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 = 12 temporal layers
- [ ] Space layers: reshape (B, T×S, D) → (B×T, S, D), attend over S tokens per frame
  - S = 256 latents + 8 registers + 1 signal token + 1 action token = 266
- [ ] Time layers: reshape (B, T×S, D) → (B×S, T, D), attend over T frames
  - Apply 1D RoPE for temporal distance
  - Apply block-causal mask: frame t attends to frames ≤ t
- [ ] Should NEVER materialize a (T×S) × (T×S) matrix

### Input Sequence (Per Frame)
- [ ] Action tokens: **1** (multi-component: movement + abilities + mouse, summed)
- [ ] Signal token: **1** (τ level + step size d, both as discrete embeddings)
- [ ] Latent tokens: **256** from frozen tokenizer, linearly projected to 768
- [ ] Register tokens: **8** learned tokens
- [ ] **Total: 266 tokens/frame**

### Positional Encoding
- [ ] RoPE for spatial positions (space layers)
- [ ] 1D RoPE for temporal positions (time layers)
- [ ] Applied to Q and K only

### Frozen Tokenizer Integration
- [ ] Tokenizer weights are **FROZEN** (no gradients)
- [ ] Latent tokens are **pre-computed** and stored on disk
- [ ] Linear projection from tokenizer bottleneck (Db=32) to dynamics (768) IS trainable
- [ ] At inference: tokenizer runs live to encode frame → project → feed to dynamics

### Shortcut Forcing Objective
- [ ] X-prediction: predict clean latents ẑ₁ directly (not velocity)
- [ ] Input: corrupted latents z̃ = (1-τ)z₀ + τz₁
- [ ] Kmax = **64** (finest step size dmin = 1/64)
- [ ] Step size d sampled as 1/U({1, 2, 4, 8, ..., 64})
- [ ] Signal level τ sampled from grid reachable by d: τ ~ U({0, d, 2d, ..., 1-d})
- [ ] Each frame gets independent τ and d (diffusion forcing)

### Loss Computation
- [ ] Flow loss (d = dmin): L = ||ẑ₁ - z₁||²
- [ ] Bootstrap loss (d > dmin):
  - Teacher call 1: b' = (f(z̃, τ, d/2, a) - z_τ) / (1-τ)
  - Intermediate: z' = z̃ + b' × d/2
  - Teacher call 2: b'' = (f(z', τ+d/2, d/2, a) - z') / (1-(τ+d/2))
  - Target: v_target = sg(b' + b'') / 2
  - Loss: (1-τ)² × ||(ẑ₁ - z̃)/(1-τ) - v_target||²
- [ ] Teacher calls use `torch.no_grad()` and stop gradient
- [ ] Ramp loss weight: w(τ) = 0.9τ + 0.1 (upweights signal, downweights noise)
- [ ] RunningRMS loss normalization

### Context Corruption
- [ ] Variable noise augmentation: τ_ctx ~ U(0, 0.3) per sample
- [ ] Applied to past context frames during training (diffusion forcing handles it naturally)
- [ ] At inference: τ_ctx = 0.1 for robustness

### Batch Alternation
- [ ] Short batches: T=32, batch=2, accum=32 (effective=64) — 80% of training
- [ ] Long batches: T=64, batch=1, accum=64 (effective=64) — 20% of training
- [ ] Alternation keeps effective batch constant for consistent loss scale

### Training Config
- [ ] Optimizer: **AdamW**, lr=**3e-4**, betas=(0.9, 0.95), weight_decay=**0.1**
- [ ] LR schedule: **WSD** (warmup 2000 steps, stable, optional decay)
- [ ] Precision: **bf16** autocast (NOT fp16)
- [ ] Gradient checkpointing: every 2 layers
- [ ] Gradient clipping: global norm 1.0
- [ ] torch.compile: enabled (with fallback if unstable)
- [ ] 8-bit AdamW: enabled if bitsandbytes available

### Infrastructure
- [ ] SIGTERM handler: save checkpoint on signal
- [ ] Checkpoint saving: periodic (every 20K steps) + best (by validation loss)
- [ ] Cooperative queue yielding: check squeue for pending jobs after checkpoint, requeue if others waiting
- [ ] Logging: track flow loss and bootstrap loss separately

### Start Frame Generation
- [ ] **30% of training:** treat sequences as independent (no temporal attention)
- [ ] Implementation: zero out or mask temporal attention for 30% of batch

### Verification Before Training
- [ ] Print total param count — should be **~170M**
- [ ] Print input breakdown: "1 action + 1 signal + 256 latent + 8 register = 266 tokens/frame × T frames"
- [ ] Print attention shapes:
  - Space: (B×T, heads, 266, 266)
  - Time: (B×266, heads, T, T)
- [ ] Verify GQA: KV dim = 4 heads × 64 = 256 (not 12 × 64 = 768)
- [ ] Run 10 steps at T=32 B=2, check VRAM — should be **8-9 GB**
- [ ] Run 10 steps at T=64 B=1, check VRAM — should be **8-10 GB**
- [ ] Verify tokenizer is frozen: `assert all(not p.requires_grad for p in tokenizer.parameters())`
- [ ] Run 100 steps, verify bootstrap loss is finite (no explosions to 1e14)
- [ ] Decode 1-step denoised latents through frozen tokenizer — should be recognizable game frames

### Evaluation Gates (Pass Before Proceeding)
- [ ] **1-step PSNR > 15 dB** (flow loss test)
- [ ] **16-frame rollout PSNR > 8 dB** (context temporal compression works)
- [ ] **32-frame rollout PSNR > 6 dB** (gate for agent finetuning)
- [ ] **64-frame rollout:** produces recognizable LoL states, not cyan blobs
- [ ] **Qualitative:** watch 3-second rollout — champions move, minions exist, terrain stable

**Gate: If 32-frame rollout PSNR > 8 dB, proceed to agent finetuning.**

---

## Data Pipeline

### Frames
- [ ] YouTube videos extracted to `/mnt/storage/ahriuwu/frames/frames/`
- [ ] ~1531 video directories, ~10.5M total frames
- [ ] Frame format: **JPEG** at **352×640** (League 16:9 aspect)
- [ ] Frames named: `frame_000000.jpg`, `frame_000001.jpg`, ... (6-digit zero-padded)

### Tokenizer Latents (Pre-computed)
- [ ] Encode all frames through frozen tokenizer before dynamics training
- [ ] Store as `.npy` or `.pt`: (T, 256, 32) — sequence of latent tokens
- [ ] Location: `/mnt/storage/ahriuwu/latents/` or similar
- [ ] Dynamics training loads pre-computed latents (faster than on-the-fly encoding)

### Replay Data
- [ ] 30 hours of labeled gameplay (League replays with actions)
- [ ] Location: Windows `C:\Users\daniz\Documents\League of Legends\Replays\`
- [ ] Extracted via `scripts/decode_replay_movement.py`
- [ ] Includes: frame images + action sequences (movement, abilities, mouse)

---

## Infrastructure & Slurm

### Tokenizer Job
```bash
sbatch --job-name=tokenizer-med --partition=main --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G \
  --wrap="bash -c \"set +u && source /home/dani/miniconda3/etc/profile.d/conda.sh && \
  conda activate ml && set -u && cd /home/dani/Repos/ahriuwu && \
  PYTHONUNBUFFERED=1 python scripts/train_transformer_tokenizer.py \
  --frames-dir /mnt/storage/ahriuwu/frames/frames \
  --batch-size 2 --gradient-accumulation 16 --gradient-checkpointing \
  --sequence-length 16 --model-size medium --epochs 50 \
  --lr 3e-4 --warmup-steps 5000 --step-save-interval 20000 \
  --wandb --wandb-project ahriuwu\""
```

### Dynamics Job
```bash
sbatch --job-name=dynamics-baseline --partition=main --gres=gpu:1 \
  --cpus-per-task=4 --mem=40G --time=72:00:00 \
  --wrap="bash -c \"set +u && source /home/dani/miniconda3/etc/profile.d/conda.sh && \
  conda activate ml && set -u && cd /home/dani/Repos/ahriuwu && \
  PYTHONUNBUFFERED=1 python scripts/train_dynamics.py \
  --latent-dir <path-to-latents> \
  --batch-size 2 --gradient-accumulation-short 32 --gradient-accumulation-long 64 \
  --gradient-checkpointing --sequence-length-short 32 --sequence-length-long 64 \
  --long-ratio 0.2 --epochs 50 --lr 3e-4 --warmup-steps 2000 \
  --decay-steps 50000 --step-save-interval 20000 \
  --wandb --wandb-project ahriuwu\""
```

### Key Env Fixes
- [ ] Bash wrapper: `bash -c "set +u && source conda.sh && conda activate ml && set -u && ..."`
  - Prevents "source: not found" error (`--wrap` uses `/bin/sh`)
  - Prevents conda unbound variable errors
- [ ] `PYTHONUNBUFFERED=1` for real-time slurm log output
- [ ] wandb login: API key in `~/.netrc` (run `wandb login` once on desktop)

---

## Desktop Setup

- [ ] SSH to desktop: `ssh desktop` (Linux side) or `ssh windows` (Windows side)
- [ ] Dual-boot: Windows (League, replay recording), Linux (training)
- [ ] Frames mounted: `/mnt/storage/ahriuwu/` (shared NTFS)
- [ ] Replays on Windows: `C:\Users\daniz\Documents\League of Legends\Replays\`
- [ ] Copy to Linux: `scp windows:/path/to/replay /tmp/`

---

**Status:** Ready to begin tokenizer factored attention fix.
