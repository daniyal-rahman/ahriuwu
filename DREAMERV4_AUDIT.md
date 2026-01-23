# DreamerV4 Implementation Audit: ahriuwu Repository

This document provides a comprehensive comparison between the DreamerV4 paper design decisions and the current implementation in this repository, which adapts the architecture for League of Legends gameplay.

---

## Executive Summary

| Category | Paper Coverage | Notes |
|----------|---------------|-------|
| Tokenizer Architecture | **Partially Implemented** | Two tokenizers: CNN (baseline) + Block-causal Transformer (DreamerV4-style) |
| Dynamics Model | **Implemented** | Factorized attention, x-prediction, shortcut forcing |
| Diffusion/Flow Matching | **Implemented** | Linear schedule, ramp weighting, diffusion forcing |
| Agent Architecture | **Implemented** | Agent tokens, cross-attention pattern |
| Prediction Heads | **Implemented** | Reward, Policy, Value heads with MTP and twohot |
| Training Procedure | **Partial** | Phase 1 complete, Phase 2/3 scaffolded |

---

## 1. TOKENIZER ARCHITECTURE

### 1.1 Encoder Architecture

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Block-causal transformer | `TransformerEncoder` with `create_encoder_mask()` | ✅ Matches |
| 16×16 patches | `PatchEmbed` with `patch_size=16` | ✅ Matches |
| Learned latent tokens | `self.latent_tokens = nn.Parameter(...)` | ✅ Matches |
| Causal attention in time | Encoder mask: latents see current+past frames causally | ✅ Matches |
| Latents attend to all modalities | Encoder mask: `mask[latent_start:latent_end, :causal_end] = True` | ✅ Matches |
| Pre-layer RMSNorm | `RMSNorm` in `TransformerBlock` | ✅ Matches |
| RoPE | `RotaryEmbedding2D` (optional via `use_rope=True`) | ✅ Matches |
| SwiGLU | `SwiGLU` FFN | ✅ Matches |
| QKNorm | `QKNorm` class with per-head RMSNorm | ✅ Matches |
| Attention logit soft capping | `soft_cap_attention()` with cap=50.0 | ✅ Matches |
| MAE training with p ~ U(0, 0.9) | `mask_ratio` parameter, sampled in training script | ✅ Matches |

**Paper Unknowns vs Implementation:**

| Unknown from Paper | Implementation Choice |
|--------------------|----------------------|
| Exact number of encoder layers | Configurable: tiny=4, small=6, medium=8, large=12 |
| Model dimension | Configurable: tiny=256, small=512, medium=768, large=1024 |
| Number of attention heads | Configurable: tiny=4, small=8, medium=12, large=16 |
| Latent token count per frame | Fixed: 256 (all sizes) |
| Feed-forward dimension ratio | `hidden_dim = int(dim * 8/3)` rounded to multiple of 64 |
| Patch embedding method | Conv2d with kernel=patch_size, stride=patch_size |
| Positional encoding | 2D sinusoidal (default) or RoPE (optional) |

### 1.2 Decoder Architecture

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Block-causal transformer | `TransformerDecoder` with `create_decoder_mask()` | ✅ Matches |
| Fresh patch queries | `self.patch_queries = nn.Parameter(...)` | ✅ Matches |
| Each modality attends within itself + to latents | Decoder mask: patches see own-frame patches + ALL latents | ✅ Matches |
| Latents only attend within themselves | Decoder mask: latents see all latents causally | ✅ Matches |

**Decoder Configuration:**

| Size | Layers | Shares hyperparams with encoder |
|------|--------|--------------------------------|
| tiny | 4 | Yes |
| small | 6 | Yes |
| medium | 8 | Yes |
| large | 12 | Yes |

### 1.3 Tokenizer Bottleneck

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Linear projection | `self.proj = nn.Linear(embed_dim, latent_dim)` | ✅ Matches |
| Tanh activation | `torch.tanh(x)` after projection | ✅ Matches |
| Minecraft: (512×16) → (256×32) | Implementation: per-token `embed_dim → latent_dim` | ⚠️ Different structure |

**Bottleneck Dimensions:**

| Size | Embed Dim → Latent Dim | Tokens | Total bottleneck size |
|------|------------------------|--------|----------------------|
| tiny | 256 → 16 | 256 | 256 × 16 = 4,096 |
| small | 512 → 32 | 256 | 256 × 32 = 8,192 |
| medium | 768 → 48 | 256 | 256 × 48 = 12,288 |
| large | 1024 → 64 | 256 | 256 × 64 = 16,384 |

**Note:** Paper describes reshaping 512×16 → 256×32. Implementation uses per-token linear projection without reshaping. Functional difference is minimal as both produce 256 tokens of compressed representation.

### 1.4 Tokenizer Training

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Loss: L_MSE + 0.2 × L_LPIPS | `TokenizerLoss(mse_weight=1.0, lpips_weight=0.2)` | ✅ Matches |
| MAE dropout: p ~ U(0, 0.9) | `mask_ratio` sampled per batch, curriculum warmup | ✅ Matches |
| Dropped patches → learned embedding | `self.mask_embed = nn.Parameter(...)` | ✅ Matches |
| Loss normalization by running RMS | `RunningRMS` class implemented | ✅ Available |
| LPIPS backbone | VGG (via lpips library or `VGGPerceptualLoss`) | ✅ Matches (VGG) |

**Training Hyperparameters (from scripts):**

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Optimizer | AdamW (betas=(0.9, 0.95), weight_decay=0.01) |
| Gradient clipping | max_norm=1.0 |
| MAE warmup | 50,000 steps |
| Mixed precision | bfloat16 (CUDA), float16 (MPS) |

---

## 2. CNN TOKENIZER (BASELINE/ALTERNATIVE)

This implementation provides an additional CNN-based tokenizer not present in DreamerV4:

| Component | Architecture |
|-----------|-------------|
| Encoder | Conv stem + 4 ResBlocks with 2× downsampling |
| Decoder | 4 ResBlockUp with 2× upsampling + Conv head |
| Activation | GELU |
| Normalization | BatchNorm |
| Output | Sigmoid ([0, 1]) |

**CNN Model Sizes:**

| Size | Latent Dim | Base Channels | Approx Params |
|------|------------|---------------|---------------|
| tiny | 128 | 32 | ~1.5M |
| small | 256 | 64 | ~6M |
| medium | 384 | 96 | ~14M |
| large | 512 | 128 | ~25M |

**Purpose:** Simpler/faster baseline for comparison with transformer tokenizer.

---

## 3. DYNAMICS MODEL ARCHITECTURE

### 3.1 Core Transformer

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| 1.6B parameters (Minecraft) | Scalable: tiny ~10M to large ~500M | ⚠️ Smaller scale |
| Block-causal (causal in time) | `TemporalAttention` with causal mask | ✅ Matches |
| 2D transformer (time + space) | `SpatialAttention` + `TemporalAttention` | ✅ Matches |
| Pre-layer RMSNorm | `RMSNorm` | ✅ Matches |
| RoPE | **Not in dynamics** (only tokenizer) | ⚠️ Optional |
| SwiGLU | `SwiGLU` FFN | ✅ Matches |
| QKNorm | `QKNorm` class with per-head RMSNorm | ✅ Matches |
| Attention logit soft capping | `soft_cap_attention()` with cap=50.0 | ✅ Matches |
| Temporal attention every 4 layers | `temporal_every=4` (default) | ✅ Matches |
| GQA (Grouped Query Attention) | `num_kv_heads` parameter for grouped KV | ✅ Matches |

**Dynamics Model Sizes:**

| Size | Model Dim | Layers | Heads | Approx Params |
|------|-----------|--------|-------|---------------|
| tiny | 256 | 6 | 4 | ~10M |
| small | 512 | 12 | 8 | ~150M |
| medium | 768 | 18 | 12 | ~500M |
| large | 512 | 24 | 8 | ~500M |

### 3.2 Spatial vs Temporal Attention

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| 3 space layers per 1 time layer | `temporal_every=4` (configurable) | ✅ Matches |
| Space: attention within timestep | `SpatialAttention`: reshape (B,T,S,D)→(B*T,S,D) | ✅ Matches |
| Time: causal attention across frames | `TemporalAttention`: reshape (B,T,S,D)→(B*S,T,D), causal mask | ✅ Matches |

**Factorization Benefits:**
- Without factorization: O((T·S)²) = O((64·256)²) ≈ 260M ops
- With factorization: O(T²·S + T·S²) ≈ 20K ops per position
- **~13,000× speedup** vs dense attention

### 3.3 Input Token Structure

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Interleaved: actions, signal/step, representations | z tokens + action conditioning (broadcast) | ⚠️ Simplified |
| Actions: components summed | `embed_actions()`: sum of movement + ability embeddings | ✅ Matches |
| Nz=256 spatial tokens (Minecraft) | `spatial_tokens = 256` (16×16) | ✅ Matches |
| 30% training treats videos as separate images | `independent_frames=True` mode (30% default) | ✅ Matches |
| Register tokens | `num_register_tokens=8` (default) | ✅ Matches |

**Action Space Implementation:**

| Component | Classes | Embedding |
|-----------|---------|-----------|
| Movement | 18 (20° apart) | `nn.Embedding(18, model_dim)` |
| Q, W, E, R, D, F, item, B | 2 each (binary) | `nn.Embedding(2, model_dim)` |

All embeddings are summed and broadcast to spatial tokens.

### 3.4 Sequence Parameters

| Paper (Minecraft) | Implementation | Status |
|-------------------|----------------|--------|
| Context length C = 192 | `max_seq_len = 256` (configurable) | ✅ Comparable |
| Batch length T2 = 256 (long) | Configurable via training script | ✅ Available |
| Batch length T1 = 64 (short) | Configurable via training script | ✅ Available |
| Nz = 256 spatial tokens | 256 (16×16 grid) | ✅ Matches |

---

## 4. SHORTCUT FORCING OBJECTIVE

### 4.1 Core Formulation

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| X-prediction | `x_prediction_loss()` predicts clean z_0 | ✅ Matches |
| K=4 inference steps | Configurable in `ShortcutForcing` | ✅ Available |
| Context corruption τ_ctx = 0.1 | `tau_ctx=0.1` in diffusion forcing | ✅ Matches |
| Ramp weight: w(τ) = 0.9τ + 0.1 | `ramp_weight(tau) = 1.0 - 0.9 * tau` (inverted convention) | ✅ Matches |
| Kmax defines step grid | `k_max=64` default | ✅ Implemented |

### 4.2 Flow Matching Term (d = dmin)

| Paper | Implementation | Status |
|-------|----------------|--------|
| Standard MSE: \|\|ẑ₁ - z₁\|\|² | `F.mse_loss(pred, target)` | ✅ Matches |

### 4.3 Bootstrap Term (d > dmin)

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Two-step distillation | `ShortcutForcing.compute_loss()` | ✅ Implemented |
| Stopped gradients | `avg_velocity.detach()` | ✅ Matches |
| Velocity space loss | `b_student - avg_velocity` | ✅ Matches |
| τ² scaling | `tau_weight = tau_idx ** 2` | ✅ Matches |

### 4.4 Diffusion Forcing

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Different signal levels per timestep | `sample_diffusion_forcing_timesteps()` | ✅ Implemented |
| Horizon-based noise schedule | Random horizon h, linear ramp after | ✅ Matches |

---

## 5. AGENT ARCHITECTURE

### 5.1 Task Conditioning

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Agent tokens as additional modality | `self.agent_token = nn.Parameter(...)` | ✅ Matches |
| Agent tokens attend to all modalities | `AgentCrossAttention` queries z tokens | ✅ Matches |
| No other modalities attend to agents | Z tokens processed separately in main blocks | ✅ Matches |
| Task embeddings | `self.task_embed = nn.Embedding(num_tasks, model_dim)` | ✅ Matches |
| One-hot task indicators | Integer task_id → embedding | ✅ Matches |

### 5.2 Policy Head

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| MLP head on agent tokens | `PolicyHead.mlp` (2-layer SiLU) | ✅ Matches |
| MTP with L=8 | `mtp_length=8` default | ✅ Matches |
| One output layer per MTP distance | `self.heads = nn.ModuleList([...])` | ✅ Matches |
| Categorical for mouse (121 classes) | **Different:** Movement = 18 classes (LoL adaptation) | ⚠️ Game-specific |
| 23 binary for keyboard | **Different:** 8 binary abilities (Q,W,E,R,D,F,item,B) | ⚠️ Game-specific |

### 5.3 Reward Head

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| MLP head on agent tokens | `RewardHead.mlp` (2-layer SiLU) | ✅ Matches |
| MTP with L=8 | `mtp_length=8` default | ✅ Matches |
| Symexp twohot output | `twohot_encode/decode`, `symlog/symexp` | ✅ Matches |
| Bucket range | -20 to +20 (255 buckets) | ✅ Matches |
| Zero-init output weights | `nn.init.zeros_(head.weight)` | ✅ Matches |

### 5.4 Value Head

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Added in Phase 3 | Separate `ValueHead` class | ✅ Matches |
| Symexp twohot output | Same as RewardHead | ✅ Matches |
| TD-learning with λ-returns | `compute_lambda_returns()` | ✅ Implemented |
| γ = 0.997 | `gamma=0.997` default | ✅ Matches |
| Zero-init output weights | `nn.init.zeros_(self.mlp[-1].weight)` | ✅ Matches |

**Paper Unknowns vs Implementation:**

| Unknown | Implementation |
|---------|----------------|
| MLP hidden dimensions | 256 (configurable) |
| Number of MLP layers | 2 hidden layers |
| λ value for λ-returns | 0.95 |
| Symexp scale | Buckets from -20 to +20 in symlog space |
| Twohot bin count | 255 |

---

## 6. TRAINING PROCEDURE

### 6.1 Phase 1: World Model Pretraining

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Tokenizer trained first | `train_tokenizer.py`, `train_transformer_tokenizer.py` | ✅ Implemented |
| Dynamics trained on tokenized data | `train_dynamics.py` with pre-tokenized latents | ✅ Implemented |
| Loss normalization by running RMS | `RunningRMS` class available | ✅ Available |

**Training Configuration:**

| Parameter | Tokenizer | Dynamics |
|-----------|-----------|----------|
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Betas | (0.9, 0.95) | (0.9, 0.95) |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | bfloat16/float16 | bfloat16/float16 |

### 6.2 Phase 2: Agent Finetuning

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Behavior cloning loss | `PolicyHead.log_prob()` for BC | ✅ Scaffolded |
| Reward modeling loss | `twohot_loss()` | ✅ Scaffolded |
| Dynamics loss continues | Can be enabled in training | ✅ Available |
| 50% uniform + 50% relevant mixture | **Not implemented** | ❌ Missing |

### 6.3 Phase 3: Imagination Training

| Paper Specification | This Implementation | Status |
|---------------------|---------------------|--------|
| Policy + value updated; transformer frozen | Separate ValueHead, freeze logic TBD | ⚠️ Partial |
| PMPO objective | **Not implemented** | ❌ Missing |
| Frozen behavioral prior | **Not implemented** | ❌ Missing |
| α = 0.5, β = 0.3 | **Not implemented** | ❌ Missing |
| Rollouts from dataset contexts | **Not implemented** | ❌ Missing |

---

## 7. ACTION SPACE DETAILS

### Minecraft (Paper) vs League of Legends (Implementation)

| Aspect | Minecraft | This Implementation |
|--------|-----------|---------------------|
| Movement | Mouse (121 classes, foveated) | 18 discrete directions (20° apart) |
| Actions | 23 binary keyboard keys | 8 binary: Q, W, E, R, D, F, item, B |
| Frame rate | 20 FPS | Variable (from video source) |
| Representation | Categorical + independent Bernoulli | Same pattern |

---

## 8. DATA PREPROCESSING

| Paper (Minecraft) | Implementation (LoL) | Status |
|-------------------|---------------------|--------|
| 360×640 → zero pad to 384×640 | 256×256 (square) | ⚠️ Different |
| Patch size 16×16 → 960 patches | Patch size 16×16 → 256 patches | ⚠️ Fewer patches |
| Image normalization | [0, 1] range, RGB | ✅ Standard |
| 20 FPS | Variable | ⚠️ Different |

---

## 9. CRITICAL GAPS AND MISSING FEATURES

### High Priority (Affects Model Quality) - ✅ ALL IMPLEMENTED

| Feature | Status | Implementation |
|---------|--------|----------------|
| QKNorm | ✅ Implemented | `QKNorm` class in dynamics.py and transformer_tokenizer.py |
| Attention soft capping | ✅ Implemented | `soft_cap_attention()` with default cap=50.0 (Gemma 2) |
| GQA (Grouped Query Attention) | ✅ Implemented | `num_kv_heads` parameter on all attention classes |
| Register tokens | ✅ Implemented | `num_register_tokens` parameter (default=8) |
| 30% independent frame training | ✅ Implemented | `independent_frames` flag + `--independent-frame-ratio` arg |

### Medium Priority (Phase 2/3)

| Missing Feature | Impact | Difficulty |
|-----------------|--------|------------|
| PMPO objective | Required for RL fine-tuning | High |
| Frozen behavioral prior | Prevents policy collapse | Medium |
| Data mixture (50/50 uniform/relevant) | Better BC training | Low |
| Imagination rollout infrastructure | Required for Phase 3 | High |

### Low Priority (Evaluation/Tooling)

| Missing Feature | Impact | Difficulty |
|-----------------|--------|------------|
| FVD metric | Video generation quality | Medium |
| Multi-step generation visualization | Debugging | Low |
| Automatic hyperparameter tuning | Convenience | Medium |

---

## 10. PARAMETER COUNT COMPARISON

### Paper (Minecraft, 1.6B total)

| Component | Estimated Params |
|-----------|-----------------|
| Tokenizer | ~100-200M (estimated) |
| Dynamics | ~1.4B |
| Heads | ~10-50M |

### This Implementation (Small config)

| Component | Params |
|-----------|--------|
| Transformer Tokenizer | ~40M |
| CNN Tokenizer | ~6M |
| Dynamics Model | ~150M |
| Agent blocks (optional) | ~20M |
| Heads | ~1.5M |
| **Total (Transformer path)** | **~210M** |
| **Total (CNN path)** | **~175M** |

---

## 11. SUMMARY: DESIGN DECISIONS

### Matches Paper

1. ✅ Block-causal transformer tokenizer with proper attention masks
2. ✅ MAE training with p ~ U(0, 0.9)
3. ✅ Bottleneck with linear → tanh
4. ✅ Loss: MSE + 0.2 × LPIPS (VGG)
5. ✅ Factorized spatial/temporal attention
6. ✅ Temporal attention every 4 layers
7. ✅ X-prediction diffusion objective
8. ✅ Ramp loss weighting
9. ✅ Diffusion forcing with per-timestep noise
10. ✅ Shortcut forcing with bootstrap loss
11. ✅ Agent tokens with asymmetric attention
12. ✅ MTP (Multi-Token Prediction) with L=8
13. ✅ Symexp twohot representation
14. ✅ λ-returns with γ=0.997, λ=0.95
15. ✅ RMSNorm, SwiGLU throughout
16. ✅ RoPE (optional in tokenizer)
17. ✅ QKNorm for attention stability
18. ✅ Attention logit soft capping (cap=50.0)
19. ✅ GQA (Grouped Query Attention) with configurable num_kv_heads
20. ✅ Register tokens (default=8) for improved information flow
21. ✅ 30% independent frame training mode

### Diverges from Paper

1. ⚠️ Smaller model scale (~200M vs 1.6B)
2. ⚠️ Different input resolution (256×256 vs 360×640)
3. ⚠️ Game-specific action space (LoL vs Minecraft)
4. ⚠️ Phase 2/3 not fully implemented (PMPO, behavioral prior)

### Implementation-Specific Additions

1. ➕ CNN tokenizer baseline
2. ➕ Factorized action embeddings (movement + abilities)
3. ➕ "No action" embedding for unlabeled data
4. ➕ Curriculum learning for MAE mask ratio
5. ➕ Step-based checkpointing

---

## 12. RECOMMENDED NEXT STEPS

### ✅ Completed (Quality Improvements)

All DreamerV4 stability features have been implemented:

1. ~~Add QKNorm to attention layers~~ → ✅ `QKNorm` class
2. ~~Add attention logit soft capping~~ → ✅ `soft_cap_attention()` with cap=50.0
3. ~~Implement 30% independent frame training~~ → ✅ `--independent-frame-ratio`
4. ~~Add register tokens~~ → ✅ `--num-register-tokens`
5. ~~Implement GQA~~ → ✅ `--num-kv-heads`

### Short-term (Scale & Efficiency)

1. Test larger model configurations (medium, large)
2. Tune hyperparameters for LoL-specific data
3. Benchmark training throughput with different batch sizes

### Medium-term (Phase 2/3)

1. Implement data mixture sampling (50/50 uniform/relevant)
2. Build imagination rollout infrastructure
3. Implement PMPO objective
4. Add frozen behavioral prior

### Long-term (Evaluation)

1. Add FVD metric for video quality
2. Build multi-step generation visualization
3. Add comprehensive evaluation suite

---

## 13. RUNNING TRAINING (PAPER-ALIGNED)

This section provides the exact commands to train models following DreamerV4 design decisions.

### 13.1 Phase 1a: Tokenizer Training

**Option A: Transformer Tokenizer (Recommended - matches DreamerV4)**

```bash
# Small config (~40M params) - matches paper architecture
python scripts/train_transformer_tokenizer.py \
    --data-dir data/processed/frames \
    --model-size small \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --lpips-weight 0.2 \
    --mae-warmup-steps 50000 \
    --gradient-checkpointing

# Medium config (~80M params) - for more capacity
python scripts/train_transformer_tokenizer.py \
    --data-dir data/processed/frames \
    --model-size medium \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --lpips-weight 0.2 \
    --mae-warmup-steps 50000 \
    --gradient-checkpointing
```

**Option B: CNN Tokenizer (Simpler baseline)**

```bash
# Small config (~6M params)
python scripts/train_tokenizer.py \
    --data-dir data/processed/frames \
    --model-size small \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --lpips-weight 0.2
```

### 13.2 Phase 1b: Pre-tokenize Frames

After tokenizer training, convert frames to latent sequences:

```bash
# For transformer tokenizer
python scripts/pretokenize_frames.py \
    --frames-dir data/processed/frames \
    --output-dir data/processed/latents_transformer \
    --checkpoint checkpoints/transformer_tokenizer_best.pt \
    --tokenizer-type transformer \
    --batch-size 16

# For CNN tokenizer
python scripts/pretokenize_frames.py \
    --frames-dir data/processed/frames \
    --output-dir data/processed/latents_cnn \
    --checkpoint checkpoints/tokenizer_best.pt \
    --tokenizer-type cnn \
    --batch-size 32

# Pack latents for faster I/O (recommended)
python scripts/pack_latents.py \
    --input-dir data/processed/latents_transformer \
    --output-dir data/processed/latents_transformer_packed
```

### 13.3 Phase 1c: Dynamics Model Training (DreamerV4-Aligned)

**Full DreamerV4 configuration with all stability features:**

```bash
# Small dynamics model (~62M params) with ALL DreamerV4 features
python scripts/train_dynamics.py \
    --latents-dir data/processed/latents_transformer_packed \
    --packed \
    --tokenizer-type transformer \
    --latent-dim 32 \
    --model-size small \
    --epochs 100 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --gradient-checkpointing \
    --shortcut-forcing \
    --shortcut-k-max 64 \
    --alternating-lengths \
    --seq-len-short 32 \
    --seq-len-long 64 \
    --batch-size-short 4 \
    --batch-size-long 2 \
    --long-ratio 0.1 \
    --soft-cap 50.0 \
    --num-register-tokens 8 \
    --independent-frame-ratio 0.3 \
    --save-steps 5000 \
    --log-interval 50
```

**With GQA for memory efficiency (recommended for larger models):**

```bash
# Medium dynamics model with GQA (8 Q heads, 4 KV heads)
python scripts/train_dynamics.py \
    --latents-dir data/processed/latents_transformer_packed \
    --packed \
    --tokenizer-type transformer \
    --latent-dim 32 \
    --model-size medium \
    --epochs 100 \
    --lr 1e-4 \
    --gradient-checkpointing \
    --shortcut-forcing \
    --alternating-lengths \
    --seq-len-short 32 \
    --seq-len-long 64 \
    --batch-size-short 2 \
    --batch-size-long 1 \
    --long-ratio 0.1 \
    --soft-cap 50.0 \
    --num-register-tokens 8 \
    --num-kv-heads 4 \
    --independent-frame-ratio 0.3 \
    --save-steps 5000
```

**With action conditioning (for behavioral cloning setup):**

```bash
python scripts/train_dynamics.py \
    --latents-dir data/processed/latents_transformer_packed \
    --packed \
    --tokenizer-type transformer \
    --latent-dim 32 \
    --model-size small \
    --epochs 100 \
    --use-actions \
    --features-dir data/processed \
    --shortcut-forcing \
    --alternating-lengths \
    --seq-len-short 32 \
    --seq-len-long 64 \
    --batch-size-short 4 \
    --batch-size-long 2 \
    --soft-cap 50.0 \
    --num-register-tokens 8 \
    --independent-frame-ratio 0.3 \
    --gradient-checkpointing
```

**With agent tokens (for Phase 2 preparation):**

```bash
python scripts/train_dynamics.py \
    --latents-dir data/processed/latents_transformer_packed \
    --packed \
    --tokenizer-type transformer \
    --latent-dim 32 \
    --model-size small \
    --epochs 100 \
    --use-actions \
    --use-agent-tokens \
    --features-dir data/processed \
    --shortcut-forcing \
    --alternating-lengths \
    --soft-cap 50.0 \
    --num-register-tokens 8 \
    --independent-frame-ratio 0.3 \
    --gradient-checkpointing
```

### 13.4 CNN Tokenizer Path (Simpler Alternative)

```bash
# Dynamics with CNN tokenizer latents
python scripts/train_dynamics.py \
    --latents-dir data/processed/latents_cnn_packed \
    --packed \
    --tokenizer-type cnn \
    --latent-dim 256 \
    --model-size small \
    --epochs 100 \
    --shortcut-forcing \
    --alternating-lengths \
    --seq-len-short 32 \
    --seq-len-long 64 \
    --batch-size-short 4 \
    --batch-size-long 2 \
    --soft-cap 50.0 \
    --num-register-tokens 8 \
    --independent-frame-ratio 0.3 \
    --gradient-checkpointing
```

### 13.5 Key Arguments Explained

| Argument | DreamerV4 Alignment | Default | Notes |
|----------|---------------------|---------|-------|
| `--soft-cap 50.0` | ✅ Required | 50.0 | Gemma 2 style attention capping |
| `--num-register-tokens 8` | ✅ Required | 8 | Information routing tokens |
| `--independent-frame-ratio 0.3` | ✅ Required | 0.3 | Prevents temporal shortcuts |
| `--shortcut-forcing` | ✅ Required | off | Enables few-step inference |
| `--alternating-lengths` | ✅ Recommended | off | DreamerV4 Section 3.4 |
| `--long-ratio 0.1` | ✅ Recommended | 0.1 | 10% long, 90% short batches |
| `--num-kv-heads N` | Optional | None | GQA for memory efficiency |
| `--no-qk-norm` | ❌ Avoid | enabled | QKNorm is important for stability |
| `--gradient-checkpointing` | Recommended | on | Reduces memory usage |

### 13.6 Resuming Training

```bash
# Resume from latest checkpoint
python scripts/train_dynamics.py \
    --resume checkpoints/run_YYYYMMDD_HHMMSS_dynamics_transformer32_small/dynamics_latest.pt \
    --epochs 200

# Resume from specific epoch
python scripts/train_dynamics.py \
    --resume checkpoints/run_YYYYMMDD_HHMMSS_dynamics_transformer32_small/dynamics_epoch_050.pt \
    --epochs 200
```

### 13.7 Monitoring Training

Key metrics to watch:
- **Loss**: Should steadily decrease; spikes may indicate learning rate issues
- **Grad Norm**: Should stay < 1.0 most of the time (clipped at 1.0)
- **Pred Std**: Should be > 0.01; very low values indicate mode collapse

Warning signs:
- `HIGH_LOSS`: Loss > 1.0 - may need lower learning rate
- `GRAD_CLIP`: Gradients frequently hitting clip threshold
- `NAN/INF`: Training instability - try lower learning rate or check data
- Very low `Pred Std` (< 0.01): Mode collapse - check learning rate, data diversity
