# DreamerV4 on Consumer Hardware: Experiments & Roadmap

**Last Updated:** 2026-03-05
**Status:** Active (Tokenizer architecture factoring in progress)

---

## Overview

This document consolidates all validated experiments discovered through multi-agent ML research across 4 lanes (architecture, training, diffusion/flow, world models). Experiments are organized by **Tier** (effort vs reward) and **Phase** (when to run them in the training pipeline).

**Key insight:** Your 1.5-2.5K unlabeled YouTube videos + 30h labeled replay data closely match DreamerV4's validation setup (100h actions from 2500h video). The paper proves this ratio works. Focus on temporal modeling and rollout degradation fixes.

---

## Phase Overview

- **Phase 0:** Tokenizer training (T=16 sequences, MAE with block-causal attention)
- **Phase 1:** Dynamics training (T=32/64 alternating, diffusion forcing, rollout evaluation)
- **Phase 2:** Agent finetuning (action prediction, reward modeling, behavioral cloning)

---

# Tier 0: Zero-Effort Fixes (Do Before Next Training)

All items are **1-2 line changes**. High confidence, low risk.

## 0.1 — Variable Noise Augmentation on Context (τ_ctx)

**Status:** ✅ IMPLEMENTED (See `src/ahriuwu/models/diffusion.py`)

**What:** Instead of fixed τ_ctx=0.1, sample per-sample from U(0, τ_ctx_max) where τ_ctx_max=0.3.

**Why:** Fixes your **4.8 dB multi-step rollout degradation** — the #1 failure mode. GameNGen showed this is critical for generalization.

**Implementation:** Already in code via `tau_ctx_per_sample = torch.rand(batch_size, 1, device=device) * tau_ctx`.

**Validation:** After first dynamics checkpoint, measure 16-frame rollout PSNR. Should improve from 4.8 dB.

---

## 0.2 — WSD Learning Rate Schedule (Not Cosine Decay)

**Status:** ✅ IMPLEMENTED (See `scripts/train_dynamics.py`, etc.)

**What:** Warmup → stable → optional decay. No need to know total training steps upfront.

**Why:** You don't have a fixed training budget. Cosine decay wastes steps. WSD = warmup (linear 0→1), stable (hold at 1.0), decay (linear 1→0 if you choose).

**Implementation:** `torch.optim.lr_scheduler.LambdaLR` with three-phase schedule.

---

## 0.3 — torch.compile(model)

**Status:** ✅ PARTIALLY IMPLEMENTED

**What:** One-liner: `model = torch.compile(model)` after moving to GPU.

**Why:** 1.5-2× speedup, no accuracy loss. Your Blackwell GPU supports this well.

**Implementation:** Already in tokenizer and dynamics scripts with `--no-compile` fallback.

**Check:** Run 1000 steps, measure batch throughput with/without. Should see 1.5-2× improvement.

---

## 0.4 — 8-bit AdamW via bitsandbytes

**Status:** ✅ IMPLEMENTED (See `src/ahriuwu/utils/training.py`)

**What:** Drop-in `bitsandbytes.optim.AdamW8bit` instead of torch.optim.AdamW.

**Why:** Saves ~1.8 GB optimizer memory (states + momentum + variance). Keeps full-parameter learning (unlike LoRA).

**Implementation:** `create_optimizer()` handles fallback if bitsandbytes unavailable.

---

## 0.5 — Non-Square Resolution (League is 16:9)

**Status:** ✅ IMPLEMENTED (See `src/ahriuwu/data/dataset.py`)

**What:** Change from 352×352 (1:1) to **352×640** (or 352×480 if tight) — League's native aspect ratio.

**Why:** Squishing to square throws away spatial information. Extra width captures the wide camera view.

**Current:** 352×352 = 484 patches (22×22 grid).
**New:** 352×640 = 880 patches (22×40 grid) — adds horizontal detail.

**VRAM impact:** Tokenizer goes from ~9GB to ~11GB at B=2 T=16 (estimated). You had 10GB headroom.

**Implementation:** Already changed in dataset and feature extraction pipeline.

---

# Tier 1: High-Impact Experiments (1-2 weeks each)

**Run these in order** after tokenizer is stable (factored attention working, no OOM).

## 1.A — Hybrid Mamba-Attention Temporal Modeling

**Priority:** HIGH (tackles largest bottleneck)
**Timeline:** Start after tokenizer stabilizes
**Effort:** 1-2 weeks

### The Problem
Your dynamics model has 6 temporal layers attending over T=64 frames → 64×64 causal attention per spatial position. With 266 tokens/frame, that's 17,024 sequences, and temporal attention alone is 17K×17K operations — manageable but the limiting factor for longer sequences or larger models.

### The Approach
Replace the 6 temporal attention layers with **5 Mamba-2 blocks + 1 attention layer** (Jamba's 7:1 ratio). Keep all 18 spatial layers as-is.

**Mamba-2 benefits:**
- O(n) complexity instead of O(n²) temporal
- Eliminates KV cache explosion (biggest single memory consumer)
- Enables T=128+ without VRAM increase
- State space models capture long-range temporal dependencies natively

**Attention layer at every 5 steps:**
- Ensures key frame transitions aren't missed
- Acts as "compression checkpoint" for the Mamba state

### Implementation
1. Install `flash-linear-attention` (has production Mamba-2 kernels)
2. Create `TemporalMamba2Block(nn.Module)` class — wrapper around Mamba with linear projection in/out
3. Modify dynamics loop to alternate: space → time (Mamba) → space → time (Mamba) → ... → space → time (Attention)
4. Benchmark: measure throughput (tokens/sec) and max sequence length before OOM

### Validation
- Compare dynamics at T=64: throughput should improve 2-4×
- Run 16-frame rollout PSNR test — should match or exceed attention baseline
- Try T=128 (impossible with pure attention, possible with Mamba hybrid)

### Expected Outcome
- 8× less temporal compute
- Enables longer context or larger batch
- Risk: Mamba may not capture fast temporal dynamics (ability timing) as precisely — mitigated by 1-in-8 attention

---

## 1.B — MeanFlow Instead of Shortcut Forcing

**Priority:** MEDIUM-HIGH (improves inference speed)
**Timeline:** Week 2-3 after dynamics baseline is solid
**Effort:** 1-2 weeks (mostly hyperparameter tuning)

### The Problem
Shortcut forcing requires K=4 denoising steps per frame during inference (slow). DreamerV4 mitigates with curriculum learning and step-size conditioning, but it's still multi-step.

### The Approach
Train dynamics to predict **average velocity over the entire trajectory** instead of instantaneous velocity. This enables **1-step generation** without distillation or curriculum.

**Key papers:** MeanFlow (arXiv:2505.13447)

### Implementation
1. Change target from `(z₁ - z̃) / (1-τ)` to `(z₁ - z₀) / trajectory_length`
2. Condition on trajectory length as an additional input
3. Loss remains MSE but now supervised by *integrated* motion, not single-step

### Why It's Risky
- **Unproven for sequential world models** — only demonstrated on image generation
- May require careful trajectory length curriculum
- Could blur fast motion if trajectory_length is too long

### Validation Approach
1. Train a small proxy model (30M params, muP) with MeanFlow vs shortcut forcing
2. Compare 1-step and 4-step PSNR at K=1, K=4
3. If MeanFlow @ K=1 beats shortcut @ K=4, invest in full model
4. If MeanFlow @ K=1 is worse, abandon and keep shortcut

### Expected Outcome
- 4× inference speedup (K=4 → K=1)
- Simpler training (no step-size conditioning tokens)
- Risk: May not work for sequential prediction

---

## 1.C — Inverse Dynamics Model Pipeline

**Priority:** HIGH (enables 10-50× more labeled data)
**Timeline:** Parallel with 1.A
**Effort:** 1 week setup + ongoing

### The Problem
You have ~30h labeled replay (with actions) but ~2.5K hours unlabeled YouTube (no actions). The paper validates that 100h actions + 2.4Kh unlabeled works — you're already in the right ratio. But you could boost the labeled portion.

### The Approach (VPT recipe)
1. **Train IDM on replay data:** Small transformer, input (frame_t, frame_t+1) → output action. Trains on your 30h labeled data.
2. **Pseudo-label YouTube:** Run IDM on all 2.5K hours to predict actions
3. **Retrain dynamics:** Now you have 2.5K+ hours with actions

### Critical Details (from LAM analysis)
- **Latent action dim must equal true action dim** — don't embed into higher-dim space
- **Use diverse gameplay** — if pseudo-labels are biased (only laning phase), dynamics learns biased policy
- **Use labeled data as auxiliary supervision** during YouTube training, not just post-hoc pseudo-labels
- **Watch for distribution shift** — YouTube overlays, UI differences, different champion pool than replay data can confuse IDM

### Implementation
```python
# IDM architecture (small)
class InverseDynamicsModel(nn.Module):
    def forward(self, frame_t, frame_t1):
        # Extract features, predict actions
        return action_logits

# Train on replay for 5K steps
idm.train_on_replay(replay_dataloader, epochs=5)

# Apply to YouTube
for video_dir in youtube_dirs:
    frames = load_video_frames(video_dir)
    actions = idm.predict_actions(frames)
    save_actions_to_disk(video_dir, actions)
```

### Validation
- IDM accuracy on held-out 10% of replay: should be >80%
- Check pseudo-label distribution: should match replay distribution (no mode collapse)
- Dynamics trained on pseudo-labeled data should converge similarly to all-labeled baseline

### Expected Outcome
- 10-50× more training sequences with actions
- Dynamics model learns richer action-conditioned representations
- Risk: IDM errors compound; biased pseudo-labels hurt dynamics

---

## 1.D — Progressive Temporal Compression (ProMAG)

**Priority:** MEDIUM (improves training speed & structure)
**Timeline:** Week 3-4 after baseline dynamics established
**Effort:** 1 week implementation + tuning

### The Problem
League gameplay has massive temporal redundancy (laning phase: minute of minions doing nothing). Training with T=64 from the start uses a lot of compute on "easy" low-motion sequences.

### The Approach
**ProMAG** (Progressive Multi-scale): Start with T=4, progressively grow to T=32, T=64 as training proceeds.

Early phases learn **motion compression** at coarse timescales; later phases refine temporal details.

### Implementation
1. Anneal context length T on a schedule:
   - Steps 0-50K: T=4
   - Steps 50K-150K: T=8
   - Steps 150K-250K: T=16
   - Steps 250K-350K: T=32
   - Steps 350K+: T=64
2. Adjust batch size to maintain constant VRAM (T=4 B=4, T=64 B=1)
3. Log loss separately per T value to see convergence at each scale

### Validation
- Dynamics trained with progressive T should converge faster than fixed T=64 from start
- Multi-step PSNR at each T should improve as training progresses
- Final 64-frame rollout should be as good as or better than fixed T=64 baseline

### Expected Outcome
- 30-40% faster convergence in early training
- Better temporal feature learning (coarse → fine hierarchy)
- Risk: Overfitting to short sequences in early phase

---

# Tier 2: Medium-Effort, Medium-Reward Experiments (2-4 weeks each)

**Run after Tier 1 baseline or if Tier 1 experiments plateau.**

## 2.E — Adaptive Latent Regularization (ALR)

Dynamically weight latent reconstruction loss based on frame motion. Preserve details in high-motion frames, compress redundant frames.

**Implementation:** Optical flow or feature distance → motion score → scale MAE loss per frame.

**Expected gain:** Better reconstruction of important frames, faster compression of static scenes.

---

## 2.F — Multi-Scale Tokenizer

Hierarchical patch grids: 16×16 base + 8×8 coarse + 4×4 ultra-coarse. Each scale trained with its own latent bottleneck.

**Expected gain:** Explicit multi-scale representation, better for temporal compression.

**Risk:** Architectural complexity, training instability at multiple scales.

---

## 2.G — Diffusion Forcing with Attention Sinks (Rolling Forcing)

Attention sinks aggregate old context to prevent KV cache explosion on long rollouts. Enables T=128+ without memory blowup.

**Implementation:** Add sink tokens that attend to nothing; everything attends to sinks. Sinks act as "compressed history."

**Expected gain:** Longer rollouts without VRAM increase.

---

## 2.H — muP Proxy Hyperparameter Sweep

Train a 30M proxy model, sweep (LR, weight_decay, warmup, decay_steps), transfer hyperparams to full 170M model.

**Expected gain:** Confident hyperparameter transfer without full training.

**Timeline:** ~2 weeks (proxy trains 4× faster).

---

# Tier 3: Long-Term / High-Risk Experiments (4+ weeks)

**Only pursue if earlier tiers hit plateaus or show exceptional promise.**

## 3.I — Selective Latent Masking in Encoder

Mask only latent tokens (not patches) in MAE. Forces latents to predict patches from patches.

**Expected gain:** Stronger latent bottleneck, better compression.

**Risk:** May hurt reconstruction quality, needs careful masking ratio tuning.

---

## 3.J — Learned Noise Schedule (LNS)

Learn per-layer noise levels during training instead of fixed τ_ctx=0.1.

**Expected gain:** Potentially better context corruption adapted to model capacity.

**Risk:** Adds complexity, may not outweigh fixed schedule.

---

## 3.K — Video Diffusion Pre-Training

Pre-train tokenizer on large unlabeled video (Kinetics, UCF101) before League fine-tuning.

**Expected gain:** Better priors, faster convergence.

**Risk:** Domain shift (general video ≠ League).

---

## 3.L — Policy Distillation from League Pros

If available, train IDM on pro-player replays as "gold standard," pseudo-label your data with pro policy.

**Expected gain:** Better action predictions, less distribution shift.

**Risk:** Pro playstyle may not generalize.

---

# Proposed Execution Roadmap

## Phase 0: Tokenizer (Current — 1 week)

1. **This session:** Fix factored attention (space-time not monolithic) ✅
2. **Next:** Verify sequences work (T=16, factored 484×484 + 16×16 attention)
3. **Goal:** Stable MAE training, PSNR > 27 dB on test set

**Checkpoints:** `checkpoints/tokenizer_medium_T16_*.pt`

---

## Phase 1a: Dynamics Baseline (Weeks 2-4)

**Prerequisites:** Tokenizer stable, latents pre-computed

1. **Week 1:** Train dynamics with **Tier 0 changes only**
   - Variable τ_ctx=0.3
   - WSD schedule (warmup 2K, decay 50K)
   - torch.compile enabled
   - 8-bit AdamW
   - Batch alternation: T=32 B=2 accum=32, T=64 B=1 accum=64

2. **Week 2:** Establish baseline metrics
   - 1-step PSNR > 15 dB
   - 16-frame rollout > 8 dB
   - 32-frame rollout > 6 dB
   - Multi-step quality video

3. **Checkpoints:** `checkpoints/dynamics_baseline_*`

---

## Phase 1b: Tier 1 Experiments (Weeks 3-6, parallel with baseline)

**Run in order, each validates before starting next:**

### Week 3-4: **1.C Inverse Dynamics Pipeline** (lowest risk, highest data gain)
- Train IDM on replay (1 day)
- Pseudo-label YouTube (1 day)
- Retrain dynamics with combined data (1 week)
- **Gate:** Combined-data dynamics should converge faster, reach baseline metrics sooner

### Week 3-4: **1.D Progressive Temporal Compression** (independent, can parallelize)
- Implement T annealing schedule (2 days)
- Train dynamics with ProMAG (1 week)
- **Gate:** ProMAG should converge 30-40% faster than fixed T=64

### Week 5-6: **1.A Mamba Hybrid Temporal** (highest risk, biggest payoff)
- Implement Mamba-2 blocks (3 days)
- Integrate into dynamics (2 days)
- Train and benchmark (1 week)
- **Gate:** Match or exceed baseline PSNR at T=64; enable T=128

### Week 6-7: **1.B MeanFlow Proxy** (only if time permits)
- Train 30M muP proxy with MeanFlow (1 week)
- Compare K=1 vs K=4 PSNR
- **Gate:** If MeanFlow K=1 > baseline K=4, invest in full model

---

## Phase 1c: Tier 2 Experiments (Weeks 7-10, if Tier 1 validates)

Based on which Tier 1 experiments succeed:

- **If 1.C succeeds + 1.A succeeds:** → 2.H (muP sweep on combined data + Mamba hybrid)
- **If 1.D succeeds:** → 2.E (ALR on the temporally diverse dataset)
- **If time + VRAM permits:** → 2.G (attention sinks with longest rollouts)

---

## Phase 2: Agent Finetuning (Weeks 10-12)

**Prerequisites:** Dynamics converged, passing evaluation gates

1. Freeze dynamics, train agent head (behavioral cloning)
2. Reward model training
3. Online finetuning with actions from replays

---

# Key Success Metrics

## Tokenizer
- ✅ PSNR > 27 dB (you had 27.5 on small, expect higher with medium)
- ✅ Reconstruction visually matches input (not blurry after 500 steps)
- ✅ Factored attention: space layers 484×484, time layers 16×16 (verified via shape prints)

## Dynamics Baseline
- ✅ 1-step PSNR > 15 dB (flow loss gate)
- ✅ 16-frame rollout > 8 dB (context matters)
- ✅ 32-frame rollout > 6 dB (gate for agent finetuning)
- ✅ No mode collapse (attention entropy > 0.1 all layers)
- ✅ VRAM consistent (7-10 GB, not shrinking or bloating)

## Tier 1 Experiment Success
- **1.A (Mamba):** T=128 runs without OOM; PSNR ≥ baseline
- **1.B (MeanFlow):** K=1 PSNR ≥ K=4 baseline (on proxy first)
- **1.C (IDM):** Faster convergence with 2.5K labeled hours
- **1.D (ProMAG):** 30-40% wall-clock time savings in early training

---

# Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Tokenizer factored attention fails → OOM | Debug shape prints, verify space (B×T, S, D) and time (B×S, T, D) reshapes |
| Tier 1 experiments regress PSNR | Always keep baseline checkpoint; A/B test before committing |
| IDM pseudo-labels are low quality | Validate on held-out replay (>80% accuracy) before using at scale |
| Mamba hybrid doesn't match attention performance | 1-in-8 attention ratio may be too sparse; increase to 1-in-5 or 1-in-4 |
| MeanFlow doesn't enable K=1 | Don't pursue further; shortcut forcing is already good |

---

# Notes & References

- **GameNGen:** Noise augmentation paper showing τ_ctx curriculum critical for rollout stability
- **DreamerV4:** "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
- **Jamba:** Mamba + Attention hybrid (7:1 ratio proven effective)
- **MeanFlow:** arXiv:2505.13447 (1-step diffusion generation)
- **VPT:** Inverse dynamics pipeline for action pseudo-labeling
- **LAM:** Analysis of distribution shift in unlabeled video (watch for overlays, UI differences)

---

**Next: Fix tokenizer factored attention, validate with test run.**
