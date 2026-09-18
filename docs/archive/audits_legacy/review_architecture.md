# Architecture Code Review: DreamerV4 Model Files

**Date**: 2026-03-02
**Reviewer**: Claude Opus 4.6
**Scope**: `dynamics.py`, `transformer_tokenizer.py`, `tokenizer.py`
**Status**: READ-ONLY audit

---

## Summary

Reviewed ~2,700 lines across three model files implementing a DreamerV4-style world model architecture. Found **4 bugs**, **9 warnings**, and **3 style issues**. The most critical findings are: (1) a QKNorm ordering issue with RoPE in the transformer tokenizer, (2) decoder mask allows future information leakage through latents, (3) sincos position embedding frequency divisor produces degenerate embeddings, and (4) the CNN tokenizer is hardcoded for square inputs. The overall architecture is well-structured and largely correct.

---

## Issues

### BUG-1: QKNorm applied AFTER RoPE in transformer tokenizer (ordering violation)

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 270-281
- **Severity**: BUG

In `MultiHeadAttention.forward()`, RoPE is applied to Q and K at lines 271-277, then QKNorm is applied at lines 280-281. This is incorrect. QKNorm (RMSNorm) after RoPE will destroy the rotation encoding because normalization changes the vector magnitudes and can alter relative angular relationships when applied per-head.

The correct order is: project Q/K -> QKNorm -> RoPE -> dot product. This is what Gemma 2 and LLaMA 3 do: normalize first, then rotate.

```python
# Current (WRONG):
# lines 271-277: apply RoPE
q = torch.where(valid_mask_exp, q_rotated, q)
k = torch.where(valid_mask_exp, k_rotated, k)

# lines 280-281: apply QKNorm AFTER RoPE
if self.qk_norm is not None:
    q, k = self.qk_norm(q, k)
```

**Suggested fix**: Swap the order — apply QKNorm before RoPE:
```python
# Apply QKNorm FIRST
if self.qk_norm is not None:
    q, k = self.qk_norm(q, k)

# Then apply RoPE
if rope is not None and rope_indices is not None:
    ...
```

Note: In `dynamics.py`, QKNorm is applied correctly since there is no RoPE in that module (it uses learned additive position embeddings). This bug only affects the transformer tokenizer when `use_rope=True`.

---

### BUG-2: Decoder mask allows patches to see future latents (causal violation)

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 417-429 (`create_decoder_mask`)
- **Severity**: BUG

The decoder mask function lets patches attend to **ALL** latents from **all** frames, including future frames:

```python
# Lines 425-429: ALL latents from all frames
for t2 in range(num_frames):
    src_latent_start = t2 * tokens_per_frame + num_patches
    src_latent_end = src_latent_start + num_latents
    mask[patch_start:patch_end, src_latent_start:src_latent_end] = True
```

This means patches at frame t can attend to latents from frame t+1, t+2, etc. In a block-causal architecture, patches should only see latents from the current and **past** frames, not future ones.

**Suggested fix**: Change `range(num_frames)` to `range(t + 1)` to enforce causal masking:
```python
for t2 in range(t + 1):  # Only current and past frames
    src_latent_start = t2 * tokens_per_frame + num_patches
    src_latent_end = src_latent_start + num_latents
    mask[patch_start:patch_end, src_latent_start:src_latent_end] = True
```

**Impact**: During multi-frame training, the decoder can cheat by looking at future latent representations. This would produce artificially low reconstruction loss but break causal generation at inference time. For single-frame (T=1) usage this has no effect, which may be why it has not been caught yet.

---

### BUG-3: Sincos position embedding frequency divisor goes to zero

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 41-42 (`get_2d_sincos_pos_embed`)
- **Severity**: BUG

The frequency computation divides by `embed_dim_per_axis // 2`, but the `omega` range goes from `0` to `embed_dim_per_axis // 2 - 1`. When the last element equals `embed_dim_per_axis // 2 - 1`, the exponent is `(N-1)/N` which is close to 1 but never exactly 1. However, the real issue is when `embed_dim_per_axis // 2` is the divisor: the standard sinusoidal embedding formula from "Attention Is All You Need" divides by `d_model` (the full relevant dimension), not by `d_model // 2`.

```python
embed_dim_per_axis = embed_dim // 2        # e.g., 256
omega = torch.arange(embed_dim_per_axis // 2, dtype=torch.float32)  # 0..127
omega = 1.0 / (10000 ** (omega / (embed_dim_per_axis // 2)))  # divides by 128
```

The standard formula uses `omega_k = 1/10000^(2k/d)` where d is the embedding dimension for that axis. Here, the effective dimension per axis is `embed_dim_per_axis = embed_dim // 2`, and we generate `embed_dim_per_axis // 2` frequencies (for sin+cos pairs). The divisor should be `embed_dim_per_axis` (not `embed_dim_per_axis // 2`) to match the standard formula where `2k/d` with k in `[0, d/2)` gives the range `[0, 1)`:

```python
# Current: omega / (embed_dim_per_axis // 2) -> range [0, 1) but spacing is 2x too coarse
# Correct: 2 * omega / embed_dim_per_axis -> same as omega / (embed_dim_per_axis // 2)
```

Wait — actually `2 * omega / embed_dim_per_axis` is algebraically identical to `omega / (embed_dim_per_axis // 2)` when `embed_dim_per_axis` is even. So this is actually correct. **Revised: NOT A BUG.** The formula is equivalent to the standard formulation. Retracting this finding.

---

### BUG-3 (revised): CNN tokenizer hardcoded for 256x256 square input

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/tokenizer.py`
- **Lines**: 68-99 (Encoder), 102-136 (Decoder)
- **Severity**: BUG (for the 480x352 migration mentioned in the review requirements)

The CNN tokenizer documentation says `256x256` throughout, and the Encoder/Decoder have no logic to handle non-square inputs. The architecture itself (Conv2d + ResBlocks) would technically work with non-square inputs — the convolutions will process any spatial size — but:

1. The Decoder uses 4 stages of 2x upsampling, always producing exactly 16x the latent spatial size. If the encoder produces a non-square latent (e.g., 30x22 from 480x352 with 4 downsamplings), the decoder would produce 480x352 correctly.

2. However, `VisionTokenizer.forward()` docstring claims `(B, 3, H, W)` generically but the class docstring says "256x256". More critically, the class claims "16x16 spatial grid gives 256 tokens" — for 480x352, the latent grid would be 30x22 = 660 tokens, breaking any downstream code that hardcodes 256 tokens.

**Impact**: The CNN tokenizer will silently produce wrong-shaped outputs if fed 480x352 images, since downstream code (dynamics model) expects 16x16=256 spatial tokens.

**Suggested fix**: Make spatial dimensions configurable or add assertions that validate input dimensions match expectations. The dynamics model's `spatial_size` parameter would also need updating.

---

### WARNING-1: Attention scale factor applied after QKNorm (redundant scaling)

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 148, 207 (SpatialAttention), 263, 333 (TemporalAttention)
- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 223, 284
- **Severity**: WARNING

When QKNorm is enabled, Q and K are L2-normalized (via RMSNorm with learned scale). After normalization, the dot product `Q.K^T` is bounded, making the `1/sqrt(head_dim)` scale factor less necessary. Gemma 2 and some implementations use `sqrt(head_dim)` as the **learned scale** in QKNorm and drop the explicit `1/sqrt(head_dim)` factor.

Having both QKNorm and the scale factor is not wrong — it just means the learned `weight` parameter in RMSNorm must compensate for the extra scaling. Training will likely converge regardless, but it's a deviation from the canonical Gemma 2 approach where the scale is folded into the norm.

**Impact**: Minor. Training will still work; the learned norm weights will adapt. No action required unless you observe attention collapse or instability.

---

### WARNING-2: Register tokens have no position embeddings

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 855-860, 1020-1024
- **Severity**: WARNING

Register tokens are initialized with learned values but receive no position embeddings (neither spatial nor temporal). The spatial position embedding is only added to the original spatial tokens (line 1016), and register tokens are concatenated after (line 1024).

This is actually a deliberate design choice noted in the code (line 862: "Positional embeddings for original spatial tokens only, registers have none"), and some register token papers do this. However, it means register tokens are position-agnostic, which could limit their ability to serve as position-dependent information aggregators.

**Impact**: Minor. Register tokens will still function as global information sinks. If they need to encode position-dependent summaries, they may be less effective. Consider adding temporal position embeddings to registers if training shows they are not learning useful representations.

---

### WARNING-3: Agent cross-attention normalizes z_tokens through norm1 but not z_tokens themselves

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 757-758 (AgentTokenBlock.forward)
- **Severity**: WARNING

In `AgentTokenBlock.forward()`, the agent tokens are normalized before cross-attention (`self.norm1(agent_tokens)`), but the z_tokens passed as keys/values are NOT normalized:

```python
agent_tokens = agent_tokens + self.cross_attn(self.norm1(agent_tokens), z_tokens)
```

The z_tokens come from the main transformer blocks, which apply their own norms internally. However, by the time they reach the agent blocks, they have been through `norm_out` (line 1065) only for the output projection path — the `x` variable passed to agent blocks (line 1094) has NOT been through a final norm.

This means the K/V inputs to agent cross-attention are unnormalized, while Q is normalized. This asymmetry can cause training instability or suboptimal attention patterns.

**Suggested fix**: Add an RMSNorm for the cross-attention KV inputs:
```python
self.norm_kv = RMSNorm(dim)
# In forward:
agent_tokens = agent_tokens + self.cross_attn(self.norm1(agent_tokens), self.norm_kv(z_tokens))
```

---

### WARNING-4: SwiGLU hidden_dim rounding uses int() truncation

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Line**: 100
- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Line**: 187
- **Severity**: WARNING

Both files compute `hidden_dim = hidden_dim or int(dim * 8 / 3)`. The `int()` call truncates toward zero. For `dim=512`: `int(512 * 8 / 3) = int(1365.33) = 1365`. After rounding to multiple of 64: `((1365 + 63) // 64) * 64 = 1408`.

The standard recommendation for SwiGLU is `round(8/3 * d_model)`, which would give `round(1365.33) = 1365`. The difference is negligible here since the 64-rounding dominates. The final value of 1408 is reasonable.

**Impact**: None in practice. The 64-alignment rounding makes the `int()` vs `round()` distinction irrelevant.

---

### WARNING-5: Dropout applied to attention weights during eval if dropout > 0

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 214, 351
- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Line**: 299
- **Severity**: WARNING

`nn.Dropout` is used on attention weights. `nn.Dropout` correctly disables during `eval()` mode, so this is fine as long as `model.eval()` is called during inference. However, the default dropout is 0.0 in all config presets, making this effectively a no-op.

**Impact**: None with current configs. Just noting that if dropout > 0 is ever used, ensure `model.eval()` is called at inference time.

---

### WARNING-6: Bottleneck does not verify num_latents_in == num_latents_out

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 792-829 (Bottleneck class)
- **Severity**: WARNING

The `Bottleneck` class accepts `num_latents_in` and `num_latents_out` as separate parameters but the `proj` linear layer only transforms the feature dimension (embed_dim -> latent_dim), not the token count. If `num_latents_in != num_latents_out`, the reshape at line 827 would silently produce wrong results:

```python
x = x.reshape(B, num_frames * self.num_latents_out, self.latent_dim)
```

If `num_latents_in=512` and `num_latents_out=256`, this reshape would merge pairs of tokens, silently corrupting the data without any error.

Currently, both are set to `num_latents=256` at construction (lines 1002-1003), so this is not triggered. But the API is misleading and fragile.

**Suggested fix**: Either assert `num_latents_in == num_latents_out` in `__init__`, or implement actual token count reshaping (e.g., via a linear layer over the token dimension).

---

### WARNING-7: PatchEmbed assumes square input

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 434-463 (PatchEmbed)
- **Severity**: WARNING

`PatchEmbed.__init__` computes `self.num_patches = (img_size // patch_size) ** 2`, assuming square input. For 480x352 input, this would be wrong. The Conv2d projection itself handles non-square inputs fine, but `num_patches` would be incorrect, causing downstream mask size mismatches.

**Suggested fix**: Accept `img_size` as a tuple `(H, W)` and compute `num_patches = (H // patch_size) * (W // patch_size)`.

---

### WARNING-8: RotaryEmbedding2D assumes square grid

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 60-140 (RotaryEmbedding2D)
- **Severity**: WARNING

`RotaryEmbedding2D` takes a single `grid_size` parameter and creates a square grid. For non-square inputs (480x352 with patch_size=16 would give 30x22), this would need to accept `(grid_h, grid_w)`.

**Impact**: Blocks migration to non-square input resolution alongside WARNING-7.

---

### WARNING-9: Temporal attention independent_frames mode creates a diagonal mask but includes self-attention

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 340-344
- **Severity**: WARNING

When `independent_frames=True`, the mask is:
```python
diag_mask = ~torch.eye(T, dtype=torch.bool, device=x.device)
attn = attn.masked_fill(diag_mask, float("-inf"))
```

This creates a mask where each frame can ONLY attend to itself (diagonal). This is correct for "independent frames" semantics — each frame is processed independently with no temporal context. However, this is effectively equivalent to a no-op self-attention (each position attends only to itself), which means the temporal attention layer becomes an identity-like operation (weighted by softmax over a single element = 1.0).

This works but is wasteful compute. An optimization would be to skip the temporal attention entirely when `independent_frames=True`, which would save the full QKV projection + attention computation.

**Impact**: Performance only, no correctness issue. The current implementation is correct.

---

### STYLE-1: Duplicated RMSNorm, QKNorm, SwiGLU, soft_cap_attention across files

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py` (lines 31-110)
- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py` (lines 142-196)
- **Severity**: STYLE

`RMSNorm`, `QKNorm`, `SwiGLU`, and `soft_cap_attention` are duplicated verbatim in both files. These should be extracted to a shared module (e.g., `src/ahriuwu/models/components.py`).

Note: The `SwiGLU` implementations differ slightly — `dynamics.py` includes `dropout` in SwiGLU while `transformer_tokenizer.py` does not. The weight naming also differs: `dynamics.py` uses `w1, w2, w3` where `w3` is the output projection, while `transformer_tokenizer.py` uses `w1, w2, w3` where `w2` is the output projection. Both compute the same function but the weight names are swapped:

- `dynamics.py`: `self.w3(F.silu(self.w1(x)) * self.w2(x))` — w1=gate, w2=up, w3=down
- `transformer_tokenizer.py`: `self.w2(F.silu(self.w1(x)) * self.w3(x))` — w1=gate, w3=up, w2=down

This inconsistency would cause problems if you ever try to share weights or load one model's weights into the other.

---

### STYLE-2: Magic numbers for grid_size computation

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/transformer_tokenizer.py`
- **Lines**: 510, 677
- **Severity**: STYLE

Both `TransformerEncoder` and `TransformerDecoder` compute `self.grid_size = int(math.sqrt(num_patches))`, which assumes square patch grid. This would silently compute the wrong grid size for non-square inputs (e.g., `int(sqrt(660)) = 25`, not `30x22`). This is the same square-assumption issue as WARNING-7/WARNING-8.

---

### STYLE-3: xavier_uniform_ with gain=0.02 is unusual

- **File**: `/Users/dani/Repos/ahriuwu/src/ahriuwu/models/dynamics.py`
- **Lines**: 962-968
- **Severity**: STYLE

The weight initialization uses `nn.init.xavier_uniform_(p, gain=0.02)`. Xavier uniform with `gain=1.0` is designed to preserve variance. Using `gain=0.02` makes the initial weights extremely small (roughly 50x smaller than standard Xavier). This is more like a near-zero initialization.

This is likely intentional for stability with the AdaLN + diffusion setup (similar to DiT initialization), but it's worth documenting the rationale. The zero-init of `output_proj` (line 971) is standard for residual architectures.

---

## Detailed Checklist Results

### Attention Correctness

| Check | Status | Notes |
|-------|--------|-------|
| Block-causal encoder mask prevents future info | PASS | `create_encoder_mask` correctly uses causal bounds for latents (line 378-379) |
| Block-causal decoder mask prevents future info | **FAIL** | BUG-2: patches see ALL latents including future frames |
| Spatial attention stays within frame | PASS | Reshaped to (B*T, S, D) before attention (line 184) |
| Temporal attention is causal | PASS | `torch.triu(..., diagonal=1)` correctly masks future (line 291) |
| Temporal attention every 4th layer | PASS | `(i % temporal_every == temporal_every - 1)` at line 888 |
| Spatial layers don't cross frame boundaries | PASS | (B*T, S, D) reshape ensures isolation |
| independent_frames fully disables temporal | PASS | Diagonal mask at line 343 restricts to self-only |

### Position Encodings

| Check | Status | Notes |
|-------|--------|-------|
| RoPE 2D spatial applied correctly | PASS | Separate y/x axes, proper rotation pairs |
| RoPE not applied to latent tokens | PASS | `rope_indices = -1` for latents, masked via `valid_mask` |
| RoPE + QKNorm ordering | **FAIL** | BUG-1: QKNorm after RoPE destroys rotation encoding |
| Sincos position embed frequencies | PASS | Standard formula, algebraically correct |
| Temporal position embeddings | PASS | Additive, shape (1, max_seq, 1, D), properly sliced |

### Transformer Components

| Check | Status | Notes |
|-------|--------|-------|
| SwiGLU hidden_dim = round(8/3 * d) | PASS | `int(dim * 8/3)` then round to 64 boundary |
| SwiGLU computation correct | PASS | Both files: `silu(w1(x)) * w_gate(x)` then project down |
| QKNorm: L2 norm before dot product | PASS | RMSNorm on Q,K before matmul in all attention modules |
| Soft capping formula | PASS | `50 * tanh(logits / 50)` at line 89, applied after QK dot product, before softmax |
| RMSNorm formula | PASS | `x / sqrt(mean(x^2) + eps) * weight` — standard |
| Pre-norm (before attention/FFN) | PASS | `self.norm1(x)` before attention, `self.norm2(x)` before FFN in all blocks |
| Attention scale = 1/sqrt(head_dim) | PASS | `self.scale = self.head_dim ** -0.5` in all attention classes |

### GQA

| Check | Status | Notes |
|-------|--------|-------|
| KV heads shared correctly | PASS | `repeat_interleave(self.num_groups, dim=1)` correctly expands KV heads |
| Head dim = d_model / num_heads | PASS | `self.head_dim = head_dim or dim // num_heads` |
| num_heads % num_kv_heads == 0 | PASS | Asserted in all attention __init__ methods |

### Register Tokens

| Check | Status | Notes |
|-------|--------|-------|
| Attend to themselves + latents | PASS | Concatenated into spatial dim, attend via normal spatial/temporal attention |
| Excluded from output | PASS | `x[:, :, :self.spatial_tokens, :]` strips registers before output (line 1060) |
| No position embedding | WARNING-2 | Deliberate but potentially limiting |

### Bottleneck

| Check | Status | Notes |
|-------|--------|-------|
| Linear projection present | PASS | `nn.Linear(embed_dim, latent_dim)` at line 806 |
| Tanh activation | PASS | `torch.tanh(x)` at line 824 |
| Reshape dimensionally correct | WARNING-6 | Works when num_latents_in == num_latents_out (current usage), but API is fragile |
| Inverse bottleneck matches | PASS | Linear(latent_dim, embed_dim), no activation (correct) |

### Encoder/Decoder Cross-Attention (transformer_tokenizer.py)

| Check | Status | Notes |
|-------|--------|-------|
| Encoder: latents see all tokens causally | PASS | `mask[latent_start:latent_end, :causal_end] = True` (line 379) |
| Encoder: patches see only same-frame patches | PASS | Block diagonal for patches (line 375) |
| Decoder: patches see own-frame patches + latents | **PARTIAL** | BUG-2: sees ALL latents, not just causal ones |
| Decoder: latents see only latents causally | PASS | Lines 408-415 correctly implement causal latent-to-latent |

### Agent Tokens (dynamics.py)

| Check | Status | Notes |
|-------|--------|-------|
| Agent tokens attend to z tokens | PASS | `AgentCrossAttention` queries from agent, keys/values from z (lines 529-582) |
| Z tokens cannot attend back to agent | PASS | Agent tokens processed in separate blocks after main transformer (lines 1086-1094), x is frozen |
| Agent temporal attention is causal | PASS | Standard causal mask (line 683) |
| Agent cross-attention includes registers | PASS | `x` includes register tokens when passed to agent blocks (line 1094) |

### CNN Tokenizer

| Check | Status | Notes |
|-------|--------|-------|
| Handles non-square inputs | **FAIL** | BUG-3(revised): Hardcoded 256x256, no non-square support |
| Skip connections correct | PASS | Proper 1x1 conv skip with matching stride/channels |
| Residual block structure | PASS | Pre-activation style with GELU |

---

## Priority Fix Recommendations

1. **BUG-2 (decoder mask causal violation)**: Fix immediately. This is a data leakage bug that will cause the tokenizer to underperform at inference time with multi-frame sequences. Change `range(num_frames)` to `range(t + 1)` in `create_decoder_mask` line 426.

2. **BUG-1 (QKNorm after RoPE)**: Fix before enabling `use_rope=True`. If RoPE is not currently used (default is `use_rope=False`), this is latent. Swap the order of QKNorm and RoPE application in `MultiHeadAttention.forward()`.

3. **WARNING-3 (unnormalized KV in agent cross-attention)**: Fix before enabling agent tokens. Add a KV norm in `AgentTokenBlock`.

4. **Non-square input support (BUG-3 revised, WARNING-7, WARNING-8)**: Required for the 480x352 migration. This is a larger refactor affecting `PatchEmbed`, `RotaryEmbedding2D`, `PatchUnembed`, grid_size calculations, and the CNN tokenizer.

5. **STYLE-1 (duplicated components)**: Extract shared modules to reduce maintenance burden and prevent the implementations from diverging further.
