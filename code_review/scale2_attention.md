# DreamerV4 Attention Mechanisms Review

## Executive Summary

This review analyzes the attention implementation in `/Users/danirahman/Repos/ahriuwu/src/ahriuwu/models/dynamics.py` against the DreamerV4 paper (Section 3.4: Efficient Transformer). The implementation correctly captures the core factorized attention architecture but is missing several paper features that primarily matter at scale (1.6B parameters) rather than at the current MVP scale (~60M parameters).

---

## 1. Factorized Attention Analysis

### Paper Specification (Section 3.4)
> "We break up the cost of dense attention over all video tokens by using separate space-only and time-only attention layers."

### Implementation Review

#### SpatialAttention (Lines 59-115)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, S, D = x.shape
    # Reshape to (B*T, S, D) for efficient batched attention
    x = x.view(B * T, S, D)
```

**CORRECT**: Spatial attention operates within frames by reshaping to `(B*T, S, D)`, where each frame's 256 spatial tokens attend only to each other. No causal mask is applied, which is correct per the paper specification that "all tokens within a time step can attend to each other."

#### TemporalAttention (Lines 118-187)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, S, D = x.shape
    # Reshape to (B*S, T, D) for efficient batched attention over time
    x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
```

**CORRECT**: Temporal attention operates across frames by reshaping to `(B*S, T, D)`, where each spatial position attends across all time steps independently.

---

## 2. Temporal Attention Frequency

### Paper Specification (Section 3.4)
> "We find that only a relatively small number of temporal layers are needed and only use temporal attention once every 4 layers, in line with recent findings."

### Implementation Review (Lines 328-343)
```python
def __init__(self, ..., temporal_every: int = 4, ...):
    ...
    for i in range(num_layers):
        # Temporal attention every temporal_every layers (on the last of each group)
        is_temporal = (i % temporal_every == temporal_every - 1)
        attn_type = "temporal" if is_temporal else "spatial"
```

**CORRECT**: The `temporal_every=4` default matches the paper exactly. For a 12-layer model:
- Layers 0, 1, 2: Spatial
- Layer 3: Temporal
- Layers 4, 5, 6: Spatial
- Layer 7: Temporal
- Layers 8, 9, 10: Spatial
- Layer 11: Temporal

This gives 3 temporal layers out of 12, exactly matching the paper's "every 4 layers" specification.

---

## 3. Causal Masking

### Paper Specification (Section 3.4)
> "Attention is masked to be causal in time, so that all tokens within a time step can attend to each other and to the past."

### Implementation Review

#### Spatial Attention: NO Causal Mask
```python
# Lines 104-107
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = F.softmax(attn, dim=-1)
attn = self.dropout(attn)
```

**CORRECT**: No causal mask is applied. Tokens within the same frame can freely attend to each other.

#### Temporal Attention: YES Causal Mask
```python
# Lines 146-150 - Pre-computed causal mask
self.register_buffer(
    "causal_mask",
    torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
)

# Lines 174-176 - Applied during forward pass
mask = self.causal_mask[:T, :T]
attn = attn.masked_fill(mask, float("-inf"))
```

**CORRECT**: Upper triangular mask (diagonal=1) blocks attention to future frames. Frame t can attend to frames 0..t but not t+1..T.

---

## 4. Attention Scaling

### Paper Standard
Standard transformer attention scaling: `1/sqrt(head_dim)`

### Implementation Review (Lines 76-77, 136-137)
```python
self.head_dim = head_dim or dim // num_heads
self.scale = self.head_dim ** -0.5
```

**CORRECT**: Both SpatialAttention and TemporalAttention use `head_dim ** -0.5 = 1/sqrt(head_dim)`.

---

## 5. Missing Paper Features

### 5.1 Grouped Query Attention (GQA)

#### Paper Specification (Section 3.4)
> "We apply GQA to all attention layers in the dynamics, where multiple query heads attend to the same key-value heads to reduce the KV cache size further."

#### Current Implementation
```python
# Lines 81-82
self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
```

**MISSING**: Standard multi-head attention where Q, K, V all have the same number of heads. No GQA implementation.

#### Severity: **LOW at current scale**
- GQA primarily reduces KV cache memory during inference
- At 60M parameters with short sequences, memory is not the bottleneck
- Paper uses GQA for 1.6B dynamics model with 192-frame context
- **When to add**: If scaling to 500M+ parameters or context > 64 frames

---

### 5.2 QK Normalization (QKNorm)

#### Paper Specification (Section 3.4)
> "We employ QKNorm and attention logit soft capping to increase training stability."

#### Current Implementation
No normalization of Q and K tensors before computing attention.

**MISSING**: Should apply RMSNorm or LayerNorm to Q and K after projection.

#### Severity: **MEDIUM**
- QKNorm prevents attention logit explosion during training
- More important at larger scales where gradients can explode
- Reference: Dehghani et al. "Scaling Vision Transformers to 22 Billion Parameters" (ICML 2023)
- **When to add**: If observing training instabilities (NaN losses, gradient spikes)

---

### 5.3 Attention Logit Soft Capping

#### Paper Specification (Section 3.4)
> "We employ QKNorm and attention logit soft capping to increase training stability."

#### Current Implementation
```python
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = F.softmax(attn, dim=-1)
```

**MISSING**: No soft capping of attention logits.

#### What it should look like
```python
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = soft_cap * torch.tanh(attn / soft_cap)  # soft_cap typically 50.0
attn = F.softmax(attn, dim=-1)
```

#### Severity: **LOW-MEDIUM**
- Prevents extreme attention values that can cause numerical instability
- Used in Gemma 2 and other large models
- Less critical at smaller scales
- **When to add**: If training becomes unstable or attention patterns become too sharp

---

### 5.4 Rotary Position Embeddings (RoPE)

#### Paper Specification (Section 3.4)
> "We start from a standard transformer with pre-layer RMSNorm, RoPE, and SwiGLU."

#### Current Implementation (Lines 315-320)
```python
# Positional embeddings
self.spatial_pos = nn.Parameter(
    torch.randn(1, 1, self.spatial_tokens, model_dim) * 0.02
)
self.temporal_pos = nn.Parameter(
    torch.randn(1, max_seq_len, 1, model_dim) * 0.02
)
```

**MISSING**: Uses learned absolute positional embeddings instead of RoPE.

#### Severity: **MEDIUM**
- RoPE provides better length generalization
- Paper emphasizes training on varying batch lengths (64 and 256 frames)
- Learned embeddings may limit generalization to longer sequences
- However, for fixed context lengths at MVP scale, learned embeddings work adequately
- **When to add**: If needing variable-length generation or training on mixed batch lengths

---

### 5.5 Flash Attention / Memory Optimizations

#### Paper Implication
Paper achieves 21 FPS inference on single H100 with 2B parameters, suggesting optimized attention.

#### Current Implementation
Standard PyTorch attention:
```python
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = F.softmax(attn, dim=-1)
out = (attn @ v)
```

**MISSING**: No Flash Attention or memory-efficient attention.

#### Severity: **LOW at current scale**
- Standard attention is fine for 60M parameters
- Flash Attention provides O(1) memory in sequence length
- **When to add**: If OOM during training with longer sequences or larger batch sizes

---

## 6. Feature Importance by Scale

| Feature | 60M Scale | 500M Scale | 1.6B Scale |
|---------|-----------|------------|------------|
| Factorized Attention | IMPLEMENTED | Critical | Critical |
| Temporal Every 4 | IMPLEMENTED | Critical | Critical |
| Causal Masking | IMPLEMENTED | Critical | Critical |
| Attention Scaling | IMPLEMENTED | Critical | Critical |
| GQA | Not needed | Helpful | Critical |
| QKNorm | Helpful | Important | Critical |
| Soft Capping | Optional | Helpful | Important |
| RoPE | Helpful | Important | Critical |
| Flash Attention | Optional | Helpful | Critical |

---

## 7. Detailed Code Analysis

### TransformerBlock (Lines 190-261)

The TransformerBlock correctly implements:
1. Pre-norm architecture with RMSNorm
2. AdaLN modulation for timestep conditioning (lines 214-218)
3. Proper residual connections with gating (lines 249, 255)

```python
# AdaLN modulation - CORRECT
shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
h = self.norm1(x)
h = h * (1 + scale1) + shift1  # Scale and shift
h = self.attn(h)
x = x + gate1 * h  # Gated residual
```

This matches DiT-style conditioning used in modern diffusion transformers.

### DynamicsTransformer (Lines 264-418)

The overall architecture correctly implements:
1. Input projection from latent to model dim
2. Separate spatial and temporal position embeddings
3. Timestep embedding for diffusion
4. Step embedding for shortcut forcing
5. Final RMSNorm before output projection
6. Zero-initialized output projection (line 361)

---

## 8. Recommendations for Scale-up

### Priority 1: Add when training becomes unstable
1. **QKNorm**: Add `nn.RMSNorm(head_dim)` applied to Q and K after projection
2. **Soft Capping**: Add `50.0 * tanh(attn / 50.0)` before softmax

### Priority 2: Add when scaling model size
1. **GQA**: Reduce KV heads to `num_heads // 4` or `num_heads // 8`
2. **Flash Attention**: Use `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`

### Priority 3: Add when scaling context length
1. **RoPE**: Replace learned positional embeddings with rotary embeddings
2. **Sliding Window Attention**: Consider for very long sequences (>256 frames)

---

## 9. Summary

### What's Correct (5/5 core features)
1. Factorized space-time attention architecture
2. Temporal attention every 4th layer
3. Causal masking only on temporal attention
4. Correct attention scaling (1/sqrt(head_dim))
5. Pre-norm transformer with RMSNorm and SwiGLU

### What's Missing (5 features)
1. GQA - Low priority at current scale
2. QKNorm - Medium priority for stability
3. Attention Logit Soft Capping - Low-medium priority
4. RoPE - Medium priority for length generalization
5. Flash Attention - Low priority at current scale

### Overall Assessment
The attention implementation is **architecturally correct** for the MVP stage. The missing features are primarily **efficiency and stability optimizations** that become critical at the paper's 1.6B parameter scale but are **not blocking** for a 60M parameter proof-of-concept. The core insight from the paper - factorized attention with sparse temporal layers - is correctly implemented.

**Recommended next steps:**
1. Continue training with current implementation
2. Add QKNorm if training becomes unstable
3. Add GQA and Flash Attention when scaling beyond 200M parameters
4. Add RoPE when training on variable-length sequences
