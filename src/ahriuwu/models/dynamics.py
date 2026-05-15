"""Dynamics Transformer for world model.

Predicts future latent frames using diffusion with factorized attention.
Architecture follows DreamerV4:

- Factorized attention: spatial within frames, temporal across frames
- Temporal attention every 4th layer (efficiency optimization)
- X-prediction objective (predicts clean data directly)
- RMSNorm, SwiGLU, RoPE (2D spatial, 1D temporal)
- Action and conditioning as explicit sequence tokens (not broadcast/AdaLN)
- Agent tokens for policy/reward prediction (Phase 2+)

References:
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (
    RMSNorm, QKNorm, SwiGLU, Attention,
    soft_cap_attention, _soft_cap_score_mod,
)
from ..constants import MOVEMENT_DIM, ABILITY_KEYS

# flex_attention imports (for AgentCrossAttention which remains bespoke)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False


def _checkpoint_block_forward(block, x, independent_frames):
    """Wrapper for gradient checkpointing that passes independent_frames as keyword arg."""
    return block(x, independent_frames=independent_frames)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with either spatial or temporal attention.

    Uses the unified ``Attention`` layer from layers.py with RoPE, GQA, QKNorm,
    soft capping, and independent frame mode.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        attn_type: Literal["spatial", "temporal"] = "spatial",
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        spatial_size: int = 16,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads)
            head_dim: Dimension per head
            attn_type: "spatial" or "temporal"
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
            spatial_size: Grid size for 2D spatial RoPE
        """
        super().__init__()
        self.attn_type = attn_type

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mode=attn_type,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            soft_cap=soft_cap,
            spatial_size=spatial_size,
            max_seq_len=max_seq_len,
            allow_flex=False,
        )

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        independent_frames: bool = False,
    ) -> torch.Tensor:
        """Forward pass (standard pre-norm transformer).

        Args:
            x: (B, T, S, D) features
            independent_frames: If True, treat frames as independent (temporal attention only)

        Returns:
            (B, T, S, D) transformed features
        """
        B, T, S, D = x.shape

        if self.attn_type == "spatial":
            # Reshape to (B*T, S, D) for spatial attention
            h = self.norm1(x).view(B * T, S, D)
            h = self.attn(h)
            x = x + h.view(B, T, S, D)
        else:
            # Reshape to (B*S, T, D) for temporal attention
            h = self.norm1(x).permute(0, 2, 1, 3).reshape(B * S, T, D)
            h = self.attn(h, independent_frames=independent_frames)
            x = x + h.view(B, S, T, D).permute(0, 2, 1, 3)

        x = x + self.ffn(self.norm2(x))
        return x


class AgentCrossAttention(nn.Module):
    """Cross-attention where agent tokens query z tokens.

    Implements the asymmetric attention pattern from DreamerV4:
    - Agent tokens can attend to all z tokens
    - Z tokens cannot attend back to agent tokens (handled by keeping them separate)

    Supports GQA, QKNorm, and soft capping.  Uses flex_attention when available,
    falls back to manual matmul otherwise.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap

        # When QKNorm is enabled, set scale=1.0 (Gemma 2 convention).
        self.scale = 1.0 if use_qk_norm else self.head_dim ** -0.5

        assert num_heads % self.num_kv_heads == 0
        self.num_groups = num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

    def forward(
        self,
        agent_tokens: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention from agent tokens to z tokens.

        Args:
            agent_tokens: (B, T, D) one agent token per frame
            z_tokens: (B, T, S, D) spatial tokens per frame

        Returns:
            (B, T, D) attended agent token features
        """
        B, T, D = agent_tokens.shape
        _, _, S, _ = z_tokens.shape

        q = self.q_proj(agent_tokens).view(B, T, self.num_heads, 1, self.head_dim)
        z_flat = z_tokens.view(B * T, S, D)
        k = self.k_proj(z_flat).view(B, T, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(z_flat).view(B, T, S, self.num_kv_heads, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        dtype = v.dtype
        q = q.to(dtype).view(B * T, self.num_heads, 1, self.head_dim)
        k = k.to(dtype).reshape(B * T, self.num_kv_heads, S, self.head_dim)
        v = v.reshape(B * T, self.num_kv_heads, S, self.head_dim)

        if _FLEX_AVAILABLE:
            score_mod = _soft_cap_score_mod if self.soft_cap is not None else None
            out = flex_attention(
                q, k, v, score_mod=score_mod,
                scale=self.scale,
                enable_gqa=(self.num_groups > 1),
            )
        else:
            # Manual fallback with GQA expansion
            if self.num_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
                k = k.reshape(k.shape[0], self.num_heads, k.shape[3], k.shape[4])
                v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
                v = v.reshape(v.shape[0], self.num_heads, v.shape[3], v.shape[4])
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.soft_cap is not None:
                attn = soft_cap_attention(attn, self.soft_cap)
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.squeeze(2).view(B, T, -1)
        return self.out_proj(out)


class AgentTokenBlock(nn.Module):
    """Processing block for agent tokens.

    1. Cross-attention to z tokens (agent sees everything)
    2. Self-attention across time (causal) via unified Attention
    3. FFN

    Supports GQA, QKNorm, and soft capping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
    ):
        """Initialize agent token block.

        Args:
            dim: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads for GQA
            head_dim: Dimension per head
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization
            soft_cap: Soft cap value for attention logits
        """
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.cross_attn = AgentCrossAttention(
            dim, num_heads, num_kv_heads, head_dim, dropout,
            use_qk_norm=use_qk_norm, soft_cap=soft_cap
        )

        self.norm2 = RMSNorm(dim)
        self.self_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mode="temporal",
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            soft_cap=soft_cap,
            max_seq_len=max_seq_len,
            allow_flex=False,
        )

        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Process agent tokens.

        Args:
            agent_tokens: (B, T, D) agent tokens
            z_tokens: (B, T, S, D) spatial z tokens

        Returns:
            (B, T, D) processed agent tokens
        """
        # Cross-attention to z tokens
        agent_tokens = agent_tokens + self.cross_attn(self.norm1(agent_tokens), self.norm_kv(z_tokens))

        # Self-attention across time (causal)
        agent_tokens = agent_tokens + self.self_attn(self.norm2(agent_tokens))

        # FFN
        agent_tokens = agent_tokens + self.ffn(self.norm3(agent_tokens))

        return agent_tokens


class DynamicsTransformer(nn.Module):
    """Dynamics model for world model training.

    Predicts clean latent frames from noisy inputs using diffusion.
    Uses factorized attention with temporal attention every Nth layer.

    Architecture:
    - Input projection: latent tokens to model dim
    - Register tokens for improved information flow
    - 2D RoPE for spatial attention, 1D RoPE for temporal attention
    - Action and conditioning (tau + step_size) as explicit sequence tokens
    - Transformer blocks with factorized attention (GQA, QKNorm, soft capping)
    - Output projection back to latent dim

    Spatial sequence per time step:
        [latent_tokens (with 2D RoPE), register_tokens (no RoPE),
         action_token (no RoPE), condition_token (no RoPE)]

    Reference: DreamerV4 Section 3.2
    """

    def __init__(
        self,
        latent_dim: int = 32,
        spatial_size: int = 16,  # 16x16 = 256 spatial tokens
        model_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        temporal_every: int = 4,  # Temporal attention every N layers
        dropout: float = 0.0,
        max_seq_len: int = 256,  # Max frames in sequence
        # Stability and efficiency features
        use_qk_norm: bool = True,
        soft_cap: float | None = 50.0,
        # Register tokens
        num_register_tokens: int = 8,
        # Agent token settings (Phase 2+)
        use_agent_tokens: bool = False,
        num_tasks: int = 1,  # For multi-task conditioning
        agent_layers: int = 4,  # Number of agent token processing layers
        # Action conditioning
        use_actions: bool = False,
        # Game-time conditioning (Phase 2+: lets the model condition on
        # where in the game we are — laning, mid, late). Optional so YT
        # pre-training without game-time labels still works (falls back
        # to a learned ``no_game_time`` embedding).
        use_game_time: bool = False,
        game_time_bucket_seconds: float = 30.0,
        game_time_num_buckets: int = 120,  # 30s × 120 = 60min ceiling
        # Train-time dropout: with this probability replace gt_emb with the
        # no_gt_embed even when game_time is provided. Prevents the model
        # from crutching on game_time when fine-tuning on action-labeled
        # data, so the same checkpoint still works at inference when no
        # game_time is available (YT-distribution or anonymous rollouts).
        # 0 disables.
        gt_dropout: float = 0.1,
        # Memory efficiency
        gradient_checkpointing: bool = False,
    ):
        """Initialize dynamics transformer.

        Args:
            latent_dim: Dimension of input latent tokens (from tokenizer)
            spatial_size: Size of spatial grid (16 for 16x16)
            model_dim: Model hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads (query heads)
            num_kv_heads: Number of KV heads for GQA (None = same as num_heads, MHA)
            head_dim: Dimension per head (default: model_dim // num_heads)
            temporal_every: Add temporal attention every N layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            use_qk_norm: Whether to use QK normalization for attention stability
            soft_cap: Soft cap value for attention logits (None = no capping)
            num_register_tokens: Number of register tokens (0 = disabled)
            use_agent_tokens: Enable agent tokens for Phase 2+
            num_tasks: Number of tasks for multi-task conditioning
            agent_layers: Number of agent token processing layers
            use_actions: Enable action conditioning with factorized embeddings
            gradient_checkpointing: Use gradient checkpointing to save memory (~2x reduction)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.spatial_tokens = spatial_size * spatial_size  # 256
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.temporal_every = temporal_every
        self.use_agent_tokens = use_agent_tokens
        self.use_actions = use_actions
        self.use_game_time = use_game_time
        self.game_time_bucket_seconds = float(game_time_bucket_seconds)
        self.game_time_num_buckets = int(game_time_num_buckets)
        self.gt_dropout = float(gt_dropout)
        self.num_register_tokens = num_register_tokens
        self.use_qk_norm = use_qk_norm
        self.soft_cap = soft_cap
        self.gradient_checkpointing = gradient_checkpointing

        # Count extra tokens appended after latent+register tokens
        # Always have 1 conditioning token (tau + step_size)
        self.num_extra_tokens = 1  # condition token
        if use_actions:
            self.num_extra_tokens += 1  # action token

        # Total spatial tokens per time step: latent + register + action + condition
        self.total_spatial_tokens = self.spatial_tokens + num_register_tokens + self.num_extra_tokens

        # Input projection: (B, T, C, H, W) -> (B, T, S, D)
        self.input_proj = nn.Linear(latent_dim, model_dim)

        # Register tokens (learnable, shared across all frames)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, model_dim) * 0.02
            )
        else:
            self.register_tokens = None

        # Discrete embeddings for tau and step_size (DreamerV4 uses lookup tables)
        # tau lives on a grid of k_max levels: {0, 1/k_max, ..., (k_max-1)/k_max}
        # step_size is power of 2: {1, 2, 4, ..., k_max} -> index by log2(d)
        self.k_max = 64
        self.num_tau_levels = self.k_max  # 64 discrete levels
        self.num_step_sizes = int(math.log2(self.k_max)) + 1  # 7: d=1,2,4,8,16,32,64
        self.tau_embed = nn.Embedding(self.num_tau_levels, model_dim)
        self.step_embed = nn.Embedding(self.num_step_sizes, model_dim)
        # Project concatenated [tau_emb, step_emb] to model_dim
        self.cond_proj = nn.Linear(model_dim * 2, model_dim)

        # Factorized action embeddings -> produces one action token per time step
        if use_actions:
            self.action_embed = nn.ModuleDict({
                'movement': nn.Linear(MOVEMENT_DIM, model_dim),  # continuous (x, y) -> D
                **{k: nn.Embedding(2, model_dim) for k in ABILITY_KEYS}
            })
            # Learned "no action" embedding for unlabeled videos
            self.no_action_embed = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Game-time conditioning: discrete embedding lookup mirroring the
        # tau/step_size pattern. game_time (seconds) → bucket index → embed →
        # added to every spatial token of that frame (same path as tau_emb /
        # step_emb). When game_time is None at forward time (e.g. YT pretrain
        # where we don't have absolute game time), we substitute a learned
        # no_gametime embedding so the model still has a stable input there
        # instead of receiving NaN/zero.
        if use_game_time:
            self.gt_embed = nn.Embedding(self.game_time_num_buckets, model_dim)
            self.no_gt_embed = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Transformer blocks with GQA, QKNorm, soft capping, and RoPE
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Temporal attention every temporal_every layers (on the last of each group)
            is_temporal = (i % temporal_every == temporal_every - 1)
            attn_type = "temporal" if is_temporal else "spatial"

            self.blocks.append(
                TransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attn_type=attn_type,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_qk_norm=use_qk_norm,
                    soft_cap=soft_cap,
                    spatial_size=spatial_size,
                )
            )

        # Output projection
        self.norm_out = RMSNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, latent_dim)

        # Agent token components (Phase 2+)
        if use_agent_tokens:
            # Learnable agent token (one per frame, initialized from parameters)
            self.agent_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

            # Task embedding for multi-task conditioning
            self.task_embed = nn.Embedding(num_tasks, model_dim)

            # Agent token temporal position embedding
            self.agent_temporal_pos = nn.Parameter(
                torch.randn(1, max_seq_len, model_dim) * 0.02
            )

            # Agent token processing blocks with GQA, QKNorm, soft capping
            self.agent_blocks = nn.ModuleList([
                AgentTokenBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_qk_norm=use_qk_norm,
                    soft_cap=soft_cap,
                )
                for _ in range(agent_layers)
            ])

            # Output normalization for agent tokens
            self.agent_norm_out = RMSNorm(model_dim)

        # Initialize weights
        self._init_weights()

    def embed_actions(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum factorized action embeddings.

        Args:
            actions: Dict with keys 'movement' (B, T, 2) float, ability keys
                     (B, T) long, and OPTIONAL 'cursor_valid' (B, T) bool. When
                     cursor_valid is False (frames before any cursor observed),
                     movement values are NaN — we substitute the learned
                     ``no_action_embed`` for those frames so NaN doesn't
                     propagate through the projection.

        Returns:
            (B, T, D) summed action embedding
        """
        movement = actions['movement']
        cursor_valid = actions.get('cursor_valid')
        if cursor_valid is not None:
            # Replace NaN-valued (invalid) movement with zeros before the
            # linear projection. The cursor_valid mask then swaps out the
            # zero-projection for the learned no_action_embed.
            safe_movement = torch.where(
                cursor_valid.unsqueeze(-1), movement, torch.zeros_like(movement)
            )
            emb = self.action_embed['movement'](safe_movement)  # (B, T, D)
            no_act = self.no_action_embed.expand_as(emb)
            emb = torch.where(cursor_valid.unsqueeze(-1), emb, no_act)
        else:
            # Legacy / fully-labeled path — assume movement has no NaN.
            emb = self.action_embed['movement'](movement)

        # Add all ability key embeddings (binary, always defined)
        for key in ABILITY_KEYS:
            emb = emb + self.action_embed[key](actions[key])

        return emb

    def _build_condition_token(
        self,
        tau: torch.Tensor,
        step_size: torch.Tensor | None,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Build conditioning token from discrete tau and step_size embeddings.

        Args:
            tau: Diffusion timesteps, shape (B,) or (B, T), values in [0, 1)
                 Must be grid-aligned: multiples of 1/k_max
            step_size: Step sizes as integers {1, 2, 4, ..., k_max}, shape (B,) or (B, T).
                       None defaults to d=1 (finest).
            B: Batch size
            T: Sequence length

        Returns:
            (B, T, D) conditioning token
        """
        # Convert continuous tau to discrete index: tau * k_max -> integer in [0, k_max)
        # NOTE: This quantization to k_max discrete bins is intentional, matching the
        # DreamerV4 paper's "discrete signal levels" design. Even when tau is sampled
        # continuously, embedding it via a discrete lookup table is the intended approach.
        tau_idx = (tau * self.k_max).long().clamp(0, self.num_tau_levels - 1)
        tau_emb = self.tau_embed(tau_idx)  # (B, D) or (B, T, D)

        # Convert step_size (integer power of 2) to index via log2
        if step_size is not None:
            step_idx = torch.log2(step_size.float()).long().clamp(0, self.num_step_sizes - 1)
        else:
            step_idx = torch.zeros(B, dtype=torch.long, device=tau.device)  # d=1 -> log2(1)=0
        step_emb = self.step_embed(step_idx)  # (B, D) or (B, T, D)

        # Expand to (B, T, D) if needed
        if tau_emb.dim() == 2:
            tau_emb = tau_emb.unsqueeze(1).expand(-1, T, -1)
        if step_emb.dim() == 2:
            step_emb = step_emb.unsqueeze(1).expand(-1, T, -1)

        # Concatenate and project: [tau_emb, step_emb] -> (B, T, D)
        cond = torch.cat([tau_emb, step_emb], dim=-1)  # (B, T, 2*D)
        cond = self.cond_proj(cond)  # (B, T, D)

        return cond

    @staticmethod
    def _init_module(module: nn.Module):
        """Type-based weight initialization (standard transformer).

        Dispatches by module type — rename-safe, no string matching.
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_weights(self):
        """Initialize all weights, then apply residual scaling overrides.

        Type-based defaults via apply(), then explicit overrides for:
        - Residual output paths: scaled by 1/sqrt(2*num_layers) for deep networks
        - Final output_proj: small init for residual learning (output starts near input)
        """
        # Type-based defaults: embeddings get normal, linears get xavier
        self.apply(self._init_module)

        # Residual scaling: attention out_proj and SwiGLU w3 in each block
        num_layers = len(self.blocks)
        residual_scale = 1.0 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.out_proj.weight, gain=residual_scale)
            nn.init.xavier_uniform_(block.ffn.w3.weight, gain=residual_scale)

        # Agent blocks (if present)
        if hasattr(self, 'agent_blocks'):
            for block in self.agent_blocks:
                nn.init.xavier_uniform_(block.cross_attn.out_proj.weight, gain=residual_scale)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight, gain=residual_scale)
                nn.init.xavier_uniform_(block.ffn.w3.weight, gain=residual_scale)

        # Final output: small but nonzero for residual learning
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)

    def forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        step_size: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        actions: dict[str, torch.Tensor] | None = None,
        game_time: torch.Tensor | None = None,
        independent_frames: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict clean latents from noisy input.

        Args:
            z_tau: Noisy latents, shape (B, T, C, H, W)
            tau: Diffusion timesteps, shape (B,) or (B, T)
            step_size: Optional step size for shortcut forcing, shape (B,) or (B, T).
                       Integer powers of 2: {1, 2, 4, 8, 16, 32, 64}. None defaults to d=1.
            task_id: Optional task ID for multi-task conditioning, shape (B,)
            actions: Optional action dict with 'movement', 'target', and ability keys.
                     Each value is (B, T) tensor of class indices.
            game_time: Optional per-frame game time in seconds, shape (B, T) or (B,).
                       Float. Only consumed when the model was built with
                       use_game_time=True; otherwise ignored. When use_game_time
                       is True but this is None, the learned ``no_gt_embed`` is
                       substituted so YT-pretrained checkpoints can still run.
            independent_frames: If True, treat frames as independent (no temporal context).
                               Use with 30% probability during training to prevent
                               temporal shortcut learning (DreamerV4 Section 3.2).

        Returns:
            If use_agent_tokens=False:
                z_0_pred: Predicted clean latents, shape (B, T, C, H, W)
            If use_agent_tokens=True:
                tuple of:
                    z_0_pred: Predicted clean latents, shape (B, T, C, H, W)
                    agent_out: Agent token outputs, shape (B, T, D) for heads
        """
        B, T, C, H, W = z_tau.shape
        assert H == W == self.spatial_size, f"Expected {self.spatial_size}x{self.spatial_size}, got {H}x{W}"

        # Reshape to (B, T, S, C) where S = H*W
        x = z_tau.view(B, T, C, -1).permute(0, 1, 3, 2)  # (B, T, S, C)

        # Project to model dim
        x = self.input_proj(x)  # (B, T, S, D) where S = spatial_tokens

        # Add register tokens if enabled
        if self.register_tokens is not None:
            # Expand register tokens to match batch and time: (1, 1, R, D) -> (B, T, R, D)
            registers = self.register_tokens.expand(B, T, -1, -1)
            # Concatenate: (B, T, S+R, D)
            x = torch.cat([x, registers], dim=2)

        # Build action token and append to sequence
        if self.use_actions:
            if actions is not None:
                action_token = self.embed_actions(actions)  # (B, T, D)
            else:
                action_token = self.no_action_embed.expand(B, T, -1)
            # (B, T, D) -> (B, T, 1, D) and concatenate
            x = torch.cat([x, action_token.unsqueeze(2)], dim=2)

        # Build conditioning (tau + step_size).
        # At our scale, attention-only conditioning is too weak (158x gradient gap).
        # We add tau directly to all tokens for strong denoising signal, and add
        # step_size directly for strong shortcut signal. Both are also projected
        # together into an appended token for cross-attention.
        tau_idx = (tau * self.k_max).long().clamp(0, self.num_tau_levels - 1)
        tau_emb = self.tau_embed(tau_idx)
        if tau_emb.dim() == 2:
            tau_emb = tau_emb.unsqueeze(1).expand(-1, T, -1)
        if step_size is not None:
            step_idx = torch.log2(step_size.float()).long().clamp(0, self.num_step_sizes - 1)
        else:
            step_idx = torch.zeros(B, dtype=torch.long, device=tau.device)
        step_emb = self.step_embed(step_idx)
        if step_emb.dim() == 2:
            step_emb = step_emb.unsqueeze(1).expand(-1, T, -1)

        # Additive: separate paths so step_size gradient isn't suppressed by tau
        x = x + tau_emb.unsqueeze(2) + step_emb.unsqueeze(2)

        # Game-time conditioning. Discretize seconds → bucket index → lookup,
        # then add to all spatial tokens of each frame. We add along the same
        # additive path as tau/step so the gradient signal is on equal footing
        # with the other scalar conditioners (DreamerV4-style). When the
        # caller didn't supply game_time, fall back to the learned
        # no_gt_embed so the path is still occupied (avoids relying on the
        # model to ignore an all-zero contribution it never saw at train).
        if self.use_game_time:
            if game_time is not None:
                # Accept (B,) by broadcasting to (B, T). Common during single-
                # frame inference where the same game_time applies to all T.
                gt = game_time
                if gt.dim() == 1:
                    gt = gt.unsqueeze(1).expand(-1, T)
                gt_idx = (gt.float() / self.game_time_bucket_seconds).long().clamp(
                    0, self.game_time_num_buckets - 1
                )
                gt_emb = self.gt_embed(gt_idx)  # (B, T, D)
                # Train-time dropout. Per-batch coin flip: with probability
                # gt_dropout, drop the whole batch's game_time signal and
                # substitute no_gt_embed. Keeps the model from leaning on
                # game_time so the same weights serve action-labeled (gt
                # known) and zero-shot (gt unknown) inference.
                if self.training and self.gt_dropout > 0 and torch.rand(1).item() < self.gt_dropout:
                    gt_emb = self.no_gt_embed.expand(B, T, -1)
            else:
                gt_emb = self.no_gt_embed.expand(B, T, -1)  # (1,1,D) -> (B,T,D)
            x = x + gt_emb.unsqueeze(2)

        # Also append combined token for attention pathway
        cond_token = self._build_condition_token(tau, step_size, B, T)
        x = torch.cat([x, cond_token.unsqueeze(2)], dim=2)

        # x is now (B, T, total_spatial_tokens + 1, D)
        # Layout: [latent+cond, register+cond, action?+cond, cond_token]

        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    _checkpoint_block_forward,
                    block, x, independent_frames,
                    use_reentrant=False,
                )
            else:
                x = block(x, independent_frames=independent_frames)

        # Strip extra tokens (register, action, condition) before output projection
        # Only keep latent tokens
        x_spatial = x[:, :, :self.spatial_tokens, :]  # (B, T, S, D)

        # Output projection for z prediction
        z_out = self.norm_out(x_spatial)
        z_out = self.output_proj(z_out)  # (B, T, S, C)

        # Reshape back to (B, T, C, H, W)
        z_0_pred = z_out.permute(0, 1, 3, 2).view(B, T, C, H, W)

        # Process agent tokens if enabled
        if self.use_agent_tokens:
            # Initialize agent tokens: expand to (B, T, D)
            agent_tokens = self.agent_token.expand(B, T, -1).clone()

            # Add task embedding if provided
            if task_id is not None:
                task_emb = self.task_embed(task_id)  # (B, D)
                agent_tokens = agent_tokens + task_emb.unsqueeze(1)

            # Add temporal position embedding
            agent_tokens = agent_tokens + self.agent_temporal_pos[:, :T, :]

            # Process through agent blocks
            # Agent tokens attend to z tokens (x includes all tokens), but z tokens don't see agent tokens
            for agent_block in self.agent_blocks:
                if self.gradient_checkpointing and self.training:
                    agent_tokens = checkpoint(
                        agent_block,
                        agent_tokens, x,
                        use_reentrant=False,
                    )
                else:
                    agent_tokens = agent_block(agent_tokens, x)

            # Output normalization
            agent_out = self.agent_norm_out(agent_tokens)

            return z_0_pred, agent_out

        return z_0_pred

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dynamics(
    size: str = "small",
    latent_dim: int = 32,
    use_agent_tokens: bool = False,
    num_tasks: int = 1,
    agent_layers: int = 4,
    use_actions: bool = False,
    use_game_time: bool = False,
    game_time_bucket_seconds: float = 30.0,
    game_time_num_buckets: int = 120,
    gt_dropout: float = 0.1,
    # New DreamerV4 features
    use_qk_norm: bool = True,
    soft_cap: float | None = 50.0,
    num_register_tokens: int = 8,
    num_kv_heads: int | None = None,
    # Memory efficiency
    gradient_checkpointing: bool = False,
) -> DynamicsTransformer:
    """Create dynamics model with preset sizes.

    Args:
        size: One of "tiny", "small", "medium", "large"
        latent_dim: Dimension of latent tokens (must match tokenizer)
        use_agent_tokens: Enable agent tokens for Phase 2+
        num_tasks: Number of tasks for multi-task conditioning
        agent_layers: Number of agent token processing layers
        use_actions: Enable action conditioning with factorized embeddings
        use_qk_norm: Whether to use QK normalization for attention stability
        soft_cap: Soft cap value for attention logits (None = no capping)
        num_register_tokens: Number of register tokens (0 = disabled)
        num_kv_heads: Number of KV heads for GQA (None = MHA, same as num_heads)
        gradient_checkpointing: Use gradient checkpointing to save memory (~2x reduction)

    Returns:
        DynamicsTransformer instance
    """
    configs = {
        "tiny": {
            "model_dim": 256,
            "num_layers": 6,
            "num_heads": 4,
        },
        "small": {
            "model_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
        },
        "medium": {
            "model_dim": 768,
            "num_layers": 18,
            "num_heads": 12,
        },
        "large": {
            "model_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    resolved = {
        "latent_dim": latent_dim,
        "spatial_size": 16,  # 256 latent tokens = 16x16 spatial grid
        "use_agent_tokens": use_agent_tokens,
        "num_tasks": num_tasks,
        "agent_layers": agent_layers,
        "use_actions": use_actions,
        "use_game_time": use_game_time,
        "game_time_bucket_seconds": game_time_bucket_seconds,
        "game_time_num_buckets": game_time_num_buckets,
        "gt_dropout": gt_dropout,
        "use_qk_norm": use_qk_norm,
        "soft_cap": soft_cap,
        "num_register_tokens": num_register_tokens,
        "num_kv_heads": num_kv_heads,
        "gradient_checkpointing": gradient_checkpointing,
        **configs[size],
    }
    model = DynamicsTransformer(**resolved)
    # Self-describing checkpoint: save_checkpoint reads model.config so any
    # downstream loader knows the exact init args without trusting CLI flags.
    model.config = {"size_preset": size, **resolved}
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing dynamics transformer...")

    model = create_dynamics("small", latent_dim=32)
    print(f"Parameters (base): {model.get_num_params():,}")
    print(f"  QKNorm: {model.use_qk_norm}")
    print(f"  Soft cap: {model.soft_cap}")
    print(f"  Register tokens: {model.num_register_tokens}")

    # Test forward pass
    B, T, C, H, W = 2, 8, 32, 16, 16
    z_tau = torch.randn(B, T, C, H, W)
    tau = torch.rand(B)

    z_pred = model(z_tau, tau)
    print(f"Input shape: {z_tau.shape}")
    print(f"Output shape: {z_pred.shape}")
    print(f"Tau shape: {tau.shape}")

    # Test with sequence tau
    tau_seq = torch.rand(B, T)
    z_pred_seq = model(z_tau, tau_seq)
    print(f"Sequence tau shape: {tau_seq.shape}")
    print(f"Output shape: {z_pred_seq.shape}")

    print("\n--- Testing independent frames mode (30% training) ---")
    z_pred_indep = model(z_tau, tau, independent_frames=True)
    print(f"Output shape (independent frames): {z_pred_indep.shape}")

    print("\n--- Testing with GQA (4 KV heads vs 8 Q heads) ---")
    model_gqa = create_dynamics("small", latent_dim=32, num_kv_heads=4)
    print(f"Parameters (GQA): {model_gqa.get_num_params():,}")
    z_pred_gqa = model_gqa(z_tau, tau)
    print(f"Output shape (GQA): {z_pred_gqa.shape}")

    print("\n--- Testing with game_time conditioning ---")
    model_gt = create_dynamics("small", latent_dim=32, use_game_time=True)
    n_gt = model_gt.get_num_params() - model.get_num_params()
    print(f"Parameters (with game_time): {model_gt.get_num_params():,} "
          f"(+{n_gt:,} for gt_embed + no_gt_embed)")
    game_time = torch.rand(B, T) * 1800.0  # 0..30 minutes
    z_pred_gt = model_gt(z_tau, tau, game_time=game_time)
    print(f"With game_time (B, T)={tuple(game_time.shape)} -> {tuple(z_pred_gt.shape)}")
    # Per-batch scalar game_time (single frame inference cadence)
    z_pred_gt_b = model_gt(z_tau, tau, game_time=torch.full((B,), 300.0))
    print(f"With game_time (B,) -> {tuple(z_pred_gt_b.shape)}")
    # Missing game_time → falls back to no_gt_embed without crashing
    z_pred_gt_none = model_gt(z_tau, tau, game_time=None)
    print(f"With game_time=None -> {tuple(z_pred_gt_none.shape)} (no_gt_embed)")

    print("\n--- Testing with actions ---")
    model_actions = create_dynamics("small", latent_dim=32, use_actions=True)
    print(f"Parameters (with actions): {model_actions.get_num_params():,}")

    # Create mock actions
    actions = {
        'movement': torch.rand(B, T, MOVEMENT_DIM),  # continuous (x, y) in [0, 1]
        **{k: torch.randint(0, 2, (B, T)) for k in ABILITY_KEYS}
    }

    z_pred_actions = model_actions(z_tau, tau, actions=actions)
    print(f"Output shape (with actions): {z_pred_actions.shape}")

    # Test without actions (should use no_action_embed)
    z_pred_no_actions = model_actions(z_tau, tau, actions=None)
    print(f"Output shape (no actions): {z_pred_no_actions.shape}")

    print("\n--- Testing with agent tokens ---")
    model_agent = create_dynamics("small", latent_dim=32, use_agent_tokens=True)
    print(f"Parameters (with agent tokens): {model_agent.get_num_params():,}")

    z_pred_agent, agent_out = model_agent(z_tau, tau)
    print(f"Output z shape: {z_pred_agent.shape}")
    print(f"Agent output shape: {agent_out.shape}")

    print("\n--- Testing with both actions and agent tokens ---")
    model_both = create_dynamics("small", latent_dim=32, use_actions=True, use_agent_tokens=True)
    print(f"Parameters (both): {model_both.get_num_params():,}")

    z_pred_both, agent_both = model_both(z_tau, tau, actions=actions)
    print(f"Output z shape: {z_pred_both.shape}")
    print(f"Agent output shape: {agent_both.shape}")

    print("\n--- Testing with no register tokens ---")
    model_no_reg = create_dynamics("small", latent_dim=32, num_register_tokens=0)
    print(f"Parameters (no registers): {model_no_reg.get_num_params():,}")
    z_pred_no_reg = model_no_reg(z_tau, tau)
    print(f"Output shape (no registers): {z_pred_no_reg.shape}")

    print("\nAll tests passed!")
