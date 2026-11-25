"""Transformer encoder, EMA target encoder, JEPA predictor, and 3D masking.

Strictly follows docs/MODEL_CONTRACT.md using JAX and Flax NNX.
"""

from __future__ import annotations

from typing import Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from models.cnn3d import SiameseVoxelCNN, count_params


class PreLNTransformerBlock(nnx.Module):
    """Pre-LayerNorm Transformer Block with Multi-Head Self-Attention and MLP.

    Architecture:
      1. Pre-LN Self-Attention: x = x + MHA(LN(x))
      2. Pre-LN MLP: x = x + Linear(GELU(Linear(LN(x))))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.linear1 = nnx.Linear(dim, dim * mlp_ratio, rngs=rngs)
        self.linear2 = nnx.Linear(dim * mlp_ratio, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.norm1(x)
        h = self.mha(h)
        x = x + h

        h = self.norm2(x)
        h = self.linear1(h)
        h = jax.nn.gelu(h)
        h = self.linear2(h)
        x = x + h

        return x


class MultiLODBackbone(nnx.Module):
    """Multi-LOD Concentric Shell Transformer Encoder Backbone.

    Processes 3 concentric voxel shells with Siamese 3D ConvNeXt, adds tiled
    spatial and scale positional embeddings, optionally gathers visible tokens,
    and processes them through 4 Pre-LN Transformer blocks.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        num_shells: int = 3,
        spatial_size: int = 16,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.num_shells = num_shells
        self.spatial_size = spatial_size
        self.grid_tokens = (spatial_size // 4) ** 3
        self.cnn = SiameseVoxelCNN(embed_dim=dim, spatial_size=spatial_size, rngs=rngs)

        # Positional and scale embeddings
        self.spatial_pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, self.grid_tokens, dim)) * 0.02
        )
        self.scale_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_shells, 1, dim)) * 0.02
        )

        # Transformer blocks and final normalization
        self.blocks = nnx.List(
            [
                PreLNTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

    def encode_tokens(
        self,
        tokens: jax.Array,
        visible_idx: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Add positional + scale embeddings and process visible tokens.

        Args:
            tokens: (B, num_shells * 64, dim) voxel tokens from 3D CNN.
            visible_idx: Optional (B, N_vis) integer indices of unmasked tokens.

        Returns:
            Encoded tokens (B, N_vis, dim) or (B, num_shells * 64, dim).
        """
        # Combine spatial (1, grid_tokens, dim) and scale (1, num_shells, 1, dim) embeddings
        spatial = self.spatial_pos_embed[...].reshape((1, 1, self.grid_tokens, self.dim))
        scale = self.scale_embed[...]
        full_pos = (spatial + scale).reshape((1, self.num_shells * self.grid_tokens, self.dim))
        tokens = tokens + full_pos

        if visible_idx is not None:
            # Gather ONLY visible tokens so context encoder never attends to targets
            tokens = jnp.take_along_axis(
                tokens,
                jnp.expand_dims(visible_idx, axis=-1),
                axis=1,
            )

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)

    def __call__(
        self,
        x: jax.Array,
        visible_idx: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Encode voxel shell inputs to spatial representation tokens.

        Args:
            x: (B, num_shells, 16, 16, 16, 16) or (B, 16, 16, 16, 16) float32.
            visible_idx: Optional (B, N_vis) integer indices.

        Returns:
            Encoded tokens tensor.
        """
        if x.ndim == 6:
            tokens = self.cnn(x)
        else:
            tokens = self.cnn.encode_single_shell(x)
        return self.encode_tokens(tokens, visible_idx=visible_idx)


class JEPAPredictor(nnx.Module):
    """Joint-Embedding Predictive Architecture (JEPA) Predictor.

    Takes visible context tokens and masked query tokens at target positions,
    and predicts target representation latents.
    """

    def __init__(
        self,
        enc_dim: int = 256,
        pred_dim: int = 192,
        depth: int = 2,
        num_heads: int = 6,
        num_shells: int = 3,
        grid_tokens: int = 64,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.num_shells = num_shells
        self.grid_tokens = grid_tokens

        self.input_proj = nnx.Linear(enc_dim, pred_dim, rngs=rngs)
        self.mask_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, pred_dim)) * 0.02
        )
        self.spatial_pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, grid_tokens, pred_dim)) * 0.02
        )
        self.scale_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_shells, 1, pred_dim)) * 0.02
        )

        self.blocks = nnx.List(
            [
                PreLNTransformerBlock(
                    dim=pred_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(pred_dim, rngs=rngs)
        self.output_proj = nnx.Linear(pred_dim, enc_dim, rngs=rngs)

    def __call__(
        self,
        context_tokens: jax.Array,
        visible_idx: jax.Array,
        target_idx: jax.Array,
    ) -> jax.Array:
        """Predict representations at masked target positions.

        Args:
            context_tokens: (B, N_vis, enc_dim) from context encoder.
            visible_idx: (B, N_vis) integer indices.
            target_idx: (B, N_tgt) integer indices.

        Returns:
            (B, N_tgt, enc_dim) predicted target tokens.
        """
        # Project context tokens
        ctx = self.input_proj(context_tokens)

        # Full positional embeddings for predictor
        spatial = self.spatial_pos_embed[...].reshape((1, 1, self.grid_tokens, self.pred_dim))
        scale = self.scale_embed[...]
        pred_pos = (spatial + scale).reshape((1, self.num_shells * self.grid_tokens, self.pred_dim))

        # Add predictor pos embeddings to context tokens
        ctx_pos = jnp.take_along_axis(
            pred_pos,
            jnp.expand_dims(visible_idx, axis=-1),
            axis=1,
        )
        ctx = ctx + ctx_pos

        # Create masked query tokens at target positions with their pos embeddings
        batch_size, n_tgt = target_idx.shape
        mask_tokens = jnp.broadcast_to(
            self.mask_token[...],
            (batch_size, n_tgt, self.pred_dim),
        )
        tgt_pos = jnp.take_along_axis(
            pred_pos,
            jnp.expand_dims(target_idx, axis=-1),
            axis=1,
        )
        tgt_queries = mask_tokens + tgt_pos

        # Concatenate along sequence axis: (B, N_vis + N_tgt, pred_dim)
        seq = jnp.concatenate([ctx, tgt_queries], axis=1)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)

        # Slice out target positions and project back to enc_dim
        n_vis = visible_idx.shape[1]
        pred_targets = seq[:, n_vis:, :]
        return self.output_proj(pred_targets)


def _get_block_tokens() -> jax.Array:
    """Precompute the 8 disjoint 2x2x2 block token partitions of a 4x4x4 grid."""
    bx, by, bz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij")
    bx, by, bz = bx.reshape(-1), by.reshape(-1), bz.reshape(-1)  # (8,)
    dx, dy, dz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij")
    dx, dy, dz = dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)  # (8,)
    x = bx[:, None] * 2 + dx[None, :]
    y = by[:, None] * 2 + dy[None, :]
    z = bz[:, None] * 2 + dz[None, :]
    return (x * 16 + y * 4 + z).astype(jnp.int32)  # shape (8, 8)


_BLOCK_TOKENS: jax.Array = _get_block_tokens()


def sample_3d_jepa_masks(
    key: jax.Array,
    batch_size: int,
    num_shells: int = 3,
    cross_scale_prob: float = 0.3,
) -> Tuple[jax.Array, jax.Array]:
    """Pure JAX (JIT-safe) 3D masking function returning fixed-shape token index arrays.

    Samples contiguous 2x2x2 blocks on the 4x4x4 grid (8 blocks of 8 tokens each).
    Selects 4 blocks per shell for targets (~50% = 32 per shell), leaving 4 blocks
    for visible context.

    For num_shells=3, with probability cross_scale_prob, masks blocks in LOD-0
    while keeping those same coordinates visible in LOD-1 and LOD-2.

    Args:
        key: JAX PRNG key.
        batch_size: Batch dimension B.
        num_shells: Number of concentric LOD shells (1 or 3).
        cross_scale_prob: Probability of cross-scale complementary masking.

    Returns:
        (visible_idx, target_idx) tuple of integer arrays with shapes:
          visible_idx: (B, 32 * num_shells)
          target_idx:  (B, 32 * num_shells)
    """

    def sample_single(k: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if num_shells == 3:
            k1, k2, k3 = jax.random.split(k, 3)
            is_cross = jax.random.uniform(k1) < cross_scale_prob

            # Independent block permutations per shell: shape (3, 8)
            shell_keys = jax.random.split(k2, 3)
            indep_perms = jax.vmap(lambda sk: jax.random.permutation(sk, 8))(shell_keys)

            # Cross-scale: LOD-0 uses p0[:4] as tgt; LOD-1 and LOD-2 use p0[4:] as tgt
            p0 = jax.random.permutation(k3, 8)
            cross_p1 = jnp.concatenate([p0[4:], p0[:4]])
            cross_perms = jnp.stack([p0, cross_p1, cross_p1], axis=0)

            perms = jax.lax.select(is_cross, cross_perms, indep_perms)
        else:
            shell_keys = jax.random.split(k, num_shells)
            perms = jax.vmap(lambda sk: jax.random.permutation(sk, 8))(shell_keys)

        shell_offsets = jnp.arange(num_shells)[:, None] * 64
        tgt_blocks = perms[:, :4]  # (num_shells, 4)
        vis_blocks = perms[:, 4:]  # (num_shells, 4)

        tgt_tokens = (_BLOCK_TOKENS[tgt_blocks].reshape(num_shells, 32) + shell_offsets).reshape(-1)
        vis_tokens = (_BLOCK_TOKENS[vis_blocks].reshape(num_shells, 32) + shell_offsets).reshape(-1)

        return vis_tokens, tgt_tokens

    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample_single)(keys)


class SpatialJEPA(nnx.Module):
    """Joint-Embedding Predictive Architecture for Multi-LOD Concentric Shells.

    Maintains:
      - context_encoder: MultiLODBackbone trained via gradients
      - target_encoder: MultiLODBackbone updated via Exponential Moving Average (EMA)
      - predictor: JEPAPredictor trained to predict target latents from context
    """

    def __init__(self, num_shells: int = 3, spatial_size: int = 16, *, rngs: nnx.Rngs) -> None:
        self.num_shells = num_shells
        self.spatial_size = spatial_size
        self.context_encoder = MultiLODBackbone(
            num_shells=num_shells, spatial_size=spatial_size, rngs=rngs
        )
        self.target_encoder = nnx.clone(self.context_encoder)
        self.predictor = JEPAPredictor(
            num_shells=num_shells, grid_tokens=(spatial_size // 4) ** 3, rngs=rngs
        )

    def update_ema(self, momentum: float) -> None:
        """Update target encoder parameters in-place via EMA: target = m * target + (1-m) * context."""
        tgt_state = nnx.state(self.target_encoder, nnx.Param)
        ctx_state = nnx.state(self.context_encoder, nnx.Param)
        new_tgt_state = jax.tree.map(
            lambda t, c: momentum * t + (1.0 - momentum) * c,
            tgt_state,
            ctx_state,
        )
        nnx.update(self.target_encoder, new_tgt_state)

    def __call__(
        self,
        x: jax.Array,
        visible_idx: jax.Array,
        target_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Compute JEPA predictions, normalized EMA targets, and context tokens.

        Args:
            x: (B, num_shells, 16, 16, 16, 16) voxel input.
            visible_idx: (B, N_vis) unmasked token indices.
            target_idx: (B, N_tgt) masked token indices to predict.

        Returns:
            Tuple of:
              - pred_targets: (B, N_tgt, 256) predictions from predictor
              - norm_ema_targets: (B, N_tgt, 256) per-token normalized target latents
              - context_tokens: (B, N_vis, 256) encoded visible representations
        """
        # Context encoding of unmasked tokens
        context_tokens = self.context_encoder(x, visible_idx=visible_idx)
        pred_targets = self.predictor(context_tokens, visible_idx, target_idx)

        # Target encoding (wrapped in stop_gradient so target encoder receives zero gradients)
        all_target_tokens = jax.lax.stop_gradient(self.target_encoder(x, visible_idx=None))
        ema_targets = jnp.take_along_axis(
            all_target_tokens,
            jnp.expand_dims(target_idx, axis=-1),
            axis=1,
        )

        # Per-token LayerNorm: (t - mean) / sqrt(var + 1e-6)
        mean = jnp.mean(ema_targets, axis=-1, keepdims=True)
        var = jnp.var(ema_targets, axis=-1, keepdims=True)
        norm_ema_targets = (ema_targets - mean) / jnp.sqrt(var + 1e-6)

        return pred_targets, norm_ema_targets, context_tokens

