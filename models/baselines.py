"""Baseline models for Squared Sense: Single-LOD JEPA and Generative 3D-MAE.

Strictly follows docs/MODEL_CONTRACT.md using channel-last (..., X, Y, Z, C) conventions.
"""

from __future__ import annotations

from typing import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from models.cnn3d import count_params
from models.transformer import (
    JEPAPredictor,
    MultiLODBackbone,
    PreLNTransformerBlock,
    SpatialJEPA,
)


class SingleLODJEPA(nnx.Module):
    """Single-LOD JEPA Baseline (Baseline A).

    Restricts representation learning to a single LOD shell (e.g. LOD-0 or LOD-2),
    yielding 64 tokens instead of 192.
    Wraps SpatialJEPA(num_shells=1) so parameter count matches Stage 1A (~4,648,672 params).
    """

    def __init__(self, lod_index: int = 0, *, rngs: nnx.Rngs) -> None:
        self.lod_index = lod_index
        self.jepa = SpatialJEPA(num_shells=1, rngs=rngs)

    @property
    def context_encoder(self) -> MultiLODBackbone:
        return self.jepa.context_encoder

    @property
    def target_encoder(self) -> MultiLODBackbone:
        return self.jepa.target_encoder

    @property
    def predictor(self) -> JEPAPredictor:
        return self.jepa.predictor

    def update_ema(self, momentum: float) -> None:
        """Update target encoder parameters in-place via EMA."""
        self.jepa.update_ema(momentum)

    def count_trainable_params(self) -> int:
        """Count trainable parameters in context encoder and predictor."""
        return count_params(self.context_encoder) + count_params(self.predictor)

    def __call__(
        self,
        x: jax.Array,
        visible_idx: jax.Array,
        target_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass on sliced single LOD shell.

        Args:
            x: (B, 3, 16, 16, 16, 16) or (B, 1, 16, 16, 16, 16) or (B, 16, 16, 16, 16).
            visible_idx: (B, N_vis) integer indices (e.g. 32 visible tokens).
            target_idx: (B, N_tgt) integer indices (e.g. 32 target tokens).

        Returns:
            Tuple of:
              - pred_targets: (B, N_tgt, 256) predictions
              - norm_ema_targets: (B, N_tgt, 256) normalized EMA targets
              - context_tokens: (B, N_vis, 256) visible context representations
        """
        if x.ndim == 6:
            x_single = x[:, self.lod_index : self.lod_index + 1]
        elif x.ndim == 5:
            x_single = jnp.expand_dims(x, axis=1)
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got shape {x.shape}")

        return self.jepa(x_single, visible_idx, target_idx)


class Generative3DMAE(nnx.Module):
    """Generative 3D Masked Autoencoder (3D-MAE) baseline (Baseline B).

    Reuses MultiLODBackbone(num_shells=3) encoder (~3.647M params) and adds
    a lightweight 3D voxel decoder (~1.150M - 1.478M params) reconstructing
    masked 4x4x4x16 voxel patches.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        dec_dim: int = 192,
        dec_depth: int = 2,
        dec_heads: int = 6,
        num_shells: int = 3,
        patch_size: int = 4,
        channels: int = 16,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.embed_dim = embed_dim
        self.dec_dim = dec_dim
        self.num_shells = num_shells
        self.patch_size = patch_size
        self.channels = channels
        self.patch_dim = (patch_size**3) * channels  # 4 * 4 * 4 * 16 = 1024

        # 1. Multi-LOD Backbone Encoder (num_shells=3)
        self.encoder = MultiLODBackbone(
            dim=embed_dim, depth=4, num_heads=8, num_shells=num_shells, rngs=rngs
        )

        # 2. Lightweight 3D voxel decoder
        self.decoder_input_proj = nnx.Linear(embed_dim, dec_dim, rngs=rngs)
        self.mask_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, dec_dim)) * 0.02
        )
        self.decoder_spatial_pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, 64, dec_dim)) * 0.02
        )
        self.decoder_scale_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_shells, 1, dec_dim)) * 0.02
        )

        self.decoder_blocks = nnx.List(
            [
                PreLNTransformerBlock(
                    dim=dec_dim,
                    num_heads=dec_heads,
                    mlp_ratio=4,
                    rngs=rngs,
                )
                for _ in range(dec_depth)
            ]
        )
        self.decoder_norm = nnx.LayerNorm(dec_dim, rngs=rngs)
        self.pred_head = nnx.Linear(dec_dim, self.patch_dim, rngs=rngs)

    def patchify(self, x: jax.Array) -> jax.Array:
        """Convert (B, num_shells, 16, 16, 16, 16) into (B, num_shells * 64, 1024)."""
        b = x.shape[0]
        # Reshape: (B, 3, 4, 4, 4, 4, 4, 4, 16)
        x_reshaped = x.reshape(
            (b, self.num_shells, self.patch_size, self.patch_size, self.patch_size, self.patch_size, self.patch_size, self.patch_size, self.channels)
        )
        # Transpose to (B, shell, gx, gy, gz, px, py, pz, C)
        x_trans = x_reshaped.transpose((0, 1, 2, 4, 6, 3, 5, 7, 8))
        return x_trans.reshape((b, self.num_shells * 64, self.patch_dim))

    def __call__(
        self,
        x: jax.Array,
        visible_idx: jax.Array,
        target_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Encode visible tokens, decode masked targets, and compute patch MSE loss.

        Args:
            x: (B, 3, 16, 16, 16, 16) input voxel shells.
            visible_idx: (B, N_vis) integer indices of visible tokens.
            target_idx: (B, N_tgt) integer indices of masked target tokens.

        Returns:
            Tuple of:
              - pred_patches: (B, N_tgt, 1024) reconstructed voxel patches
              - target_patches: (B, N_tgt, 1024) ground-truth voxel patches
              - mse_loss: scalar MSE loss strictly over masked target patches
        """
        # 1. Encode ONLY visible tokens
        context_tokens = self.encoder(x, visible_idx=visible_idx)  # (B, N_vis, 256)

        # 2. Project to decoder dimension
        ctx = self.decoder_input_proj(context_tokens)  # (B, N_vis, 192)

        # 3. Add decoder positional + scale embeddings
        spatial = self.decoder_spatial_pos_embed[...].reshape((1, 1, 64, self.dec_dim))
        scale = self.decoder_scale_embed[...]
        full_dec_pos = (spatial + scale).reshape((1, self.num_shells * 64, self.dec_dim))

        ctx_pos = jnp.take_along_axis(
            full_dec_pos,
            jnp.expand_dims(visible_idx, axis=-1),
            axis=1,
        )
        ctx = ctx + ctx_pos

        # 4. Create masked query tokens at target positions
        batch_size, n_tgt = target_idx.shape
        mask_tokens = jnp.broadcast_to(
            self.mask_token[...],
            (batch_size, n_tgt, self.dec_dim),
        )
        tgt_pos = jnp.take_along_axis(
            full_dec_pos,
            jnp.expand_dims(target_idx, axis=-1),
            axis=1,
        )
        tgt_queries = mask_tokens + tgt_pos

        # 5. Full sequence in decoder: visible context + masked query tokens
        seq = jnp.concatenate([ctx, tgt_queries], axis=1)  # (B, N_vis + N_tgt, 192)
        for block in self.decoder_blocks:
            seq = block(seq)
        seq = self.decoder_norm(seq)

        # 6. Slice target positions and reconstruct patches
        n_vis = visible_idx.shape[1]
        pred_tokens = seq[:, n_vis:, :]
        pred_patches = self.pred_head(pred_tokens)  # (B, N_tgt, 1024)

        # 7. Extract ground-truth target patches and compute MSE loss
        all_patches = self.patchify(x)  # (B, 192, 1024)
        target_patches = jnp.take_along_axis(
            all_patches,
            jnp.expand_dims(target_idx, axis=-1),
            axis=1,
        )  # (B, N_tgt, 1024)

        mse_loss = jnp.mean(jnp.square(pred_patches - target_patches))

        return pred_patches, target_patches, mse_loss


def _get_block_tokens_32() -> jax.Array:
    """Precompute the 64 disjoint 2x2x2 block token partitions of an 8x8x8 grid."""
    bx, by, bz = jnp.meshgrid(jnp.arange(4), jnp.arange(4), jnp.arange(4), indexing="ij")
    bx, by, bz = bx.reshape(-1), by.reshape(-1), bz.reshape(-1)  # (64,)
    dx, dy, dz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij")
    dx, dy, dz = dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)  # (8,)
    x = bx[:, None] * 2 + dx[None, :]
    y = by[:, None] * 2 + dy[None, :]
    z = bz[:, None] * 2 + dz[None, :]
    return (x * 64 + y * 8 + z).astype(jnp.int32)  # shape (64, 8)


_BLOCK_TOKENS_32: jax.Array = _get_block_tokens_32()


def sample_3d_jepa_masks_32(
    key: jax.Array,
    batch_size: int,
) -> Tuple[jax.Array, jax.Array]:
    """Pure JAX 3D block masking over an 8x8x8 token grid (64 blocks of 8 tokens).

    Selects 32 blocks (256 tokens) as target and 32 blocks (256 tokens) as visible context.
    """
    def sample_single(k: jax.Array) -> Tuple[jax.Array, jax.Array]:
        perm = jax.random.permutation(k, 64)
        tgt_blocks = perm[:32]
        vis_blocks = perm[32:]
        tgt_tokens = _BLOCK_TOKENS_32[tgt_blocks].reshape(-1)
        vis_tokens = _BLOCK_TOKENS_32[vis_blocks].reshape(-1)
        return vis_tokens, tgt_tokens

    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample_single)(keys)


class Uniform32JEPA(nnx.Module):
    """Uniform 32^3 Spatial JEPA Baseline (Baseline B4).

    Operates on a single uniform 1x-resolution 32x32x32x16 voxel grid (matching the
    32^3 bounding volume of LOD-1 without 2x strided downsampling), producing
    8x8x8 = 512 spatial tokens of dimension 256.
    """

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.jepa = SpatialJEPA(num_shells=1, spatial_size=32, rngs=rngs)

    @property
    def context_encoder(self) -> MultiLODBackbone:
        return self.jepa.context_encoder

    @property
    def target_encoder(self) -> MultiLODBackbone:
        return self.jepa.target_encoder

    @property
    def predictor(self) -> JEPAPredictor:
        return self.jepa.predictor

    def update_ema(self, momentum: float) -> None:
        """Update target encoder parameters in-place via EMA."""
        self.jepa.update_ema(momentum)

    def count_trainable_params(self) -> int:
        """Count trainable parameters in context encoder and predictor."""
        return count_params(self.context_encoder) + count_params(self.predictor)

    def __call__(
        self,
        x: jax.Array,
        visible_idx: jax.Array,
        target_idx: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass on uniform 32x32x32x16 voxel grid.

        Args:
            x: (B, 32, 32, 32, 16) or (B, 1, 32, 32, 32, 16) float32 tensor.
            visible_idx: (B, 256) visible token indices.
            target_idx: (B, 256) target token indices.
        """
        if x.ndim == 5:
            x_in = jnp.expand_dims(x, axis=1)
        elif x.ndim == 6:
            x_in = x
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got shape {x.shape}")

        return self.jepa(x_in, visible_idx, target_idx)

