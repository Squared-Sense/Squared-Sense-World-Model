"""Siamese 3D ConvNeXt encoder with Squeeze-and-Excitation (SE) and CoordConv3D.

Strictly follows docs/MODEL_CONTRACT.md using channel-last layout (..., X, Y, Z, C).
"""

from __future__ import annotations

from typing import Sequence, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def count_params(module: nnx.Module) -> int:
    """Count total trainable parameters in an NNX module.

    Sums the .size over jax.tree_util.tree_leaves of nnx.state(module, nnx.Param).
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(module, nnx.Param)))


class CoordConv3D(nnx.Module):
    """Appends normalized 3D coordinates [-1, 1] along the channel dimension.

    Creates a constant (non-trainable) (1, 16, 16, 16, 3) float32 meshgrid of
    normalized (u, v, w) in [-1, 1] coordinates and concatenates along axis=-1:
    (N, 16, 16, 16, 16) -> (N, 16, 16, 16, 19). Stored as a plain JAX array.
    """

    def __init__(self, size: int = 16) -> None:
        coords = jnp.linspace(-1.0, 1.0, size, dtype=jnp.float32)
        grid_x, grid_y, grid_z = jnp.meshgrid(coords, coords, coords, indexing="ij")
        grid = jnp.stack([grid_x, grid_y, grid_z], axis=-1)  # (size, size, size, 3)
        self.grid = jnp.expand_dims(grid, axis=0)  # Plain JAX array, NOT nnx.Param

    def __call__(self, x: jax.Array) -> jax.Array:
        """Concatenate normalized coordinates to input voxel tensor.

        Args:
            x: Input tensor of shape (N, 16, 16, 16, C).

        Returns:
            Tensor of shape (N, 16, 16, 16, C + 3).
        """
        batch_size = x.shape[0]
        grid = jnp.broadcast_to(self.grid, (batch_size, *self.grid.shape[1:]))
        return jnp.concatenate([x, grid], axis=-1)


class ConvNeXt3DSEBlock(nnx.Module):
    """ConvNeXt 3D block with Squeeze-and-Excitation (SE) channel attention.

    Architecture:
      1. Depthwise 3D Conv (3x3x3, feature_group_count=dim)
      2. LayerNorm
      3. Pointwise MLP: Linear(dim, dim * expand) -> GELU -> Linear(dim * expand, dim)
      4. Squeeze-and-Excitation (SE): Global avg pool -> Linear(dim, dim // se_reduction)
         -> GELU -> Linear(dim // se_reduction, dim) -> Sigmoid -> scale
      5. Residual skip connection: return x + h
    """

    def __init__(
        self,
        dim: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        expand: int = 4,
        se_reduction: int = 4,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        k = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.dwconv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=k,
            padding="SAME",
            feature_group_count=dim,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp1 = nnx.Linear(dim, dim * expand, rngs=rngs)
        self.mlp2 = nnx.Linear(dim * expand, dim, rngs=rngs)
        self.se_fc1 = nnx.Linear(dim, dim // se_reduction, rngs=rngs)
        self.se_fc2 = nnx.Linear(dim // se_reduction, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.mlp1(h)
        h = jax.nn.gelu(h)
        h = self.mlp2(h)

        # Squeeze-and-Excitation
        se = jnp.mean(h, axis=(1, 2, 3), keepdims=True)
        se = self.se_fc1(se)
        se = jax.nn.gelu(se)
        se = self.se_fc2(se)
        se = jax.nn.sigmoid(se)
        h = h * se

        return x + h


class SiameseVoxelCNN(nnx.Module):
    """Siamese 3D ConvNeXt encoder for concentric multi-LOD voxel shells.

    Encodes each shell from (N, 16, 16, 16, 16) into 64 tokens of dimension 256.
    Across 3 concentric LOD shells: (B, 3, 16, 16, 16, 16) -> (B, 192, 256).
    """

    def __init__(
        self,
        in_channels: int = 16,
        embed_dim: int = 256,
        spatial_size: int = 16,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.spatial_size = spatial_size
        self.grid_tokens = (spatial_size // 4) ** 3
        self.coord_conv = CoordConv3D(size=spatial_size)

        # Stem: (16 + 3 = 19) -> 64, stride 2 -> (N, 8, 8, 8, 64)
        self.stem_conv = nnx.Conv(
            in_features=in_channels + 3,
            out_features=64,
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding="VALID",
            rngs=rngs,
        )
        self.stem_norm = nnx.LayerNorm(64, rngs=rngs)

        # Stage 1: 2 x ConvNeXt3DSEBlock(64) -> (N, 8, 8, 8, 64)
        self.stage1_block0 = ConvNeXt3DSEBlock(64, rngs=rngs)
        self.stage1_block1 = ConvNeXt3DSEBlock(64, rngs=rngs)

        # Downsample: 64 -> 128, stride 2 -> (N, 4, 4, 4, 128)
        self.downsample_conv = nnx.Conv(
            in_features=64,
            out_features=128,
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding="VALID",
            rngs=rngs,
        )
        self.downsample_norm = nnx.LayerNorm(128, rngs=rngs)

        # Stage 2: 2 x ConvNeXt3DSEBlock(128) -> (N, 4, 4, 4, 128)
        self.stage2_block0 = ConvNeXt3DSEBlock(128, rngs=rngs)
        self.stage2_block1 = ConvNeXt3DSEBlock(128, rngs=rngs)

        # Projection: 128 -> 256 -> (N, 4, 4, 4, 256) -> flatten to (N, 64, 256)
        self.proj = nnx.Conv(
            in_features=128,
            out_features=embed_dim,
            kernel_size=(1, 1, 1),
            rngs=rngs,
        )

    def encode_single_shell(self, shell: jax.Array) -> jax.Array:
        """Encode a batch of individual voxel shells.

        Args:
            shell: Tensor of shape (N, 16, 16, 16, 16).

        Returns:
            Tokens of shape (N, 64, 256).
        """
        h = self.coord_conv(shell)
        h = self.stem_conv(h)
        h = self.stem_norm(h)
        h = self.stage1_block0(h)
        h = self.stage1_block1(h)
        h = self.downsample_conv(h)
        h = self.downsample_norm(h)
        h = self.stage2_block0(h)
        h = self.stage2_block1(h)
        h = self.proj(h)
        batch_size = h.shape[0]
        return h.reshape((batch_size, self.grid_tokens, h.shape[-1]))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode batch of multi-LOD concentric voxel shells.

        Args:
            x: Tensor of shape (B, 3, 16, 16, 16, 16).

        Returns:
            Tokens of shape (B, 192, 256).
        """
        b, lod, sx, sy, sz, c = x.shape
        x_flat = x.reshape((b * lod, sx, sy, sz, c))
        tokens = self.encode_single_shell(x_flat)
        return tokens.reshape((b, lod * self.grid_tokens, tokens.shape[-1]))

