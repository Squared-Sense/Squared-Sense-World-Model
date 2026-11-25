"""Dual-LSTM temporal integrator with ego-motion 3D shifting and query cross-attention.

Strictly follows docs/MODEL_CONTRACT.md using channel-last (..., X, Y, Z, C) conventions.
"""

from __future__ import annotations

from typing import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as ndimage


def ego_motion_shift_3d(
    tensor_5d: jax.Array,
    delta_pos: jax.Array,
    scale: float,
) -> jax.Array:
    """Apply ego-motion compensatory coordinate shift to 3D spatial memory.

    Shifts coordinates by delta_pos / (8.0 * scale) using trilinear interpolation
    (order=1) with constant 0.0 boundary padding. Fully differentiable and JIT-compatible.

    Args:
        tensor_5d: (B, 2, 2, 2, C) channel-last tensor for one LOD shell.
        delta_pos: (B, 3) agent displacement (dx, dy, dz) from state[:, 3:6].
        scale: LOD geometric scale in (1.0, 2.0, 4.0).

    Returns:
        Shifted tensor of identical shape (B, 2, 2, 2, C).
    """
    gx, gy, gz = jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij")
    base_coords = jnp.stack([gx, gy, gz], axis=0).astype(jnp.float32)  # (3, 2, 2, 2)

    def shift_vol(vol_3d: jax.Array, delta: jax.Array) -> jax.Array:
        # Physical displacement shifts coordinates by delta / (8.0 * scale)
        shift = delta / (8.0 * scale)
        coords = base_coords + shift[:, None, None, None]
        return ndimage.map_coordinates(
            vol_3d, coords, order=1, mode="constant", cval=0.0
        )

    # Vectorize across channel dimension (axis=-1) then batch dimension (axis=0)
    shift_sample = lambda sample_4d, d: jax.vmap(
        lambda v: shift_vol(v, d), in_axes=-1, out_axes=-1
    )(sample_4d)
    return jax.vmap(shift_sample)(tensor_5d, delta_pos)


class LSTMCellNNX(nnx.Module):
    """Standard Long Short-Term Memory (LSTM) cell in Flax NNX."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gates = nnx.Linear(
            in_features + hidden_features, 4 * hidden_features, rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        state: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        """Update LSTM cell state: (x, (h, c)) -> (h_new, c_new)."""
        h, c = state
        combined = jnp.concatenate([x, h], axis=-1)
        gates = self.gates(combined)
        h_dim = self.hidden_features

        i = jax.nn.sigmoid(gates[..., :h_dim])
        f = jax.nn.sigmoid(gates[..., h_dim : 2 * h_dim])
        g = jnp.tanh(gates[..., 2 * h_dim : 3 * h_dim])
        o = jax.nn.sigmoid(gates[..., 3 * h_dim :])

        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return h_new, c_new


class QueryCrossAttentionBlock(nnx.Module):
    """Pre-LN Cross-Attention block querying spatial memory with state query token."""

    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm_mlp = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp1 = nnx.Linear(dim, dim * mlp_ratio, rngs=rngs)
        self.mlp2 = nnx.Linear(dim * mlp_ratio, dim, rngs=rngs)

    def __call__(self, q: jax.Array, kv: jax.Array) -> jax.Array:
        """Cross-attention: queries q (B, 1, D) over keys/values kv (B, 24, D)."""
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn = self.mha(inputs_q=q_norm, inputs_k=kv_norm, inputs_v=kv_norm)
        q = q + attn

        mlp_out = self.mlp2(jax.nn.gelu(self.mlp1(self.norm_mlp(q))))
        q = q + mlp_out
        return q


class DualLSTMTemporalIntegrator(nnx.Module):
    """Dual-LSTM temporal model fusing 3D spatial memory with proprioceptive state.

    Components:
      - 24 coarse spatial slots pooled from (B, 192, 256) tokens
      - Shared spatial_lstm with ego-motion shift compensation
      - State proprioceptive LSTM producing query token
      - Pre-LN Query Cross-Attention fusing state with spatial memory
      - Action-conditioned future predictor predicting next pooled latent z_hat_next
      - Auxiliary Inverse Dynamics Model (IDM) head predicting action from transitions
    """

    def __init__(
        self,
        token_dim: int = 256,
        spatial_hidden: int = 192,
        state_dim: int = 12,
        state_hidden: int = 64,
        num_actions: int = 12,
        num_heads: int = 6,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.token_dim = token_dim
        self.spatial_hidden = spatial_hidden
        self.state_dim = state_dim
        self.state_hidden = state_hidden
        self.num_actions = num_actions

        # 1. Spatial and state LSTMs
        self.spatial_lstm = LSTMCellNNX(token_dim, spatial_hidden, rngs=rngs)
        self.state_lstm = LSTMCellNNX(state_dim, state_hidden, rngs=rngs)
        self.state_proj = nnx.Linear(state_hidden, spatial_hidden, rngs=rngs)

        # 2. Query Cross-Attention block
        self.cross_attn = QueryCrossAttentionBlock(
            dim=spatial_hidden, num_heads=num_heads, mlp_ratio=2, rngs=rngs
        )

        # 3. Action-conditioned future predictor
        self.action_embed = nnx.Embed(num_actions, 64, rngs=rngs)
        self.pred_mlp1 = nnx.Linear(spatial_hidden + 64, spatial_hidden, rngs=rngs)
        self.pred_mlp2 = nnx.Linear(spatial_hidden, token_dim, rngs=rngs)

        # 4. Auxiliary Inverse Dynamics Model (IDM) head: [z_t, z_{t+1}] (512) -> logits (12)
        self.idm_mlp1 = nnx.Linear(token_dim * 2, 64, rngs=rngs)
        self.idm_mlp2 = nnx.Linear(64, num_actions, rngs=rngs)

    def init_hidden(
        self, batch_size: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Initialize hidden states for spatial memory and state LSTM."""
        h_sp = jnp.zeros(
            (batch_size, 3, 2, 2, 2, self.spatial_hidden), dtype=jnp.float32
        )
        c_sp = jnp.zeros(
            (batch_size, 3, 2, 2, 2, self.spatial_hidden), dtype=jnp.float32
        )
        h_st = jnp.zeros((batch_size, self.state_hidden), dtype=jnp.float32)
        c_st = jnp.zeros((batch_size, self.state_hidden), dtype=jnp.float32)
        return (h_sp, c_sp, h_st, c_st)

    def pool_spatial(self, tokens: jax.Array) -> jax.Array:
        """Average-pool 2x2x2 blocks from (B, 192, 256) to 24 coarse slots (B, 24, 256)."""
        b = tokens.shape[0]
        if tokens.ndim == 3 and tokens.shape[1] == 192:
            x = tokens.reshape((b, 3, 4, 4, 4, self.token_dim))
            x_blocks = x.reshape((b, 3, 2, 2, 2, 2, 2, 2, self.token_dim))
            return jnp.mean(x_blocks, axis=(3, 5, 7)).reshape((b, 24, self.token_dim))
        elif tokens.ndim == 2:
            # Latent vector broadcast to slots during autoregressive rollout
            return jnp.broadcast_to(tokens[:, None, :], (b, 24, self.token_dim))
        else:
            return tokens.reshape((b, 24, self.token_dim))

    def step(
        self,
        tokens_t: jax.Array,
        state_t: jax.Array,
        action_t: jax.Array,
        hidden: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        disable_ego_shift: bool = False,
    ) -> Tuple[jax.Array, jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        """Run single temporal integration step.

        Args:
            tokens_t: (B, 192, 256) spatial tokens or (B, 256) pooled latent.
            state_t: (B, 12) proprioceptive state.
            action_t: (B,) discrete action index.
            hidden: (h_sp, c_sp, h_st, c_st) hidden state tuple.

        Returns:
            Tuple of:
              - z_hat_next: (B, 256) predicted next pooled latent
              - c_t: (B, 192) fused context representation
              - next_hidden: updated hidden state tuple
        """
        h_sp, c_sp, h_st, c_st = hidden
        b = tokens_t.shape[0]
        delta_pos = state_t[:, 3:6]

        # 1. Ego-motion shifting across LOD scales (1.0, 2.0, 4.0)
        if disable_ego_shift:
            h_sp_shifted = h_sp
            c_sp_shifted = c_sp
        else:
            scales = (1.0, 2.0, 4.0)
            h_shifts = [ego_motion_shift_3d(h_sp[:, l], delta_pos, scales[l]) for l in range(3)]
            c_shifts = [ego_motion_shift_3d(c_sp[:, l], delta_pos, scales[l]) for l in range(3)]
            h_sp_shifted = jnp.stack(h_shifts, axis=1)
            c_sp_shifted = jnp.stack(c_shifts, axis=1)

        # 2. Update spatial memory with pooled slots
        slots_flat = self.pool_spatial(tokens_t)  # (B, 24, 256)
        h_flat = h_sp_shifted.reshape((b, 24, self.spatial_hidden))
        c_flat = c_sp_shifted.reshape((b, 24, self.spatial_hidden))

        h_next_flat, c_next_flat = self.spatial_lstm(slots_flat, (h_flat, c_flat))
        h_sp_next = h_next_flat.reshape((b, 3, 2, 2, 2, self.spatial_hidden))
        c_sp_next = c_next_flat.reshape((b, 3, 2, 2, 2, self.spatial_hidden))

        # 3. Update proprioceptive state LSTM
        h_st_next, c_st_next = self.state_lstm(state_t, (h_st, c_st))
        q = self.state_proj(h_st_next)[:, None, :]  # (B, 1, 192)

        # 4. Cross-attention fusing state query with spatial memory
        q_fused = self.cross_attn(q, h_next_flat)
        c_t = q_fused[:, 0, :]  # (B, 192)

        # 5. Predict next-step latent conditioned on action
        act_emb = self.action_embed(action_t)  # (B, 64)
        cat = jnp.concatenate([c_t, act_emb], axis=-1)  # (B, 256)
        z_hat_next = self.pred_mlp2(jax.nn.gelu(self.pred_mlp1(cat)))

        next_hidden = (h_sp_next, c_sp_next, h_st_next, c_st_next)
        return z_hat_next, c_t, next_hidden

    def rollout(
        self,
        tokens_seq: jax.Array,
        state_seq: jax.Array,
        action_seq: jax.Array,
        k_steps: int = 4,
        disable_ego_shift: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """Unroll sequence over trajectory and perform k_steps autoregressive predictions.

        Args:
            tokens_seq: (B, T, 192, 256) sequence of voxel shell tokens.
            state_seq: (B, T, 12) sequence of states.
            action_seq: (B, T) sequence of actions.
            k_steps: Number of forward prediction rollout steps.

        Returns:
            Tuple of:
              - pred_future: (B, T_valid, k_steps, 256) future rollout predictions
              - idm_logits: (B, T - 1, num_actions) inverse dynamics logits
        """
        b, T, _, _ = tokens_seq.shape
        hidden = self.init_hidden(b)

        # 1. Step through observed trajectory
        z_hat_list = []
        c_list = []
        hidden_list = []
        for t in range(T):
            z_hat, c_t, hidden = self.step(
                tokens_seq[:, t],
                state_seq[:, t],
                action_seq[:, t],
                hidden,
                disable_ego_shift=disable_ego_shift,
            )
            z_hat_list.append(z_hat)
            c_list.append(c_t)
            hidden_list.append(hidden)

        # 2. Compute IDM logits on consecutive pooled latents
        z_pooled = jnp.mean(tokens_seq, axis=2)  # (B, T, 256)
        idm_list = []
        for t in range(T - 1):
            pair = jnp.concatenate([z_pooled[:, t], z_pooled[:, t + 1]], axis=-1)
            logits = self.idm_mlp2(jax.nn.gelu(self.idm_mlp1(pair)))
            idm_list.append(logits)
        idm_logits = jnp.stack(idm_list, axis=1)  # (B, T-1, num_actions)

        # 3. Autoregressive k_steps rollouts
        T_valid = T - k_steps + 1
        rollouts = []
        for t in range(T_valid):
            h_roll = hidden_list[t]
            cur_z = z_hat_list[t]
            k_preds = [cur_z]
            for step_k in range(1, k_steps):
                act_k = action_seq[:, t + step_k]
                st_k = state_seq[:, t + step_k]
                cur_z, _, h_roll = self.step(
                    cur_z, st_k, act_k, h_roll, disable_ego_shift=disable_ego_shift
                )
                k_preds.append(cur_z)
            rollouts.append(jnp.stack(k_preds, axis=1))

        pred_future = jnp.stack(rollouts, axis=1)  # (B, T_valid, k_steps, 256)
        return pred_future, idm_logits

