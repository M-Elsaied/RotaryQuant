"""Fused Metal kernels for the IsoQuant decode pipeline.

The decode path is a three-kernel pipeline around a standard softmax:

  Kernel A: fused_qk_dot_3bit  — Q·Kᵀ from packed 3-bit KV (no K reconstruction)
            softmax            — mx.softmax over the scores
  Kernel C: fused_value_accum  — weighted V sum from packed storage (no V reconstruction)
  Kernel D: wht_so4_inverse    — structured inverse rotation on aggregated output (once)

The key insight: inverse rotation moves from O(T × d²) to O(d²) by accumulating
in rotated space and rotating the output once.

Usage:
    from .fused_kv_decode_kernels import (
        fused_qk_dot,
        fused_value_accum,
        fused_inverse_rotate,
        fused_isoquant_attention,
    )
"""

from __future__ import annotations

import mlx.core as mx

_MAX_KERNEL_HEAD_DIM = 512

# ---------------------------------------------------------------------------
# Metal shader sources
# ---------------------------------------------------------------------------

# Kernel A: Fused Q·Kᵀ — one threadgroup per token, 32 threads cooperate.
# Decodes 3-bit packed K on the fly, never materialises full K.
# Centroids in constant buffer for fast cache access.
_FUSED_QK_DOT_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint token_idx = threadgroup_position_in_grid.x;
    uint q_head_idx = threadgroup_position_in_grid.y;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint T_stride = storage_stride[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    // Cache query in threadgroup memory
    threadgroup float q_local[512];  // max head_dim
    for (uint i = lane; i < D; i += TG_SIZE) {
        q_local[i] = q[q_head_idx * D + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pointer to this token's packed K for this head
    const device uint8_t* packed_base = K_packed + (kv_head_idx * T_stride + token_idx) * PACKED_WORDS * 3;

    float acc = 0.0f;

    // Striped loop over dimensions (8 values per 3 bytes)
    for (uint w = lane; w < PACKED_WORDS; w += TG_SIZE) {
        // Read 3 bytes = 24 bits = 8 × 3-bit indices
        uint byte0 = packed_base[w * 3 + 0];
        uint byte1 = packed_base[w * 3 + 1];
        uint byte2 = packed_base[w * 3 + 2];
        uint word = byte0 | (byte1 << 8) | (byte2 << 16);

        uint base_dim = w * VALS_PER_WORD;

        // Unpack 8 indices and lookup centroids, accumulate dot product
        for (uint i = 0; i < VALS_PER_WORD; i++) {
            uint idx = (word >> (i * 3)) & 0x7;
            float k_val = centroids[idx];
            float q_val = q_local[base_dim + i];
            acc += k_val * q_val;
        }
    }

    // SIMD reduction
    acc = simd_sum(acc);

    if (lane == 0) {
        float norm_val = norms[(kv_head_idx * T_stride + token_idx)];
        scores[q_head_idx * T + token_idx] = acc * norm_val;
    }
"""

# Kernel C: Fused weighted value accumulation in ROTATED space.
# One threadgroup per head, all threads cooperate over d_k.
# Accumulates sum_j(a_j * dequant(V_packed[j])) without inverse rotation.
_FUSED_VALUE_ACCUM_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint q_head_idx = threadgroup_position_in_grid.x;
    uint kv_head_idx = kv_head_map[q_head_idx];

    uint D = head_dim[0];
    uint T = seq_len[0];
    uint T_stride = storage_stride[0];
    uint VALS_PER_WORD = 8;
    uint PACKED_WORDS = D / VALS_PER_WORD;
    uint TG_SIZE = threads_per_threadgroup.x;

    // Each thread accumulates partial sums for its assigned dimensions
    // We process all tokens, accumulating into thread-local registers
    for (uint d = lane; d < D; d += TG_SIZE) {
        float sum = 0.0f;
        uint w_idx = d / VALS_PER_WORD;
        uint bit_pos = d % VALS_PER_WORD;

        for (uint t = 0; t < T; t++) {
            float attn_w = attn_weights[q_head_idx * T + t];
            if (attn_w == 0.0f) continue;

            float norm_val = norms[kv_head_idx * T_stride + t];
            const device uint8_t* packed_base = V_packed + (kv_head_idx * T_stride + t) * PACKED_WORDS * 3;

            // Read the 3-byte word containing this dimension
            uint byte0 = packed_base[w_idx * 3 + 0];
            uint byte1 = packed_base[w_idx * 3 + 1];
            uint byte2 = packed_base[w_idx * 3 + 2];
            uint word = byte0 | (byte1 << 8) | (byte2 << 16);

            uint idx = (word >> (bit_pos * 3)) & 0x7;
            float v_val = centroids[idx] * norm_val;
            sum += attn_w * v_val;
        }

        output[q_head_idx * D + d] = sum;
    }
"""

# ---------------------------------------------------------------------------
# Kernel cache (lazy init)
# ---------------------------------------------------------------------------

_fused_kernel_cache: dict[str, any] = {}


def _get_fused_qk_kernel():
    key = "fused_qk_v2"
    if key not in _fused_kernel_cache:
        _fused_kernel_cache[key] = mx.fast.metal_kernel(
            name="fused_qk_dot_3bit",
            input_names=[
                "K_packed",
                "centroids",
                "norms",
                "q",
                "kv_head_map",
                "head_dim",
                "seq_len",
                "storage_stride",
            ],
            output_names=["scores"],
            source=_FUSED_QK_DOT_SOURCE,
        )
    return _fused_kernel_cache[key]


def _get_fused_value_kernel():
    key = "fused_val_v2"
    if key not in _fused_kernel_cache:
        _fused_kernel_cache[key] = mx.fast.metal_kernel(
            name="fused_value_accum_3bit",
            input_names=[
                "V_packed",
                "centroids",
                "norms",
                "attn_weights",
                "kv_head_map",
                "head_dim",
                "seq_len",
                "storage_stride",
            ],
            output_names=["output"],
            source=_FUSED_VALUE_ACCUM_SOURCE,
        )
    return _fused_kernel_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_indices_3bit(indices: mx.array) -> mx.array:
    """Pack uint8 quantisation indices (0-7) into 3-bit packed format.

    Input: (num_heads, seq_len, head_dim) uint8, values in [0, 7]
    Output: (num_heads, seq_len, packed_bytes) uint8

    Every 8 indices → 3 bytes (24 bits).
    """
    H, S, D = indices.shape
    assert D % 8 == 0, f"head_dim {D} must be divisible by 8"
    packed_words = D // 8

    # Reshape to groups of 8
    idx = indices.reshape(H, S, packed_words, 8).astype(mx.uint32)

    # Pack 8 × 3-bit values into 24 bits
    word = mx.zeros((H, S, packed_words), dtype=mx.uint32)
    for i in range(8):
        word = word | (idx[..., i] << (i * 3))

    # Split 24-bit word into 3 bytes
    byte0 = (word & 0xFF).astype(mx.uint8)
    byte1 = ((word >> 8) & 0xFF).astype(mx.uint8)
    byte2 = ((word >> 16) & 0xFF).astype(mx.uint8)

    # Interleave bytes: [b0, b1, b2, b0, b1, b2, ...]
    packed = mx.stack([byte0, byte1, byte2], axis=-1).reshape(H, S, packed_words * 3)
    return packed


def fused_qk_dot(
    K_packed: mx.array,
    centroids: mx.array,
    norms: mx.array,
    q: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    storage_stride: int | None = None,
) -> mx.array:
    """Compute attention scores Q·Kᵀ from 3-bit packed K without reconstruction.

    Args:
        K_packed: (num_heads, seq_len, packed_bytes) uint8 — 3-bit packed indices
        centroids: (num_centroids,) float32 — Lloyd-Max centroids (typically 8 for 3-bit)
        norms: (num_heads, seq_len) float32 — stored ||k|| norms
        q: (num_heads, head_dim) float32 — query vectors (one per head)
        num_heads, seq_len, head_dim: dimensions
        storage_stride: buffer stride along T axis (defaults to seq_len)

    Returns:
        scores: (num_heads, seq_len) float32 — attention scores
    """
    if head_dim > _MAX_KERNEL_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds kernel maximum ({_MAX_KERNEL_HEAD_DIM})"
        )
    kernel = _get_fused_qk_kernel()
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ss = mx.array(
        [storage_stride if storage_stride is not None else seq_len], dtype=mx.uint32
    )

    (scores,) = kernel(
        inputs=[
            K_packed.reshape(-1),
            centroids.reshape(-1),
            norms.reshape(-1),
            q.reshape(-1),
            kv_head_map.reshape(-1),
            hd,
            sl,
            ss,
        ],
        output_shapes=[(num_heads * seq_len,)],
        output_dtypes=[mx.float32],
        grid=(32 * seq_len, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return scores.reshape(num_heads, seq_len)


def fused_value_accum(
    V_packed: mx.array,
    centroids: mx.array,
    norms: mx.array,
    attn_weights: mx.array,
    kv_head_map: mx.array,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    storage_stride: int | None = None,
) -> mx.array:
    """Compute attention-weighted value sum from 3-bit packed V in rotated space.

    Accumulates sum_j(a_j * dequant(V[j])) without applying inverse rotation.
    The caller must apply inverse rotation once on the output.

    Args:
        V_packed: (num_heads, seq_len, packed_bytes) uint8 — 3-bit packed indices
        centroids: (num_centroids,) float32 — Lloyd-Max centroids
        norms: (num_heads, seq_len) float32 — stored ||v|| norms
        attn_weights: (num_heads, seq_len) float32 — softmax attention weights
        num_heads, seq_len, head_dim: dimensions
        storage_stride: buffer stride along T axis (defaults to seq_len)

    Returns:
        output_rotated: (num_heads, head_dim) float32 — attention output in rotated space
    """
    if head_dim > _MAX_KERNEL_HEAD_DIM:
        raise ValueError(
            f"head_dim {head_dim} exceeds kernel maximum ({_MAX_KERNEL_HEAD_DIM})"
        )
    kernel = _get_fused_value_kernel()
    hd = mx.array([head_dim], dtype=mx.uint32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ss = mx.array(
        [storage_stride if storage_stride is not None else seq_len], dtype=mx.uint32
    )

    (output,) = kernel(
        inputs=[
            V_packed.reshape(-1),
            centroids.reshape(-1),
            norms.reshape(-1),
            attn_weights.reshape(-1),
            kv_head_map.reshape(-1),
            hd,
            sl,
            ss,
        ],
        output_shapes=[(num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(min(head_dim, 128) * num_heads, 1, 1),
        threadgroup=(min(head_dim, 128), 1, 1),
    )
    return output.reshape(num_heads, head_dim)


def fused_inverse_rotate(
    output_rotated: mx.array,
    block_matrices: mx.array,
    use_hadamard: bool = True,
) -> mx.array:
    """Apply inverse rotation (SO(4) blocks + WHT) on aggregated output.

    This is Kernel D — applied ONCE on the attention-weighted sum, not per-token.
    Uses the existing optimized Metal backend from isoquant_metal_kernels.

    Args:
        output_rotated: (num_heads, head_dim) float32 — output in rotated space
        block_matrices: (num_heads, num_blocks, 4, 4) — forward block matrices
            (inverse for SO(4) is the transpose; WHT is self-inverse)
        use_hadamard: whether WHT was applied in forward rotation

    Returns:
        output: (num_heads, head_dim) float32 — output in original space
    """
    try:
        from .isoquant_metal_kernels import metal_rotate_inverse
    except ModuleNotFoundError:
        from .mlx_isoquant import structured_rotate_inverse

        return structured_rotate_inverse(output_rotated, block_matrices, use_hadamard)

    # Add seq_len=1 dimension for the rotation kernel interface
    x = output_rotated[:, None, :]  # (H, 1, D)
    block_matrices_t = mx.swapaxes(block_matrices, -2, -1)  # transpose for inverse
    result = metal_rotate_inverse(x, block_matrices_t, use_hadamard=use_hadamard)
    return result[:, 0, :]  # (H, D)


def fused_isoquant_attention(
    q: mx.array,
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    rotation_matrices: mx.array,
    block_matrices: mx.array,
    use_hadamard: bool,
    scale: float,
    mask: mx.array | None = None,
    num_kv_heads: int | None = None,
    storage_stride: int | None = None,
) -> mx.array:
    """Full fused IsoQuant attention: Q·Kᵀ → softmax → V·a → inverse rotate.

    This is the complete 4-kernel pipeline replacing the current
    reconstruct_keys() + get_values() + SDPA path.

    The query is rotated forward into the same space as the stored keys,
    so attention scores are computed in rotated space (valid because
    isometric rotation preserves inner products).

    Args:
        q: (batch, num_q_heads, 1, head_dim) float32 — query (decode = single token)
        K_packed: (num_kv_heads, seq_len, packed_bytes) uint8
        V_packed: (num_kv_heads, seq_len, packed_bytes) uint8
        centroids: (num_centroids,) float32
        k_norms: (num_kv_heads, seq_len) float32
        v_norms: (num_kv_heads, seq_len) float32
        rotation_matrices: (num_kv_heads, head_dim, head_dim) float32
        block_matrices: (num_kv_heads, num_blocks, 4, 4) float32
        use_hadamard: bool
        scale: float — attention scale (1/sqrt(d_k))
        mask: optional attention mask
        num_kv_heads: KV head count (defaults to num_q_heads if None)
        storage_stride: buffer stride along T axis (defaults to seq_len)

    Returns:
        output: (batch, num_q_heads, 1, head_dim) — attention output
    """
    B, H_q, _, D = q.shape
    H_kv = num_kv_heads if num_kv_heads is not None else H_q
    T = K_packed.shape[1]
    repeats = H_q // H_kv if H_q != H_kv else 1

    # Build KV head mapping: query head i reads from KV head i // repeats
    kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

    # Step 1: Rotate query forward into compressed space
    R_T = mx.swapaxes(rotation_matrices, -2, -1)  # (H_kv, D, D)
    q_flat = q[0, :, 0, :]  # (H_q, D) — single decode token
    if repeats > 1:
        R_T_exp = mx.repeat(R_T, repeats, axis=0)
    else:
        R_T_exp = R_T
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)  # (H_q, D)

    # Step 2: Kernel A — fused Q·Kᵀ from packed storage (no K materialisation)
    scores = fused_qk_dot(
        K_packed,
        centroids,
        k_norms,
        q_rot,
        kv_head_map,
        num_heads=H_q,
        seq_len=T,
        head_dim=D,
        storage_stride=storage_stride,
    )
    scores = scores * scale  # (H_q, T)

    # Apply mask if provided
    if mask is not None:
        m = mask
        while m.ndim > scores.ndim:
            m = m.squeeze(0)
        scores = scores + m

    # Step 3: Kernel B — softmax
    scores = mx.softmax(scores, axis=-1)  # (H_q, T)

    # Step 4: Kernel C — fused value accumulation in rotated space
    output_rot = fused_value_accum(
        V_packed,
        centroids,
        v_norms,
        scores,
        kv_head_map,
        num_heads=H_q,
        seq_len=T,
        head_dim=D,
        storage_stride=storage_stride,
    )  # (H_q, D) — still in rotated space

    # Step 5: Kernel D — single inverse rotation on aggregated output
    if block_matrices is not None:
        if repeats > 1:
            # Group query heads back to KV-head space for rotation
            output_groups = output_rot.reshape(H_kv, repeats, D)
            rotated = []
            for g in range(repeats):
                rotated.append(
                    fused_inverse_rotate(
                        output_groups[:, g, :], block_matrices, use_hadamard
                    )
                )
            output = mx.stack(rotated, axis=1).reshape(H_q, D)
        else:
            output = fused_inverse_rotate(output_rot, block_matrices, use_hadamard)
    else:
        # Dense fallback
        if repeats > 1:
            R_exp = mx.repeat(rotation_matrices, repeats, axis=0)
        else:
            R_exp = rotation_matrices
        output = mx.matmul(output_rot[:, None, :], R_exp)[:, 0, :]

    # Reshape to match SDPA output: (B, H_q, 1, D)
    return output[None, :, None, :]
