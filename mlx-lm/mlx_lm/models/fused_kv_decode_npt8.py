"""Phase 3 v1: NPT=8 single-pass fused IsoQuant attention (head_dim=256).

Single-pass implementation: one threadgroup per query head, serial loop over
the full sequence length T. NOT the T-tiled + FA2-merge design from the spec
(``docs/superpowers/specs/2026-04-24-fused-npt8-tiled-design.md``).

The three-kernel pipeline in ``fused_kv_decode_kernels`` handles
``head_dim=128`` (NPT=4). This module provides a single-pass variant for
``head_dim=256`` (NPT=8) using online softmax with in-kernel inverse
SO(4) rotation and optional WHT.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["fused_attention_npt8"]

_npt8_kernel_cache: dict[tuple[bool], mx.fast.MetalKernel] | None = None


_NPT8_FUSED_ATTENTION_SOURCE = """
    // Grid: (32, H_q, 1) — 32 threads per TG, one TG per query head
    // Threadgroup: (32, 1, 1)
    // NPT=8: each thread owns 8 consecutive dims = one full packed word

    uint lane = thread_position_in_threadgroup.x;
    uint q_head = threadgroup_position_in_grid.y;
    uint kv_head = kv_head_map[q_head];
    uint T = seq_len[0];
    uint T_stride = storage_stride[0];
    uint use_mask = has_mask[0];

    // Each thread owns 8 consecutive dimensions
    uint dim_base = lane * 8;

    // NPT=8: each thread reads exactly one packed word (word index = lane)
    uint w_byte = lane * 3;

    // Load rotated query values into registers
    float q_r[8];
    for (uint i = 0; i < 8; i++) {
        q_r[i] = q[q_head * 256 + dim_base + i];
    }

    // Online softmax state
    float m_run = -1e38f;
    float l_run = 0.0f;
    float O_r[8];
    for (uint i = 0; i < 8; i++) O_r[i] = 0.0f;

    // Pre-compute base offsets for K and V packed data
    uint packed_words = 256 / 8;  // = 32
    uint stride_bytes = packed_words * 3;  // = 96
    uint kv_k_base = kv_head * T_stride * stride_bytes;
    uint kv_v_base = kv_head * T_stride * stride_bytes;

    // === Main loop: single pass over all KV tokens ===
    for (uint t = 0; t < T; t++) {
        // --- Decode K[t] for this thread's 8 dims ---
        uint k_off = kv_k_base + t * stride_bytes + w_byte;
        uint kw = uint(K_packed[k_off]) | (uint(K_packed[k_off+1]) << 8) | (uint(K_packed[k_off+2]) << 16);
        float k_norm = k_norms[kv_head * T_stride + t];

        // Partial dot product over 8 dimensions
        float partial = 0.0f;
        for (uint i = 0; i < 8; i++) {
            float k_val = centroids[(kw >> (i * 3)) & 0x7] * k_norm;
            partial += q_r[i] * k_val;
        }

        // Full dot product via SIMD reduction
        float score = simd_sum(partial) * scale_val[0];

        if (use_mask) score += mask_data[q_head * T + t];

        // --- Online softmax update ---
        float m_new = max(m_run, score);
        float corr = exp(m_run - m_new);
        float es = exp(score - m_new);

        // --- Decode V[t] and accumulate ---
        uint v_off = kv_v_base + t * stride_bytes + w_byte;
        uint vw = uint(V_packed[v_off]) | (uint(V_packed[v_off+1]) << 8) | (uint(V_packed[v_off+2]) << 16);
        float v_norm = v_norms[kv_head * T_stride + t];

        for (uint i = 0; i < 8; i++) {
            float v_val = centroids[(vw >> (i * 3)) & 0x7] * v_norm;
            O_r[i] = O_r[i] * corr + es * v_val;
        }
        l_run = l_run * corr + es;
        m_run = m_new;
    }

    // === Normalize ===
    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint i = 0; i < 8; i++) O_r[i] *= inv_l;

    // === Inverse rotation ===
    // NPT=8: each thread owns 2 consecutive 4x4 blocks
    uint block_a = lane * 2;
    uint block_b = lane * 2 + 1;

    // Block A: dims [0..3] of this thread's 8
    uint bo_a = (kv_head * 64 + block_a) * 16;
    float ra0 = blocks_t[bo_a+ 0]*O_r[0] + blocks_t[bo_a+ 1]*O_r[1] + blocks_t[bo_a+ 2]*O_r[2] + blocks_t[bo_a+ 3]*O_r[3];
    float ra1 = blocks_t[bo_a+ 4]*O_r[0] + blocks_t[bo_a+ 5]*O_r[1] + blocks_t[bo_a+ 6]*O_r[2] + blocks_t[bo_a+ 7]*O_r[3];
    float ra2 = blocks_t[bo_a+ 8]*O_r[0] + blocks_t[bo_a+ 9]*O_r[1] + blocks_t[bo_a+10]*O_r[2] + blocks_t[bo_a+11]*O_r[3];
    float ra3 = blocks_t[bo_a+12]*O_r[0] + blocks_t[bo_a+13]*O_r[1] + blocks_t[bo_a+14]*O_r[2] + blocks_t[bo_a+15]*O_r[3];

    // Block B: dims [4..7] of this thread's 8
    uint bo_b = (kv_head * 64 + block_b) * 16;
    float rb0 = blocks_t[bo_b+ 0]*O_r[4] + blocks_t[bo_b+ 1]*O_r[5] + blocks_t[bo_b+ 2]*O_r[6] + blocks_t[bo_b+ 3]*O_r[7];
    float rb1 = blocks_t[bo_b+ 4]*O_r[4] + blocks_t[bo_b+ 5]*O_r[5] + blocks_t[bo_b+ 6]*O_r[6] + blocks_t[bo_b+ 7]*O_r[7];
    float rb2 = blocks_t[bo_b+ 8]*O_r[4] + blocks_t[bo_b+ 9]*O_r[5] + blocks_t[bo_b+10]*O_r[6] + blocks_t[bo_b+11]*O_r[7];
    float rb3 = blocks_t[bo_b+12]*O_r[4] + blocks_t[bo_b+13]*O_r[5] + blocks_t[bo_b+14]*O_r[6] + blocks_t[bo_b+15]*O_r[7];

    O_r[0] = ra0; O_r[1] = ra1; O_r[2] = ra2; O_r[3] = ra3;
    O_r[4] = rb0; O_r[5] = rb1; O_r[6] = rb2; O_r[7] = rb3;

    if (USE_HADAMARD) {
        threadgroup float sa[256];
        threadgroup float sb[256];

        for (uint i = 0; i < 8; i++) sa[dim_base + i] = O_r[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* src = sa;
        threadgroup float* dst = sb;

        for (uint stride = 1; stride < 256; stride <<= 1) {
            for (uint i = 0; i < 8; i++) {
                uint idx = dim_base + i;
                uint partner = idx ^ stride;
                float sv = src[idx];
                float pv = src[partner];
                dst[idx] = ((idx & stride) == 0) ? (sv + pv) : (pv - sv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* tmp = src; src = dst; dst = tmp;
        }

        float wht_norm = 1.0f / sqrt(256.0f);
        for (uint i = 0; i < 8; i++) {
            output[q_head * 256 + dim_base + i] = src[dim_base + i] * wht_norm;
        }
    } else {
        for (uint i = 0; i < 8; i++) {
            output[q_head * 256 + dim_base + i] = O_r[i];
        }
    }
"""


def _get_npt8_kernel(use_hadamard: bool) -> mx.fast.MetalKernel:
    global _npt8_kernel_cache
    if _npt8_kernel_cache is None:
        _npt8_kernel_cache = {}

    key = (use_hadamard,)
    if key not in _npt8_kernel_cache:
        source = _NPT8_FUSED_ATTENTION_SOURCE.replace(
            "USE_HADAMARD", "1" if use_hadamard else "0"
        )
        _npt8_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"fused_attention_npt8_had{int(use_hadamard)}",
            input_names=[
                "K_packed",
                "V_packed",
                "centroids",
                "k_norms",
                "v_norms",
                "q",
                "kv_head_map",
                "blocks_t",
                "scale_val",
                "seq_len",
                "storage_stride",
                "has_mask",
                "mask_data",
            ],
            output_names=["output"],
            source=source,
        )
    return _npt8_kernel_cache[key]


def fused_attention_npt8(
    K_packed: mx.array,
    V_packed: mx.array,
    centroids: mx.array,
    k_norms: mx.array,
    v_norms: mx.array,
    q_rot: mx.array,
    kv_head_map: mx.array,
    *,
    blocks_t: mx.array,
    scale: float,
    use_hadamard: bool,
    mask: mx.array | None,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    storage_stride: int | None = None,
) -> mx.array:
    """NPT=8 single-pass fused attention for head_dim=256.

    Single Metal kernel dispatch replacing the 3-kernel pipeline
    (fused_qk_dot + softmax + fused_value_accum) with online softmax
    and in-kernel inverse rotation. Full T is processed in one pass
    per threadgroup.
    """
    assert head_dim == 256, f"NPT=8 kernel requires head_dim=256, got {head_dim}"

    kernel = _get_npt8_kernel(use_hadamard)

    if mask is not None:
        m = mask
        while m.ndim > 2:
            m = m.squeeze(0)
        if m.shape[0] == 1 and num_heads > 1:
            m = mx.broadcast_to(m, (num_heads, seq_len))
        mask_flat = m.reshape(-1).astype(mx.float32)
        has_mask_val = mx.array([1], dtype=mx.uint32)
    else:
        mask_flat = mx.zeros((1,), dtype=mx.float32)
        has_mask_val = mx.array([0], dtype=mx.uint32)

    scale_arr = mx.array([scale], dtype=mx.float32)
    sl = mx.array([seq_len], dtype=mx.uint32)
    ss = mx.array(
        [storage_stride if storage_stride is not None else seq_len], dtype=mx.uint32
    )

    (output,) = kernel(
        inputs=[
            K_packed.reshape(-1),
            V_packed.reshape(-1),
            centroids.reshape(-1),
            k_norms.reshape(-1),
            v_norms.reshape(-1),
            q_rot.reshape(-1),
            kv_head_map.reshape(-1),
            blocks_t.reshape(-1),
            scale_arr,
            sl,
            ss,
            has_mask_val,
            mask_flat,
        ],
        output_shapes=[(num_heads * head_dim,)],
        output_dtypes=[mx.float32],
        grid=(32, num_heads, 1),
        threadgroup=(32, 1, 1),
    )
    return output.reshape(num_heads, head_dim)
