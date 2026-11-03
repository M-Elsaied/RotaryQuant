"""Phase 3 v1: NPT=8 single-pass fused kernel for head_dim=256.

This is a full-sequence single-pass implementation (one threadgroup per head,
serial loop over T). It is NOT the T-tiled + FA2-merge design from the spec.

The reference uses ``fused_value_accum`` (serial T loop) — not
``fused_value_accum_tiled`` — for primary equivalence so failures localize
to the new kernel. A secondary test cross-checks the tiled V path.

See: docs/superpowers/specs/2026-04-24-fused-npt8-tiled-design.md (target design).
"""

import os

os.environ.setdefault("ISOQUANT_BITS", "3")

import mlx.core as mx
import numpy as np
import pytest
from conftest_npt8 import (
    _identity_blocks,
    _python_inverse_rotation,
    _random_so4_blocks,
    _ref_3kernel_stable,
    _synthetic_d256,
)
from mlx_lm.models.fused_kv_decode_kernels import (
    fused_qk_dot,
)
from mlx_lm.models.fused_kv_decode_npt8 import fused_attention_npt8

metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@metal_only
def test_fused_npt8_matches_stable_3kernel_no_inverse_rotation() -> None:
    t, h_kv, h_q, d = 256, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)
    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    out = fused_attention_npt8(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_fused_npt8_also_matches_tiled_v_accum_path() -> None:
    from mlx_lm.models.fused_kv_decode_tiled import fused_value_accum_tiled

    t, h_kv, h_q, d = 256, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)
    scores = fused_qk_dot(k_p, c, nk, q, kv_map, h_q, t, d) * scale
    attn = mx.softmax(scores, axis=-1)
    ref_tiled = fused_value_accum_tiled(
        v_p, c, nv, attn, kv_map, h_q, t, d, tile_size=128
    )
    out = fused_attention_npt8(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref_tiled, out)
    np.testing.assert_allclose(
        np.asarray(ref_tiled), np.asarray(out), rtol=1e-3, atol=1e-4
    )


@metal_only
def test_fused_npt8_with_mask() -> None:
    """NPT=8 kernel handles attention mask correctly."""

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)

    rng = np.random.default_rng(99)
    mask = mx.array(rng.standard_normal((h_q, t)).astype(np.float32))

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask)
    out = fused_attention_npt8(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=mask,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_fused_npt8_with_storage_stride() -> None:
    """NPT=8 kernel works with padded buffers (storage_stride > seq_len)."""

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks_t = _identity_blocks(h_kv, d)

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)

    padded_t = t + 64
    pad_k = mx.zeros((h_kv, 64, k_p.shape[2]), dtype=mx.uint8)
    pad_v = mx.zeros((h_kv, 64, v_p.shape[2]), dtype=mx.uint8)
    k_padded = mx.concatenate([k_p, pad_k], axis=1)
    v_padded = mx.concatenate([v_p, pad_v], axis=1)
    nk_padded = mx.concatenate([nk, mx.zeros((h_kv, 64))], axis=1)
    nv_padded = mx.concatenate([nv, mx.zeros((h_kv, 64))], axis=1)

    out = fused_attention_npt8(
        k_padded,
        v_padded,
        c,
        nk_padded,
        nv_padded,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        storage_stride=padded_t,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_fused_npt8_inverse_rotation_non_identity() -> None:
    """NPT=8 kernel applies inverse SO(4) rotation with random blocks."""

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks, blocks_t = _random_so4_blocks(h_kv, d, seed=77)

    ref_rot = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    mx.eval(ref_rot)

    repeats = h_q // h_kv
    ref_rot_grouped = ref_rot.reshape(h_kv, repeats, d)
    inv_parts = []
    for g in range(repeats):
        group_out = ref_rot_grouped[:, g, :]
        inv_parts.append(
            _python_inverse_rotation(group_out, blocks, use_hadamard=False)
        )
    ref = mx.stack(inv_parts, axis=1).reshape(h_q, d)

    out = fused_attention_npt8(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_fused_npt8_with_hadamard() -> None:
    """NPT=8 kernel applies inverse SO(4) + WHT (Hadamard) correctly."""

    t, h_kv, h_q, d = 128, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks, blocks_t = _random_so4_blocks(h_kv, d, seed=88)

    ref_rot = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    mx.eval(ref_rot)

    repeats = h_q // h_kv
    ref_rot_grouped = ref_rot.reshape(h_kv, repeats, d)
    inv_parts = []
    for g in range(repeats):
        group_out = ref_rot_grouped[:, g, :]
        inv_parts.append(_python_inverse_rotation(group_out, blocks, use_hadamard=True))
    ref = mx.stack(inv_parts, axis=1).reshape(h_q, d)

    out = fused_attention_npt8(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        blocks_t=blocks_t,
        scale=scale,
        use_hadamard=True,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-2, atol=1e-3)


@metal_only
def test_npt8_cache_level_dispatch() -> None:
    """IsoQuantKVCache.fused_attention dispatches NPT=8 when ISOQUANT_USE_NPT8_FUSED=1 and D=256."""
    from unittest import mock

    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
    from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

    h_kv, d, seq_len, bit_width = 2, 256, 32, 3
    cache = IsoQuantKVCache(
        num_heads=h_kv,
        head_dim=d,
        bit_width=bit_width,
        codebook_dir=get_default_codebook_dir(),
    )
    if cache._fallback_cache is not None:
        pytest.skip("IsoQuant fallback — no TQ codebook for D=256, 3-bit")

    rng = np.random.default_rng(42)
    h_q = h_kv * 8  # GQA repeats=8
    keys = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))
    values = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    queries = mx.array(rng.normal(size=(1, h_q, 1, d)).astype(np.float32))
    scale = d**-0.5

    # Dense path (reconstruct + SDPA)
    keys_dense = cache.reconstruct_keys()
    values_dense = cache.get_values()
    output_dense = scaled_dot_product_attention(
        queries, keys_dense, values_dense, cache=None, scale=scale, mask=None
    )
    mx.eval(output_dense)

    # 3-kernel fused path (env flag OFF)
    with mock.patch.dict(os.environ, {"ISOQUANT_USE_NPT8_FUSED": "0"}):
        output_3kernel = cache.fused_attention(queries, scale=scale, mask=None)
    mx.eval(output_3kernel)

    # NPT=8 fused path (env flag ON) — assert dispatch actually happens
    with (
        mock.patch.dict(os.environ, {"ISOQUANT_USE_NPT8_FUSED": "1"}),
        mock.patch.object(
            cache, "_fused_attention_npt8", wraps=cache._fused_attention_npt8
        ) as npt8_spy,
    ):
        output_npt8 = cache.fused_attention(queries, scale=scale, mask=None)
    mx.eval(output_npt8)
    assert npt8_spy.call_count == 1, (
        f"_fused_attention_npt8 was not dispatched (called {npt8_spy.call_count} times)"
    )

    # Both fused paths should match dense within tolerance
    np.testing.assert_allclose(
        np.asarray(output_3kernel), np.asarray(output_dense), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(output_npt8), np.asarray(output_dense), rtol=1e-2, atol=1e-2
    )
    # NPT=8 and 3-kernel should match each other tightly
    np.testing.assert_allclose(
        np.asarray(output_npt8), np.asarray(output_3kernel), rtol=1e-3, atol=1e-3
    )


def test_npt8_cache_level_prealloc_mode() -> None:
    """NPT=8 dispatch works correctly in prealloc cache mode."""
    from unittest import mock

    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
    from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

    h_kv, d, seq_len, bit_width = 2, 256, 32, 3
    cache_concat = IsoQuantKVCache(
        num_heads=h_kv,
        head_dim=d,
        bit_width=bit_width,
        codebook_dir=get_default_codebook_dir(),
    )
    if cache_concat._fallback_cache is not None:
        pytest.skip("IsoQuant fallback — no TQ codebook for D=256, 3-bit")

    rng = np.random.default_rng(55)
    h_q = h_kv * 4
    keys = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))
    values = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))

    cache_concat.update_and_fetch(keys, values)
    cache_concat.finalize_deferred_prefill()

    cache_prealloc = IsoQuantKVCache(
        num_heads=h_kv,
        head_dim=d,
        bit_width=bit_width,
        codebook_dir=get_default_codebook_dir(),
    )
    with mock.patch.dict(os.environ, {"ISOQUANT_CACHE_MODE": "prealloc"}):
        cache_prealloc._cache_mode = "prealloc"
    cache_prealloc.update_and_fetch(keys, values)
    cache_prealloc.finalize_deferred_prefill()

    queries = mx.array(rng.normal(size=(1, h_q, 1, d)).astype(np.float32))
    scale = d**-0.5

    with (
        mock.patch.dict(os.environ, {"ISOQUANT_USE_NPT8_FUSED": "1"}),
        mock.patch.object(
            cache_prealloc,
            "_fused_attention_npt8",
            wraps=cache_prealloc._fused_attention_npt8,
        ) as npt8_spy,
    ):
        out_concat = cache_concat.fused_attention(queries, scale=scale, mask=None)
        out_prealloc = cache_prealloc.fused_attention(queries, scale=scale, mask=None)
    mx.eval(out_concat, out_prealloc)

    assert npt8_spy.call_count == 1, "NPT=8 dispatch not triggered in prealloc mode"
    np.testing.assert_allclose(
        np.asarray(out_prealloc), np.asarray(out_concat), rtol=1e-3, atol=1e-3
    )
