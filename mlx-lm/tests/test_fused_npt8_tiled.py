"""Phase 3b: tiled NPT=8 fused kernel for head_dim=256."""

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
from mlx_lm.models.fused_kv_decode_npt8 import fused_attention_npt8
from mlx_lm.models.fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@metal_only
def test_tiled_matches_v1_single_pass() -> None:
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)
    blocks_t = mx.swapaxes(blocks, -2, -1)

    ref = fused_attention_npt8(
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
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_matches_3kernel_reference() -> None:
    t, h_kv, h_q, d = 1024, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=256,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_with_partial_last_tile() -> None:
    t, h_kv, h_q, d = 300, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)
    blocks_t = mx.swapaxes(blocks, -2, -1)

    ref = fused_attention_npt8(
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
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_with_mask() -> None:
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)

    rng = np.random.default_rng(99)
    mask = mx.array(rng.standard_normal((h_q, t)).astype(np.float32))

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask)
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=mask,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_with_storage_stride() -> None:
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)

    padded_t = t + 32
    pad_k = mx.zeros((h_kv, 32, k_p.shape[2]), dtype=mx.uint8)
    pad_v = mx.zeros((h_kv, 32, v_p.shape[2]), dtype=mx.uint8)
    k_padded = mx.concatenate([k_p, pad_k], axis=1)
    v_padded = mx.concatenate([v_p, pad_v], axis=1)
    nk_padded = mx.concatenate([nk, mx.zeros((h_kv, 32))], axis=1)
    nv_padded = mx.concatenate([nv, mx.zeros((h_kv, 32))], axis=1)

    out = fused_attention_npt8_tiled(
        k_padded,
        v_padded,
        c,
        nk_padded,
        nv_padded,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
        storage_stride=padded_t,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_with_rotation() -> None:
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks, _ = _random_so4_blocks(h_kv, d, seed=77)

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

    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_with_hadamard() -> None:
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks, _ = _random_so4_blocks(h_kv, d, seed=88)

    ref_rot = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    mx.eval(ref_rot)

    repeats = h_q // h_kv
    ref_rot_grouped = ref_rot.reshape(h_kv, repeats, d)
    inv_parts = []
    for g in range(repeats):
        group_out = ref_rot_grouped[:, g, :]
        inv_parts.append(_python_inverse_rotation(group_out, blocks, use_hadamard=True))
    ref = mx.stack(inv_parts, axis=1).reshape(h_q, d)

    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=True,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


def test_npt8_tiled_cache_level_dispatch() -> None:
    from unittest import mock

    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
    from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

    h_kv, d, bit_width = 2, 256, 3
    seq_len = IsoQuantKVCache._NPT8_TILED_T_THRESHOLD + 32
    cache = IsoQuantKVCache(
        num_heads=h_kv,
        head_dim=d,
        bit_width=bit_width,
        codebook_dir=get_default_codebook_dir(),
    )
    if cache._fallback_cache is not None:
        pytest.skip("IsoQuant fallback — no TQ codebook for D=256, 3-bit")

    rng = np.random.default_rng(123)
    h_q = h_kv * 4
    keys = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))
    values = mx.array(rng.normal(size=(1, h_kv, seq_len, d)).astype(np.float16))
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    queries = mx.array(rng.normal(size=(1, h_q, 1, d)).astype(np.float32))
    scale = d**-0.5

    keys_dense = cache.reconstruct_keys()
    values_dense = cache.get_values()
    output_dense = scaled_dot_product_attention(
        queries, keys_dense, values_dense, cache=None, scale=scale, mask=None
    )
    mx.eval(output_dense)

    with (
        mock.patch.dict(os.environ, {"ISOQUANT_USE_NPT8_FUSED": "1"}),
        mock.patch.object(
            cache,
            "_fused_attention_npt8_tiled",
            wraps=cache._fused_attention_npt8_tiled,
        ) as tiled_spy,
    ):
        output_tiled = cache.fused_attention(queries, scale=scale, mask=None)
    mx.eval(output_tiled)

    assert tiled_spy.call_count == 1, "NPT=8 tiled dispatch not triggered"
    np.testing.assert_allclose(
        np.asarray(output_tiled), np.asarray(output_dense), rtol=1e-2, atol=1e-2
    )


@metal_only
def test_tiled_single_tile() -> None:
    """tile_size >= seq_len → num_tiles=1; merge-of-one should be identity."""
    t, h_kv, h_q, d = 64, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)
    blocks_t = mx.swapaxes(blocks, -2, -1)

    ref = fused_attention_npt8(
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
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=256,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_mha_no_gqa() -> None:
    """h_kv == h_q (MHA, no GQA) — mx.take expansion is identity."""
    t, h_kv, h_q, d = 512, 4, 4, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks = _identity_blocks(h_kv, d)

    ref = _ref_3kernel_stable(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
    out = fused_attention_npt8_tiled(
        k_p,
        v_p,
        c,
        nk,
        nv,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=None,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)


@metal_only
def test_tiled_combined_mask_rotation_stride() -> None:
    """Mask + random SO(4) rotation + padded storage_stride — all at once."""
    t, h_kv, h_q, d = 512, 2, 16, 256
    k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d256(t, h_kv, h_q)
    scale = float(1.0 / np.sqrt(d))
    blocks, _ = _random_so4_blocks(h_kv, d, seed=55)

    rng = np.random.default_rng(77)
    mask = mx.array(rng.standard_normal((h_q, t)).astype(np.float32))

    padded_t = t + 64
    pad_k = mx.zeros((h_kv, 64, k_p.shape[2]), dtype=mx.uint8)
    pad_v = mx.zeros((h_kv, 64, v_p.shape[2]), dtype=mx.uint8)
    k_padded = mx.concatenate([k_p, pad_k], axis=1)
    v_padded = mx.concatenate([v_p, pad_v], axis=1)
    nk_padded = mx.concatenate([nk, mx.zeros((h_kv, 64))], axis=1)
    nv_padded = mx.concatenate([nv, mx.zeros((h_kv, 64))], axis=1)

    ref_rot = _ref_3kernel_stable(
        k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask
    )
    mx.eval(ref_rot)
    repeats = h_q // h_kv
    ref_grouped = ref_rot.reshape(h_kv, repeats, d)
    inv_parts = []
    for g in range(repeats):
        inv_parts.append(
            _python_inverse_rotation(ref_grouped[:, g, :], blocks, use_hadamard=False)
        )
    ref = mx.stack(inv_parts, axis=1).reshape(h_q, d)

    out = fused_attention_npt8_tiled(
        k_padded,
        v_padded,
        c,
        nk_padded,
        nv_padded,
        q,
        kv_map,
        block_matrices=blocks,
        scale=scale,
        use_hadamard=False,
        mask=mask,
        num_heads=h_q,
        seq_len=t,
        head_dim=d,
        tile_size=128,
        storage_stride=padded_t,
    )
    mx.eval(ref, out)
    np.testing.assert_allclose(np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4)
