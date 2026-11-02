"""Phase 6: NPT=16 single-pass fused kernel for head_dim=512 (Kimi MLA).

Tests verify the NPT=16 kernel against the 3-kernel reference pipeline,
with and without SO(4) inverse rotation and Hadamard, and with masks
(MLA pe_scores).
"""

import os

os.environ.setdefault("ISOQUANT_BITS", "3")

import mlx.core as mx
import numpy as np
import pytest
from conftest_npt16 import (
    _identity_blocks_d512,
    _python_inverse_rotation,
    _random_so4_blocks_d512,
    _ref_3kernel_stable_d512,
    _synthetic_d512,
    _synthetic_d512_shared_kv,
)
from mlx_lm.models.fused_kv_decode_npt16 import fused_attention_npt16

metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@metal_only
class TestNPT16MatchesReference:
    """NPT=16 kernel matches 3-kernel pipeline (no inverse rotation)."""

    def test_identity_blocks_no_mask(self):
        t, h_kv, h_q, d = 128, 1, 64, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks_t = _identity_blocks_d512(h_kv)
        ref = _ref_3kernel_stable_d512(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4
        )

    def test_shared_kv_mla_style(self):
        """K=V (MLA shared latent) — same result as independent K/V."""
        t, h_kv, h_q, d = 64, 1, 64, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512_shared_kv(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks_t = _identity_blocks_d512(h_kv)
        ref = _ref_3kernel_stable_d512(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4
        )


@metal_only
class TestNPT16WithMask:
    """NPT=16 kernel handles additive mask (pe_scores in MLA)."""

    def test_random_mask(self):
        t, h_kv, h_q, d = 128, 1, 64, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks_t = _identity_blocks_d512(h_kv)
        rng = np.random.default_rng(99)
        mask = mx.array(rng.standard_normal((h_q, t)).astype(np.float32))
        ref = _ref_3kernel_stable_d512(
            k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask
        )
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4
        )

    def test_causal_mask(self):
        """Large negative values in mask should zero out future tokens."""
        t, h_kv, h_q, d = 32, 1, 4, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks_t = _identity_blocks_d512(h_kv)
        causal = np.zeros((h_q, t), dtype=np.float32)
        causal[:, t // 2 :] = -1e9
        mask = mx.array(causal)
        ref = _ref_3kernel_stable_d512(
            k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale, mask
        )
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4
        )


@metal_only
class TestNPT16InverseRotation:
    """NPT=16 kernel applies SO(4) inverse rotation correctly."""

    def test_random_so4_blocks(self):
        t, h_kv, h_q, d = 64, 1, 64, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks, blocks_t = _random_so4_blocks_d512(h_kv, seed=42)
        ref_rot = _ref_3kernel_stable_d512(
            k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale
        )
        mx.eval(ref_rot)
        ref = _python_inverse_rotation(ref_rot, blocks, use_hadamard=False)
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-3
        )

    def test_so4_plus_hadamard(self):
        t, h_kv, h_q, d = 64, 1, 64, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks, blocks_t = _random_so4_blocks_d512(h_kv, seed=77)
        ref_rot = _ref_3kernel_stable_d512(
            k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale
        )
        mx.eval(ref_rot)
        ref = _python_inverse_rotation(ref_rot, blocks, use_hadamard=True)
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=5e-3, atol=5e-3
        )


@metal_only
class TestNPT16SequenceLengths:
    """NPT=16 handles various sequence lengths."""

    @pytest.mark.parametrize("t", [1, 4, 16, 64, 256, 512])
    def test_various_lengths(self, t):
        h_kv, h_q, d = 1, 4, 512
        k_p, v_p, c, nk, nv, q, kv_map = _synthetic_d512(t, h_kv, h_q)
        scale = float(1.0 / np.sqrt(d))
        blocks_t = _identity_blocks_d512(h_kv)
        ref = _ref_3kernel_stable_d512(k_p, v_p, c, nk, nv, q, kv_map, h_q, t, d, scale)
        out = fused_attention_npt16(
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
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-3, atol=1e-4
        )


@metal_only
class TestNPT16StorageStride:
    """NPT=16 handles storage_stride (prealloc cache mode)."""

    def test_with_storage_stride(self):
        t, h_kv, h_q, d = 32, 1, 4, 512
        rng = np.random.default_rng(11)
        cap = 64
        indices_k = np.zeros((h_kv, cap, 512), dtype=np.uint8)
        indices_v = np.zeros((h_kv, cap, 512), dtype=np.uint8)
        indices_k[:, :t, :] = rng.integers(0, 8, (h_kv, t, 512))
        indices_v[:, :t, :] = rng.integers(0, 8, (h_kv, t, 512))
        nk = np.zeros((h_kv, cap), dtype=np.float32)
        nv = np.zeros((h_kv, cap), dtype=np.float32)
        nk[:, :t] = rng.standard_normal((h_kv, t))
        nv[:, :t] = rng.standard_normal((h_kv, t))
        centroids = mx.array(np.linspace(-1.5, 1.5, 8, dtype=np.float32))
        q_rot = mx.array(rng.standard_normal((h_q, 512)).astype(np.float32))
        kv_head_map = mx.arange(h_q, dtype=mx.uint32) // (h_q // h_kv)
        from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit

        k_p = pack_indices_3bit(mx.array(indices_k))
        v_p = pack_indices_3bit(mx.array(indices_v))
        nk_mx = mx.array(nk)
        nv_mx = mx.array(nv)
        mx.eval(k_p, v_p, nk_mx, nv_mx, centroids, q_rot, kv_head_map)

        blocks_t = _identity_blocks_d512(h_kv)
        scale = float(1.0 / np.sqrt(d))
        out = fused_attention_npt16(
            k_p,
            v_p,
            centroids,
            nk_mx,
            nv_mx,
            q_rot,
            kv_head_map,
            blocks_t=blocks_t,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=t,
            head_dim=d,
            storage_stride=cap,
        )

        k_p_tight = pack_indices_3bit(mx.array(indices_k[:, :t, :]))
        v_p_tight = pack_indices_3bit(mx.array(indices_v[:, :t, :]))
        nk_tight = mx.array(nk[:, :t])
        nv_tight = mx.array(nv[:, :t])
        mx.eval(k_p_tight, v_p_tight, nk_tight, nv_tight)
        ref = fused_attention_npt16(
            k_p_tight,
            v_p_tight,
            centroids,
            nk_tight,
            nv_tight,
            q_rot,
            kv_head_map,
            blocks_t=blocks_t,
            scale=scale,
            use_hadamard=False,
            mask=None,
            num_heads=h_q,
            seq_len=t,
            head_dim=d,
        )
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.asarray(ref), np.asarray(out), rtol=1e-5, atol=1e-5
        )
