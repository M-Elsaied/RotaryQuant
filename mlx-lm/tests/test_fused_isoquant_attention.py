"""Correctness test: fused IsoQuant attention vs dense reconstruct+SDPA path.

Verifies that computing attention in rotated space with a single inverse
rotation on the output produces identical results to the existing
reconstruct_keys() + get_values() + scaled_dot_product_attention() path.
"""

import os
import unittest
from unittest import mock

import numpy as np
import mlx.core as mx

from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir


def _make_populated_cache(
    num_heads: int = 4,
    head_dim: int = 128,
    seq_len: int = 32,
    bit_width: int = 3,
    seed: int = 42,
) -> IsoQuantKVCache:
    """Create an IsoQuantKVCache with compressed data populated."""
    cache = IsoQuantKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        bit_width=bit_width,
        codebook_dir=get_default_codebook_dir(),
    )
    # Skip if fallback was activated (no codebook)
    if cache._fallback_cache is not None:
        return cache

    rng = np.random.default_rng(seed)
    keys = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )
    values = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )

    cache.update_and_fetch(keys, values)
    # Cache starts in deferred mode — finalize to compress the FP16 buffer
    cache.finalize_deferred_prefill()
    return cache


class TestFusedIsoQuantAttention(unittest.TestCase):
    """Compare fused rotated-space attention against dense reconstruct+SDPA."""

    def test_supports_fused_attention_flag(self):
        cache = _make_populated_cache()
        if cache._fallback_cache is not None:
            self.skipTest("No codebook for head_dim=128, bit_width=3")
        self.assertTrue(cache.supports_fused_attention)

    @unittest.skipUnless(mx.metal.is_available(), "Metal required")
    def test_fused_matches_dense_no_mask(self):
        """Core correctness: fused path == dense path without mask."""
        cache = _make_populated_cache(num_heads=4, seq_len=16, seed=100)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")
        self.assertTrue(cache.supports_fused_attention)

        rng = np.random.default_rng(200)
        queries = mx.array(rng.normal(size=(1, 4, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # Dense path: reconstruct full K, V then SDPA
        keys_dense = cache.reconstruct_keys()  # (1, H, T, D)
        values_dense = cache.get_values()  # (1, H, T, D)
        output_dense = scaled_dot_product_attention(
            queries,
            keys_dense,
            values_dense,
            cache=None,
            scale=scale,
            mask=None,
        )
        mx.eval(output_dense)

        # Fused path: compute in rotated space, single inverse rotate
        output_fused = cache.fused_attention(queries, scale=scale, mask=None)
        mx.eval(output_fused)

        self.assertEqual(output_dense.shape, output_fused.shape)
        np.testing.assert_allclose(
            np.array(output_fused, dtype=np.float32),
            np.array(output_dense, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused attention output differs from dense path",
        )
        self.assertIs(
            cache._fused_metal_ok,
            True,
            "Fused attention unexpectedly fell back instead of using Metal kernels",
        )

    def test_fused_matches_dense_with_mask(self):
        """Correctness with causal-style mask applied."""
        cache = _make_populated_cache(num_heads=2, seq_len=20, seed=300)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        rng = np.random.default_rng(400)
        queries = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # Create a mask that blocks the last 5 tokens
        T = 20
        mask = mx.zeros((1, 1, 1, T))
        mask_np = np.zeros((1, 1, 1, T), dtype=np.float32)
        mask_np[:, :, :, -5:] = -1e9
        mask = mx.array(mask_np)

        # Dense
        keys_dense = cache.reconstruct_keys()
        values_dense = cache.get_values()
        output_dense = scaled_dot_product_attention(
            queries,
            keys_dense,
            values_dense,
            cache=None,
            scale=scale,
            mask=mask,
        )
        mx.eval(output_dense)

        # Fused
        output_fused = cache.fused_attention(queries, scale=scale, mask=mask)
        mx.eval(output_fused)

        self.assertEqual(output_dense.shape, output_fused.shape)
        np.testing.assert_allclose(
            np.array(output_fused, dtype=np.float32),
            np.array(output_dense, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused attention with mask differs from dense path",
        )

    def test_fused_matches_dense_gqa(self):
        """Correctness with grouped-query attention (H_q > H_kv)."""
        # 8 query heads, 2 KV heads → 4× GQA
        cache = _make_populated_cache(num_heads=2, seq_len=12, seed=500)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        rng = np.random.default_rng(600)
        H_q = 8
        queries = mx.array(rng.normal(size=(1, H_q, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # Dense path — need to expand KV heads for comparison
        keys_dense = cache.reconstruct_keys()  # (1, 2, T, D)
        values_dense = cache.get_values()  # (1, 2, T, D)
        # Repeat KV heads to match query heads
        keys_expanded = mx.repeat(keys_dense, 4, axis=1)  # (1, 8, T, D)
        values_expanded = mx.repeat(values_dense, 4, axis=1)  # (1, 8, T, D)
        output_dense = scaled_dot_product_attention(
            queries,
            keys_expanded,
            values_expanded,
            cache=None,
            scale=scale,
            mask=None,
        )
        mx.eval(output_dense)

        # Fused path — handles GQA internally
        output_fused = cache.fused_attention(queries, scale=scale, mask=None)
        mx.eval(output_fused)

        self.assertEqual(output_dense.shape, output_fused.shape)
        np.testing.assert_allclose(
            np.array(output_fused, dtype=np.float32),
            np.array(output_dense, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused GQA attention differs from dense path",
        )

    def test_fused_matches_dense_longer_sequence(self):
        """Stress test with longer sequence (64 tokens)."""
        cache = _make_populated_cache(num_heads=4, seq_len=64, seed=700)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        rng = np.random.default_rng(800)
        queries = mx.array(rng.normal(size=(1, 4, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        keys_dense = cache.reconstruct_keys()
        values_dense = cache.get_values()
        output_dense = scaled_dot_product_attention(
            queries,
            keys_dense,
            values_dense,
            cache=None,
            scale=scale,
            mask=None,
        )
        mx.eval(output_dense)

        output_fused = cache.fused_attention(queries, scale=scale, mask=None)
        mx.eval(output_fused)

        self.assertEqual(output_dense.shape, output_fused.shape)
        np.testing.assert_allclose(
            np.array(output_fused, dtype=np.float32),
            np.array(output_dense, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Fused attention differs at T=64",
        )

    def test_packed_kv_cache_reused_and_invalidated(self):
        """Packed 3-bit bytes are built once at finalize, reused across decode
        steps, and extended (not rebuilt) when a decode token is appended."""
        rng = np.random.default_rng(1400)
        queries = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # The fused Metal encoder writes packed bytes directly and never calls
        # pack_indices_3bit; pin the MLX encoder so the pack path is observable
        # and the counts are the same with and without Metal.
        with mock.patch.dict(os.environ, {"ISOQUANT_FUSED_ENCODE": "0"}), mock.patch(
            "mlx_lm.models.fused_kv_decode_kernels.pack_indices_3bit",
            wraps=__import__(
                "mlx_lm.models.fused_kv_decode_kernels",
                fromlist=["pack_indices_3bit"],
            ).pack_indices_3bit,
        ) as pack_mock:
            # Prefill + finalize: keys and values are packed once each.
            cache = _make_populated_cache(num_heads=2, seq_len=8, seed=1300)
            if cache._fallback_cache is not None:
                self.skipTest("No codebook")
            self.assertEqual(
                pack_mock.call_count,
                2,
                "finalize_deferred_prefill should pack keys and values exactly once",
            )

            # Attention on an unchanged cache reuses the packed bytes.
            out1 = cache.fused_attention(queries, scale=scale, mask=None)
            mx.eval(out1)
            out2 = cache.fused_attention(queries, scale=scale, mask=None)
            mx.eval(out2)
            self.assertEqual(
                pack_mock.call_count,
                2,
                "Packed 3-bit KV was recomputed despite unchanged cache state",
            )

            # One decode token: only the appended token is packed (keys and
            # values), and the packed caches are extended in place.
            new_keys = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float16))
            new_values = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float16))
            cache.update_and_fetch(new_keys, new_values)
            self.assertEqual(
                pack_mock.call_count,
                4,
                "A decode write should pack only the appended token's keys and values",
            )

            out3 = cache.fused_attention(queries, scale=scale, mask=None)
            mx.eval(out3)
            self.assertEqual(
                pack_mock.call_count,
                4,
                "fused_attention re-packed the cache after a decode write instead "
                "of reusing the incrementally maintained packed KV",
            )

    @unittest.skipUnless(mx.metal.is_available(), "Metal required")
    def test_metal_path_actually_executes(self):
        """Verify the Metal-fused path ran, not the silent MLX-ops fallback."""
        cache = _make_populated_cache(num_heads=2, seq_len=8, seed=1500)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")
        self.assertTrue(cache.supports_fused_attention)

        # Reset latch so the Metal path is attempted
        cache._fused_metal_ok = None

        rng = np.random.default_rng(1600)
        queries = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        output = cache.fused_attention(queries, scale=scale, mask=None)
        mx.eval(output)

        self.assertTrue(
            cache._fused_metal_ok is True,
            f"Metal path did not execute (_fused_metal_ok={cache._fused_metal_ok}). "
            "The fused_attention call silently fell back to the MLX-ops path.",
        )

    def test_fully_fused_single_kernel_matches_dense(self):
        """Verify single-kernel fully_fused_attention matches dense SDPA."""
        cache = _make_populated_cache(num_heads=4, seq_len=32, seed=1700)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        from mlx_lm.models.fused_kv_decode_kernels import (
            fully_fused_attention,
            pack_indices_3bit,
        )

        rng = np.random.default_rng(1800)
        queries = mx.array(rng.normal(size=(1, 4, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # Dense path
        keys_dense = cache.reconstruct_keys()
        values_dense = cache.get_values()
        output_dense = scaled_dot_product_attention(
            queries,
            keys_dense,
            values_dense,
            cache=None,
            scale=scale,
            mask=None,
        )
        mx.eval(output_dense)

        # Single-kernel path (bypass cache method, call kernel directly)
        H_q, D = 4, 128
        H_kv = cache.num_heads
        T = cache.compressed_keys["indices"].shape[1]
        repeats = H_q // H_kv

        R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)

        k_packed = pack_indices_3bit(cache.compressed_keys["indices"])
        v_packed = pack_indices_3bit(cache.compressed_values["indices"])
        k_norms = cache.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
        v_norms = cache.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
        centroids = cache.compressor.centroids.reshape(-1).astype(mx.float32)
        kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

        output_fused = fully_fused_attention(
            K_packed=k_packed,
            V_packed=v_packed,
            centroids=centroids,
            k_norms=k_norms,
            v_norms=v_norms,
            q_rot=q_rot,
            kv_head_map=kv_head_map,
            blocks_t=cache.block_matrices_t,
            scale=scale,
            num_heads=H_q,
            seq_len=T,
            head_dim=D,
            use_hadamard=cache._use_hadamard,
            mask=None,
        )
        mx.eval(output_fused)

        # Reshape to match SDPA output
        output_fused_full = output_fused[None, :, None, :]

        self.assertEqual(output_dense.shape, output_fused_full.shape)
        np.testing.assert_allclose(
            np.array(output_fused_full, dtype=np.float32),
            np.array(output_dense, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Single-kernel fully fused attention differs from dense SDPA",
        )

    def test_fully_fused_single_dispatch_count(self):
        """Verify fully_fused_attention() uses exactly 1 mx.fast.metal_kernel dispatch."""
        cache = _make_populated_cache(num_heads=2, seq_len=8, seed=1900)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        rng = np.random.default_rng(2000)
        queries = mx.array(rng.normal(size=(1, 2, 1, 128)).astype(np.float32))
        scale = 128**-0.5
        H_q = 2
        D = 128
        T = 8

        from mlx_lm.models.fused_kv_decode_kernels import (
            fully_fused_attention,
            pack_indices_3bit,
        )

        k_packed = pack_indices_3bit(cache.compressed_keys["indices"])
        v_packed = pack_indices_3bit(cache.compressed_values["indices"])
        k_norms = cache.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
        v_norms = cache.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
        centroids = cache.compressor.centroids.reshape(-1).astype(mx.float32)
        kv_head_map = mx.arange(H_q, dtype=mx.uint32)

        R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T), axis=1)

        dispatch_count = 0
        original_metal_kernel = mx.fast.metal_kernel

        def counting_metal_kernel(*args, **kwargs):
            nonlocal dispatch_count
            result = original_metal_kernel(*args, **kwargs)
            original_call = result

            def counting_call(*call_args, **call_kwargs):
                nonlocal dispatch_count
                dispatch_count += 1
                return original_call(*call_args, **call_kwargs)

            return counting_call

        from mlx_lm.models import fused_kv_decode_kernels

        saved_cache = dict(fused_kv_decode_kernels._fused_kernel_cache)
        fused_kv_decode_kernels._fused_kernel_cache.clear()

        try:
            with mock.patch.object(
                mx.fast, "metal_kernel", side_effect=counting_metal_kernel
            ):
                output = fully_fused_attention(
                    K_packed=k_packed,
                    V_packed=v_packed,
                    centroids=centroids,
                    k_norms=k_norms,
                    v_norms=v_norms,
                    q_rot=q_rot,
                    kv_head_map=kv_head_map,
                    blocks_t=cache.block_matrices_t,
                    scale=scale,
                    num_heads=H_q,
                    seq_len=T,
                    head_dim=D,
                    use_hadamard=cache._use_hadamard,
                )
                mx.eval(output)
        finally:
            fused_kv_decode_kernels._fused_kernel_cache = saved_cache

        self.assertEqual(
            dispatch_count,
            1,
            f"Expected 1 Metal kernel dispatch (single fused), got {dispatch_count}",
        )

    def test_fallback_when_not_supported(self):
        """Fused attention gracefully falls back when conditions aren't met."""
        # Use bit_width=2 (not 3) — supports_fused_attention should be False
        cache = _make_populated_cache(num_heads=4, seq_len=8, bit_width=2, seed=900)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")
        self.assertFalse(cache.supports_fused_attention)

        rng = np.random.default_rng(1000)
        queries = mx.array(rng.normal(size=(1, 4, 1, 128)).astype(np.float32))
        scale = 128**-0.5

        # Should fall back to dense path without error
        output = cache.fused_attention(queries, scale=scale, mask=None)
        mx.eval(output)
        self.assertEqual(output.shape, (1, 4, 1, 128))

    def test_attention_scores_preserve_inner_products(self):
        """Verify that rotating Q and K preserves Q·Kᵀ scores.

        This is the mathematical foundation: for orthogonal R,
        (Rq)^T(Rk) = q^T R^T R k = q^T k.
        """
        cache = _make_populated_cache(num_heads=2, seq_len=8, seed=1100)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook")

        rng = np.random.default_rng(1200)
        q = mx.array(rng.normal(size=(2, 1, 128)).astype(np.float32))

        # Scores in original space: q @ k^T
        k_indices = cache.compressed_keys["indices"]
        k_norms = cache.compressed_keys["x_norm"]
        k_rot_quant = cache.compressor.centroids[k_indices]
        k_rot_scaled = k_rot_quant * k_norms.astype(mx.float32)

        # Inverse rotate to get original-space K
        k_original = cache._rotate_inverse(k_rot_quant) * k_norms.astype(mx.float32)
        scores_original = mx.matmul(q, mx.swapaxes(k_original, -2, -1))

        # Rotate Q forward, scores in rotated space: (Rq) @ (Rk)^T
        R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
        q_rot = mx.matmul(q, R_T)
        scores_rotated = mx.matmul(q_rot, mx.swapaxes(k_rot_scaled, -2, -1))

        mx.eval(scores_original, scores_rotated)
        np.testing.assert_allclose(
            np.array(scores_rotated),
            np.array(scores_original),
            atol=1e-4,
            rtol=1e-4,
            err_msg="Rotated-space scores differ from original-space scores",
        )


if __name__ == "__main__":
    unittest.main()
