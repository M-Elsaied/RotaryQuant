"""Correctness tests: fused Metal compress+pack vs Python _compress_batch + pack_indices_3bit."""

import unittest

import numpy as np
import mlx.core as mx

from mlx_lm.models.fused_kv_compress import fused_compress_and_pack
from mlx_lm.models.mlx_turboquant import (
    get_default_codebook_dir,
    load_codebook,
    quantize_scalar,
)
from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit


def _load_codebook_3bit(head_dim):
    import os

    cb_dir = get_default_codebook_dir()
    path = os.path.join(cb_dir, f"dim_{head_dim}_3bit.npz")
    if not os.path.exists(path):
        return None, None
    return load_codebook(head_dim, 3, cb_dir)


def _make_rotation_components(num_heads, head_dim, seed=42):
    rng = np.random.default_rng(seed)
    num_blocks = head_dim // 4
    blocks = []
    for _ in range(num_heads):
        head_blocks = []
        for _ in range(num_blocks):
            A = rng.normal(size=(4, 4))
            Q, _ = np.linalg.qr(A)
            head_blocks.append(Q)
        blocks.append(np.stack(head_blocks))
    return mx.array(np.stack(blocks).astype(np.float32))


def _python_compress_batch(x, block_matrices, centroids, boundaries, use_hadamard=True):
    """Reproduce _compress_batch logic: normalize -> FWHT -> SO(4) -> quantize."""
    from mlx_lm.models.mlx_isoquant import structured_rotate_forward

    block_matrices_t = mx.swapaxes(block_matrices, -2, -1)
    x_f32 = x.astype(mx.float32)
    x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
    x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8, dtype=mx.float32))
    x_rot = structured_rotate_forward(x_unit, block_matrices_t, use_hadamard)
    indices, _ = quantize_scalar(x_rot, centroids, boundaries)
    return {
        "indices": indices.astype(mx.uint8),
        "x_norm": x_norm.astype(mx.float16),
    }


@unittest.skipUnless(mx.metal.is_available(), "Metal required")
class TestFusedKVCompressD256(unittest.TestCase):
    """Verify fused Metal kernel matches Python reference path for D=256."""

    D = 256
    H_kv = 4

    @classmethod
    def setUpClass(cls):
        cls.centroids, cls.boundaries = _load_codebook_3bit(cls.D)
        if cls.centroids is None:
            return
        cls.block_matrices = _make_rotation_components(cls.H_kv, cls.D)

    def _skip_if_no_codebook(self):
        if self.centroids is None:
            self.skipTest(f"No codebook for dim_{self.D}_3bit")

    def _random_input(self, num_heads=None, seq_len=1, seed=123):
        H = num_heads or self.H_kv
        rng = np.random.default_rng(seed)
        return mx.array(rng.normal(size=(H, seq_len, self.D)).astype(np.float32))

    def test_fused_vs_python_indices(self):
        self._skip_if_no_codebook()
        x = self._random_input()
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=256: Fused indices differ from Python path",
        )

    def test_fused_vs_python_norms(self):
        self._skip_if_no_codebook()
        x = self._random_input()
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, fused_norms, _ = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["x_norm"], fused_norms)
        np.testing.assert_allclose(
            np.array(fused_norms),
            np.array(ref["x_norm"]),
            atol=1e-3,
            rtol=1e-2,
            err_msg="D=256: Fused norms differ from Python path",
        )

    def test_fused_vs_python_packed(self):
        self._skip_if_no_codebook()
        x = self._random_input()
        fused_packed, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        ref_packed = pack_indices_3bit(fused_indices)
        mx.eval(fused_packed, ref_packed)
        np.testing.assert_array_equal(
            np.array(fused_packed),
            np.array(ref_packed),
            err_msg="D=256: Fused packing differs from pack_indices_3bit",
        )

    def test_roundtrip_dequant(self):
        self._skip_if_no_codebook()
        from mlx_lm.models.mlx_isoquant import structured_rotate_inverse

        x = self._random_input(seed=999)
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, fused_norms, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(fused_indices, fused_norms, ref["indices"], ref["x_norm"])

        x_quant_fused = self.centroids[fused_indices.astype(mx.int32)]
        x_inv_fused = structured_rotate_inverse(
            x_quant_fused, self.block_matrices, True
        )
        x_recon_fused = x_inv_fused * fused_norms

        x_quant_ref = self.centroids[ref["indices"].astype(mx.int32)]
        x_inv_ref = structured_rotate_inverse(x_quant_ref, self.block_matrices, True)
        x_recon_ref = x_inv_ref * ref["x_norm"]

        mx.eval(x_recon_fused, x_recon_ref)

        err_fused = np.mean(np.abs(np.array(x_recon_fused) - np.array(x)))
        err_ref = np.mean(np.abs(np.array(x_recon_ref) - np.array(x)))
        self.assertAlmostEqual(
            err_fused,
            err_ref,
            places=3,
            msg=f"D=256: Reconstruction errors differ: fused={err_fused:.6f} vs ref={err_ref:.6f}",
        )

    def test_batch_shapes(self):
        self._skip_if_no_codebook()
        packed_bytes = self.D // 8 * 3
        for seq_len in [1, 4]:
            x = self._random_input(seq_len=seq_len)
            packed, norms, indices = fused_compress_and_pack(
                x, self.block_matrices, self.centroids, self.boundaries
            )
            mx.eval(packed, norms, indices)
            self.assertEqual(
                packed.shape,
                (self.H_kv, seq_len, packed_bytes),
                f"D=256: packed shape wrong for seq_len={seq_len}",
            )
            self.assertEqual(
                norms.shape,
                (self.H_kv, seq_len, 1),
                f"D=256: norms shape wrong for seq_len={seq_len}",
            )
            self.assertEqual(
                indices.shape,
                (self.H_kv, seq_len, self.D),
                f"D=256: indices shape wrong for seq_len={seq_len}",
            )

    def test_fwht_stage_correctness(self):
        self._skip_if_no_codebook()
        x = self._random_input(seed=777)
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=256: FWHT stage produces different quantization indices",
        )

    def test_boundary_edge_values(self):
        self._skip_if_no_codebook()
        boundaries_np = np.array(self.boundaries)
        rng = np.random.default_rng(42)
        x = mx.array(rng.normal(size=(self.H_kv, 1, self.D)).astype(np.float32))
        x = x * float(boundaries_np.max() - boundaries_np.min()) * 2

        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=256: Boundary edge case: fused indices differ",
        )


@unittest.skipUnless(mx.metal.is_available(), "Metal required")
class TestFusedKVCompressD128(unittest.TestCase):
    """Verify fused Metal kernel matches Python reference path for D=128."""

    D = 128
    H_kv = 4

    @classmethod
    def setUpClass(cls):
        cls.centroids, cls.boundaries = _load_codebook_3bit(cls.D)
        if cls.centroids is None:
            return
        cls.block_matrices = _make_rotation_components(cls.H_kv, cls.D)

    def _skip_if_no_codebook(self):
        if self.centroids is None:
            self.skipTest(f"No codebook for dim_{self.D}_3bit")

    def _random_input(self, num_heads=None, seq_len=1, seed=123):
        H = num_heads or self.H_kv
        rng = np.random.default_rng(seed)
        return mx.array(rng.normal(size=(H, seq_len, self.D)).astype(np.float32))

    def test_fused_vs_python_indices(self):
        self._skip_if_no_codebook()
        x = self._random_input()
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=128: Fused indices differ from Python path",
        )

    def test_fused_vs_python_norms(self):
        self._skip_if_no_codebook()
        x = self._random_input()
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, fused_norms, _ = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["x_norm"], fused_norms)
        np.testing.assert_allclose(
            np.array(fused_norms),
            np.array(ref["x_norm"]),
            atol=1e-3,
            rtol=1e-2,
            err_msg="D=128: Fused norms differ from Python path",
        )

    def test_fused_vs_python_packed(self):
        """Fused packed must match pack_indices_3bit — tests cooperative packing."""
        self._skip_if_no_codebook()
        x = self._random_input()
        fused_packed, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        ref_packed = pack_indices_3bit(fused_indices)
        mx.eval(fused_packed, ref_packed)
        np.testing.assert_array_equal(
            np.array(fused_packed),
            np.array(ref_packed),
            err_msg="D=128: Fused cooperative packing differs from pack_indices_3bit",
        )

    def test_roundtrip_dequant(self):
        self._skip_if_no_codebook()
        from mlx_lm.models.mlx_isoquant import structured_rotate_inverse

        x = self._random_input(seed=999)
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, fused_norms, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(fused_indices, fused_norms, ref["indices"], ref["x_norm"])

        x_quant_fused = self.centroids[fused_indices.astype(mx.int32)]
        x_inv_fused = structured_rotate_inverse(
            x_quant_fused, self.block_matrices, True
        )
        x_recon_fused = x_inv_fused * fused_norms

        x_quant_ref = self.centroids[ref["indices"].astype(mx.int32)]
        x_inv_ref = structured_rotate_inverse(x_quant_ref, self.block_matrices, True)
        x_recon_ref = x_inv_ref * ref["x_norm"]

        mx.eval(x_recon_fused, x_recon_ref)

        err_fused = np.mean(np.abs(np.array(x_recon_fused) - np.array(x)))
        err_ref = np.mean(np.abs(np.array(x_recon_ref) - np.array(x)))
        self.assertAlmostEqual(
            err_fused,
            err_ref,
            places=3,
            msg=f"D=128: Reconstruction errors differ: fused={err_fused:.6f} vs ref={err_ref:.6f}",
        )

    def test_batch_shapes(self):
        self._skip_if_no_codebook()
        packed_bytes = self.D // 8 * 3  # 48 for D=128
        for seq_len in [1, 4]:
            x = self._random_input(seq_len=seq_len)
            packed, norms, indices = fused_compress_and_pack(
                x, self.block_matrices, self.centroids, self.boundaries
            )
            mx.eval(packed, norms, indices)
            self.assertEqual(
                packed.shape,
                (self.H_kv, seq_len, packed_bytes),
                f"D=128: packed shape wrong for seq_len={seq_len}",
            )
            self.assertEqual(
                norms.shape,
                (self.H_kv, seq_len, 1),
                f"D=128: norms shape wrong for seq_len={seq_len}",
            )
            self.assertEqual(
                indices.shape,
                (self.H_kv, seq_len, self.D),
                f"D=128: indices shape wrong for seq_len={seq_len}",
            )

    def test_fwht_stage_correctness(self):
        self._skip_if_no_codebook()
        x = self._random_input(seed=777)
        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=128: FWHT stage produces different quantization indices",
        )

    def test_boundary_edge_values(self):
        self._skip_if_no_codebook()
        boundaries_np = np.array(self.boundaries)
        rng = np.random.default_rng(42)
        x = mx.array(rng.normal(size=(self.H_kv, 1, self.D)).astype(np.float32))
        x = x * float(boundaries_np.max() - boundaries_np.min()) * 2

        ref = _python_compress_batch(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        _, _, fused_indices = fused_compress_and_pack(
            x, self.block_matrices, self.centroids, self.boundaries
        )
        mx.eval(ref["indices"], fused_indices)
        np.testing.assert_array_equal(
            np.array(fused_indices),
            np.array(ref["indices"]),
            err_msg="D=128: Boundary edge case: fused indices differ",
        )


class TestFusedEncodeIntegration(unittest.TestCase):
    """Integration tests through IsoQuantKVCache.update_and_fetch."""

    def _make_cache(self, fused_encode="0", bit_width=3, head_dim=256, num_heads=4):
        from unittest import mock

        env = {
            "ISOQUANT_FUSED_ENCODE": fused_encode,
            "ISOQUANT_CACHE_MODE": "concat_append",
        }
        with mock.patch.dict("os.environ", env):
            from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
            from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

            cache = IsoQuantKVCache(
                num_heads=num_heads,
                head_dim=head_dim,
                bit_width=bit_width,
                codebook_dir=get_default_codebook_dir(),
            )
        return cache

    def _skip_if_fallback(self, cache):
        if cache._fallback_cache is not None:
            self.skipTest("No codebook available")

    def _run_fused_vs_unfused(self, head_dim):
        """Common logic: compare fused vs unfused decode through update_and_fetch."""
        cache_fused = self._make_cache(fused_encode="1", head_dim=head_dim)
        cache_unfused = self._make_cache(fused_encode="0", head_dim=head_dim)
        self._skip_if_fallback(cache_fused)
        self._skip_if_fallback(cache_unfused)

        self.assertTrue(cache_fused._use_fused_encode)
        self.assertFalse(cache_unfused._use_fused_encode)

        rng = np.random.default_rng(42)
        prefill_k = mx.array(rng.normal(size=(1, 4, 8, head_dim)).astype(np.float16))
        prefill_v = mx.array(rng.normal(size=(1, 4, 8, head_dim)).astype(np.float16))
        cache_fused.update_and_fetch(prefill_k, prefill_v)
        cache_unfused.update_and_fetch(prefill_k, prefill_v)
        cache_fused.finalize_deferred_prefill()
        cache_unfused.finalize_deferred_prefill()

        for i in range(3):
            dk = mx.array(rng.normal(size=(1, 4, 1, head_dim)).astype(np.float16))
            dv = mx.array(rng.normal(size=(1, 4, 1, head_dim)).astype(np.float16))
            cache_fused.update_and_fetch(dk, dv)
            cache_unfused.update_and_fetch(dk, dv)

        mx.eval(
            cache_fused.compressed_keys["indices"],
            cache_unfused.compressed_keys["indices"],
        )
        np.testing.assert_array_equal(
            np.array(cache_fused.compressed_keys["indices"]),
            np.array(cache_unfused.compressed_keys["indices"]),
            err_msg=f"D={head_dim}: Fused encode produces different cache indices",
        )
        np.testing.assert_array_equal(
            np.array(cache_fused._packed_keys_cache),
            np.array(cache_unfused._packed_keys_cache),
            err_msg=f"D={head_dim}: Fused encode produces different packed cache",
        )

    def test_fused_vs_unfused_d256(self):
        self._run_fused_vs_unfused(256)

    def test_fused_vs_unfused_d128(self):
        self._run_fused_vs_unfused(128)

    def test_fallback_on_failure(self):
        from unittest import mock

        cache = self._make_cache(fused_encode="1")
        self._skip_if_fallback(cache)
        self.assertTrue(cache._use_fused_encode)

        rng = np.random.default_rng(99)
        prefill_k = mx.array(rng.normal(size=(1, 4, 4, 256)).astype(np.float16))
        prefill_v = mx.array(rng.normal(size=(1, 4, 4, 256)).astype(np.float16))
        cache.update_and_fetch(prefill_k, prefill_v)
        cache.finalize_deferred_prefill()

        with mock.patch(
            "mlx_lm.models.fused_kv_compress.fused_compress_and_pack",
            side_effect=RuntimeError("Metal compile failure"),
        ):
            dk = mx.array(rng.normal(size=(1, 4, 1, 256)).astype(np.float16))
            dv = mx.array(rng.normal(size=(1, 4, 1, 256)).astype(np.float16))
            cache.update_and_fetch(dk, dv)

        self.assertFalse(cache._use_fused_encode)
        self.assertIsNotNone(cache.compressed_keys)
        self.assertEqual(cache.compressed_keys["indices"].shape[1], 5)

    def test_gate_rejects_non_3bit(self):
        cache = self._make_cache(fused_encode="1", bit_width=4)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook for 4-bit")
        self.assertFalse(cache._use_fused_encode)

    def test_gate_rejects_unsupported_dim(self):
        """Fused encode must not activate for head_dim not in {128, 256}."""
        cache = self._make_cache(fused_encode="1", head_dim=64)
        if cache._fallback_cache is not None:
            self.skipTest("No codebook for dim=64")
        self.assertFalse(cache._use_fused_encode)

    def test_gate_rejects_iso_v1(self):
        from unittest import mock

        env = {
            "ISOQUANT_FUSED_ENCODE": "1",
            "ISOQUANT_CACHE_MODE": "concat_append",
        }
        with mock.patch.dict("os.environ", env):
            from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
            from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

            cache = IsoQuantKVCache(
                num_heads=4,
                head_dim=256,
                bit_width=3,
                codebook_dir=get_default_codebook_dir(),
            )
            if cache._fallback_cache is not None:
                self.skipTest("No codebook")
            cache._apply_global_mix = False
            cache._use_hadamard = False
            cache._resolve_fused_encode_gate()
            self.assertFalse(cache._use_fused_encode)


if __name__ == "__main__":
    unittest.main()
