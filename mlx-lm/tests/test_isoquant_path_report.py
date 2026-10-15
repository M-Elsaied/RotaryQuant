"""Per-layer IsoQuant path report: which attention/encode path each layer ran."""

import unittest
import warnings
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_lm.models import mlx_isoquant as iso
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

H, D = 2, 128


def _cache(layer_idx=3, head_dim=D, bit_width=3):
    return IsoQuantKVCache(
        num_heads=H,
        head_dim=head_dim,
        bit_width=bit_width,
        layer_idx=layer_idx,
        codebook_dir=get_default_codebook_dir(),
        seed=42,
    )


def _kv(T, head_dim=D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.normal(size=(1, H, T, head_dim)).astype(np.float32))
    v = mx.array(rng.normal(size=(1, H, T, head_dim)).astype(np.float32))
    return k, v


def _prefill_and_decode(cache, n_decode, read=True, head_dim=D):
    cache.update_and_fetch(*_kv(8, head_dim))
    cache.finalize_deferred_prefill()
    q = mx.array(np.random.default_rng(1).normal(size=(1, H, 1, head_dim)), mx.float32)
    for i in range(n_decode):
        cache.update_and_fetch(*_kv(1, head_dim, seed=10 + i))
        if read:
            out = cache.fused_attention(q, scale=head_dim**-0.5)
            mx.eval(out)


class PathReportTest(unittest.TestCase):
    def setUp(self):
        iso.reset_stats()

    def test_reset_clears_report(self):
        _prefill_and_decode(_cache(), 1)
        self.assertEqual(iso.kv_path_report()["n_layers"], 1)
        iso.reset_stats()
        self.assertEqual(iso.kv_path_report(), {"n_layers": 0, "totals": {}, "layers": {}})

    def test_prefill_is_recorded_as_deferred_fp16(self):
        c = _cache(layer_idx=5)
        c.update_and_fetch(*_kv(8))
        counts = iso.kv_path_report()["layers"]["5"]["counts"]
        self.assertEqual(counts, {"deferred_fp16": 1})

    def test_decode_reads_record_one_path_per_step(self):
        _prefill_and_decode(_cache(layer_idx=7), 4)
        counts = iso.kv_path_report()["layers"]["7"]["counts"]
        self.assertEqual(counts["decode_write"], 4)
        self.assertEqual(counts.get("fused_metal", 0) + counts.get("fused_mlx", 0), 4)
        self.assertNotIn("raw_passthrough", counts)

    def test_metal_failure_is_recorded_and_violates(self):
        c = _cache(layer_idx=2)
        with patch.object(
            IsoQuantKVCache,
            "_fused_attention_metal",
            side_effect=RuntimeError("boom"),
        ):
            _prefill_and_decode(c, 3)
        report = iso.kv_path_report()
        layer = report["layers"]["2"]
        self.assertEqual(layer["counts"]["fused_mlx"], 3)
        self.assertNotIn("fused_metal", layer["counts"])
        self.assertIn("fused_metal: RuntimeError: boom", layer["errors"])
        problems = iso.fused_path_violations(report)
        self.assertTrue(any("fused_mlx" in p for p in problems))
        self.assertTrue(any("boom" in p for p in problems))

    def test_metal_success_has_no_violations(self):
        def fake_metal(self, q_rot, scale, mask, H_q, H_kv, T, D, repeats):
            return mx.zeros((H_q, D))

        with patch.object(IsoQuantKVCache, "_fused_attention_metal", fake_metal):
            _prefill_and_decode(_cache(layer_idx=4), 3)
        report = iso.kv_path_report()
        self.assertEqual(report["layers"]["4"]["counts"]["fused_metal"], 3)
        self.assertEqual(iso.fused_path_violations(report), [])

    def test_write_without_read_is_raw_passthrough(self):
        _prefill_and_decode(_cache(layer_idx=6), 3, read=False)
        report = iso.kv_path_report()
        self.assertEqual(report["layers"]["6"]["counts"]["raw_passthrough"], 3)
        self.assertTrue(iso.fused_path_violations(report))

    def test_reconstruct_after_finalize_is_unfused_dense(self):
        c = _cache(layer_idx=8)
        c.update_and_fetch(*_kv(8))
        mx.eval(c.reconstruct_keys())  # deferred read: not counted again
        c.finalize_deferred_prefill()
        c.update_and_fetch(*_kv(1, seed=3))
        mx.eval(c.reconstruct_keys(), c.get_values())
        counts = iso.kv_path_report()["layers"]["8"]["counts"]
        self.assertEqual(counts["deferred_fp16"], 1)
        self.assertEqual(counts["unfused_dense"], 1)
        self.assertNotIn("raw_passthrough", counts)

    def test_non_3bit_fused_call_is_unfused_dense(self):
        c = _cache(layer_idx=9, bit_width=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _prefill_and_decode(c, 2)
        self.assertTrue(any("bit_width=4" in str(w.message) for w in caught))
        counts = iso.kv_path_report()["layers"]["9"]["counts"]
        self.assertEqual(counts["unfused_dense"], 2)
        self.assertTrue(iso.fused_path_violations(iso.kv_path_report()))

    def test_unsupported_head_dim_records_fallback_cache(self):
        c = _cache(layer_idx=11, head_dim=96)  # no shipped codebook for d=96
        c.update_and_fetch(*_kv(4, head_dim=96))
        layer = iso.kv_path_report()["layers"]["11"]
        self.assertEqual(layer["counts"], {"fp16_fallback_cache": 1})
        self.assertTrue(layer["errors"][0].startswith("fallback_cache:"))
        self.assertTrue(iso.fused_path_violations(iso.kv_path_report()))

    def test_no_layers_is_a_violation(self):
        self.assertEqual(
            iso.fused_path_violations(iso.kv_path_report()),
            ["no IsoQuant layers recorded"],
        )

    def test_encoder_errors_are_reported_not_violations(self):
        def fake_metal(self, q_rot, scale, mask, H_q, H_kv, T, D, repeats):
            return mx.zeros((H_q, D))

        with patch.object(IsoQuantKVCache, "_fused_attention_metal", fake_metal):
            _prefill_and_decode(_cache(layer_idx=13), 2)
        iso._record_error(13, "fused_encode", RuntimeError("enc"))
        report = iso.kv_path_report()
        self.assertIn("fused_encode: RuntimeError: enc", report["layers"]["13"]["errors"])
        self.assertEqual(iso.fused_path_violations(report), [])

    def test_layer_with_only_errors_is_reported(self):
        iso._record_error(21, "fallback_cache", "heads=2, head_dim=96")
        report = iso.kv_path_report()
        self.assertEqual(report["layers"]["21"]["counts"], {})
        self.assertTrue(iso.fused_path_violations(report))

    def test_errors_are_capped_and_deduplicated(self):
        e = ValueError("same")
        for _ in range(5):
            iso._record_error(1, "x", e)
        for i in range(5):
            iso._record_error(1, "y", ValueError(str(i)))
        self.assertEqual(len(iso.kv_path_report()["layers"]["1"]["errors"]), 3)

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_real_metal_path_is_fused(self):
        _prefill_and_decode(_cache(layer_idx=12), 3)
        report = iso.kv_path_report()
        self.assertEqual(report["layers"]["12"]["counts"].get("fused_metal"), 3)
        self.assertEqual(iso.fused_path_violations(report), [])


if __name__ == "__main__":
    unittest.main()
