"""Scoring through the decode path (scripts/_kv_scoring.py) on a tiny random model.

The model is a 4-layer qwen2 with head_dim 64 and random weights; the tests
check bookkeeping and which cache paths the scored positions read, not model
quality.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import mlx.core as mx
from mlx.utils import tree_map

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _kv_scoring  # noqa: E402
from mlx_lm.models import mlx_isoquant  # noqa: E402
from mlx_lm.models.base import scaled_dot_product_attention  # noqa: E402
from mlx_lm.models.cache import create_causal_mask, make_prompt_cache  # noqa: E402
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache  # noqa: E402
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir  # noqa: E402

N_TOKENS = 40
PREFIX = 20
# Every layer quantized; the MLX encoder keeps CPU and Metal runs identical.
ENV = {"TURBOQUANT_SKIP_LAYERS": "0", "ISOQUANT_FUSED_ENCODE": "0"}


def _tiny_model(seed=0):
    from mlx_lm.models import qwen2

    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=256,
    )
    mx.random.seed(seed)
    model = qwen2.Model(args)
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
    mx.eval(model.parameters())
    return model


class KVScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.tokens = [int(t) for t in np.random.default_rng(1).integers(0, 256, N_TOKENS)]

    def setUp(self):
        self._env = mock.patch.dict(os.environ, ENV)
        self._env.start()
        mlx_isoquant.reset_stats()

    def tearDown(self):
        self._env.stop()

    def score(self, kv_type, **kw):
        kw.setdefault("prefix_tokens", PREFIX)
        return _kv_scoring.score_tokens(self.model, self.tokens, kv_type, **kw)

    # ----- bookkeeping on the default backend

    def test_prefix_decode_matches_single_pass_on_default_backend(self):
        single = self.score("default", mode=_kv_scoring.SINGLE_PASS)
        prefix = self.score("default")
        self.assertEqual(single["n_scored"], N_TOKENS - 1)
        self.assertEqual(prefix["n_scored"], N_TOKENS - PREFIX - 1)
        self.assertEqual(prefix["prefix_tokens"], PREFIX)
        # single-pass position i predicts token i+1; prefix mode scores tokens PREFIX+1..N-1
        np.testing.assert_allclose(
            prefix["token_nll"], single["token_nll"][PREFIX:], atol=1e-3, rtol=1e-4
        )
        self.assertFalse(prefix["compressed_path"]["checked"])

    def test_chunked_prefix_decode_matches_token_by_token(self):
        one = self.score("default", chunk=1)
        four = self.score("default", chunk=4)
        self.assertEqual(one["n_scored"], four["n_scored"])
        np.testing.assert_allclose(four["token_nll"], one["token_nll"], atol=1e-3, rtol=1e-4)

    def test_resolve_prefix_bounds(self):
        self.assertEqual(_kv_scoring.resolve_prefix(40, 10, None), 29)
        self.assertEqual(_kv_scoring.resolve_prefix(40, 100, None), 1)
        self.assertEqual(_kv_scoring.resolve_prefix(40, None, 39), 38)
        self.assertEqual(_kv_scoring.resolve_prefix(40, None, None), 20)
        with self.assertRaises(ValueError):
            _kv_scoring.resolve_prefix(2, None, None)

    # ----- IsoQuant: which path did the scored positions read?

    def test_isoquant_prefix_decode_reads_compressed_cache(self):
        r = self.score("isoquant")
        checks = r["compressed_path"]
        self.assertTrue(checks["checked"])
        self.assertTrue(checks["ok"], checks["reasons"])
        self.assertEqual(checks["n_isoquant_caches"], 4)
        prefill, scoring = r["paths"]["prefill"], r["paths"]["scoring"]
        self.assertGreater(prefill.get("deferred_fp16", 0), 0)
        self.assertEqual(scoring.get("deferred_fp16", 0), 0)
        self.assertEqual(scoring.get("decode_write", 0), 4 * r["n_scored"])
        reads = scoring.get("fused_metal", 0) + scoring.get("fused_mlx", 0)
        self.assertEqual(reads, 4 * r["n_scored"])
        self.assertEqual(scoring.get("unfused_dense", 0), 0)

    def test_isoquant_single_pass_scores_fp16(self):
        r = self.score("isoquant", mode=_kv_scoring.SINGLE_PASS)
        checks = r["compressed_path"]
        self.assertFalse(checks["ok"])
        self.assertTrue(any("FP16" in x or "no decode reads" in x for x in checks["reasons"]))
        scoring = r["paths"]["scoring"]
        self.assertGreater(scoring.get("deferred_fp16", 0), 0)
        self.assertEqual(
            sum(scoring.get(p, 0) for p in ("fused_metal", "fused_mlx", "unfused_dense")), 0
        )

    def test_isoquant_chunked_scoring_uses_dense_path_and_agrees(self):
        one = self.score("isoquant", chunk=1)
        four = self.score("isoquant", chunk=4)
        self.assertTrue(four["compressed_path"]["ok"], four["compressed_path"]["reasons"])
        self.assertGreater(four["paths"]["scoring"].get("unfused_dense", 0), 0)
        self.assertEqual(four["paths"]["scoring"].get("deferred_fp16", 0), 0)
        # Same quantized cache, fused vs dense read: only rounding differs.
        self.assertLess(abs(four["nll"] - one["nll"]), 1e-2)

    def test_isoquant_without_deferral_compresses_during_prefill(self):
        r = self.score("isoquant", defer_prefill=False)
        prefill = r["paths"]["prefill"]
        self.assertEqual(prefill.get("deferred_fp16", 0), 0)
        self.assertEqual(prefill.get("decode_write", 0), 4)
        self.assertEqual(prefill.get("unfused_dense", 0), 4)
        self.assertTrue(r["compressed_path"]["ok"], r["compressed_path"]["reasons"])

    # ----- TurboQuant

    def test_turboquant_prefix_decode_reads_compressed_cache(self):
        r = self.score("turboquant")
        checks = r["compressed_path"]
        self.assertTrue(checks["ok"], checks["reasons"])
        self.assertEqual(checks["n_turboquant_caches"], 4)
        self.assertEqual(r["n_scored"], N_TOKENS - PREFIX - 1)

    def test_turboquant_quantizes_during_prefill(self):
        """TurboQuant never defers: the prefix is already quantized after prefill."""
        cache = make_prompt_cache(self.model, kv_cache_type="turboquant")
        prompt = mx.array(self.tokens[:PREFIX])[None]
        mx.eval(self.model(prompt, cache=cache))
        for c in _kv_scoring.iter_caches(cache):
            self.assertIsNone(c._fallback_cache)
            self.assertTrue(c.compressed_keys)
            self.assertEqual(c.offset, PREFIX)

    def test_turboquant_single_pass_already_reads_quantized_cache(self):
        """The asymmetry of the old harness: TurboQuant is quantized, IsoQuant is not."""
        r = self.score("turboquant", mode=_kv_scoring.SINGLE_PASS)
        self.assertTrue(r["compressed_path"]["ok"], r["compressed_path"]["reasons"])

    # ----- multi-query attention on a finalized IsoQuant cache

    def test_multi_query_fused_attention_matches_dense(self):
        cache = IsoQuantKVCache(
            num_heads=2, head_dim=64, bit_width=3, layer_idx=0,
            codebook_dir=get_default_codebook_dir(), seed=42,
        )
        if cache._fallback_cache is not None:
            self.skipTest("No codebook for head_dim=64")
        rng = np.random.default_rng(3)

        def kv(n):
            return mx.array(rng.normal(size=(1, 2, n, 64)).astype(np.float32))

        cache.update_and_fetch(kv(16), kv(16))
        cache.finalize_deferred_prefill()
        cache.update_and_fetch(kv(3), kv(3))  # three new tokens, compressed on write
        q = mx.array(rng.normal(size=(1, 4, 3, 64)).astype(np.float32))
        mask = create_causal_mask(3, 16)
        out = cache.fused_attention(q, scale=64**-0.5, mask=mask)
        ref = scaled_dot_product_attention(
            q, cache.reconstruct_keys(), cache.get_values(), cache=None, scale=64**-0.5, mask=mask
        )
        mx.eval(out, ref)
        self.assertEqual(out.shape, (1, 4, 3, 64))
        np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-5, rtol=1e-5)

    def test_reconstructed_kv_follow_model_dtype(self):
        cache = IsoQuantKVCache(
            num_heads=2, head_dim=64, bit_width=3, layer_idx=0,
            codebook_dir=get_default_codebook_dir(), seed=42,
        )
        if cache._fallback_cache is not None:
            self.skipTest("No codebook for head_dim=64")
        x = mx.random.normal((1, 2, 8, 64)).astype(mx.float32)
        cache.update_and_fetch(x, x)
        cache.finalize_deferred_prefill()
        self.assertEqual(cache.reconstruct_keys().dtype, mx.float32)
        self.assertEqual(cache.get_values().dtype, mx.float32)


if __name__ == "__main__":
    unittest.main()
