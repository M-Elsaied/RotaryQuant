"""Compressed (3-bit IsoQuant) attention versus FP16 attention on the same inputs.

The existing fused-attention tests compare the fused kernels against a dense
read of the *same quantized cache*, which checks the kernels but not the
quantization. These tests compare against ``mx.fast.scaled_dot_product_attention``
on the original FP16 keys and values, so the error they report is the
end-to-end effect of compression on one attention step.

Bounds were set from Gaussian K/V/Q on the MLX CPU backend (per-head cosine
0.954 to 0.968, relative L2 error 0.23 to 0.28 across head_dim 64 to 512, GQA 1
to 4 and T 64 to 2048) with a margin. Gaussian inputs are close to the worst
case for a near-uniform softmax (the output is a small mean of many values);
real activations will be added as a fixture when a model-derived capture is
checked in.

Slow cases (T = 32K, head_dim 256 and 512 at long T) run only with
``ISOQUANT_RUN_SLOW_TESTS=1``.
"""

import os
import unittest

import numpy as np
import mlx.core as mx

from mlx_lm.models import mlx_isoquant
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

MIN_COSINE = 0.93  # per head, worst head
MAX_REL_L2 = 0.35  # over all heads

CASES = [  # (head_dim, gqa_repeats, T)
    (64, 1, 64),
    (128, 1, 64),
    (128, 4, 64),
    (128, 8, 256),
    (128, 4, 2048),
    (64, 2, 1024),
    (256, 1, 64),
    (512, 1, 64),
]
SLOW_CASES = [
    (256, 4, 2048),
    (512, 4, 2048),
    (128, 4, 32768),
]


def _run_case(head_dim, repeats, T, seed=0):
    """Return (per-head cosine, relative L2 error, path totals) for one decode step."""
    n_kv = 2
    n_q = n_kv * repeats
    mlx_isoquant.reset_stats()
    cache = IsoQuantKVCache(
        num_heads=n_kv,
        head_dim=head_dim,
        bit_width=3,
        layer_idx=0,
        codebook_dir=get_default_codebook_dir(),
        seed=42,
    )
    if cache._fallback_cache is not None:
        return None

    rng = np.random.default_rng(seed)

    def gauss(*shape):
        return mx.array(rng.normal(size=shape).astype(np.float16))

    keys, values = gauss(1, n_kv, T, head_dim), gauss(1, n_kv, T, head_dim)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()
    k1, v1 = gauss(1, n_kv, 1, head_dim), gauss(1, n_kv, 1, head_dim)
    cache.update_and_fetch(k1, v1)

    scale = head_dim**-0.5
    q = gauss(1, n_q, 1, head_dim)
    ref = mx.fast.scaled_dot_product_attention(
        q, mx.concatenate([keys, k1], axis=2), mx.concatenate([values, v1], axis=2), scale=scale
    )
    out = cache.fused_attention(q, scale=scale)
    mx.eval(out, ref)

    a = np.array(out, dtype=np.float32)[0, :, 0, :]
    b = np.array(ref, dtype=np.float32)[0, :, 0, :]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    totals = mlx_isoquant.kv_path_report()["totals"]
    return cos, rel, totals


class CompressedVsFP16AttentionTest(unittest.TestCase):
    def _check(self, head_dim, repeats, T):
        res = _run_case(head_dim, repeats, T)
        if res is None:
            self.skipTest(f"No 3-bit codebook for head_dim={head_dim}")
        cos, rel, totals = res
        label = f"head_dim={head_dim} gqa={repeats} T={T}"
        self.assertGreaterEqual(
            float(cos.min()), MIN_COSINE, f"{label}: per-head cosine {cos.round(4).tolist()}"
        )
        self.assertLessEqual(rel, MAX_REL_L2, f"{label}: relative L2 error {rel:.4f}")
        # Compression was exercised: prefix compressed at finalize, one decode
        # write, and one decode read on a fused path.
        self.assertGreaterEqual(totals.get("decode_write", 0), 1, label)
        reads = totals.get("fused_metal", 0) + totals.get("fused_mlx", 0)
        self.assertEqual(reads, 1, f"{label}: {totals}")
        self.assertEqual(totals.get("unfused_dense", 0), 0, label)
        self.assertGreaterEqual(mlx_isoquant.stats_summary()["compress_calls"], 1, label)

    def test_cases(self):
        for head_dim, repeats, T in CASES:
            with self.subTest(head_dim=head_dim, gqa=repeats, T=T):
                self._check(head_dim, repeats, T)

    @unittest.skipUnless(
        os.environ.get("ISOQUANT_RUN_SLOW_TESTS") == "1", "set ISOQUANT_RUN_SLOW_TESTS=1"
    )
    def test_slow_cases(self):
        for head_dim, repeats, T in SLOW_CASES:
            with self.subTest(head_dim=head_dim, gqa=repeats, T=T):
                self._check(head_dim, repeats, T)

    def test_error_is_not_trivially_small(self):
        """Guard against a reference that accidentally reads the quantized cache."""
        res = _run_case(128, 1, 64)
        if res is None:
            self.skipTest("No codebook")
        _, rel, _ = res
        self.assertGreater(rel, 1e-3)


if __name__ == "__main__":
    unittest.main()
