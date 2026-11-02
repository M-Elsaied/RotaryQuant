"""Compare the Metal fused-attention path with the MLX-ops path on one cache.

Builds an IsoQuantKVCache with random K/V at two sequence lengths, runs
``fused_attention`` once on the Metal path and once with the Metal latch
forced off, and prints the largest absolute difference. Exits 0 only when
the Metal kernels ran and agree with the MLX-ops path to within 1e-3.

Usage: python scripts/metal_vs_mlx_parity.py
"""

import sys

import numpy as np
import mlx.core as mx

from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

TOL = 1e-3


def main() -> int:
    rng = np.random.default_rng(0)
    worst = 0.0
    all_metal = True
    for T in (64, 1024):
        cache = IsoQuantKVCache(
            num_heads=2,
            head_dim=128,
            bit_width=3,
            layer_idx=0,
            codebook_dir=get_default_codebook_dir(),
            seed=42,
        )

        def kv(n):
            return mx.array(rng.normal(size=(1, 2, n, 128)).astype(np.float32))

        cache.update_and_fetch(kv(T), kv(T))
        cache.finalize_deferred_prefill()
        cache.update_and_fetch(kv(1), kv(1))
        q = mx.array(rng.normal(size=(1, 8, 1, 128)).astype(np.float32))

        out_metal = cache.fused_attention(q, scale=128**-0.5)
        mx.eval(out_metal)
        metal_ran = cache._fused_metal_ok is True

        cache._fused_metal_ok = False  # force the MLX-ops path on the same cache
        out_mlx = cache.fused_attention(q, scale=128**-0.5)
        mx.eval(out_mlx)

        diff = float(mx.max(mx.abs(out_metal - out_mlx)))
        worst = max(worst, diff)
        all_metal = all_metal and metal_ran
        print(f"T={T} metal_ran={metal_ran} max_abs_diff={diff:.2e}")

    if not all_metal:
        verdict = "NOT-RUN (Metal kernels did not execute)"
    elif worst < TOL:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(f"METAL_VS_MLX {verdict} max_abs_diff={worst:.2e} tol={TOL:g}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
