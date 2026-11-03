#!/usr/bin/env python3
"""Benchmark: dense SDPA vs MLX-ops fused vs 3-kernel Metal.

Measures per-call latency for each IsoQuant attention execution path
so you can see whether the fused pipeline actually helps or whether
expert I/O dominates.

Usage:
    python -m scripts.benchmark_fused_attention [--heads 4] [--seq 128] [--dim 128] [--iters 50]
    python -m scripts.benchmark_fused_attention --sweep   # run all standard configs
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mlx.core as mx

from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir


def _make_cache(num_heads: int, head_dim: int, seq_len: int, seed: int = 42):
    cache = IsoQuantKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        bit_width=3,
        codebook_dir=get_default_codebook_dir(),
    )
    if cache._fallback_cache is not None:
        raise RuntimeError("No codebook found for this config")

    rng = np.random.default_rng(seed)
    keys = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )
    values = mx.array(
        rng.normal(size=(1, num_heads, seq_len, head_dim)).astype(np.float16)
    )
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()
    return cache


def bench_dense_sdpa(cache, queries, scale, warmup=5, iters=50):
    """Path 1: reconstruct_keys() + get_values() + SDPA."""
    for _ in range(warmup):
        k = cache.reconstruct_keys()
        v = cache.get_values()
        out = scaled_dot_product_attention(
            queries, k, v, cache=None, scale=scale, mask=None
        )
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        k = cache.reconstruct_keys()
        v = cache.get_values()
        out = scaled_dot_product_attention(
            queries, k, v, cache=None, scale=scale, mask=None
        )
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def bench_fused_mlx(cache, queries, scale, warmup=5, iters=50):
    """Path 2: MLX-ops fused (centroid gather + dense matmul, no Metal kernels)."""
    saved = cache._fused_metal_ok
    cache._fused_metal_ok = False

    for _ in range(warmup):
        out = cache.fused_attention(queries, scale=scale)
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = cache.fused_attention(queries, scale=scale)
        mx.eval(out)
    elapsed = time.perf_counter() - t0

    cache._fused_metal_ok = saved
    return elapsed / iters


def bench_3kernel_metal(cache, queries, scale, warmup=5, iters=50):
    """Path 3: Legacy 3-kernel pipeline (QK dot + softmax + V accum + inverse rot)."""
    from mlx_lm.models.fused_kv_decode_kernels import (
        fused_qk_dot,
        fused_value_accum,
        pack_indices_3bit,
    )

    if cache._packed_keys_cache is None:
        cache._packed_keys_cache = pack_indices_3bit(cache.compressed_keys["indices"])
    if cache._packed_values_cache is None:
        cache._packed_values_cache = pack_indices_3bit(
            cache.compressed_values["indices"]
        )

    B, H_q, _, D = queries.shape
    H_kv = cache.num_heads
    T = cache.compressed_keys["indices"].shape[1]
    repeats = H_q // H_kv if H_q != H_kv else 1

    k_packed = cache._packed_keys_cache
    v_packed = cache._packed_values_cache
    k_norms = cache.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
    v_norms = cache.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
    centroids = cache.compressor.centroids.reshape(-1).astype(mx.float32)
    kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

    R_T = mx.swapaxes(cache.rotation_matrices, -2, -1)
    q_flat = queries[0, :, 0, :]
    R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
    q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)

    def _run_3kernel():
        scores = fused_qk_dot(
            k_packed, centroids, k_norms, q_rot, kv_head_map, H_q, T, D
        )
        scores = scores * scale
        attn_w = mx.softmax(scores, axis=-1)
        out_rot = fused_value_accum(
            v_packed, centroids, v_norms, attn_w, kv_head_map, H_q, T, D
        )
        from mlx_lm.models.fused_kv_decode_kernels import fused_inverse_rotate

        out = fused_inverse_rotate(out_rot, cache.block_matrices, cache._use_hadamard)
        return out

    for _ in range(warmup):
        out = _run_3kernel()
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = _run_3kernel()
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def run_config(heads, seq, dim, iters, warmup):
    """Run all 3 paths for a single configuration."""
    cache = _make_cache(heads, dim, seq)
    rng = np.random.default_rng(99)
    queries = mx.array(rng.normal(size=(1, heads, 1, dim)).astype(np.float32))
    scale = dim**-0.5

    t_dense = bench_dense_sdpa(cache, queries, scale, warmup, iters)
    t_mlx = bench_fused_mlx(cache, queries, scale, warmup, iters)
    t_3kernel = bench_3kernel_metal(cache, queries, scale, warmup, iters)

    return t_dense, t_mlx, t_3kernel


def print_results(label, t_dense, t_mlx, t_3kernel):
    """Print results for one configuration."""
    print(f"  {label}")
    print(f"    Dense SDPA:           {t_dense * 1000:8.3f} ms/call")
    print(
        f"    Fused MLX-ops:        {t_mlx * 1000:8.3f} ms/call  ({t_dense / t_mlx:5.2f}x vs dense)"
    )
    if t_3kernel is not None:
        print(
            f"    3-kernel Metal:       {t_3kernel * 1000:8.3f} ms/call  ({t_dense / t_3kernel:5.2f}x vs dense)"
        )
    else:
        print("    3-kernel Metal:       FAILED")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused attention paths")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--sweep", action="store_true", help="Run standard config sweep"
    )
    args = parser.parse_args()

    if args.sweep:
        configs = [
            (4, 16, 128, "H=4 T=16 D=128"),
            (4, 64, 128, "H=4 T=64 D=128"),
            (4, 128, 128, "H=4 T=128 D=128"),
            (4, 512, 128, "H=4 T=512 D=128"),
            (8, 128, 128, "H=8 T=128 D=128"),
            (8, 2048, 128, "H=8 T=2048 D=128"),
        ]
        print(f"Sweep: {len(configs)} configs, {args.iters} iters each")
        print("=" * 72)
        print()
        for heads, seq, dim, label in configs:
            t_dense, t_mlx, t_3k = run_config(heads, seq, dim, args.iters, args.warmup)
            print_results(label, t_dense, t_mlx, t_3k)
    else:
        print(f"Config: H={args.heads}, T={args.seq}, D={args.dim}, iters={args.iters}")
        print()

        t_dense, t_mlx, t_3k = run_config(
            args.heads, args.seq, args.dim, args.iters, args.warmup
        )
        print_results(f"H={args.heads} T={args.seq} D={args.dim}", t_dense, t_mlx, t_3k)


if __name__ == "__main__":
    main()
