#!/usr/bin/env python3
"""MLX Metal kernel benchmarks -- mirrors the Mojo kernel set.

Benchmarks standard kernels (matmul, softmax, RoPE), novel IsoQuant kernels
(structured rotation forward/inverse, KV compress), and fused attention
pipelines (unfused, framework-compiled) with rigorous
statistical methodology:
  - Adaptive iteration targeting CI < 2% median (max 500)
  - BCa bootstrap CIs via scipy.stats.bootstrap (10k samples)
  - Durbin-Watson autocorrelation and Wald-Wolfowitz runs test

Usage:
    python scripts/benchmark_mlx_kernels.py --output results/mlx_kernels.json
    python scripts/benchmark_mlx_kernels.py --kernel matmul --output results/mlx_matmul.json
    python scripts/benchmark_mlx_kernels.py --kernel fused_attention --output results/mlx_fused.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Ensure mlx-lm is importable from the repo
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mlx-lm"))

# Find codebook directory (may differ in worktrees vs main repo)
# In git worktrees, generated artifacts like codebooks may only exist in the
# main working tree, so we check multiple candidate paths.
_MAIN_REPO = os.environ.get(
    "QWEN_REPO_ROOT",
    os.path.join(os.path.expanduser("~"), "QwenCoderLocal"),
)
_CODEBOOK_CANDIDATES = [
    os.path.join(_REPO_ROOT, "mlx-lm", "mlx_lm", "models", "turboquant_codebooks"),
    os.path.join(_REPO_ROOT, "turboquant-mlx", "codebooks"),
    os.path.join(_REPO_ROOT, "codebooks"),
    os.path.join(_MAIN_REPO, "mlx-lm", "mlx_lm", "models", "turboquant_codebooks"),
    os.path.join(_MAIN_REPO, "turboquant-mlx", "codebooks"),
]


def _find_codebook_dir() -> str | None:
    """Find the first existing codebook directory."""
    for path in _CODEBOOK_CANDIDATES:
        if os.path.isdir(path):
            return path
    # Also try the mlx_turboquant default
    try:
        from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

        d = get_default_codebook_dir()
        if os.path.isdir(d):
            return d
    except ImportError:
        pass
    return None


import mlx.core as mx

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

SEED = 42
_BENCH_CONFIG = {
    "max_iters": 500,
    "min_iters": 20,
    "target_ci_pct": 2.0,  # target CI width as % of median
    "bootstrap_samples": 10_000,
}


@dataclass
class BenchResult:
    name: str
    shape: str
    dtype: str
    mean_us: float = 0.0
    median_us: float = 0.0
    std_us: float = 0.0
    p5_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci95_bca: list[float] = field(default_factory=lambda: [0.0, 0.0])
    ci_converged: bool = False
    ci_width_pct: float = 0.0
    n_iterations: int = 0
    dw_statistic: float = 0.0
    runs_test_p: float = 0.0
    tflops: float | None = None
    gbs: float | None = None
    roofline_pct: float | None = None
    flops_total: int | None = None
    bytes_accessed: int | None = None
    rmse: float | None = None
    sub_timings: dict | None = None


def _durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for autocorrelation in residuals."""
    if len(residuals) < 3:
        return 2.0
    diffs = np.diff(residuals)
    denom = np.sum(residuals**2)
    if denom == 0:
        return 2.0
    return float(np.sum(diffs**2) / denom)


def _runs_test_p(data: np.ndarray) -> float:
    """Wald-Wolfowitz runs test p-value for randomness."""
    if len(data) < 10:
        return 1.0
    median = np.median(data)
    binary = data >= median
    runs = 1 + np.sum(binary[1:] != binary[:-1])
    n1 = np.sum(binary)
    n0 = len(binary) - n1
    if n0 == 0 or n1 == 0:
        return 1.0
    n = n0 + n1
    expected = 1 + 2 * n0 * n1 / n
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n**2 * (n - 1))
    if var <= 0:
        return 1.0
    z = (runs - expected) / math.sqrt(var)
    # Two-tailed p-value using normal approximation
    from scipy.stats import norm

    return float(2 * norm.sf(abs(z)))


def _bca_ci(samples: np.ndarray) -> tuple[list[float], float]:
    """BCa bootstrap 95% CI via scipy. Returns ([lo, hi], width_pct_of_median)."""
    from scipy.stats import bootstrap

    rng = np.random.default_rng(SEED)
    res = bootstrap(
        (samples,),
        statistic=np.median,
        n_resamples=_BENCH_CONFIG["bootstrap_samples"],
        confidence_level=0.95,
        method="BCa",
        random_state=rng,
    )
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    med = float(np.median(samples))
    width_pct = (hi - lo) / med * 100 if med > 0 else 0.0
    return [lo, hi], width_pct


def adaptive_bench(
    fn: Callable[[], None],
    warmup: int = 10,
    stream: mx.Stream | None = None,
) -> BenchResult:
    """Run fn adaptively until CI < target or max iters reached.

    All operations are synced via mx.eval inside fn, plus a final
    mx.synchronize() on the GPU stream.
    """
    max_iters = _BENCH_CONFIG["max_iters"]
    min_iters = _BENCH_CONFIG["min_iters"]
    target_ci_pct = _BENCH_CONFIG["target_ci_pct"]

    # Warmup
    for _ in range(warmup):
        fn()
        mx.synchronize()

    timings = []
    for i in range(max_iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # microseconds

        if i + 1 >= min_iters and (i + 1) % 10 == 0:
            arr = np.array(timings)
            _, ci_pct = _bca_ci(arr)
            if ci_pct < target_ci_pct:
                break

    arr = np.array(timings)
    ci, ci_pct = _bca_ci(arr)
    residuals = arr - np.mean(arr)

    result = BenchResult(
        name="",
        shape="",
        dtype="",
        mean_us=float(np.mean(arr)),
        median_us=float(np.median(arr)),
        std_us=float(np.std(arr)),
        p5_us=float(np.percentile(arr, 5)),
        p95_us=float(np.percentile(arr, 95)),
        p99_us=float(np.percentile(arr, 99)),
        ci95_bca=ci,
        ci_converged=ci_pct < target_ci_pct,
        ci_width_pct=round(ci_pct, 4),
        n_iterations=len(timings),
        dw_statistic=_durbin_watson(residuals),
        runs_test_p=_runs_test_p(arr),
    )
    return result


# ---------------------------------------------------------------------------
# Standard kernels
# ---------------------------------------------------------------------------


def bench_matmul(M: int, N: int, K: int, dtype: str = "float16") -> BenchResult:
    """Benchmark mx.matmul with shapes (M, K) @ (K, N)."""
    dt = mx.float16 if dtype == "float16" else mx.float32
    mx.random.seed(SEED)
    a = mx.random.normal((M, K), dtype=dt, stream=mx.gpu)
    b = mx.random.normal((K, N), dtype=dt, stream=mx.gpu)
    mx.eval(a, b)

    compiled_fn = mx.compile(lambda: mx.matmul(a, b, stream=mx.gpu))

    def run():
        c = compiled_fn()
        mx.eval(c)

    result = adaptive_bench(run)
    result.name = "matmul"
    result.shape = f"{M}x{N}x{K}"
    result.dtype = dtype
    flops = 2 * M * N * K
    itemsize = 2 if dtype == "float16" else 4
    result.flops_total = flops
    result.bytes_accessed = (M * K + K * N + M * N) * itemsize
    result.tflops = flops / (result.median_us * 1e-6) / 1e12
    return result


def bench_softmax(B: int, H: int, S: int) -> BenchResult:
    """Benchmark mx.softmax on shape (B, H, S, S)."""
    mx.random.seed(SEED)
    x = mx.random.normal((B, H, S, S), dtype=mx.float16, stream=mx.gpu)
    mx.eval(x)

    compiled_fn = mx.compile(lambda: mx.softmax(x, axis=-1, stream=mx.gpu))

    def run():
        out = compiled_fn()
        mx.eval(out)

    result = adaptive_bench(run)
    result.name = "softmax"
    result.shape = f"{B}x{H}x{S}x{S}"
    result.dtype = "float16"
    numel = B * H * S * S
    bytes_moved = 2 * numel * 2  # fp16 = 2 bytes (read + write)
    result.bytes_accessed = bytes_moved
    result.flops_total = numel * 5  # exp + sub + div + sum + max per element (approx)
    result.gbs = bytes_moved / (result.median_us * 1e-6) / 1e9
    return result


def _build_rope_freqs(
    S: int, D: int, base: float = 10000.0
) -> tuple[mx.array, mx.array]:
    """Pre-compute sin/cos frequency table for RoPE."""
    half_d = D // 2
    freqs = 1.0 / (base ** (np.arange(0, half_d, dtype=np.float32) / half_d))
    positions = np.arange(S, dtype=np.float32)
    angles = np.outer(positions, freqs)  # (S, D/2)
    cos_table = mx.array(np.cos(angles), dtype=mx.float32)
    sin_table = mx.array(np.sin(angles), dtype=mx.float32)
    mx.eval(cos_table, sin_table)
    return cos_table, sin_table


def bench_rope(B: int, H: int, S: int, D: int = 128) -> BenchResult:
    """Benchmark RoPE apply_rotary_pos_emb."""
    mx.random.seed(SEED)
    x = mx.random.normal((B, H, S, D), dtype=mx.float16, stream=mx.gpu)
    cos_table, sin_table = _build_rope_freqs(S, D)
    mx.eval(x, cos_table, sin_table)

    half = D // 2

    @mx.compile
    def apply_rope():
        x_f32 = x.astype(mx.float32)
        x1 = x_f32[..., :half]
        x2 = x_f32[..., half:]
        cos_v = cos_table[None, None, :, :]  # (1, 1, S, D/2)
        sin_v = sin_table[None, None, :, :]
        out1 = x1 * cos_v - x2 * sin_v
        out2 = x2 * cos_v + x1 * sin_v
        return mx.concatenate([out1, out2], axis=-1).astype(mx.float16)

    def run():
        out = apply_rope()
        mx.eval(out)

    result = adaptive_bench(run)
    result.name = "rope"
    result.shape = f"{B}x{H}x{S}x{D}"
    result.dtype = "float16"
    numel = B * H * S * D
    # Read x (fp16) + read cos_table (fp32) + read sin_table (fp32) + write out (fp16)
    bytes_moved = numel * 2 + 2 * (S * half * 4) + numel * 2
    result.bytes_accessed = bytes_moved
    result.flops_total = numel * 6  # 4 muls + 1 add + 1 sub per element
    result.gbs = bytes_moved / (result.median_us * 1e-6) / 1e9
    return result


# ---------------------------------------------------------------------------
# Novel kernels: IsoQuant rotation
# ---------------------------------------------------------------------------


def _get_isoquant_components(H: int, D: int, seed: int = SEED):
    """Build IsoQuant rotation components (dense + structured)."""
    try:
        from mlx_lm.models.mlx_isoquant import build_isoquant_rotation_components

        components = build_isoquant_rotation_components(H, D, seed, layer_idx=0)
        return components
    except ImportError:
        # Fallback: build dense rotation only
        rng = np.random.default_rng(seed)
        mats = []
        for _ in range(H):
            A = rng.normal(size=(D, D)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            mats.append(Q)
        return {
            "rotation_matrices": mx.array(np.stack(mats), dtype=mx.float32),
            "block_matrices": None,
            "block_matrices_t": None,
            "use_hadamard": False,
        }


def bench_rotate_forward(H: int, S: int, D: int) -> list[BenchResult]:
    """Benchmark IsoQuant forward rotation: dense vs structured (vs Metal if available)."""
    results = []
    mx.random.seed(SEED)
    x = mx.random.normal((H, S, D), dtype=mx.float32, stream=mx.gpu)
    components = _get_isoquant_components(H, D)
    R = components["rotation_matrices"]
    R_T = mx.swapaxes(R, -2, -1)
    mx.eval(x, R_T)

    # --- Dense: x @ R_T ---
    compiled_dense = mx.compile(lambda: mx.matmul(x, R_T, stream=mx.gpu))

    def run_dense():
        out = compiled_dense()
        mx.eval(out)

    r = adaptive_bench(run_dense)
    r.name = "rotate_forward_dense"
    r.shape = f"{H}x{S}x{D}"
    r.dtype = "float32"
    flops = 2.0 * H * S * D * D
    r.tflops = flops / (r.median_us * 1e-6) / 1e12
    results.append(r)

    # --- Structured: FWHT + block SO(4) ---
    blocks_t = components["block_matrices_t"]
    use_hadamard = components["use_hadamard"]
    if blocks_t is not None:
        mx.eval(blocks_t)
        try:
            from mlx_lm.models.mlx_isoquant import structured_rotate_forward

            compiled_struct = mx.compile(
                lambda: structured_rotate_forward(x, blocks_t, use_hadamard)
            )

            def run_struct():
                out = compiled_struct()
                mx.eval(out)

            r = adaptive_bench(run_struct)
            r.name = "rotate_forward_structured"
            r.shape = f"{H}x{S}x{D}"
            r.dtype = "float32"
            # Structured: FWHT = O(d log d) per vector + block = O(d) per vector
            struct_flops = H * S * (D * math.log2(D) + D * 16)
            r.tflops = struct_flops / (r.median_us * 1e-6) / 1e12
            results.append(r)
        except ImportError:
            pass

    # --- Metal kernel path ---
    block_mats = components["block_matrices"]
    if block_mats is not None:
        mx.eval(block_mats)
        try:
            from mlx_lm.models.isoquant_metal_kernels import metal_rotate_forward

            def run_metal():
                out = metal_rotate_forward(x, block_mats, use_hadamard)
                mx.eval(out)

            r = adaptive_bench(run_metal)
            r.name = "rotate_forward_metal"
            r.shape = f"{H}x{S}x{D}"
            r.dtype = "float32"
            struct_flops = H * S * (D * math.log2(D) + D * 16)
            r.tflops = struct_flops / (r.median_us * 1e-6) / 1e12
            results.append(r)
        except (ImportError, Exception):
            pass

    return results


def bench_rotate_inverse(H: int, D: int) -> list[BenchResult]:
    """Benchmark IsoQuant inverse rotation: dense vs structured (vs Metal)."""
    results = []
    S = 1  # Inverse is applied once on aggregated output
    mx.random.seed(SEED)
    x_rot = mx.random.normal((H, S, D), dtype=mx.float32, stream=mx.gpu)
    components = _get_isoquant_components(H, D)
    R = components["rotation_matrices"]
    mx.eval(x_rot, R)

    # --- Dense: x_rot @ R ---
    compiled_dense = mx.compile(lambda: mx.matmul(x_rot, R, stream=mx.gpu))

    def run_dense():
        out = compiled_dense()
        mx.eval(out)

    r = adaptive_bench(run_dense)
    r.name = "rotate_inverse_dense"
    r.shape = f"{H}x{S}x{D}"
    r.dtype = "float32"
    flops = 2.0 * H * S * D * D
    r.tflops = flops / (r.median_us * 1e-6) / 1e12
    results.append(r)

    # --- Structured ---
    blocks = components["block_matrices"]
    use_hadamard = components["use_hadamard"]
    if blocks is not None:
        mx.eval(blocks)
        try:
            from mlx_lm.models.mlx_isoquant import structured_rotate_inverse

            compiled_struct = mx.compile(
                lambda: structured_rotate_inverse(x_rot, blocks, use_hadamard)
            )

            def run_struct():
                out = compiled_struct()
                mx.eval(out)

            r = adaptive_bench(run_struct)
            r.name = "rotate_inverse_structured"
            r.shape = f"{H}x{S}x{D}"
            r.dtype = "float32"
            struct_flops = H * S * (D * math.log2(D) + D * 16)
            r.tflops = struct_flops / (r.median_us * 1e-6) / 1e12
            results.append(r)
        except ImportError:
            pass

    # --- Metal ---
    blocks_t = components["block_matrices_t"]
    if blocks_t is not None:
        mx.eval(blocks_t)
        try:
            from mlx_lm.models.isoquant_metal_kernels import metal_rotate_inverse

            def run_metal():
                out = metal_rotate_inverse(x_rot, blocks_t, use_hadamard)
                mx.eval(out)

            r = adaptive_bench(run_metal)
            r.name = "rotate_inverse_metal"
            r.shape = f"{H}x{S}x{D}"
            r.dtype = "float32"
            struct_flops = H * S * (D * math.log2(D) + D * 16)
            r.tflops = struct_flops / (r.median_us * 1e-6) / 1e12
            results.append(r)
        except (ImportError, Exception):
            pass

    return results


def bench_kv_compress(H: int, T: int, D: int) -> BenchResult:
    """Benchmark TurboQuant compress + decompress pipeline."""
    from mlx_lm.models.mlx_turboquant import TurboQuantCompressor

    mx.random.seed(SEED)
    x = mx.random.normal((H, T, D), dtype=mx.float32, stream=mx.gpu)
    rng = np.random.default_rng(SEED)
    rot_np = []
    for _ in range(H):
        A = rng.normal(size=(D, D)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        rot_np.append(Q)
    rotation = mx.array(np.stack(rot_np), dtype=mx.float32)

    codebook_dir = _find_codebook_dir()
    if codebook_dir is None:
        raise FileNotFoundError("No codebook directory found")
    compressor = TurboQuantCompressor(
        bit_width=3,
        head_dim=D,
        codebook_dir=codebook_dir,
        seed=SEED,
    )
    mx.eval(x, rotation)

    # Compute RMSE BEFORE the benchmark loop (single roundtrip)
    compressed_rmse = compressor.compress_value(x[0], rotation[0])
    decompressed_rmse = compressor.decompress_value(compressed_rmse, rotation[0])
    mx.eval(decompressed_rmse)
    diff = x[0].astype(mx.float32) - decompressed_rmse.astype(mx.float32)
    rmse = float(mx.sqrt(mx.mean(diff * diff)).item())

    # Measure compress + decompress roundtrip
    def run():
        compressed = compressor.compress_value(x[0], rotation[0])
        decompressed = compressor.decompress_value(compressed, rotation[0])
        mx.eval(decompressed)

    result = adaptive_bench(run)
    result.name = "kv_compress"
    result.shape = f"{H}x{T}x{D}"
    result.dtype = "float32"
    # GB/s: read x + write compressed + read compressed + write decompressed
    bytes_per_elem = 4
    bytes_moved = T * D * bytes_per_elem * 2  # rough: read + write roundtrip
    result.gbs = bytes_moved / (result.median_us * 1e-6) / 1e9
    result.rmse = rmse

    return result


# ---------------------------------------------------------------------------
# Fused attention benchmarks
# ---------------------------------------------------------------------------


def _create_attention_inputs(H: int, T: int, D: int):
    """Create Q, K, V arrays for attention benchmarks."""
    mx.random.seed(SEED)
    q = mx.random.normal((1, H, 1, D), dtype=mx.float16, stream=mx.gpu)
    k = mx.random.normal((1, H, T, D), dtype=mx.float16, stream=mx.gpu)
    v = mx.random.normal((1, H, T, D), dtype=mx.float16, stream=mx.gpu)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v)
    return q, k, v, scale


def bench_fused_attention_unfused(H: int, T: int, D: int) -> BenchResult:
    """Variant 1: Individual mx.eval() per op (fully unfused)."""
    q, k, v, scale = _create_attention_inputs(H, T, D)

    sub_timings: dict[str, list[float]] = {
        "qk_dot": [],
        "softmax": [],
        "attn_v": [],
    }

    def run():
        # Q @ K^T
        scores = mx.matmul(q, mx.swapaxes(k, -2, -1), stream=mx.gpu)
        mx.eval(scores)

        # softmax
        attn = mx.softmax(scores * scale, axis=-1, stream=mx.gpu)
        mx.eval(attn)

        # attn @ V
        output = mx.matmul(attn, v, stream=mx.gpu)
        mx.eval(output)

    def run_with_sub_timings():
        t0 = time.perf_counter()
        scores = mx.matmul(q, mx.swapaxes(k, -2, -1), stream=mx.gpu)
        mx.eval(scores)
        mx.synchronize()
        t1 = time.perf_counter()

        attn = mx.softmax(scores * scale, axis=-1, stream=mx.gpu)
        mx.eval(attn)
        mx.synchronize()
        t2 = time.perf_counter()

        output = mx.matmul(attn, v, stream=mx.gpu)
        mx.eval(output)
        mx.synchronize()
        t3 = time.perf_counter()

        sub_timings["qk_dot"].append((t1 - t0) * 1e6)
        sub_timings["softmax"].append((t2 - t1) * 1e6)
        sub_timings["attn_v"].append((t3 - t2) * 1e6)

    # Run main benchmark
    result = adaptive_bench(run)

    # Collect sub-timings AFTER main benchmark (shares compilation cache)
    for _ in range(5):
        run_with_sub_timings()
        mx.synchronize()
    for _ in range(10):
        run_with_sub_timings()
        mx.synchronize()

    result.name = "fused_attention_unfused"
    result.shape = f"{H}x{T}x{D}"
    result.dtype = "float16"
    result.sub_timings = {
        k: round(float(np.median(v)), 2) for k, v in sub_timings.items()
    }

    # TFLOPS: 2*H*(Q@K^T) + H*(softmax) + 2*H*(attn@V)
    flops = H * (2 * 1 * T * D + 2 * 1 * T * D)  # QK + AV
    result.tflops = flops / (result.median_us * 1e-6) / 1e12
    return result


def bench_fused_attention_compiled(H: int, T: int, D: int) -> BenchResult:
    """Variant 2: mx.compile() wraps entire attention, single mx.eval()."""
    q, k, v, scale = _create_attention_inputs(H, T, D)

    @mx.compile
    def fused_fn(q, k, v):
        scores = mx.matmul(q, mx.swapaxes(k, -2, -1), stream=mx.gpu)
        attn = mx.softmax(scores * scale, axis=-1, stream=mx.gpu)
        output = mx.matmul(attn, v, stream=mx.gpu)
        return output

    def run():
        out = fused_fn(q, k, v)
        mx.eval(out)

    result = adaptive_bench(run)
    result.name = "fused_attention_compiled"
    result.shape = f"{H}x{T}x{D}"
    result.dtype = "float16"
    flops = H * (2 * 1 * T * D + 2 * 1 * T * D)
    result.tflops = flops / (result.median_us * 1e-6) / 1e12
    return result


# ---------------------------------------------------------------------------
# Kernel registry and runner
# ---------------------------------------------------------------------------


def _result_to_json(r: BenchResult) -> dict:
    """Convert BenchResult to the framework-agnostic JSON schema."""
    throughput = {}
    if r.tflops is not None:
        throughput["tflops"] = round(r.tflops, 4)
    if r.gbs is not None:
        throughput["gbs"] = round(r.gbs, 2)
    if r.roofline_pct is not None:
        throughput["roofline_pct"] = round(r.roofline_pct, 2)
    if r.flops_total is not None:
        throughput["flops_total"] = r.flops_total
    if r.bytes_accessed is not None:
        throughput["bytes_accessed"] = r.bytes_accessed
    if r.rmse is not None:
        throughput["rmse"] = round(r.rmse, 6)

    entry = {
        "name": r.name,
        "shape": r.shape,
        "dtype": r.dtype,
        "stats": {
            "mean_us": round(r.mean_us, 2),
            "median_us": round(r.median_us, 2),
            "std_us": round(r.std_us, 2),
            "p5_us": round(r.p5_us, 2),
            "p95_us": round(r.p95_us, 2),
            "p99_us": round(r.p99_us, 2),
            "ci95_bca": [round(v, 2) for v in r.ci95_bca],
            "ci_converged": r.ci_converged,
            "ci_width_pct": r.ci_width_pct,
            "n_iterations": r.n_iterations,
            "dw_statistic": round(r.dw_statistic, 4),
            "runs_test_p": round(r.runs_test_p, 4),
        },
        "throughput": throughput,
    }
    if r.sub_timings:
        entry["sub_timings_median_us"] = r.sub_timings
    return entry


def run_matmul_suite() -> list[BenchResult]:
    """Run all matmul shape configurations."""
    results = []
    configs = [
        # Decode GEMV
        (1, 6144, 6144, "float16"),
        # Prefill QKV
        (2048, 6144, 6144, "float16"),
        # FFN up (decode + prefill)
        (1, 16384, 6144, "float16"),
        (2048, 16384, 6144, "float16"),
        # FFN down (decode + prefill)
        (1, 6144, 16384, "float16"),
        (2048, 6144, 16384, "float16"),
        # MoE expert
        (1, 11008, 5120, "float16"),
    ]
    # Square sweep
    for sz in [512, 1024, 2048, 4096, 8192]:
        configs.append((sz, sz, sz, "float16"))

    for M, N, K, dt in configs:
        print(f"  matmul {M}x{N}x{K} {dt}...")
        r = bench_matmul(M, N, K, dt)
        print(
            f"    {r.median_us:.1f} us, {r.tflops:.2f} TFLOPS, CI={r.ci_width_pct:.2f}%"
        )
        results.append(r)
    return results


def run_softmax_suite() -> list[BenchResult]:
    results = []
    configs = [
        (1, 48, 128),
        (1, 48, 512),
        (1, 48, 2048),
        (1, 48, 8192),
        (1, 8, 32768),  # reduced heads for large S to fit in GPU memory
    ]
    for B, H, S in configs:
        print(f"  softmax {B}x{H}x{S}x{S}...")
        try:
            r = bench_softmax(B, H, S)
            print(
                f"    {r.median_us:.1f} us, {r.gbs:.1f} GB/s, CI={r.ci_width_pct:.2f}%"
            )
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")
    return results


def run_rope_suite() -> list[BenchResult]:
    results = []
    for S in [128, 512, 2048, 8192]:
        print(f"  rope 1x48x{S}x128...")
        r = bench_rope(1, 48, S, 128)
        print(f"    {r.median_us:.1f} us, {r.gbs:.1f} GB/s, CI={r.ci_width_pct:.2f}%")
        results.append(r)
    return results


def run_rotation_suite() -> list[BenchResult]:
    results = []
    configs = [
        (8, 128, 128),  # Standard H=8 decode
        (8, 512, 128),  # H=8 prefill chunk
        (8, 2048, 128),  # H=8 long context
        (32, 128, 128),  # H=32 decode
    ]
    for H, S, D in configs:
        print(f"  rotate_forward H={H} S={S} D={D}...")
        fwd_results = bench_rotate_forward(H, S, D)
        for r in fwd_results:
            print(f"    {r.name}: {r.median_us:.1f} us, CI={r.ci_width_pct:.2f}%")
        results.extend(fwd_results)

    for H, D in [(8, 128), (32, 128)]:
        print(f"  rotate_inverse H={H} D={D}...")
        inv_results = bench_rotate_inverse(H, D)
        for r in inv_results:
            print(f"    {r.name}: {r.median_us:.1f} us, CI={r.ci_width_pct:.2f}%")
        results.extend(inv_results)

    return results


def run_kv_compress_suite() -> list[BenchResult]:
    results = []
    for H, T, D in [(8, 128, 128), (8, 512, 128), (8, 2048, 128)]:
        print(f"  kv_compress H={H} T={T} D={D}...")
        r = bench_kv_compress(H, T, D)
        print(
            f"    {r.median_us:.1f} us, {r.gbs:.1f} GB/s, RMSE={r.rmse:.6f}, CI={r.ci_width_pct:.2f}%"
        )
        results.append(r)
    return results


def run_fused_attention_suite() -> list[BenchResult]:
    results = []
    configs = [
        (8, 128, 128),
        (8, 512, 128),
        (8, 2048, 128),
        (32, 128, 128),
        (32, 2048, 128),
    ]

    for H, T, D in configs:
        print(f"  fused_attention H={H} T={T} D={D}...")

        print("    unfused...")
        r = bench_fused_attention_unfused(H, T, D)
        print(f"      {r.median_us:.1f} us, CI={r.ci_width_pct:.2f}%")
        results.append(r)

        print("    compiled...")
        r = bench_fused_attention_compiled(H, T, D)
        print(f"      {r.median_us:.1f} us, CI={r.ci_width_pct:.2f}%")
        results.append(r)


    return results


# Mapping from kernel name to suite runner
KERNEL_SUITES = {
    "matmul": run_matmul_suite,
    "softmax": run_softmax_suite,
    "rope": run_rope_suite,
    "rotation": run_rotation_suite,
    "kv_compress": run_kv_compress_suite,
    "fused_attention": run_fused_attention_suite,
}


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def _get_hardware_info() -> dict:
    """Collect hardware metadata, preferring roofline_calibrate if available."""
    try:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
        from roofline_calibrate import get_hardware_info

        return get_hardware_info()
    except ImportError:
        import platform

        info = {}
        try:
            info = mx.device_info()
        except Exception:
            pass
        return {
            "chip": platform.processor() or "Apple Silicon",
            "memory_gb": round(info.get("memory_size", 0) / (1024**3)),
            "macos_version": platform.mac_ver()[0],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MLX Metal kernel benchmarks -- mirrors the Mojo kernel set"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mlx_kernels.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        choices=list(KERNEL_SUITES.keys()),
        help="Run only this kernel suite (default: all)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=_BENCH_CONFIG["max_iters"],
        help="Max iterations per benchmark (default: 500)",
    )
    parser.add_argument(
        "--target-ci",
        type=float,
        default=_BENCH_CONFIG["target_ci_pct"],
        help="Target CI width %% of median (default: 2.0)",
    )
    parser.add_argument(
        "--roofline",
        type=str,
        default="results/roofline_m4max.json",
        help="Roofline calibration JSON for computing %% of peak",
    )
    args = parser.parse_args()

    _BENCH_CONFIG["max_iters"] = args.max_iters
    _BENCH_CONFIG["target_ci_pct"] = args.target_ci

    print("=== MLX Kernel Benchmarks ===")
    print(f"MLX version: {mx.__version__}")
    print(
        f"Target CI: {_BENCH_CONFIG['target_ci_pct']}%, "
        f"Max iters: {_BENCH_CONFIG['max_iters']}"
    )
    print()

    suites_to_run = (
        {args.kernel: KERNEL_SUITES[args.kernel]} if args.kernel else KERNEL_SUITES
    )

    all_results: list[BenchResult] = []
    for suite_name, suite_fn in suites_to_run.items():
        print(f"--- {suite_name} ---")
        try:
            results = suite_fn()
            all_results.extend(results)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
        print()

    # Compute roofline_pct from calibration
    peak_tflops = None
    peak_bw_gbs = None
    if os.path.exists(args.roofline):
        with open(args.roofline) as f:
            cal = json.load(f).get("calibration", {})
        dtype_key = "peak_fp16_tflops"  # default to FP16
        peak_tflops = cal.get(dtype_key, cal.get("peak_fp32_tflops"))
        peak_bw_gbs = cal.get("peak_memory_bandwidth_gbs")

    dw_warnings = []
    for r in all_results:
        # Roofline percentage
        if r.flops_total and r.bytes_accessed and peak_tflops and peak_bw_gbs:
            ai = r.flops_total / r.bytes_accessed  # arithmetic intensity
            ceiling_tflops = min(peak_tflops, peak_bw_gbs * ai / 1000)
            if r.tflops is not None and ceiling_tflops > 0:
                r.roofline_pct = (r.tflops / ceiling_tflops) * 100.0
            elif r.gbs is not None and peak_bw_gbs > 0:
                r.roofline_pct = (r.gbs / peak_bw_gbs) * 100.0
        elif r.gbs is not None and peak_bw_gbs and peak_bw_gbs > 0:
            r.roofline_pct = (r.gbs / peak_bw_gbs) * 100.0
        # Flag DW < 0.3 (severe positive autocorrelation)
        if r.dw_statistic < 0.3:
            dw_warnings.append(
                f"  WARNING: {r.name} {r.shape} DW={r.dw_statistic:.4f} < 0.3 (autocorrelated)"
            )

    if dw_warnings:
        print("\nDurbin-Watson autocorrelation warnings:")
        for w in dw_warnings:
            print(w)

    # Build JSON output
    hw = _get_hardware_info()
    output = {
        "framework": "mlx",
        "framework_version": str(mx.__version__),
        "compilation": {
            "mode": "jit",
            "optimization": "mx.compile()",
            "graph_compilation": "enabled",
        },
        "hardware": hw,
        "kernels": [_result_to_json(r) for r in all_results],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(all_results)} results to {args.output}")

    # Summary
    converged = sum(1 for r in all_results if r.ci_converged)
    print(f"CI converged: {converged}/{len(all_results)}")


if __name__ == "__main__":
    main()
