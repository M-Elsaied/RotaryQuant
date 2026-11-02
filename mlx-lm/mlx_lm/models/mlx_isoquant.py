# Copyright © 2026 — IsoQuant KV rotation via isoclinic decomposition of SO(4).
# SAFETY: V-cache dequant uses R.T (orthogonal inverse). Isoclinic matrices are
# orthogonal by construction; inverse_rotate is implicit via matmul with R.
#
# Math: each 4D block is rotated by T(v) = q_L * v * conj(q_R) where q_L, q_R
# are independent unit quaternions. This gives the full SO(4) on each block
# (no fixed axis). Paper: arXiv:2603.28430 "IsoQuant" (March 2026).

from __future__ import annotations

import os
import sys
import time

import numpy as np
import mlx.core as mx

from .mlx_turboquant import (
    TurboQuantCompressor,
    TurboQuantKVCache,
    get_default_codebook_dir,
    quantize_scalar,
)

_METAL_BACKEND = None


# ---------------------------------------------------------------------------
# Module-level instrumentation. Aggregated across all per-layer cache
# instances. Reset before each benchmark run via ``reset_stats()`` and read
# after via ``stats_summary()``. Modelled on ``mx.get_peak_memory()`` —
# global state is fine because there is one cache pool per generation run.
# ---------------------------------------------------------------------------


class IsoQuantStats:
    """Counters for IsoQuant cache activity (write, read, fused metal)."""

    __slots__ = (
        "prefill_calls",
        "prefill_tokens",
        "decode_calls",
        "decode_tokens",
        "compress_calls",
        "decompress_calls",
        "read_keys_calls",
        "read_values_calls",
        "finalize_calls",
        "fused_metal_attempts",
        "fused_metal_failures",
        "packed_cache_hits",
        "packed_cache_misses",
        "fallback_invocations",
        "unfused_fallback_calls",
        "packed_key_repack_ms",
        "packed_value_repack_ms",
        "fused_qk_ms",
        "fused_softmax_ms",
        "fused_value_ms",
        "fused_value_tiled_ms",
        "fused_inverse_ms",
        "fused_single_kernel_ms",
        "fused_metal_total_ms",
        "decode_key_compress_ms",
        "decode_value_compress_ms",
        "decode_concat_ms",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        for f in self.__slots__:
            setattr(self, f, 0)

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.__slots__}
        # Derived rates — guard divisions
        packed_total = self.packed_cache_hits + self.packed_cache_misses
        d["packed_cache_hit_rate"] = (
            round(self.packed_cache_hits / packed_total, 4) if packed_total else None
        )
        fused_total = self.fused_metal_attempts
        d["fused_metal_success_rate"] = (
            round(1 - self.fused_metal_failures / fused_total, 4)
            if fused_total
            else None
        )
        d["packed_repack_ms_total"] = round(
            self.packed_key_repack_ms + self.packed_value_repack_ms, 3
        )
        d["fused_kernel_ms_total"] = round(
            self.fused_qk_ms
            + self.fused_softmax_ms
            + self.fused_value_ms
            + self.fused_value_tiled_ms
            + self.fused_inverse_ms
            + self.fused_single_kernel_ms,
            3,
        )
        d["decode_write_ms_total"] = round(
            self.decode_key_compress_ms
            + self.decode_value_compress_ms
            + self.decode_concat_ms,
            3,
        )
        return d


_GLOBAL_STATS = IsoQuantStats()


def get_stats() -> IsoQuantStats:
    return _GLOBAL_STATS


def reset_stats() -> None:
    _GLOBAL_STATS.reset()
    reset_path_report()


def stats_summary() -> dict:
    return _GLOBAL_STATS.to_dict()


# ---------------------------------------------------------------------------
# Per-layer path record: which code path each IsoQuant layer actually ran.
#
#   deferred_fp16        attention over uncompressed prefill KV (no compression)
#   fp16_fallback_cache  layer runs a plain KVCache (unsupported shape/codebook)
#   fused_metal          decode attention via the packed Metal kernels
#   fused_mlx            decode attention via the MLX-ops fallback (K/V materialised)
#   unfused_dense        dense reconstruction of compressed K/V + SDPA
#   fused_empty          fused_attention called on an empty cache (zeros returned)
#   decode_write         compressed decode-time write
#   encode_fused / encode_mlx   which encoder produced a decode-time write
#
# kv_path_report() also derives raw_passthrough = decode writes that were not
# followed by a compressed read, i.e. the model attended over the returned
# chunk only.
# ---------------------------------------------------------------------------

DECODE_READ_PATHS = ("fused_metal", "fused_mlx", "unfused_dense")
NON_FUSED_PATHS = ("fused_mlx", "unfused_dense", "fp16_fallback_cache", "fused_empty")
# Error sources that affect the attention path (encoder errors are reported only).
ATTENTION_ERROR_SOURCES = ("fused_metal", "fallback_cache")
_MAX_ERRORS_PER_LAYER = 3

_PATH_COUNTS: dict[int, dict[str, int]] = {}
_PATH_ERRORS: dict[int, list[str]] = {}


def _layer_key(layer_idx) -> int:
    return -1 if layer_idx is None else int(layer_idx)


def _record_path(layer_idx, path: str, n: int = 1) -> None:
    counts = _PATH_COUNTS.setdefault(_layer_key(layer_idx), {})
    counts[path] = counts.get(path, 0) + n


def _record_error(layer_idx, where: str, exc: BaseException | str) -> None:
    errors = _PATH_ERRORS.setdefault(_layer_key(layer_idx), [])
    detail = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    msg = f"{where}: {detail}"[:300]
    if msg not in errors and len(errors) < _MAX_ERRORS_PER_LAYER:
        errors.append(msg)


def reset_path_report() -> None:
    _PATH_COUNTS.clear()
    _PATH_ERRORS.clear()


def kv_path_report() -> dict:
    """Per-layer path counts and totals for all IsoQuant caches since last reset."""
    layers = {}
    totals: dict[str, int] = {}
    for idx in sorted(set(_PATH_COUNTS) | set(_PATH_ERRORS)):
        counts = dict(_PATH_COUNTS.get(idx, {}))
        reads = sum(counts.get(p, 0) for p in DECODE_READ_PATHS)
        writes = counts.get("decode_write", 0)
        if writes > reads:
            counts["raw_passthrough"] = writes - reads
        layers[str(idx)] = {"counts": counts, "errors": list(_PATH_ERRORS.get(idx, []))}
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    return {"n_layers": len(layers), "totals": totals, "layers": layers}


def fused_path_violations(report: dict) -> list[str]:
    """Reasons a run did not use the fused Metal path for every decode read.

    Decode-time encoder fallbacks (encode_mlx, fused_encode errors) are reported
    in kv_path_report() but are not violations.
    """
    if not report.get("layers"):
        return ["no IsoQuant layers recorded"]
    problems = []
    for idx, layer in report["layers"].items():
        counts = layer["counts"]
        bad = {p: counts[p] for p in NON_FUSED_PATHS + ("raw_passthrough",) if counts.get(p)}
        if bad:
            problems.append(f"layer {idx}: {bad}")
        elif not counts.get("fused_metal"):
            problems.append(f"layer {idx}: no fused_metal decode reads")
        for err in layer["errors"]:
            if err.split(":", 1)[0] in ATTENTION_ERROR_SOURCES:
                problems.append(f"layer {idx}: {err}")
    return problems


def _get_metal_backend():
    """Lazy-load the optional Metal structured-rotation backend.

    Kept behind an explicit opt-in because correctness should match the dense
    path first; performance closure remains a separate measurement task.
    """
    global _METAL_BACKEND
    if _METAL_BACKEND is None:
        from .isoquant_metal_kernels import metal_rotate_forward, metal_rotate_inverse

        _METAL_BACKEND = (metal_rotate_forward, metal_rotate_inverse)
    return _METAL_BACKEND


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, last axis 4, (w,x,y,z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([ow, ox, oy, oz], axis=-1)


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., 1:] *= -1.0
    return out


def _isoclinic_vec_np(q_L: np.ndarray, q_R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Isoclinic SO(4) rotation: v' = q_L * v * conj(q_R)."""
    return _quat_mul_np(_quat_mul_np(q_L, v), _quat_conj_np(q_R))


def _isoclinic_block_matrix(q_L: np.ndarray, q_R: np.ndarray) -> np.ndarray:
    """4x4 orthogonal matrix M where M @ v = q_L * v * conj(q_R).

    Two independent unit quaternions give the full SO(4) per block —
    no fixed axis, all 4 dimensions mixed. This is the isoclinic
    decomposition from arXiv:2603.28430.
    """
    M = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        e = np.zeros(4, dtype=np.float32)
        e[i] = 1.0
        M[:, i] = _isoclinic_vec_np(q_L, q_R, e)
    return M


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=4).astype(np.float32)
    return g / (np.linalg.norm(g) + 1e-8)


def _build_isoclinic_block_rotation_matrix(
    head_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Block-diagonal SO(4) rotation without any global pre-mixing."""
    blocks = _build_isoclinic_block_matrices(head_dim, rng)
    return _block_matrices_to_dense(blocks)


def _build_isoclinic_block_matrices(
    head_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Independent 4x4 SO(4) blocks for a single head."""
    n_blocks = head_dim // 4
    blocks = np.zeros((n_blocks, 4, 4), dtype=np.float32)
    for b in range(n_blocks):
        q_L = _random_unit_quat(rng)
        q_R = _random_unit_quat(rng)
        blocks[b] = _isoclinic_block_matrix(q_L, q_R)
    return blocks


def _block_matrices_to_dense(blocks: np.ndarray) -> np.ndarray:
    """Dense block-diagonal matrix from 4x4 block matrices."""
    n_blocks = blocks.shape[0]
    head_dim = n_blocks * 4
    R_block = np.zeros((head_dim, head_dim), dtype=np.float32)
    for b, blk in enumerate(blocks):
        sl = slice(b * 4, (b + 1) * 4)
        R_block[sl, sl] = blk
    return R_block


def _hadamard_matrix(d: int) -> np.ndarray:
    """Normalized Walsh-Hadamard matrix of size d (must be power of 2).

    O(d log d) global mixing matrix for decorrelating all dimensions before
    block rotation.
    """
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2.0)
    return H.astype(np.float32)


def _combine_block_rotation_with_hadamard(
    R_block: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Combine block SO(4) rotations with a global Hadamard mix.

    Since ``R_block`` is block diagonal, each 4-row block only depends on the
    matching 4 rows of ``H``. Building the product blockwise avoids the large
    dense matmul path that emitted suspicious runtime warnings on Apple Silicon.
    """
    head_dim = R_block.shape[0]
    combined = np.zeros((head_dim, head_dim), dtype=np.float32)
    for start in range(0, head_dim, 4):
        stop = start + 4
        blk = R_block[start:stop, start:stop]
        H_chunk = H[start:stop, :]
        block_rows = np.zeros((4, head_dim), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                block_rows[i, :] += blk[i, j] * H_chunk[j, :]
        combined[start:stop, :] = block_rows
    return combined


def build_isoquant_rotation_matrix(
    head_dim: int, rng: np.random.Generator, apply_global_mix: bool = True
) -> np.ndarray:
    """Single head rotation matrix for IsoQuant.

    ``apply_global_mix=True`` builds the current WHT+SO(4) path.
    ``False`` preserves the legacy block-only SO(4) path for prompt-cache
    compatibility with older serialized ``iso_v1`` artifacts.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"IsoQuant expects head_dim divisible by 4, got {head_dim}")
    return _build_isoquant_rotation_components_for_head(
        head_dim, rng, apply_global_mix=apply_global_mix
    )[0]


def _build_isoquant_rotation_components_for_head(
    head_dim: int, rng: np.random.Generator, apply_global_mix: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return dense and structured rotation components for a single head.

    The cache uses row vectors. Forward rotation is `x @ R.T`, where
    `R = R_block @ H` when Hadamard mixing is enabled. The structured forward
    path is therefore `x @ H @ R_block.T`.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"IsoQuant expects head_dim divisible by 4, got {head_dim}")
    blocks = _build_isoclinic_block_matrices(head_dim, rng)
    blocks_t = np.swapaxes(blocks, -2, -1)
    R_block = _block_matrices_to_dense(blocks)
    is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
    use_hadamard = bool(apply_global_mix and is_pow2)
    if use_hadamard:
        H = _hadamard_matrix(head_dim)
        dense = _combine_block_rotation_with_hadamard(R_block, H)
    else:
        dense = R_block
    return dense, blocks, blocks_t, use_hadamard


def build_isoquant_rotation_matrices(
    num_heads: int,
    head_dim: int,
    seed: int,
    layer_idx: int = 0,
    apply_global_mix: bool = True,
) -> mx.array:
    rng = np.random.default_rng(seed + (layer_idx or 0) * 1000)
    mats = [
        build_isoquant_rotation_matrix(head_dim, rng, apply_global_mix=apply_global_mix)
        for _ in range(num_heads)
    ]
    return mx.array(np.stack(mats, dtype=np.float32))


def build_isoquant_rotation_components(
    num_heads: int,
    head_dim: int,
    seed: int,
    layer_idx: int = 0,
    apply_global_mix: bool = True,
) -> dict[str, mx.array | bool]:
    """Structured per-head IsoQuant rotation state plus dense fallback matrix."""
    rng = np.random.default_rng(seed + (layer_idx or 0) * 1000)
    dense_mats = []
    block_mats = []
    block_mats_t = []
    use_hadamard = None
    for _ in range(num_heads):
        dense, blocks, blocks_t, head_use_hadamard = (
            _build_isoquant_rotation_components_for_head(
                head_dim, rng, apply_global_mix=apply_global_mix
            )
        )
        dense_mats.append(dense)
        block_mats.append(blocks)
        block_mats_t.append(blocks_t)
        if use_hadamard is None:
            use_hadamard = head_use_hadamard
    return {
        "rotation_matrices": mx.array(np.stack(dense_mats, dtype=np.float32)),
        "block_matrices": mx.array(np.stack(block_mats, dtype=np.float32)),
        "block_matrices_t": mx.array(np.stack(block_mats_t, dtype=np.float32)),
        "use_hadamard": bool(use_hadamard),
    }


def _fwht_last_axis(x: mx.array) -> mx.array:
    """Exact normalized Walsh-Hadamard transform on the last axis."""
    dim = x.shape[-1]
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"FWHT expects power-of-2 last dimension, got {dim}")
    y = x.astype(mx.float32)
    scale = mx.array(np.sqrt(2.0), dtype=mx.float32)
    stride = 1
    while stride < dim:
        shape = y.shape[:-1] + (dim // (2 * stride), 2, stride)
        paired = y.reshape(shape)
        even = paired[..., 0, :]
        odd = paired[..., 1, :]
        y = mx.stack([(even + odd) / scale, (even - odd) / scale], axis=-2)
        y = y.reshape(x.shape)
        stride *= 2
    return y


def _apply_so4_blocks_last_axis(x: mx.array, blocks: mx.array) -> mx.array:
    """Apply per-head 4x4 block matvecs to the last axis.

    Args:
        x: shape (heads, ..., head_dim)
        blocks: shape (heads, n_blocks, 4, 4)
    """
    head_dim = x.shape[-1]
    n_blocks = head_dim // 4
    heads = x.shape[0]
    trailing = x.shape[1:-1]
    flat = 1
    for dim in trailing:
        flat *= dim
    x_blocks = x.reshape((heads, flat, n_blocks, 1, 4))
    block_view = blocks[:, None, :, :, :]
    rotated = mx.matmul(x_blocks, block_view).squeeze(-2)
    return rotated.reshape(x.shape)


def structured_rotate_forward(
    x: mx.array, block_matrices_t: mx.array, use_hadamard: bool
) -> mx.array:
    """Row-vector forward rotate matching `x @ R.T`."""
    y = _fwht_last_axis(x) if use_hadamard else x.astype(mx.float32)
    return _apply_so4_blocks_last_axis(y, block_matrices_t)


def structured_rotate_inverse(
    x_rot: mx.array, block_matrices: mx.array, use_hadamard: bool
) -> mx.array:
    """Row-vector inverse rotate matching `x_rot @ R`."""
    y = _apply_so4_blocks_last_axis(x_rot, block_matrices)
    return _fwht_last_axis(y) if use_hadamard else y


class IsoQuantKVCache(TurboQuantKVCache):
    """
    KV cache using IsoQuant-style rotations: a global Walsh-Hadamard pre-mix
    followed by independent isoclinic SO(4) rotations on each 4-D block.

    Reuses TurboQuant Lloyd–Max quantisation and asymmetric attention; only the
    orthogonal R differs (Hadamard + isoclinic blocks vs dense QR).
    """

    _PREALLOC_STEP = 256

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        bit_width: int = 4,
        max_seq_len: int = 32768,
        layer_idx: int | None = None,
        codebook_dir: str = "codebooks",
        seed: int = 42,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            bit_width=bit_width,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            codebook_dir=codebook_dir,
            seed=seed,
        )
        # Deferred prefill: accumulate FP16 KV during prefill, bulk-compress
        # at the prefill→decode transition.  Zero error compounding during prefill.
        self._deferred = True
        self._fp16_keys: list[mx.array] = []
        self._fp16_values: list[mx.array] = []
        self._apply_global_mix = True
        self.block_matrices = None
        self.block_matrices_t = None
        self._use_hadamard = False
        self._use_structured_runtime = (
            os.environ.get("ISOQUANT_USE_STRUCTURED_MLX", "0") == "1"
        )
        self._use_metal_runtime = os.environ.get("ISOQUANT_USE_METAL", "0") == "1"
        self._profile_fused_timing = (
            os.environ.get("ISOQUANT_PROFILE_METAL", "0") == "1"
        )
        self._metal_runtime_error: str | None = None
        self._fused_metal_ok: bool | None = (
            None  # None=untested, True=works, False=failed
        )
        self._packed_keys_cache: mx.array | None = None
        self._packed_values_cache: mx.array | None = None
        # Runtime defaults graduated by §3.4 (2026-05-05): combined =
        # FUSED_ENCODE=1 + CACHE_MODE=prealloc. Evidence:
        # artifacts/branch_c_profiling/write_path_ablation_paired_v2.json,
        # artifacts/branch_c_profiling/write_path_quality_gate_gemma4/.
        # FUSED_ENCODE introduces measurable numerical drift but per-task
        # quality holds; opt out with ISOQUANT_FUSED_ENCODE=0 if bit-
        # reproducibility vs the prior default is required.
        self._cache_mode = os.environ.get("ISOQUANT_CACHE_MODE", "prealloc")
        self._fused_encode_requested = (
            os.environ.get("ISOQUANT_FUSED_ENCODE", "1") == "1"
        )
        self._use_fused_encode = False
        self._fused_encode_verified = False
        if self._fallback_cache is None:
            try:
                self._set_rotation_components(self.num_heads, self.head_dim)
            except Exception:
                self._activate_fallback_cache(self.num_heads, self.head_dim)
        self._resolve_fused_encode_gate()

    def _activate_fallback_cache(self, n_kv_heads: int, head_dim: int) -> None:
        exc = sys.exc_info()[1]  # set when called from an except block
        reason = f"heads={n_kv_heads}, head_dim={head_dim}"
        if exc is not None:
            reason += f"; {type(exc).__name__}: {exc}"
        _record_error(self.layer_idx, "fallback_cache", reason)
        super()._activate_fallback_cache(n_kv_heads, head_dim)

    @classmethod
    def from_state(cls, state: dict, meta_state=None, **kwargs):
        if meta_state is None:
            raise ValueError(
                "meta_state is required for from_state reconstruction. "
                "Cannot restore IsoQuant cache without version, bit_width, "
                "num_heads, and head_dim."
            )
        obj = cls.__new__(cls)
        obj.meta_state = meta_state
        obj.state = state
        return obj

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    "iso_v2",
                    self.bit_width,
                    self.num_heads,
                    self.head_dim,
                    self.layer_idx if self.layer_idx is not None else -1,
                    self.codebook_dir,
                    self.seed,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        version = v[0]
        if version not in {"iso_v1", "iso_v2"}:
            raise ValueError(
                f"IsoQuantKVCache expects meta_state iso_v1/iso_v2, got {version!r}"
            )
        self.bit_width = int(v[1])
        self.num_heads = int(v[2])
        self.head_dim = int(v[3])
        layer_idx_val = int(v[4])
        self.layer_idx = layer_idx_val if layer_idx_val != -1 else None
        self.max_seq_len = 32768
        self.codebook_dir = (
            os.path.abspath(v[5]) if len(v) > 5 and v[5] else get_default_codebook_dir()
        )
        from .mlx_turboquant import _DEFAULT_TQ_SEED

        self.seed = int(v[6]) if len(v) >= 7 else _DEFAULT_TQ_SEED
        self._apply_global_mix = version == "iso_v2"
        self._deferred = False
        self._fp16_keys = []
        self._fp16_values = []
        self._fallback_cache = None
        self._warned_fallback = False
        self.block_matrices = None
        self.block_matrices_t = None
        self._use_hadamard = False
        self._use_structured_runtime = (
            os.environ.get("ISOQUANT_USE_STRUCTURED_MLX", "0") == "1"
        )
        self._use_metal_runtime = os.environ.get("ISOQUANT_USE_METAL", "0") == "1"
        self._profile_fused_timing = (
            os.environ.get("ISOQUANT_PROFILE_METAL", "0") == "1"
        )
        self._metal_runtime_error = None
        self._fused_metal_ok = None
        self._packed_keys_cache = None
        self._packed_values_cache = None
        # Runtime defaults graduated by §3.4 (2026-05-05): combined =
        # FUSED_ENCODE=1 + CACHE_MODE=prealloc. Evidence:
        # artifacts/branch_c_profiling/write_path_ablation_paired_v2.json,
        # artifacts/branch_c_profiling/write_path_quality_gate_gemma4/.
        # FUSED_ENCODE introduces measurable numerical drift but per-task
        # quality holds; opt out with ISOQUANT_FUSED_ENCODE=0 if bit-
        # reproducibility vs the prior default is required.
        self._cache_mode = os.environ.get("ISOQUANT_CACHE_MODE", "prealloc")
        self._fused_encode_requested = (
            os.environ.get("ISOQUANT_FUSED_ENCODE", "1") == "1"
        )
        self._use_fused_encode = False
        self._fused_encode_verified = False
        try:
            self.compressor = TurboQuantCompressor(
                bit_width=self.bit_width,
                head_dim=self.head_dim,
                codebook_dir=self.codebook_dir,
                seed=self.seed,
            )
            self._set_rotation_components(self.num_heads, self.head_dim)
        except Exception:
            self._activate_fallback_cache(self.num_heads, self.head_dim)
        self._resolve_fused_encode_gate()
        if self._cache_mode == "prealloc" and getattr(self, "compressed_keys", None):
            from .fused_kv_decode_kernels import pack_indices_3bit

            self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
            self._packed_values_cache = pack_indices_3bit(
                self.compressed_values["indices"]
            )
            self._extend_buffers_by_step()

    def _set_rotation_components(self, num_heads: int, head_dim: int) -> None:
        components = build_isoquant_rotation_components(
            num_heads,
            head_dim,
            self.seed,
            self.layer_idx or 0,
            apply_global_mix=self._apply_global_mix,
        )
        self.rotation_matrices = components["rotation_matrices"]
        self.block_matrices = components["block_matrices"]
        self.block_matrices_t = components["block_matrices_t"]
        self._use_hadamard = bool(components["use_hadamard"])

    def _resolve_fused_encode_gate(self) -> None:
        """Evaluate all preconditions for fused Metal encode path.

        Requires: env var set, D in {128, 256}, 3-bit quantization,
        iso_v2 (FWHT enabled), and no fallback cache.
        """
        if not getattr(self, "_fused_encode_requested", False):
            self._use_fused_encode = False
            return
        self._use_fused_encode = (
            self.head_dim in (128, 256)
            and self.bit_width == 3
            and getattr(self, "_use_hadamard", False)
            and getattr(self, "_apply_global_mix", True)
            and self._fallback_cache is None
        )

    def _rotate_forward(self, x: mx.array) -> mx.array:
        q_t = mx.swapaxes(self.rotation_matrices, -2, -1)
        if self._use_metal_runtime and self.block_matrices_t is not None:
            try:
                metal_forward, _ = _get_metal_backend()
                return metal_forward(x, self.block_matrices, self._use_hadamard)
            except Exception as exc:
                self._use_metal_runtime = False
                self._metal_runtime_error = str(exc)
        if not self._use_structured_runtime or self.block_matrices_t is None:
            return mx.matmul(x, q_t)
        return structured_rotate_forward(x, self.block_matrices_t, self._use_hadamard)

    def _rotate_inverse(self, x_rot: mx.array) -> mx.array:
        if self._use_metal_runtime and self.block_matrices is not None:
            try:
                _, metal_inverse = _get_metal_backend()
                return metal_inverse(x_rot, self.block_matrices_t, self._use_hadamard)
            except Exception as exc:
                self._use_metal_runtime = False
                self._metal_runtime_error = str(exc)
        if not self._use_structured_runtime or self.block_matrices is None:
            return mx.matmul(x_rot, self.rotation_matrices)
        return structured_rotate_inverse(x_rot, self.block_matrices, self._use_hadamard)

    def _decompress_batch(self, compressed: dict[str, mx.array]) -> mx.array:
        _GLOBAL_STATS.decompress_calls += 1
        indices = compressed["indices"]
        x_rot_quant = self.compressor.centroids[indices]
        x_hat_unit = self._rotate_inverse(x_rot_quant)
        return x_hat_unit * compressed["x_norm"]

    def _invalidate_fused_caches(self) -> None:
        self._packed_keys_cache = None
        self._packed_values_cache = None

    def _ensure_buffer_capacity(self, additional: int) -> None:
        """Extend all 6 preallocated buffers if offset + additional > capacity.

        If packed caches are missing (e.g. after from_state restore), rebuild
        them from the compressed indices first.
        """
        # Lazy rebuild of packed cache (handles from_state restore in prealloc mode)
        if self._packed_keys_cache is None or self._packed_values_cache is None:
            from .fused_kv_decode_kernels import pack_indices_3bit

            self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
            self._packed_values_cache = pack_indices_3bit(
                self.compressed_values["indices"]
            )

        needed = self.offset + additional
        current_cap = self.compressed_keys["indices"].shape[1]

        if needed <= current_cap:
            return

        n_steps = (
            needed - current_cap + self._PREALLOC_STEP - 1
        ) // self._PREALLOC_STEP
        extend_by = n_steps * self._PREALLOC_STEP

        H = self.compressed_keys["indices"].shape[0]
        D_idx = self.compressed_keys["indices"].shape[2]
        D_norm = self.compressed_keys["x_norm"].shape[2]
        packed_D = self._packed_keys_cache.shape[2]

        pad_idx = mx.zeros((H, extend_by, D_idx), dtype=mx.uint8)
        pad_norm = mx.zeros((H, extend_by, D_norm), dtype=mx.float16)
        pad_packed = mx.zeros((H, extend_by, packed_D), dtype=mx.uint8)

        self.compressed_keys["indices"] = mx.concatenate(
            [self.compressed_keys["indices"], pad_idx], axis=1
        )
        self.compressed_keys["x_norm"] = mx.concatenate(
            [self.compressed_keys["x_norm"], pad_norm], axis=1
        )
        self.compressed_values["indices"] = mx.concatenate(
            [self.compressed_values["indices"], pad_idx], axis=1
        )
        self.compressed_values["x_norm"] = mx.concatenate(
            [self.compressed_values["x_norm"], pad_norm], axis=1
        )
        self._packed_keys_cache = mx.concatenate(
            [self._packed_keys_cache, pad_packed], axis=1
        )
        self._packed_values_cache = mx.concatenate(
            [self._packed_values_cache, pad_packed], axis=1
        )

    def _extend_buffers_by_step(self) -> None:
        """Extend compressed + packed buffers by _PREALLOC_STEP zeros."""
        H = self.compressed_keys["indices"].shape[0]
        D_idx = self.compressed_keys["indices"].shape[2]
        D_norm = self.compressed_keys["x_norm"].shape[2]
        packed_D = self._packed_keys_cache.shape[2]

        pad_idx = mx.zeros((H, self._PREALLOC_STEP, D_idx), dtype=mx.uint8)
        pad_norm = mx.zeros((H, self._PREALLOC_STEP, D_norm), dtype=mx.float16)
        pad_packed = mx.zeros((H, self._PREALLOC_STEP, packed_D), dtype=mx.uint8)

        self.compressed_keys["indices"] = mx.concatenate(
            [self.compressed_keys["indices"], pad_idx], axis=1
        )
        self.compressed_keys["x_norm"] = mx.concatenate(
            [self.compressed_keys["x_norm"], pad_norm], axis=1
        )
        self.compressed_values["indices"] = mx.concatenate(
            [self.compressed_values["indices"], pad_idx], axis=1
        )
        self.compressed_values["x_norm"] = mx.concatenate(
            [self.compressed_values["x_norm"], pad_norm], axis=1
        )
        self._packed_keys_cache = mx.concatenate(
            [self._packed_keys_cache, pad_packed], axis=1
        )
        self._packed_values_cache = mx.concatenate(
            [self._packed_values_cache, pad_packed], axis=1
        )
        mx.eval(
            self.compressed_keys["indices"],
            self.compressed_keys["x_norm"],
            self.compressed_values["indices"],
            self.compressed_values["x_norm"],
            self._packed_keys_cache,
            self._packed_values_cache,
        )

    def _profile_mx_call(self, stat_name: str, fn):
        if not self._profile_fused_timing:
            return fn()

        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        setattr(
            _GLOBAL_STATS,
            stat_name,
            getattr(_GLOBAL_STATS, stat_name) + (time.perf_counter() - t0) * 1000.0,
        )
        return out

    def _profile_mx_eval(self, stat_name: str, *arrays) -> None:
        if not self._profile_fused_timing:
            mx.eval(*arrays)
            return

        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*arrays)
        mx.synchronize()
        setattr(
            _GLOBAL_STATS,
            stat_name,
            getattr(_GLOBAL_STATS, stat_name) + (time.perf_counter() - t0) * 1000.0,
        )

    def finalize_deferred_prefill(self) -> None:
        """Bulk-compress accumulated FP16 KV into IsoQuant at the prefill→decode boundary.

        During prefill, KV is stored in FP16 (zero error compounding).
        This method compresses the entire buffer in one pass, then switches
        to incremental compression for the decode phase.
        """
        if not self._deferred:
            return
        _GLOBAL_STATS.finalize_calls += 1
        self._deferred = False

        if self._fallback_cache is not None:
            # Fallback path — nothing to compress
            self._fp16_keys.clear()
            self._fp16_values.clear()
            return

        if not self._fp16_keys:
            return

        # Concatenate all prefill chunks: each is (heads, seq_chunk, dim)
        all_keys = mx.concatenate(self._fp16_keys, axis=1)
        all_values = mx.concatenate(self._fp16_values, axis=1)

        # Bulk compress
        self.compressed_keys = self._compress_batch(all_keys)
        self.compressed_values = self._compress_batch(all_values)
        # Phase 2: pre-populate packed cache once at finalize so decode appends
        # extend rather than rebuild. Replaces the prior _invalidate_fused_caches()
        # call which forced a lazy rebuild on every fused_attention.
        from .fused_kv_decode_kernels import pack_indices_3bit

        self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
        self._packed_values_cache = pack_indices_3bit(self.compressed_values["indices"])
        mx.eval(
            *[v for v in self.compressed_keys.values() if isinstance(v, mx.array)],
            *[v for v in self.compressed_values.values() if isinstance(v, mx.array)],
            self._packed_keys_cache,
            self._packed_values_cache,
        )

        if self._cache_mode == "prealloc":
            self._extend_buffers_by_step()

        # Free the FP16 buffer
        self._fp16_keys.clear()
        self._fp16_values.clear()

    def _compress_batch(self, x: mx.array) -> dict[str, mx.array]:
        """Batched MLX quantization for (heads, seq, dim), avoiding Python per-head loops."""
        _GLOBAL_STATS.compress_calls += 1
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8, dtype=mx.float32))
        x_rot = self._rotate_forward(x_unit)
        indices, _ = quantize_scalar(
            x_rot, self.compressor.centroids, self.compressor.boundaries
        )
        return {
            "indices": indices.astype(mx.uint8),
            "x_norm": x_norm.astype(mx.float16),
        }

    def _ensure_runtime_shape(self, n_kv_heads: int, head_dim: int) -> None:
        if self._fallback_cache is not None:
            return
        if n_kv_heads == self.num_heads and head_dim == self.head_dim:
            return

        try:
            self.num_heads = int(n_kv_heads)
            self.head_dim = int(head_dim)
            self.compressor = TurboQuantCompressor(
                bit_width=self.bit_width,
                head_dim=self.head_dim,
                codebook_dir=self.codebook_dir,
                seed=self.seed,
            )
            self._set_rotation_components(self.num_heads, self.head_dim)
            self._resolve_fused_encode_gate()
            self._invalidate_fused_caches()
            self._fused_metal_ok = None
        except Exception:
            # Fallback for heterogeneous-cache layers without matching codebooks.
            self._activate_fallback_cache(n_kv_heads, head_dim)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array, offset: int | None = None
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache using batched MLX kernels only.

        This keeps Python out of the per-head hot path for IsoQuant.
        """
        batch_size, n_kv_heads, seq_len, _ = keys.shape
        assert batch_size == 1, "IsoQuantKVCache currently assumes batch=1"
        if offset is not None:
            self._seq_len = offset
            self.offset = offset
        self._ensure_runtime_shape(n_kv_heads, int(keys.shape[-1]))
        if self._fallback_cache is not None:
            _GLOBAL_STATS.fallback_invocations += 1
            _record_path(self.layer_idx, "fp16_fallback_cache")
            out = self._fallback_cache.update_and_fetch(keys, values)
            self._sync_fallback_offset()
            return out

        k_batch = keys[0]  # (heads, seq, dim)
        v_batch = values[0]

        if self._deferred:
            # Prefill phase: accumulate FP16, no compression (zero error compounding).
            _GLOBAL_STATS.prefill_calls += 1
            _GLOBAL_STATS.prefill_tokens += seq_len
            _record_path(self.layer_idx, "deferred_fp16")
            self._fp16_keys.append(k_batch)
            self._fp16_values.append(v_batch)
            self._seq_len += seq_len
            self.offset += seq_len
            return keys, values

        # Decode phase (post-finalize): compress incrementally.
        _GLOBAL_STATS.decode_calls += 1
        _GLOBAL_STATS.decode_tokens += seq_len
        _record_path(self.layer_idx, "decode_write")

        new_packed_k = None
        new_packed_v = None

        if self._use_fused_encode:
            try:
                from .fused_kv_compress import fused_compress_and_pack

                def _fused_encode(x_batch):
                    packed, norms, indices = fused_compress_and_pack(
                        x_batch,
                        self.block_matrices,
                        self.compressor.centroids,
                        self.compressor.boundaries,
                    )
                    return packed, {"indices": indices, "x_norm": norms}

                new_packed_k, new_compressed_keys = self._profile_mx_call(
                    "decode_key_compress_ms",
                    lambda: _fused_encode(k_batch),
                )
                new_packed_v, new_compressed_vals = self._profile_mx_call(
                    "decode_value_compress_ms",
                    lambda: _fused_encode(v_batch),
                )
                if not getattr(self, "_fused_encode_verified", False):
                    mx.eval(
                        new_packed_k,
                        new_compressed_keys["indices"],
                        new_compressed_keys["x_norm"],
                        new_packed_v,
                        new_compressed_vals["indices"],
                        new_compressed_vals["x_norm"],
                    )
                    self._fused_encode_verified = True
            except Exception as e:
                _record_error(self.layer_idx, "fused_encode", e)
                self._use_fused_encode = False
                new_packed_k = None
                new_packed_v = None

        _record_path(
            self.layer_idx, "encode_fused" if new_packed_k is not None else "encode_mlx"
        )

        if new_packed_k is None:
            new_compressed_keys = self._profile_mx_call(
                "decode_key_compress_ms",
                lambda: self._compress_batch(k_batch),
            )
            new_compressed_vals = self._profile_mx_call(
                "decode_value_compress_ms",
                lambda: self._compress_batch(v_batch),
            )

        from .fused_kv_decode_kernels import pack_indices_3bit

        if self._cache_mode == "prealloc":
            self._ensure_buffer_capacity(seq_len)
            pos = self.offset
            end = pos + seq_len
            if new_packed_k is None:
                new_packed_k = pack_indices_3bit(new_compressed_keys["indices"])
                new_packed_v = pack_indices_3bit(new_compressed_vals["indices"])
            self._profile_mx_eval(
                "decode_concat_ms",
                new_compressed_keys["indices"],
                new_compressed_keys["x_norm"],
                new_compressed_vals["indices"],
                new_compressed_vals["x_norm"],
                new_packed_k,
                new_packed_v,
            )
            self.compressed_keys["indices"][:, pos:end, :] = new_compressed_keys[
                "indices"
            ]
            self.compressed_keys["x_norm"][:, pos:end, :] = new_compressed_keys[
                "x_norm"
            ]
            self.compressed_values["indices"][:, pos:end, :] = new_compressed_vals[
                "indices"
            ]
            self.compressed_values["x_norm"][:, pos:end, :] = new_compressed_vals[
                "x_norm"
            ]
            self._packed_keys_cache[:, pos:end, :] = new_packed_k
            self._packed_values_cache[:, pos:end, :] = new_packed_v
        else:

            def _concat_compressed(existing, new):
                if not existing:
                    return new
                for key in new:
                    existing[key] = mx.concatenate([existing[key], new[key]], axis=1)
                return existing

            self.compressed_keys = _concat_compressed(
                self.compressed_keys, new_compressed_keys
            )
            self.compressed_values = _concat_compressed(
                self.compressed_values, new_compressed_vals
            )
            if new_packed_k is None:
                new_packed_k = pack_indices_3bit(new_compressed_keys["indices"])
                new_packed_v = pack_indices_3bit(new_compressed_vals["indices"])
            if self._packed_keys_cache is None:
                self._packed_keys_cache = new_packed_k
                self._packed_values_cache = new_packed_v
            else:
                self._packed_keys_cache = mx.concatenate(
                    [self._packed_keys_cache, new_packed_k], axis=1
                )
                self._packed_values_cache = mx.concatenate(
                    [self._packed_values_cache, new_packed_v], axis=1
                )
            self._profile_mx_eval(
                "decode_concat_ms",
                *[v for v in self.compressed_keys.values() if isinstance(v, mx.array)],
                *[
                    v
                    for v in self.compressed_values.values()
                    if isinstance(v, mx.array)
                ],
                self._packed_keys_cache,
                self._packed_values_cache,
            )
        self._seq_len += seq_len
        self.offset += seq_len
        return keys, values

    def reconstruct_keys(self) -> mx.array:
        """Full key sequence for SDPA; supports deferred FP16 prefill before bulk compress."""
        _GLOBAL_STATS.read_keys_calls += 1
        if self._fallback_cache is not None:
            return self._get_fallback_keys()
        if self._deferred and self._fp16_keys:
            full = mx.concatenate(self._fp16_keys, axis=1)
            return full[None, ...].astype(self._dtype)
        _record_path(self.layer_idx, "unfused_dense")
        if self._cache_mode == "prealloc":
            sliced = {
                k: v[:, : self.offset, :] for k, v in self.compressed_keys.items()
            }
        else:
            sliced = self.compressed_keys
        x_hat = self._decompress_batch(sliced)
        return x_hat[None, ...].astype(self._dtype)

    def get_values(self) -> mx.array:
        """Full value sequence for SDPA (pair with ``reconstruct_keys``)."""
        _GLOBAL_STATS.read_values_calls += 1
        if self._fallback_cache is not None:
            return self._get_fallback_values()
        if self._deferred and self._fp16_values:
            full = mx.concatenate(self._fp16_values, axis=1)
            return full[None, ...].astype(self._dtype)
        if self._cache_mode == "prealloc":
            sliced = {
                k: v[:, : self.offset, :] for k, v in self.compressed_values.items()
            }
        else:
            sliced = self.compressed_values
        x_hat = self._decompress_batch(sliced)
        return x_hat[None, ...].astype(self._dtype)

    # ------------------------------------------------------------------
    # Fused decode pipeline (Section 6.6)
    # ------------------------------------------------------------------
    #
    # The fused path eliminates the per-value inverse rotation by computing
    # attention entirely in rotated space and applying a single inverse
    # rotation on the aggregated output.
    #
    # Cost reduction: O(T × d²) → O(d²) for rotation.
    # ------------------------------------------------------------------

    @property
    def supports_fused_attention(self) -> bool:
        """True if the fused decode pipeline can be used."""
        return (
            self._fallback_cache is None
            and not self._deferred
            and self.compressed_keys is not None
            and self.compressed_values is not None
            and self.bit_width == 3
        )

    def fused_attention(
        self,
        queries: mx.array,
        scale: float,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Fused IsoQuant attention: compute in rotated space, inverse rotate once.

        Two execution paths:
          Metal-fused: packed 3-bit decode in-register, no K/V materialisation.
          MLX-ops:     centroid gather + dense matmul (correct but no bandwidth win).

        The Metal path is tried first; on failure it falls back to MLX-ops
        and latches ``_fused_metal_ok = False`` so subsequent calls skip the
        attempt.

        Args:
            queries: (B, num_heads, 1, head_dim) — decode query (single token)
            scale: attention scale factor (1/sqrt(d_k))
            mask: optional attention mask

        Returns:
            output: (B, num_heads, 1, head_dim) — attention output
        """
        if self.compressed_keys is None or self._seq_len == 0:
            _record_path(self.layer_idx, "fused_empty")
            return mx.zeros_like(queries)

        if not self.supports_fused_attention:
            import warnings

            if self.bit_width != 3:
                warnings.warn(
                    f"IsoQuant fused path unavailable: bit_width={self.bit_width} "
                    f"(only 3-bit supported). Using unfused O(T*d^2) path.",
                    stacklevel=2,
                )
            _GLOBAL_STATS.unfused_fallback_calls += 1
            keys = self.reconstruct_keys()
            values = self.get_values()
            from .base import scaled_dot_product_attention

            return scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=scale, mask=mask
            )

        B, H_q, _, D = queries.shape
        H_kv = self.num_heads
        if self._cache_mode == "prealloc":
            T = self.offset
        else:
            T = self.compressed_keys["indices"].shape[1]
        repeats = H_q // H_kv if H_q != H_kv else 1

        # --- Rotate query forward (always dense — one vector, O(d²)) ---
        R_T = mx.swapaxes(self.rotation_matrices, -2, -1)
        q_flat = queries[0, :, 0, :]
        R_T_exp = mx.repeat(R_T, repeats, axis=0) if repeats > 1 else R_T
        q_rot = mx.squeeze(mx.matmul(q_flat[:, None, :], R_T_exp), axis=1)  # (H_q, D)

        # --- Try Metal-fused kernels (packed 3-bit, no materialisation) ---
        if self._fused_metal_ok is not False:
            _GLOBAL_STATS.fused_metal_attempts += 1
            try:
                out = self._fused_attention_metal(
                    q_rot, scale, mask, H_q, H_kv, T, D, repeats
                )
                self._fused_metal_ok = True
                _record_path(self.layer_idx, "fused_metal")
                return out[None, :, None, :].astype(queries.dtype)
            except Exception as e:
                _GLOBAL_STATS.fused_metal_failures += 1
                _record_error(self.layer_idx, "fused_metal", e)
                self._fused_metal_ok = False

        # --- MLX-ops fallback (centroid gather + dense matmul) ---
        _record_path(self.layer_idx, "fused_mlx")
        out = self._fused_attention_mlx(q_rot, scale, mask, H_q, H_kv, T, D, repeats)
        return out[None, :, None, :].astype(queries.dtype)

    # ----- Metal-fused path (Kernels A/C/D) -----

    # Data-driven threshold (sweep_threshold.py, 100 iters, M4 Max, H=8 D=128):
    # The 3-kernel path wins at all practical T because it parallelizes QK dot
    # across T tokens with grid=(32*T, H).  The single kernel processes T serially
    # in one 32-thread group, losing ~30-50% even at T=16.
    # Disabled (threshold=0) until the single kernel gains T-parallel tiling.
    _SINGLE_KERNEL_T_THRESHOLD = 0
    _NPT8_TILED_T_THRESHOLD = 512

    def _fused_attention_metal(
        self,
        q_rot: mx.array,
        scale: float,
        mask: mx.array | None,
        H_q: int,
        H_kv: int,
        T: int,
        D: int,
        repeats: int,
    ) -> mx.array:
        """Execute attention via Metal kernels on packed 3-bit storage."""
        from .fused_kv_decode_kernels import pack_indices_3bit

        total_t0 = time.perf_counter() if self._profile_fused_timing else None

        if self._packed_keys_cache is None:
            _GLOBAL_STATS.packed_cache_misses += 1
            if self._cache_mode == "prealloc":
                valid_k_idx = self.compressed_keys["indices"][:, : self.offset, :]
            else:
                valid_k_idx = self.compressed_keys["indices"]
            self._packed_keys_cache = self._profile_mx_call(
                "packed_key_repack_ms",
                lambda: pack_indices_3bit(valid_k_idx),
            )
            if self._cache_mode == "prealloc":
                cap = self.compressed_keys["indices"].shape[1]
                cur = self._packed_keys_cache.shape[1]
                if cur < cap:
                    pad = mx.zeros(
                        (
                            self._packed_keys_cache.shape[0],
                            cap - cur,
                            self._packed_keys_cache.shape[2],
                        ),
                        dtype=mx.uint8,
                    )
                    self._packed_keys_cache = mx.concatenate(
                        [self._packed_keys_cache, pad], axis=1
                    )
        else:
            _GLOBAL_STATS.packed_cache_hits += 1

        if self._packed_values_cache is None:
            _GLOBAL_STATS.packed_cache_misses += 1
            if self._cache_mode == "prealloc":
                valid_v_idx = self.compressed_values["indices"][:, : self.offset, :]
            else:
                valid_v_idx = self.compressed_values["indices"]
            self._packed_values_cache = self._profile_mx_call(
                "packed_value_repack_ms",
                lambda: pack_indices_3bit(valid_v_idx),
            )
            if self._cache_mode == "prealloc":
                cap = self.compressed_values["indices"].shape[1]
                cur = self._packed_values_cache.shape[1]
                if cur < cap:
                    pad = mx.zeros(
                        (
                            self._packed_values_cache.shape[0],
                            cap - cur,
                            self._packed_values_cache.shape[2],
                        ),
                        dtype=mx.uint8,
                    )
                    self._packed_values_cache = mx.concatenate(
                        [self._packed_values_cache, pad], axis=1
                    )
        else:
            _GLOBAL_STATS.packed_cache_hits += 1

        k_packed = self._packed_keys_cache
        v_packed = self._packed_values_cache

        k_norms = self.compressed_keys["x_norm"][:, :, 0].astype(mx.float32)
        v_norms = self.compressed_values["x_norm"][:, :, 0].astype(mx.float32)
        centroids = self.compressor.centroids.reshape(-1).astype(mx.float32)
        kv_head_map = mx.arange(H_q, dtype=mx.uint32) // repeats

        if self._cache_mode == "prealloc":
            storage_stride = k_packed.shape[1]
        else:
            storage_stride = T

        if D == 256 and os.environ.get("ISOQUANT_USE_NPT8_FUSED", "0") == "1":
            if T >= self._NPT8_TILED_T_THRESHOLD:
                out = self._fused_attention_npt8_tiled(
                    k_packed,
                    v_packed,
                    centroids,
                    k_norms,
                    v_norms,
                    q_rot,
                    kv_head_map,
                    scale,
                    mask,
                    H_q,
                    H_kv,
                    T,
                    D,
                    storage_stride,
                )
            else:
                out = self._fused_attention_npt8(
                    k_packed,
                    v_packed,
                    centroids,
                    k_norms,
                    v_norms,
                    q_rot,
                    kv_head_map,
                    scale,
                    mask,
                    H_q,
                    H_kv,
                    T,
                    D,
                    storage_stride,
                )
        elif D == 512 and os.environ.get("ISOQUANT_USE_NPT16_FUSED", "0") == "1":
            out = self._fused_attention_npt16(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                q_rot,
                kv_head_map,
                scale,
                mask,
                H_q,
                H_kv,
                T,
                D,
                storage_stride,
            )
        elif T <= self._SINGLE_KERNEL_T_THRESHOLD:
            out = self._fused_attention_single_kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                q_rot,
                kv_head_map,
                scale,
                mask,
                H_q,
                T,
                D,
                storage_stride,
            )
        else:
            out = self._fused_attention_3kernel(
                k_packed,
                v_packed,
                centroids,
                k_norms,
                v_norms,
                q_rot,
                kv_head_map,
                scale,
                mask,
                H_q,
                H_kv,
                T,
                D,
                repeats,
                storage_stride,
            )
        if total_t0 is not None:
            mx.eval(out)
            mx.synchronize()
            _GLOBAL_STATS.fused_metal_total_ms += (
                time.perf_counter() - total_t0
            ) * 1000.0
        return out

    def _fused_attention_single_kernel(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        T,
        D,
        storage_stride,
    ) -> mx.array:
        """Single fully-fused kernel: QK + online softmax + V + inverse rotation."""
        from .fused_kv_decode_kernels import fully_fused_attention

        return self._profile_mx_call(
            "fused_single_kernel_ms",
            lambda: fully_fused_attention(
                K_packed=k_packed,
                V_packed=v_packed,
                centroids=centroids,
                k_norms=k_norms,
                v_norms=v_norms,
                q_rot=q_rot,
                kv_head_map=kv_head_map,
                blocks_t=self.block_matrices_t,
                scale=scale,
                num_heads=H_q,
                seq_len=T,
                head_dim=D,
                use_hadamard=self._use_hadamard,
                mask=mask,
                storage_stride=storage_stride,
            ),
        )

    def _fused_attention_npt8(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        H_kv,
        T,
        D,
        storage_stride,
    ) -> mx.array:
        """NPT=8 single-pass fused kernel for head_dim=256."""
        from .fused_kv_decode_npt8 import fused_attention_npt8

        return self._profile_mx_call(
            "fused_single_kernel_ms",
            lambda: fused_attention_npt8(
                K_packed=k_packed,
                V_packed=v_packed,
                centroids=centroids,
                k_norms=k_norms,
                v_norms=v_norms,
                q_rot=q_rot,
                kv_head_map=kv_head_map,
                blocks_t=self.block_matrices_t,
                scale=scale,
                use_hadamard=self._use_hadamard,
                mask=mask,
                num_heads=H_q,
                seq_len=T,
                head_dim=D,
                storage_stride=storage_stride,
            ),
        )

    def _fused_attention_npt16(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        H_kv,
        T,
        D,
        storage_stride,
    ) -> mx.array:
        """NPT=16 single-pass fused kernel for head_dim=512."""
        from .fused_kv_decode_npt16 import fused_attention_npt16

        return self._profile_mx_call(
            "fused_single_kernel_ms",
            lambda: fused_attention_npt16(
                K_packed=k_packed,
                V_packed=v_packed,
                centroids=centroids,
                k_norms=k_norms,
                v_norms=v_norms,
                q_rot=q_rot,
                kv_head_map=kv_head_map,
                blocks_t=self.block_matrices_t,
                scale=scale,
                use_hadamard=self._use_hadamard,
                mask=mask,
                num_heads=H_q,
                seq_len=T,
                head_dim=D,
                storage_stride=storage_stride,
            ),
        )

    def _fused_attention_npt8_tiled(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        H_kv,
        T,
        D,
        storage_stride,
    ) -> mx.array:
        """NPT=8 tiled fused kernel with FA2-style merge for long sequences."""
        from .fused_kv_decode_npt8_tiled import fused_attention_npt8_tiled

        try:
            tile_size = int(os.environ.get("ISOQUANT_NPT8_TILE_SIZE", "256"))
        except ValueError:
            tile_size = 256

        return self._profile_mx_call(
            "fused_single_kernel_ms",
            lambda: fused_attention_npt8_tiled(
                K_packed=k_packed,
                V_packed=v_packed,
                centroids=centroids,
                k_norms=k_norms,
                v_norms=v_norms,
                q_rot=q_rot,
                kv_head_map=kv_head_map,
                block_matrices=self.block_matrices,
                scale=scale,
                use_hadamard=self._use_hadamard,
                mask=mask,
                num_heads=H_q,
                seq_len=T,
                head_dim=D,
                tile_size=tile_size,
                storage_stride=storage_stride,
            ),
        )

    # Crossover above which T-parallel tiling of fused_value_accum wins
    # over the serial-T baseline. Default chosen from the analyst's M4 Max
    # heuristic (128-256 sweet spot); override via ISOQUANT_VACCUM_TILE.
    _T_TILED_VALUE_ACCUM_THRESHOLD = 256

    def _fused_attention_3kernel(
        self,
        k_packed,
        v_packed,
        centroids,
        k_norms,
        v_norms,
        q_rot,
        kv_head_map,
        scale,
        mask,
        H_q,
        H_kv,
        T,
        D,
        repeats,
        storage_stride,
    ) -> mx.array:
        """Legacy 3-kernel pipeline: QK dot + softmax + V accum + inverse rotation."""
        from .fused_kv_decode_kernels import fused_qk_dot, fused_value_accum

        scores = self._profile_mx_call(
            "fused_qk_ms",
            lambda: fused_qk_dot(
                k_packed,
                centroids,
                k_norms,
                q_rot,
                kv_head_map,
                H_q,
                T,
                D,
                storage_stride=storage_stride,
            ),
        )
        scores = scores * scale

        if mask is not None:
            m = mask
            while m.ndim > scores.ndim:
                m = m.squeeze(0)
            scores = scores + m

        attn_weights = self._profile_mx_call(
            "fused_softmax_ms",
            lambda: mx.softmax(scores, axis=-1),
        )

        if T > self._T_TILED_VALUE_ACCUM_THRESHOLD:
            import os

            from .fused_kv_decode_tiled import fused_value_accum_tiled

            try:
                tile_size = int(os.environ.get("ISOQUANT_VACCUM_TILE", 128))
            except ValueError:
                tile_size = 128
            output_rot = self._profile_mx_call(
                "fused_value_tiled_ms",
                lambda: fused_value_accum_tiled(
                    v_packed,
                    centroids,
                    v_norms,
                    attn_weights,
                    kv_head_map,
                    H_q,
                    T,
                    D,
                    tile_size=tile_size,
                    storage_stride=storage_stride,
                ),
            )
        else:
            output_rot = self._profile_mx_call(
                "fused_value_ms",
                lambda: fused_value_accum(
                    v_packed,
                    centroids,
                    v_norms,
                    attn_weights,
                    kv_head_map,
                    H_q,
                    T,
                    D,
                    storage_stride=storage_stride,
                ),
            )

        return self._profile_mx_call(
            "fused_inverse_ms",
            lambda: self._apply_inverse_rotation(
                output_rot, H_q, H_kv, D, repeats, use_metal_kernel=True
            ),
        )

    # ----- MLX-ops fallback (centroid gather + dense matmul) -----

    def _fused_attention_mlx(
        self,
        q_rot: mx.array,
        scale: float,
        mask: mx.array | None,
        H_q: int,
        H_kv: int,
        T: int,
        D: int,
        repeats: int,
    ) -> mx.array:
        """Rotated-space attention using standard MLX ops.

        Algorithmically identical to the Metal path but uses centroid gather
        + dense matmul instead of fused kernels.  Correct, but no bandwidth
        reduction — K and V are fully materialised in FP32 before the dot
        product.
        """
        if self._cache_mode == "prealloc":
            k_indices = self.compressed_keys["indices"][:, : self.offset, :]
            k_xnorm = self.compressed_keys["x_norm"][:, : self.offset, :]
            v_indices = self.compressed_values["indices"][:, : self.offset, :]
            v_xnorm = self.compressed_values["x_norm"][:, : self.offset, :]
        else:
            k_indices = self.compressed_keys["indices"]
            k_xnorm = self.compressed_keys["x_norm"]
            v_indices = self.compressed_values["indices"]
            v_xnorm = self.compressed_values["x_norm"]

        k_rot_quant = self.compressor.centroids[k_indices]
        k_rot_scaled = k_rot_quant * k_xnorm.astype(mx.float32)

        v_rot_quant = self.compressor.centroids[v_indices]
        v_rot_scaled = v_rot_quant * v_xnorm.astype(mx.float32)

        if repeats > 1:
            k_rot_scaled = mx.repeat(k_rot_scaled, repeats, axis=0)
            v_rot_scaled = mx.repeat(v_rot_scaled, repeats, axis=0)

        # Q·Kᵀ in rotated space
        scores = mx.matmul(
            q_rot[:, None, :], mx.swapaxes(k_rot_scaled, -2, -1)
        )  # (H_q, 1, T)
        scores = scores * scale

        if mask is not None:
            m = mask
            while m.ndim > scores.ndim:
                m = m.squeeze(0)
            scores = scores + m

        attn_weights = mx.softmax(scores, axis=-1)  # (H_q, 1, T)

        # Weighted value sum in rotated space
        output_rot = mx.matmul(attn_weights, v_rot_scaled)[:, 0, :]  # (H_q, D)

        return self._apply_inverse_rotation(
            output_rot, H_q, H_kv, D, repeats, use_metal_kernel=False
        )

    # ----- Shared inverse rotation -----

    def _apply_inverse_rotation(
        self,
        output_rot: mx.array,
        H_q: int,
        H_kv: int,
        D: int,
        repeats: int,
        use_metal_kernel: bool = False,
    ) -> mx.array:
        """Apply single inverse rotation on aggregated attention output.

        For GQA, groups query heads back to KV-head space, rotates per group,
        then re-interleaves.
        """
        if repeats > 1:
            output_groups = output_rot.reshape(H_kv, repeats, D)
            rotated_groups = []
            for g in range(repeats):
                group_out = output_groups[:, g, :]  # (H_kv, D)
                rotated_groups.append(
                    self._inverse_rotate_one(group_out, use_metal_kernel)
                )
            return mx.stack(rotated_groups, axis=1).reshape(H_q, D)
        else:
            return self._inverse_rotate_one(output_rot, use_metal_kernel)

    def _inverse_rotate_one(self, x: mx.array, use_metal_kernel: bool) -> mx.array:
        """Inverse-rotate (H, D) tensor.  Metal kernel or dense fallback."""
        if use_metal_kernel and self.block_matrices is not None:
            from .fused_kv_decode_kernels import fused_inverse_rotate

            return fused_inverse_rotate(x, self.block_matrices, self._use_hadamard)
        return self._rotate_inverse(x[:, None, :])[:, 0, :]

    @property
    def state(self) -> dict:
        base = super().state
        mode = getattr(self, "_cache_mode", "concat_append")
        if mode == "prealloc" and getattr(self, "offset", 0) > 0:
            T = self.offset
            base["compressed_keys"] = {
                k: v[:, :T, :] for k, v in self.compressed_keys.items()
            }
            base["compressed_values"] = {
                k: v[:, :T, :] for k, v in self.compressed_values.items()
            }
        return base

    @state.setter
    def state(self, v):
        TurboQuantKVCache.state.fset(self, v)
        mode = getattr(self, "_cache_mode", "concat_append")
        if mode == "prealloc" and self.compressed_keys:
            self._invalidate_fused_caches()
            from .fused_kv_decode_kernels import pack_indices_3bit

            self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
            self._packed_values_cache = pack_indices_3bit(
                self.compressed_values["indices"]
            )
            self._extend_buffers_by_step()

    def trim(self, n: int):
        """Trim prefix tokens, invalidating packed caches."""
        super().trim(n)
        self._invalidate_fused_caches()
        mode = getattr(self, "_cache_mode", "concat_append")
        if mode == "prealloc" and self.compressed_keys:
            from .fused_kv_decode_kernels import pack_indices_3bit

            self._packed_keys_cache = pack_indices_3bit(self.compressed_keys["indices"])
            self._packed_values_cache = pack_indices_3bit(
                self.compressed_values["indices"]
            )
            self._extend_buffers_by_step()

    def asymmetric_attention_scores(self, query, scale=1.0):
        if getattr(self, "_cache_mode", "concat_append") == "prealloc":
            saved_keys = self.compressed_keys
            saved_values = self.compressed_values
            T = self.offset
            self.compressed_keys = {
                k: v[:, :T, :] for k, v in self.compressed_keys.items()
            }
            self.compressed_values = {
                k: v[:, :T, :] for k, v in self.compressed_values.items()
            }
            try:
                return super().asymmetric_attention_scores(query, scale)
            finally:
                self.compressed_keys = saved_keys
                self.compressed_values = saved_values
        return super().asymmetric_attention_scores(query, scale)
