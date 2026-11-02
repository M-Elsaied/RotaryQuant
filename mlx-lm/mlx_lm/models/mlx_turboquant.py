"""
TurboQuant MLX Implementation (Refactored for Stateless Rotation)
========================================
Core TurboQuant algorithm for Apple Silicon using MLX.
"""

import importlib.util
import math
import os
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Any, Callable

_TQ_DEBUG = os.environ.get("TURBOQUANT_DEBUG", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_DEFAULT_TQ_SEED = 42

try:
    import turboquant_storage

    HAS_TQ_STORAGE = hasattr(turboquant_storage, "pack_3bit")
except ImportError:
    HAS_TQ_STORAGE = False


def get_default_codebook_dir() -> str:
    env_dir = os.environ.get("TURBOQUANT_CODEBOOK_DIR", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)

    module_dir = Path(__file__).resolve().parent
    for candidate in (module_dir / "turboquant_codebooks", module_dir / "codebooks"):
        if candidate.exists():
            return str(candidate)

    return os.path.abspath("codebooks")


def pack_indices(indices_mx: mx.array) -> Any:
    if not HAS_TQ_STORAGE:
        return indices_mx
    shape = indices_mx.shape
    if shape[-1] % 8 != 0:
        return indices_mx
    flat = np.array(indices_mx).flatten().astype(np.uint8)
    packed = turboquant_storage.pack_3bit(flat)
    new_shape = list(shape[:-1]) + [shape[-1] * 3 // 8]
    return packed.reshape(new_shape)


def unpack_indices(packed_indices: Any, original_shape: tuple) -> mx.array:
    if not HAS_TQ_STORAGE:
        return packed_indices
    if math.prod(packed_indices.shape) == math.prod(original_shape):
        return packed_indices

    if isinstance(packed_indices, mx.array):
        flat = np.array(packed_indices).flatten()
    else:
        flat = packed_indices.flatten()

    original_len = math.prod(original_shape)
    unpacked = turboquant_storage.unpack_3bit(flat, original_len)
    return mx.array(unpacked.reshape(original_shape))


def load_codebook(
    head_dim: int, bits: int, codebook_dir: str = "codebooks"
) -> tuple[mx.array, mx.array]:
    path = Path(codebook_dir) / f"dim_{head_dim}_{bits}bit.npz"
    if not path.exists():
        raise FileNotFoundError(f"Codebook not found at {path}")
    data = np.load(path)
    centroids = mx.array(data["centroids"], dtype=mx.float32)
    boundaries = mx.array(data["boundaries"], dtype=mx.float32)
    return centroids, boundaries


def quantize_scalar(
    x: mx.array, centroids: mx.array, boundaries: mx.array
) -> tuple[mx.array, mx.array]:
    flat_x = mx.reshape(x, (-1,))
    greater = flat_x[:, None] > boundaries[None, :]
    indices = mx.sum(greater, axis=-1)
    indices = mx.reshape(indices, x.shape)
    x_quant = centroids[indices]
    return indices, x_quant


class RotorQuantCompressor:
    """
    RotorQuant Compressor using Clifford $Cl(3,0)$ rotors.
    """

    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        codebook_dir: str | None = None,
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bit_width
        self.codebook_dir = os.path.abspath(codebook_dir or get_default_codebook_dir())
        self.centroids, self.boundaries = load_codebook(
            head_dim, bit_width, self.codebook_dir
        )

        # We need head_dim to be a multiple of 3 for exactly mapping 3D chunks.
        # Pad with zeros if necessary during runtime.
        self.padded_dim = ((head_dim + 2) // 3) * 3
        self.num_chunks = self.padded_dim // 3

    def _sandwich_product(self, x: mx.array, rotors: mx.array) -> mx.array:
        """
        Applies R * x * R_tilde using 4 non-zero components of Cl(3,0) rotors.
        x: [..., num_chunks, 3]
        rotors: [num_chunks, 4]  -> (scalar, bivector_xy, bivector_yz, bivector_zx)
        """
        # Expand rotors to match batch/seq dims of x
        R = mx.expand_dims(rotors, tuple(range(x.ndim - 2)))

        # Rotor components: R0 (scalar), R12, R23, R31 (bivectors)
        # Note: mapping corresponds to (w, z, x, y) in standard quaternion/rotor notation
        w = R[..., 0]
        x_rot = R[..., 2]  # R23
        y_rot = R[..., 3]  # R31
        z_rot = R[..., 1]  # R12

        # Vector components
        v1 = x[..., 0]
        v2 = x[..., 1]
        v3 = x[..., 2]

        # Rodrigues formula for Cl(3,0) sandwich: v' = v + 2s(u × v) + 2u × (u × v)
        # where R = s + u (scalar + bivector dual vector)
        # u × v (NOT v × u — sign matters)
        cross1 = y_rot * v3 - z_rot * v2
        cross2 = z_rot * v1 - x_rot * v3
        cross3 = x_rot * v2 - y_rot * v1

        # 2s(u × v)
        term1_1 = 2 * w * cross1
        term1_2 = 2 * w * cross2
        term1_3 = 2 * w * cross3

        # u × (u × v)
        cross2_1 = y_rot * cross3 - z_rot * cross2
        cross2_2 = z_rot * cross1 - x_rot * cross3
        cross2_3 = x_rot * cross2 - y_rot * cross1

        term2_1 = 2 * cross2_1
        term2_2 = 2 * cross2_2
        term2_3 = 2 * cross2_3

        out1 = v1 + term1_1 + term2_1
        out2 = v2 + term1_2 + term2_2
        out3 = v3 + term1_3 + term2_3

        return mx.stack([out1, out2, out3], axis=-1)

    def compress(self, x: mx.array, rotors: mx.array) -> dict:
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8))

        # Pad to multiple of 3
        pad_len = self.padded_dim - self.head_dim
        if pad_len > 0:
            zeros = mx.zeros(x_unit.shape[:-1] + (pad_len,))
            x_unit = mx.concatenate([x_unit, zeros], axis=-1)

        x_chunked = mx.reshape(x_unit, (*x_unit.shape[:-1], self.num_chunks, 3))

        # Apply rotor rotation
        x_rot_chunked = self._sandwich_product(x_chunked, rotors)

        # Flatten back — quantize ALL dims including padding so decompress
        # can inverse-rotate cleanly (truncating before quantization loses
        # rotated padding info that leaks into other dimensions on inverse).
        x_rot = mx.reshape(x_rot_chunked, (*x_rot_chunked.shape[:-2], self.padded_dim))

        indices, x_rot_quant = quantize_scalar(x_rot, self.centroids, self.boundaries)

        return {
            "indices": indices,
            "x_rot_quant": x_rot_quant,
            "x_norm": x_norm,
        }

    def decompress(
        self,
        packed_indices: Any,
        original_shape: tuple,
        x_norm: mx.array,
        rotors: mx.array,
    ) -> mx.array:
        indices = unpack_indices(packed_indices, original_shape)
        x_rot_quant = self.centroids[indices]

        # Indices are stored at padded_dim width (divisible by 3).
        # Reshape into 3D chunks, inverse-rotate, then truncate to head_dim.
        x_chunked = mx.reshape(
            x_rot_quant, (*x_rot_quant.shape[:-1], self.num_chunks, 3)
        )

        # Inverse rotation (conjugate of rotor)
        inv_rotors = mx.concatenate([rotors[..., :1], -rotors[..., 1:]], axis=-1)
        x_unrot_chunked = self._sandwich_product(x_chunked, inv_rotors)

        x_hat_unit = mx.reshape(
            x_unrot_chunked, (*x_unrot_chunked.shape[:-2], self.padded_dim)
        )
        # Truncate padding after inverse rotation (padding was part of the rotation)
        if self.padded_dim > self.head_dim:
            x_hat_unit = x_hat_unit[..., : self.head_dim]

        return x_hat_unit * x_norm


class TurboQuantCompressor:
    """
    Stateless TurboQuant Compressor (MSE + QJL).
    Takes rotation matrix Q as an argument during compression.
    """

    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        qjl_dim: int | None = None,
        codebook_dir: str = "codebooks",
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bit_width
        self.qjl_dim = qjl_dim or head_dim

        self.centroids, self.boundaries = load_codebook(
            head_dim, bit_width, codebook_dir
        )

        mx.random.seed(seed + 1000)
        self.S = mx.random.normal((self.qjl_dim, head_dim))
        self.qjl_scale = math.sqrt(math.pi / 2.0) / self.qjl_dim

    def compress(self, x: mx.array, rotation: mx.array) -> dict:
        """
        Args:
            x: shape (..., head_dim)
            rotation: shape (head_dim, head_dim) - orthogonal matrix Q
        """
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8))

        Q_T = mx.transpose(rotation)
        x_rot = mx.matmul(x_unit, Q_T)

        indices, x_rot_quant = quantize_scalar(x_rot, self.centroids, self.boundaries)

        x_hat_unit = mx.matmul(x_rot_quant, rotation)
        x_hat = x_hat_unit * x_norm

        residual = x - x_hat
        residual_norm = mx.linalg.norm(residual, axis=-1, keepdims=True)

        projected = mx.matmul(residual, mx.transpose(self.S))
        residual_signs = mx.sign(projected)

        return {
            "indices": indices,
            "x_rot_quant": x_rot_quant,
            "x_norm": x_norm,
            "residual_signs": residual_signs,
            "residual_norm": residual_norm,
        }

    def compress_value(self, x: mx.array, rotation: mx.array) -> dict:
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8))

        Q_T = mx.transpose(rotation)
        x_rot = mx.matmul(x_unit, Q_T)

        indices, _ = quantize_scalar(x_rot, self.centroids, self.boundaries)
        return {"indices": indices, "x_norm": x_norm}

    def decompress_value(self, compressed: dict, rotation: mx.array) -> mx.array:
        if "indices_shape" in compressed:
            indices = unpack_indices(compressed["indices"], compressed["indices_shape"])
        else:
            indices = compressed["indices"]
        x_rot_quant = self.centroids[indices]
        x_hat_unit = mx.matmul(x_rot_quant, rotation)
        return x_hat_unit * compressed["x_norm"]


def asymmetric_attention_scores(
    query: mx.array,
    compressed: dict,
    rotation: mx.array,
    S: mx.array,
    qjl_scale: float,
    scale: float,
) -> mx.array:
    if _TQ_DEBUG:
        mx.eval(query)
        if mx.any(mx.isnan(query)):
            print(f"[TURBOQUANT_DEBUG] query has NaNs, shape={query.shape}")

    scaled_q = query * scale

    q_rot = mx.matmul(scaled_q, mx.swapaxes(rotation, -2, -1))

    if "x_rot_quant" not in compressed:
        raise ValueError(
            "compressed must include 'x_rot_quant'; TurboQuantKVCache reconstructs it before calling."
        )

    x_rot_quant_T = mx.swapaxes(compressed["x_rot_quant"], -2, -1)
    base_scores = mx.matmul(q_rot, x_rot_quant_T)
    norm_T = mx.swapaxes(compressed["x_norm"], -2, -1)
    term1 = base_scores * norm_T

    # Term 2
    Sq = mx.matmul(scaled_q, mx.swapaxes(S, -2, -1))
    signs_T = mx.swapaxes(compressed["residual_signs"], -2, -1)
    correction = mx.matmul(Sq, signs_T)

    r_norm = mx.swapaxes(compressed["residual_norm"], -2, -1)
    term2 = (qjl_scale * r_norm) * correction

    res = term1 + term2
    if _TQ_DEBUG:
        mx.eval(res)
        if mx.any(mx.isnan(res)):
            t1 = mx.any(mx.isnan(term1)).item()
            t2 = mx.any(mx.isnan(term2)).item()
            print(f"[TURBOQUANT_DEBUG] NaNs in scores; term1={t1} term2={t2}")
    return res.astype(query.dtype)


_asym_override_state: tuple[str, float, Callable[..., mx.array]] | None = None


def get_active_asymmetric_attention_scores() -> Callable[..., mx.array]:
    """
    Baseline ``asymmetric_attention_scores`` or replacement from
    ``TURBOQUANT_ASYMMETRIC_SCORE_MODULE`` (path to a .py file defining the same API).
    """
    global _asym_override_state
    path = os.environ.get("TURBOQUANT_ASYMMETRIC_SCORE_MODULE", "").strip()
    if not path:
        return asymmetric_attention_scores
    apath = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(apath)
    except OSError:
        return asymmetric_attention_scores
    if (
        _asym_override_state is not None
        and _asym_override_state[0] == apath
        and _asym_override_state[1] == mtime
    ):
        return _asym_override_state[2]
    spec = importlib.util.spec_from_file_location(
        "turboquant_asymmetric_evolved", apath
    )
    if spec is None or spec.loader is None:
        return asymmetric_attention_scores
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "asymmetric_attention_scores", None)
    if not callable(fn):
        return asymmetric_attention_scores
    _asym_override_state = (apath, mtime, fn)
    return fn


class TurboQuantKVCache:
    """
    TurboQuant-compressed KV cache compatible with mlx-lm's KVCache interface.
    Only for standard attention layers (NOT SSM/Gated Delta layers).

    Stores: optionally packed codebook indices for keys, dense residual-sign
            tensors, float32 norms + residual norms per vector, and dense
            values.
    Reconstructs on-the-fly or computes asymmetric scores directly.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        bit_width: int = 4,  # 3 or 4 recommended
        max_seq_len: int = 32768,
        layer_idx: int | None = None,
        codebook_dir: str = "codebooks",
        seed: int = 42,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx
        self.seed = seed
        self.codebook_dir = os.path.abspath(codebook_dir or get_default_codebook_dir())
        self.compressor = None
        self.rotation_matrices = None
        self.compressed_keys: dict[str, Any] = {}
        self.compressed_values: dict[str, Any] = {}
        self.uncompressed_values: mx.array = None  # kept for backward compat
        self._fallback_cache = None
        self._warned_fallback = False
        self._seq_len = 0
        self._dtype = mx.float16  # or whatever base dtype
        self.offset = 0

        try:
            self._reconfigure_runtime_shape(num_heads, head_dim)
        except Exception:
            self._activate_fallback_cache(num_heads, head_dim)

    def _activate_fallback_cache(self, n_kv_heads: int, head_dim: int) -> None:
        from .cache import KVCache

        self._fallback_cache = KVCache()
        self.num_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        if not self._warned_fallback:
            print(
                f"[{type(self).__name__}] fallback to KVCache for heads={n_kv_heads}, dim={head_dim}"
            )
            self._warned_fallback = True

    def _sync_fallback_offset(self) -> None:
        if self._fallback_cache is None:
            return
        self.offset = self._fallback_cache.offset
        self._seq_len = self._fallback_cache.offset

    def _get_fallback_keys(self) -> mx.array:
        if self._fallback_cache is None or self._fallback_cache.keys is None:
            return None
        return self._fallback_cache.keys[..., : self._fallback_cache.offset, :]

    def _get_fallback_values(self) -> mx.array:
        if self._fallback_cache is None or self._fallback_cache.values is None:
            return None
        return self._fallback_cache.values[..., : self._fallback_cache.offset, :]

    def _reconfigure_runtime_shape(self, n_kv_heads: int, head_dim: int) -> None:
        if (
            self.compressor is not None
            and n_kv_heads == self.num_heads
            and head_dim == self.head_dim
        ):
            return
        self.num_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.compressor = TurboQuantCompressor(
            bit_width=self.bit_width,
            head_dim=self.head_dim,
            codebook_dir=self.codebook_dir,
            seed=self.seed,
        )
        rng = np.random.default_rng(self.seed + (self.layer_idx or 0) * 1000)
        rotations = []
        for _ in range(self.num_heads):
            A = rng.normal(size=(self.head_dim, self.head_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            rotations.append(Q)
        self.rotation_matrices = mx.array(np.stack(rotations), dtype=mx.float32)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array, offset: int | None = None
    ) -> tuple[mx.array, mx.array]:
        """
        Standard interface: compress new keys, store compressed rep,
        return (possibly on-the-fly reconstructed or placeholder) for attention.
        In practice, for TurboQuant we often bypass full reconstruction.
        """
        batch_size, n_kv_heads, seq_len, k_head_dim = (
            keys.shape
        )  # usually batch=1 in generation
        assert batch_size == 1, "TurboQuantKVCache currently assumes batch=1"
        # Reconstructed K/V follow the model dtype rather than a fixed float16.
        self._dtype = keys.dtype
        if self._fallback_cache is None:
            try:
                self._reconfigure_runtime_shape(n_kv_heads, int(k_head_dim))
            except Exception:
                self._activate_fallback_cache(n_kv_heads, int(k_head_dim))
        if self._fallback_cache is not None:
            out = self._fallback_cache.update_and_fetch(keys, values)
            self._sync_fallback_offset()
            return out

        new_seq_len = seq_len
        if offset is not None:
            self._seq_len = offset
            self.offset = offset
        start = self._seq_len

        # For processing, remove batch dim temporarily and cast to float32
        # keys: (1, H, S, D) -> (H, S, D)
        k_batch = keys[0].astype(mx.float32)

        v_batch = values[0].astype(mx.float32)

        new_k_indices = []
        new_k_norms = []
        new_v_indices = []
        new_v_norms = []

        for h in range(n_kv_heads):
            Q = self.rotation_matrices[h]
            # Compress keys (full TQ with residuals stored in compress but we only keep MSE)
            compressed_head = self.compressor.compress(k_batch[h], rotation=Q)
            new_k_indices.append(compressed_head["indices"])
            new_k_norms.append(compressed_head["x_norm"])
            # Compress values (MSE-only, no QJL needed)
            compressed_val = self.compressor.compress_value(v_batch[h], rotation=Q)
            new_v_indices.append(compressed_val["indices"])
            new_v_norms.append(compressed_val["x_norm"])

        new_compressed_keys = {
            "indices": mx.stack(new_k_indices, axis=0).astype(mx.uint8),
            "x_norm": mx.stack(new_k_norms, axis=0).astype(mx.float16),
        }
        new_compressed_vals = {
            "indices": mx.stack(new_v_indices, axis=0).astype(mx.uint8),
            "x_norm": mx.stack(new_v_norms, axis=0).astype(mx.float16),
        }

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

        mx.eval(
            *[v for v in self.compressed_keys.values() if isinstance(v, mx.array)],
            *[v for v in self.compressed_values.values() if isinstance(v, mx.array)],
        )

        self._seq_len += new_seq_len
        self.offset += new_seq_len

        # For compatibility: return keys/values as-is or dummy (real usage will detect cache type)
        # Many integrations keep the original for the fused path fallback
        return keys, values

    def get_values(self) -> mx.array:
        """Decompress values on-the-fly from compressed storage.
        Returns shape (1, num_kv_heads, seq_len, head_dim) in model dtype."""
        if self._fallback_cache is not None:
            return self._get_fallback_values()
        if self.compressed_values:
            indices = self.compressed_values["indices"]  # (H, S, D) uint8
            x_rot_quant = self.compressor.centroids[indices]  # (H, S, D) float32
            x_hat_unit = mx.matmul(x_rot_quant, self.rotation_matrices)
            x_hat = x_hat_unit * self.compressed_values["x_norm"]
            return x_hat[None, ...].astype(self._dtype)
        return self.uncompressed_values

    def reconstruct_keys(self) -> mx.array:
        """Decompress keys on-the-fly: indices -> centroids -> rotate back -> scale by norm.
        Returns shape (1, num_kv_heads, seq_len, head_dim) in model dtype."""
        if self._fallback_cache is not None:
            return self._get_fallback_keys()
        indices = self.compressed_keys["indices"]  # (H, S, D) uint8
        if HAS_TQ_STORAGE and not isinstance(indices, mx.array):
            indices = unpack_indices(indices, self.compressed_keys["indices_shape"])
        x_rot_quant = self.compressor.centroids[indices]  # (H, S, D) float32
        # Rotate back: (H, S, D) @ (H, D, D) -> (H, S, D)
        x_hat_unit = mx.matmul(x_rot_quant, self.rotation_matrices)
        x_hat = x_hat_unit * self.compressed_keys["x_norm"]  # (H, S, 1)
        return x_hat[None, ...].astype(self._dtype)  # (1, H, S, D)

    def asymmetric_attention_scores(
        self, query: mx.array, scale: float = 1.0
    ) -> mx.array:
        """
        Core TurboQuant method: compute asymmetric attention scores without full dequant.
        This is what replaces mx.fast.scaled_dot_product_attention for this cache.

        query shape: typically (batch, num_q_heads, seq_len, head_dim) or adjusted for GQA
        """
        batch, num_q_heads, num_queries, head_dim = query.shape
        heads_per_kv = num_q_heads // self.num_heads
        score_fn = get_active_asymmetric_attention_scores()

        # We need to broadcast self.compressed_keys if GQA is active (num_q_heads > num_kv_heads)

        # Prepare Q: (num_heads, head_dim, head_dim)
        # S: (head_dim, head_dim) -> (1, head_dim, head_dim)

        # Wait, get_active_asymmetric_attention_scores accepts a query and a dict.
        # If we just vmap it, or if it naturally broadcasts?
        # The optimized score_fn uses mx.matmul. mx.matmul broadcasts!
        # query: (1, num_q_heads, num_queries, head_dim)
        # rotation: (num_kv_heads, head_dim, head_dim) -> needs to be (num_q_heads, head_dim, head_dim)

        if heads_per_kv > 1:
            q_sample_reshaped = query.reshape(
                batch, self.num_heads, heads_per_kv, num_queries, head_dim
            )

            # Broadcast the rotation matrix
            # Q is (num_heads, head_dim, head_dim) -> (1, num_heads, 1, head_dim, head_dim)
            Q_expanded = self.rotation_matrices[None, :, None, :, :]

            # Broadcast compressed_keys
            comp_expanded = {}
            for k, v in self.compressed_keys.items():
                if k == "indices_shape":
                    # (H, S, D) -> (1, H, 1, S, D) - just keep it simple, unpack first!
                    pass
                elif k == "indices" and HAS_TQ_STORAGE and not isinstance(v, mx.array):
                    # Unpack before broadcasting
                    unpacked = unpack_indices(v, self.compressed_keys["indices_shape"])
                    comp_expanded[k] = unpacked[None, :, None, ...]
                else:
                    # v is (num_heads, seq_len, ...) -> (1, num_heads, 1, seq_len, ...)
                    comp_expanded[k] = v[None, :, None, ...]

            # We must dynamically reconstruct x_rot_quant if it was removed!
            if "x_rot_quant" not in comp_expanded and "indices" in comp_expanded:
                indices = comp_expanded["indices"]
                comp_expanded["x_rot_quant"] = self.compressor.centroids[indices]

            scores_expanded = score_fn(
                q_sample_reshaped,
                comp_expanded,
                rotation=Q_expanded,
                S=self.compressor.S,
                qjl_scale=self.compressor.qjl_scale,
                scale=scale,
            )

            # scores_expanded: (1, num_heads, heads_per_kv, num_queries, seq_len)
            scores = scores_expanded.reshape(batch, num_q_heads, num_queries, -1)
        else:
            # Broadcast Q and compressed
            Q_expanded = self.rotation_matrices[None, ...]
            comp_expanded = {}
            for k, v in self.compressed_keys.items():
                if k == "indices_shape":
                    pass
                elif k == "indices" and HAS_TQ_STORAGE and not isinstance(v, mx.array):
                    unpacked = unpack_indices(v, self.compressed_keys["indices_shape"])
                    comp_expanded[k] = unpacked[None, ...]
                else:
                    comp_expanded[k] = v[None, ...]

            # We must dynamically reconstruct x_rot_quant if it was removed!
            if "x_rot_quant" not in comp_expanded and "indices" in comp_expanded:
                indices = comp_expanded["indices"]
                comp_expanded["x_rot_quant"] = self.compressor.centroids[indices]

            scores = score_fn(
                query,
                comp_expanded,
                rotation=Q_expanded,
                S=self.compressor.S,
                qjl_scale=self.compressor.qjl_scale,
                scale=scale,
            )

        return scores

    def trim(self, n: int):
        """Trim prefix (for prefix caching / sliding window)."""
        self._seq_len = max(0, self._seq_len - n)
        self.offset = self._seq_len
        if self.compressed_keys:
            for key in self.compressed_keys:
                if key == "indices_shape":
                    old_shape = self.compressed_keys[key]
                    self.compressed_keys[key] = (
                        old_shape[0],
                        old_shape[1] - n,
                        old_shape[2],
                    )
                else:
                    self.compressed_keys[key] = self.compressed_keys[key][:, n:]
        if self.compressed_values:
            for key in self.compressed_values:
                self.compressed_values[key] = self.compressed_values[key][:, n:]
        if self.uncompressed_values is not None:
            self.uncompressed_values = self.uncompressed_values[:, :, n:, :]

    def make_mask(
        self, seq_len: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array:
        """Compatible with base.py create_attention_mask."""
        # Return causal mask or whatever your model expects
        from mlx_lm.models.base import create_causal_mask

        return create_causal_mask(seq_len, self.offset, window_size=window_size)

    @property
    def state(self) -> dict[str, Any]:
        """For serialization / checkpointing prefix cache.

        seq_len/bit_width wrapped as mx.array because mx.save_safetensors
        rejects plain Python ints (std::bad_cast).
        """
        return {
            "seq_len": mx.array(self._seq_len, dtype=mx.int32),
            "compressed_keys": self.compressed_keys,
            "compressed_values": self.compressed_values,
            "bit_width": mx.array(self.bit_width, dtype=mx.int32),
        }

    @state.setter
    def state(self, v):
        seq_len_v = v["seq_len"]
        if isinstance(seq_len_v, mx.array):
            seq_len_v = int(seq_len_v.item())
        self._seq_len = seq_len_v
        self.offset = seq_len_v
        self.compressed_keys = v["compressed_keys"]
        self.compressed_values = v.get("compressed_values", {})
        self.uncompressed_values = v.get("uncompressed_values")
        self._dtype = mx.float16

    @classmethod
    def from_state(cls, state: dict, meta_state=None, **kwargs):
        # Reconstruct from state (for prefix cache loading)
        if meta_state is None:
            raise ValueError(
                "meta_state is required for from_state reconstruction. "
                "Cannot restore cache without version, bit_width, num_heads, head_dim."
            )
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    "v2",
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
        if v is None:
            raise ValueError(
                "meta_state is required for from_state reconstruction. "
                "Cannot restore cache without version, bit_width, num_heads, head_dim."
            )
        version = v[0]
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported meta_state version: {version}")
        self.bit_width = int(v[1])
        self.num_heads = int(v[2])
        self.head_dim = int(v[3])
        layer_idx_val = int(v[4])
        self.layer_idx = layer_idx_val if layer_idx_val != -1 else None
        self.max_seq_len = 32768
        self.seed = int(v[6]) if version == "v2" and len(v) > 6 else _DEFAULT_TQ_SEED
        self.codebook_dir = (
            os.path.abspath(v[5])
            if version == "v2" and len(v) > 5 and v[5]
            else get_default_codebook_dir()
        )
        self._fallback_cache = None
        self._warned_fallback = False
        try:
            self._reconfigure_runtime_shape(self.num_heads, self.head_dim)
        except Exception:
            self._activate_fallback_cache(self.num_heads, self.head_dim)

    def is_trimmable(self):
        return True

    def size(self):
        if self._fallback_cache is not None:
            return self._fallback_cache.size()
        return self._seq_len

    @property
    def nbytes(self):
        if self._fallback_cache is not None:
            return self._fallback_cache.nbytes
        if self._seq_len == 0:
            return 0
        total = 0
        for store in (self.compressed_keys, self.compressed_values):
            if store:
                for k, v in store.items():
                    if isinstance(v, mx.array):
                        total += v.size * v.itemsize
        if self.uncompressed_values is not None:
            total += self.uncompressed_values.size * self.uncompressed_values.itemsize
        return total

    def empty(self):
        if self._fallback_cache is not None:
            return self._fallback_cache.empty()
        return self._seq_len == 0


from .cache import KVCache


class RotorQuantKVCache(KVCache):
    """
    RotorQuant KV cache. Uses sparse Clifford rotors instead of dense matrices.
    """

    def __init__(
        self,
        head_dim: int = 128,
        bit_width: int = 3,
        codebook_dir: str | None = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.codebook_dir = os.path.abspath(codebook_dir or get_default_codebook_dir())
        self.compressor = RotorQuantCompressor(
            bit_width, head_dim, self.codebook_dir, seed=seed
        )

        mx.random.seed(seed)
        raw_rotors = mx.random.normal((self.compressor.num_chunks, 4))
        self.rotors = raw_rotors / mx.linalg.norm(raw_rotors, axis=-1, keepdims=True)

        self.step = 256
        self.keys = None
        self.values = None
        self.k_norms = None
        self.v_norms = None
        self.is_deferred = False
        self.deferred_keys = None
        self.deferred_values = None
        # Multi-token updates are held in FP16 until the first single-token
        # step. Set False to quantize every update as it arrives.
        self.defer_prefill = True

    def _compress_and_store(self, keys: mx.array, values: mx.array):
        B, n_kv_heads, seq_len, head_dim = keys.shape

        k_comp = self.compressor.compress(keys, self.rotors)
        v_comp = self.compressor.compress(values, self.rotors)

        prev = self.offset

        packed_k = mx.array(pack_indices(k_comp["indices"]))
        packed_v = mx.array(pack_indices(v_comp["indices"]))

        if self.keys is None or (prev + seq_len) > self.keys.shape[2]:
            n_steps = (self.step + seq_len - 1) // self.step

            k_shape = (B, n_kv_heads, prev + n_steps * self.step, packed_k.shape[-1])
            v_shape = (B, n_kv_heads, prev + n_steps * self.step, packed_v.shape[-1])
            norm_shape = (B, n_kv_heads, prev + n_steps * self.step, 1)

            new_k = mx.zeros(k_shape, dtype=packed_k.dtype)
            new_v = mx.zeros(v_shape, dtype=packed_v.dtype)
            new_k_norm = mx.zeros(norm_shape, dtype=k_comp["x_norm"].dtype)
            new_v_norm = mx.zeros(norm_shape, dtype=v_comp["x_norm"].dtype)

            if self.keys is not None:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
                new_k_norm[..., :prev, :] = self.k_norms[..., :prev, :]
                new_v_norm[..., :prev, :] = self.v_norms[..., :prev, :]

            self.keys = new_k
            self.values = new_v
            self.k_norms = new_k_norm
            self.v_norms = new_v_norm

        self.offset += seq_len
        self.keys[..., prev : self.offset, :] = packed_k
        self.values[..., prev : self.offset, :] = packed_v
        self.k_norms[..., prev : self.offset, :] = k_comp["x_norm"]
        self.v_norms[..., prev : self.offset, :] = v_comp["x_norm"]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, n_kv_heads, seq_len, head_dim = keys.shape
        # Reconstructed K/V follow the model dtype rather than a fixed float16.
        self._dtype = keys.dtype

        if seq_len > 1 and self.defer_prefill:
            self.is_deferred = True
            if self.deferred_keys is None:
                self.deferred_keys = keys
                self.deferred_values = values
            else:
                self.deferred_keys = mx.concatenate([self.deferred_keys, keys], axis=2)
                self.deferred_values = mx.concatenate(
                    [self.deferred_values, values], axis=2
                )
            # Track total sequence length including deferred tokens
            self.offset += seq_len

            k_hat = self.deferred_keys
            v_hat = self.deferred_values

            if self.keys is not None and self.offset > 0:
                fetched_packed_k = self.keys[..., : self.offset, :]
                fetched_norm_k = self.k_norms[..., : self.offset, :]
                original_shape_k = fetched_packed_k.shape[:-1] + (
                    self.compressor.padded_dim,
                )
                past_k = self.compressor.decompress(
                    fetched_packed_k, original_shape_k, fetched_norm_k, self.rotors
                )
                k_hat = mx.concatenate([past_k, k_hat], axis=2)

                fetched_packed_v = self.values[..., : self.offset, :]
                fetched_norm_v = self.v_norms[..., : self.offset, :]
                original_shape_v = fetched_packed_v.shape[:-1] + (
                    self.compressor.padded_dim,
                )
                past_v = self.compressor.decompress(
                    fetched_packed_v, original_shape_v, fetched_norm_v, self.rotors
                )
                v_hat = mx.concatenate([past_v, v_hat], axis=2)

            return k_hat, v_hat

        # Transition to decode or just normal decode step
        if self.is_deferred:
            # Reset offset before compress — it was already tracked during deferred accumulation
            deferred_len = self.deferred_keys.shape[2]
            self.offset -= deferred_len
            self._compress_and_store(self.deferred_keys, self.deferred_values)
            self.deferred_keys = None
            self.deferred_values = None
            self.is_deferred = False

        self._compress_and_store(keys, values)

        fetched_packed_k = self.keys[..., : self.offset, :]
        fetched_norm_k = self.k_norms[..., : self.offset, :]
        original_shape_k = fetched_packed_k.shape[:-1] + (self.compressor.padded_dim,)
        k_hat = self.compressor.decompress(
            fetched_packed_k, original_shape_k, fetched_norm_k, self.rotors
        )

        fetched_packed_v = self.values[..., : self.offset, :]
        fetched_norm_v = self.v_norms[..., : self.offset, :]
        original_shape_v = fetched_packed_v.shape[:-1] + (self.compressor.padded_dim,)
        v_hat = self.compressor.decompress(
            fetched_packed_v, original_shape_v, fetched_norm_v, self.rotors
        )

        return k_hat, v_hat

    @property
    def state(self):
        # We need to save the rotors along with the KV states
        return self.keys, self.values, self.k_norms, self.v_norms, self.rotors

    @state.setter
    def state(self, v):
        self.keys, self.values, self.k_norms, self.v_norms, self.rotors = v

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return (
            self.keys.nbytes
            + self.values.nbytes
            + self.k_norms.nbytes
            + self.v_norms.nbytes
        )
