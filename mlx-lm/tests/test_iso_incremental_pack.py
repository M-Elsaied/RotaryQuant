"""Phase 2.5: preallocated-buffer correctness + A/B equivalence tests.

Tests both cache modes (concat_append and prealloc) and verifies:
- Incremental packed cache matches fresh rebuild (bit-exact)
- Prealloc mode produces identical valid data to concat_append
- Buffer extension at STEP boundary
- Fused attention equivalence across modes
- reconstruct_keys / get_values ignore padding
- State serialization round-trip excludes padding
- trim() invalidates packed caches
"""

import os

os.environ["ISOQUANT_BITS"] = "3"

import pytest

import mlx.core as mx
import numpy as np

from mlx_lm.models.fused_kv_decode_kernels import pack_indices_3bit
from mlx_lm.models.mlx_isoquant import IsoQuantKVCache
from mlx_lm.models.mlx_turboquant import get_default_codebook_dir

metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

NUM_HEADS, HEAD_DIM = 2, 256
CODEBOOK_DIR = get_default_codebook_dir()


@pytest.fixture(params=["concat_append", "prealloc"])
def cache_mode(request, monkeypatch):
    monkeypatch.setenv("ISOQUANT_CACHE_MODE", request.param)
    return request.param


def _make_cache(mode=None):
    """Build a real IsoQuant cache wired to the production codebook dir."""
    if mode is not None:
        os.environ["ISOQUANT_CACHE_MODE"] = mode
    return IsoQuantKVCache(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        bit_width=3,
        layer_idx=0,
        codebook_dir=CODEBOOK_DIR,
    )


def _gen_kv(seq_len: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    keys = mx.array(
        rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32)
    )
    values = mx.array(
        rng.standard_normal((1, NUM_HEADS, seq_len, HEAD_DIM)).astype(np.float32)
    )
    mx.eval(keys, values)
    return keys, values


def _valid_packed(cache):
    """Extract valid-range packed cache, accounting for prealloc padding."""
    T = cache.offset
    k = cache._packed_keys_cache
    v = cache._packed_values_cache
    if cache._cache_mode == "prealloc":
        k = k[:, :T, :]
        v = v[:, :T, :]
    return k, v


def test_incremental_append_matches_rebuild_short(cache_mode):
    """After 10 decode steps, incrementally-built packed cache equals fresh rebuild."""
    cache = _make_cache()
    assert cache._fallback_cache is None
    keys, values = _gen_kv(256)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=100 + step)
        cache.update_and_fetch(k_new, v_new)

    inc_keys, inc_vals = _valid_packed(cache)
    assert inc_keys is not None and inc_vals is not None

    if cache._cache_mode == "prealloc":
        ref_indices_k = cache.compressed_keys["indices"][:, : cache.offset, :]
        ref_indices_v = cache.compressed_values["indices"][:, : cache.offset, :]
    else:
        ref_indices_k = cache.compressed_keys["indices"]
        ref_indices_v = cache.compressed_values["indices"]

    ref_keys = pack_indices_3bit(ref_indices_k)
    ref_vals = pack_indices_3bit(ref_indices_v)
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)

    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))


def test_incremental_append_matches_rebuild_long(cache_mode):
    """8K total tokens: catches drift / slot-aliasing across many appends."""
    cache = _make_cache()
    assert cache._fallback_cache is None
    keys, values = _gen_kv(2048)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(8000):
        k_new, v_new = _gen_kv(1, seed=10000 + step)
        cache.update_and_fetch(k_new, v_new)

    inc_keys, inc_vals = _valid_packed(cache)
    assert inc_keys is not None and inc_vals is not None

    if cache._cache_mode == "prealloc":
        ref_indices_k = cache.compressed_keys["indices"][:, : cache.offset, :]
        ref_indices_v = cache.compressed_values["indices"][:, : cache.offset, :]
    else:
        ref_indices_k = cache.compressed_keys["indices"]
        ref_indices_v = cache.compressed_values["indices"]

    ref_keys = pack_indices_3bit(ref_indices_k)
    ref_vals = pack_indices_3bit(ref_indices_v)
    mx.eval(inc_keys, inc_vals, ref_keys, ref_vals)

    np.testing.assert_array_equal(np.asarray(inc_keys), np.asarray(ref_keys))
    np.testing.assert_array_equal(np.asarray(inc_vals), np.asarray(ref_vals))


def test_prealloc_matches_concat_append():
    """Both modes produce bit-exact identical valid compressed data."""
    results = {}
    for mode in ("concat_append", "prealloc"):
        cache = _make_cache(mode)
        keys, values = _gen_kv(256)
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        for step in range(50):
            k_new, v_new = _gen_kv(1, seed=200 + step)
            cache.update_and_fetch(k_new, v_new)

        T = cache.offset
        if mode == "prealloc":
            k_idx = cache.compressed_keys["indices"][:, :T, :]
        else:
            k_idx = cache.compressed_keys["indices"]

        mx.eval(k_idx)
        results[mode] = np.asarray(k_idx)

    np.testing.assert_array_equal(results["concat_append"], results["prealloc"])


def test_buffer_extension_at_step_boundary():
    """Buffer extends by STEP when capacity is exhausted."""
    cache = _make_cache("prealloc")
    keys, values = _gen_kv(128)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    initial_cap = cache.compressed_keys["indices"].shape[1]
    assert initial_cap == 128 + 256

    for step in range(256):
        k_new, v_new = _gen_kv(1, seed=300 + step)
        cache.update_and_fetch(k_new, v_new)

    k_new, v_new = _gen_kv(1, seed=999)
    cache.update_and_fetch(k_new, v_new)

    new_cap = cache.compressed_keys["indices"].shape[1]
    assert new_cap == 128 + 256 + 256
    assert cache.offset == 128 + 256 + 1


@metal_only
def test_fused_attention_equivalence():
    """fused_attention produces same output in both modes."""
    from mlx_lm.models.mlx_isoquant import reset_stats, get_stats

    outputs = {}
    for mode in ("concat_append", "prealloc"):
        reset_stats()
        cache = _make_cache(mode)
        keys, values = _gen_kv(128)
        cache.update_and_fetch(keys, values)
        cache.finalize_deferred_prefill()

        for step in range(10):
            k_new, v_new = _gen_kv(1, seed=500 + step)
            cache.update_and_fetch(k_new, v_new)

        assert cache.supports_fused_attention, (
            f"fused attention must be supported in {mode} mode"
        )
        queries = mx.array(
            np.random.default_rng(42)
            .standard_normal((1, NUM_HEADS, 1, HEAD_DIM))
            .astype(np.float32)
        )
        out = cache.fused_attention(queries, scale=1.0 / (HEAD_DIM**0.5))
        mx.eval(out)
        outputs[mode] = np.asarray(out)

        stats = get_stats()
        assert stats.fused_metal_attempts > 0, (
            f"fused Metal kernel was never attempted in {mode} mode"
        )
        assert stats.fused_metal_failures == 0, (
            f"fused Metal kernel failed {stats.fused_metal_failures} times in {mode} mode"
        )

    np.testing.assert_allclose(
        outputs["concat_append"],
        outputs["prealloc"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_reconstruct_values_ignore_padding():
    """reconstruct_keys and get_values return only valid tokens in prealloc mode."""
    cache = _make_cache("prealloc")
    keys, values = _gen_kv(64)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=600 + step)
        cache.update_and_fetch(k_new, v_new)

    rk = cache.reconstruct_keys()
    gv = cache.get_values()
    mx.eval(rk, gv)

    assert rk.shape[2] == 74  # 64 + 10
    assert gv.shape[2] == 74


def test_state_serialization_roundtrip(monkeypatch):
    """State serialization excludes padding; from_state restores prealloc buffers."""
    monkeypatch.setenv("ISOQUANT_CACHE_MODE", "prealloc")
    cache = _make_cache("prealloc")
    keys, values = _gen_kv(64)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=700 + step)
        cache.update_and_fetch(k_new, v_new)

    state = cache.state
    T_expected = 74
    assert state["compressed_keys"]["indices"].shape[1] == T_expected

    cache2 = IsoQuantKVCache.from_state(state, cache.meta_state)

    assert cache2.compressed_keys["indices"].shape[1] == T_expected + 256
    assert cache2.offset == T_expected


def test_trim_invalidates_packed_cache():
    """trim() invalidates packed caches and re-extends in prealloc mode."""
    cache = _make_cache("prealloc")
    keys, values = _gen_kv(128)
    cache.update_and_fetch(keys, values)
    cache.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=800 + step)
        cache.update_and_fetch(k_new, v_new)

    assert cache._packed_keys_cache is not None
    pre_offset = cache.offset
    cache.trim(5)
    assert cache._packed_keys_cache is not None  # re-extended after trim
    assert cache.offset == pre_offset - 5
    cap = cache.compressed_keys["indices"].shape[1]
    assert cap > cache.offset  # buffer has padding


@metal_only
def test_restore_then_fused_attention_and_decode(monkeypatch):
    """from_state restore followed by fused attention + continued decode.

    Proves the full pipeline works after serialization round-trip:
    1. Build cache, prefill, decode 10 steps.
    2. Serialize state.
    3. Restore via from_state.
    4. Run fused_attention on restored cache — must match original.
    5. Continue decoding 5 more steps on restored cache.
    6. Run fused_attention again — must match cache built without restore.
    """
    from mlx_lm.models.mlx_isoquant import reset_stats, get_stats

    monkeypatch.setenv("ISOQUANT_CACHE_MODE", "prealloc")

    # --- Build original cache ---
    cache_orig = _make_cache("prealloc")
    keys, values = _gen_kv(128)
    cache_orig.update_and_fetch(keys, values)
    cache_orig.finalize_deferred_prefill()

    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=900 + step)
        cache_orig.update_and_fetch(k_new, v_new)

    assert cache_orig.supports_fused_attention

    queries = mx.array(
        np.random.default_rng(42)
        .standard_normal((1, NUM_HEADS, 1, HEAD_DIM))
        .astype(np.float32)
    )
    scale = 1.0 / (HEAD_DIM**0.5)
    out_orig = cache_orig.fused_attention(queries, scale=scale)
    mx.eval(out_orig)

    # --- Serialize and restore ---
    state = cache_orig.state
    meta = cache_orig.meta_state

    reset_stats()
    cache_restored = IsoQuantKVCache.from_state(state, meta)

    assert cache_restored.supports_fused_attention
    assert cache_restored.offset == cache_orig.offset
    assert cache_restored._cache_mode == "prealloc"
    assert cache_restored._packed_keys_cache is not None

    # --- Fused attention on restored cache must match original ---
    out_restored = cache_restored.fused_attention(queries, scale=scale)
    mx.eval(out_restored)

    stats = get_stats()
    assert stats.fused_metal_attempts > 0, "fused Metal kernel was never attempted"
    assert stats.fused_metal_failures == 0, (
        f"fused Metal kernel failed {stats.fused_metal_failures} times"
    )

    np.testing.assert_allclose(
        np.asarray(out_orig),
        np.asarray(out_restored),
        rtol=1e-5,
        atol=1e-5,
        err_msg="fused attention output diverged after from_state restore",
    )

    # --- Continue decoding on restored cache ---
    for step in range(5):
        k_new, v_new = _gen_kv(1, seed=950 + step)
        cache_restored.update_and_fetch(k_new, v_new)

    assert cache_restored.offset == 128 + 10 + 5

    # --- Build a reference cache with the same full history ---
    cache_ref = _make_cache("prealloc")
    keys, values = _gen_kv(128)
    cache_ref.update_and_fetch(keys, values)
    cache_ref.finalize_deferred_prefill()
    for step in range(10):
        k_new, v_new = _gen_kv(1, seed=900 + step)
        cache_ref.update_and_fetch(k_new, v_new)
    for step in range(5):
        k_new, v_new = _gen_kv(1, seed=950 + step)
        cache_ref.update_and_fetch(k_new, v_new)

    # Fused attention on both must match
    out_continued = cache_restored.fused_attention(queries, scale=scale)
    out_ref = cache_ref.fused_attention(queries, scale=scale)
    mx.eval(out_continued, out_ref)

    np.testing.assert_allclose(
        np.asarray(out_continued),
        np.asarray(out_ref),
        rtol=1e-5,
        atol=1e-5,
        err_msg="fused attention diverged after restore + continued decode",
    )
