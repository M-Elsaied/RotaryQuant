"""Score tokens through the KV path that decode actually uses.

Two scoring modes:

``single_pass``
    One forward pass over the whole window. IsoQuant (and RotorQuant) hold a
    multi-token update in FP16 until the caches are finalized, so every scored
    position reads FP16 KV, while TurboQuant quantizes each update as it
    arrives and is scored on its quantized cache. The backends are therefore
    not compared under the same cache state. Kept for comparison with older
    artifacts.

``prefix_decode``
    Prefill the first ``prefix_tokens`` positions, finalize the caches (bulk
    compression), then feed the remaining tokens through the decode path in
    chunks of ``chunk`` tokens (default 1, the generation path) and score only
    those. After the prefix every backend quantizes each update before it is
    read, so the scored positions reflect the compressed cache. The report
    says which paths ran and whether the compressed cache was really read.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import mlx.core as mx
import numpy as np

SINGLE_PASS = "single_pass"
PREFIX_DECODE = "prefix_decode"
MODES = (PREFIX_DECODE, SINGLE_PASS)

_ISO_READ_PATHS = ("fused_metal", "fused_mlx", "unfused_dense")
_ISO_FP16_PATHS = ("deferred_fp16", "fp16_fallback_cache")


def add_arguments(parser) -> None:
    parser.add_argument(
        "--scoring",
        choices=MODES,
        default=PREFIX_DECODE,
        help="prefix_decode: prefill, finalize, then score the remaining tokens "
        "through the decode path (default). single_pass: one forward pass over "
        "the window; IsoQuant then scores FP16 KV while TurboQuant scores its "
        "quantized cache.",
    )
    parser.add_argument(
        "--score-tokens",
        type=int,
        default=128,
        help="prefix_decode: number of trailing tokens to score; the rest of the "
        "window is the prefix (clamped so the prefix holds at least one token).",
    )
    parser.add_argument(
        "--score-chunk",
        type=int,
        default=1,
        help="prefix_decode: tokens per forward call while scoring. 1 is the "
        "generation path; larger values attend densely over the dequantized cache.",
    )
    parser.add_argument(
        "--no-deferral",
        action="store_true",
        help="Quantize prefill updates as they arrive instead of holding them in FP16 "
        "until finalize (applies to every compressed backend).",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Do not exit 1 when a compressed backend did not read its compressed "
        "cache for the scored positions.",
    )


def settings_from_args(args) -> dict[str, Any]:
    return {
        "mode": args.scoring,
        "score_tokens": args.score_tokens,
        "chunk": args.score_chunk,
        "defer_prefill": not args.no_deferral,
    }


# --------------------------------------------------------------------------- caches


def iter_caches(cache: Iterable[Any]):
    """Yield every cache object, descending into CacheList containers."""
    for c in cache:
        subs = getattr(c, "caches", None)
        if isinstance(subs, (list, tuple)):
            yield from iter_caches(subs)
        else:
            yield c


def set_deferral(cache: Iterable[Any], enabled: bool) -> None:
    """Switch prefill deferral on or off for every compressed cache in ``cache``.

    RotorQuant caches expose ``defer_prefill``. IsoQuant caches start deferred
    and leave that state through ``finalize_deferred_prefill``; calling it before
    any update switches them to incremental compression with nothing to flush.
    TurboQuant caches never defer, so there is nothing to switch.
    """
    for c in iter_caches(cache):
        if hasattr(c, "defer_prefill"):
            c.defer_prefill = enabled
        elif not enabled and callable(getattr(c, "finalize_deferred_prefill", None)):
            c.finalize_deferred_prefill()


def _path_totals() -> dict[str, int]:
    try:
        from mlx_lm.models import mlx_isoquant
    except Exception:  # pragma: no cover - mlx_lm without the IsoQuant module
        return {}
    return dict(mlx_isoquant.kv_path_report().get("totals", {}))


def _diff(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    out = {k: after.get(k, 0) - before.get(k, 0) for k in sorted(keys)}
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- scoring


def _token_log_probs(logits: mx.array, targets: mx.array) -> np.ndarray:
    """Log-probability of ``targets`` (1, L) under ``logits`` (1, L, V)."""
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    picked = mx.take_along_axis(lp, targets[:, :, None], axis=-1).squeeze(-1)
    mx.eval(picked)
    return np.array(picked[0], dtype=np.float64)


def resolve_prefix(n_tokens: int, score_tokens: int | None, prefix_tokens: int | None) -> int:
    """Number of prefix tokens for a window of ``n_tokens``.

    The prefix keeps at least one token and leaves at least one token to score.
    """
    if n_tokens < 3:
        raise ValueError(f"need at least 3 tokens to score through the decode path, got {n_tokens}")
    if prefix_tokens is not None:
        p = int(prefix_tokens)
    elif score_tokens is not None:
        p = n_tokens - int(score_tokens) - 1
    else:
        p = n_tokens // 2
    return max(1, min(p, n_tokens - 2))


def score_tokens(
    model,
    token_ids,
    kv_cache_type: str = "default",
    *,
    mode: str = PREFIX_DECODE,
    score_tokens: int | None = None,
    prefix_tokens: int | None = None,
    chunk: int = 1,
    defer_prefill: bool = True,
) -> dict[str, Any]:
    """Score ``token_ids`` with the given KV backend; see the module docstring.

    Returns a dict with ``nll``, ``ppl``, ``n_scored``, ``prefix_tokens``,
    ``token_nll`` (per scored position), ``final_logits`` (last step, float32),
    ``paths`` (IsoQuant path counts during prefill and scoring) and
    ``compressed_path`` (verification that the scored positions read the
    compressed cache; ``checked`` is False for the default backend).
    """
    from mlx_lm.models.cache import finalize_deferred_kv_caches, make_prompt_cache

    if mode not in MODES:
        raise ValueError(f"unknown scoring mode {mode!r}; expected one of {MODES}")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    tokens = [int(t) for t in token_ids]
    n = len(tokens)
    prompt = mx.array(tokens)[None]
    cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)

    if mode == SINGLE_PASS:
        before = _path_totals()
        logits = model(prompt, cache=cache)
        mx.eval(logits)
        lps = _token_log_probs(logits[:, :-1, :], prompt[:, 1:])
        final_logits = logits[0, -1].astype(mx.float32)
        prefix = 0
        paths = {"prefill": {}, "scoring": _diff(before, _path_totals())}
    else:
        prefix = resolve_prefix(n, score_tokens, prefix_tokens)
        if not defer_prefill:
            set_deferral(cache, False)
        before = _path_totals()
        logits = model(prompt[:, :prefix], cache=cache)
        mx.eval(logits)
        finalize_deferred_kv_caches(cache)
        # From here on every backend quantizes an update before it is read.
        set_deferral(cache, False)
        prefill_paths = _diff(before, _path_totals())

        before = _path_totals()
        parts = []
        final_logits = None
        for start in range(prefix, n - 1, chunk):
            end = min(start + chunk, n - 1)
            logits = model(prompt[:, start:end], cache=cache)
            mx.eval(logits)
            parts.append(_token_log_probs(logits, prompt[:, start + 1 : end + 1]))
            final_logits = logits[0, -1].astype(mx.float32)
        lps = np.concatenate(parts)
        paths = {"prefill": prefill_paths, "scoring": _diff(before, _path_totals())}

    nll = float(-lps.mean())
    # Tokens written to the cache: all n in a single pass, n - 1 when the last
    # token is only a target.
    written = n if mode == SINGLE_PASS else n - 1
    verification = verify_compressed_path(kv_cache_type, cache, paths["scoring"], written)
    return {
        "mode": mode,
        "kv_cache_type": kv_cache_type,
        "n_tokens": n,
        "prefix_tokens": prefix,
        "n_scored": int(lps.size),
        "chunk": chunk if mode == PREFIX_DECODE else n,
        "defer_prefill": bool(defer_prefill),
        "nll": nll,
        "ppl": float(math.exp(nll)),
        "token_nll": [float(-x) for x in lps],
        "final_logits": final_logits,
        "paths": paths,
        "compressed_path": verification,
    }


# --------------------------------------------------------------------------- checks


def verify_compressed_path(kv_cache_type: str, cache, scoring_paths: dict[str, int], expected_offset: int) -> dict[str, Any]:
    """Did the scored positions read a compressed cache?

    IsoQuant: the scoring phase must contain decode reads and no FP16 reads.
    TurboQuant: every cache must be on its quantized store (no FP16 fallback)
    holding ``expected_offset`` tokens.
    RotorQuant: as TurboQuant, and no cache may still hold deferred FP16.
    """
    if kv_cache_type == "default":
        return {"checked": False, "ok": True, "reasons": []}

    iso, tq, rotor = [], [], []
    for c in iter_caches(cache):
        if hasattr(c, "finalize_deferred_prefill") and hasattr(c, "_deferred"):
            iso.append(c)
        elif hasattr(c, "defer_prefill") and hasattr(c, "is_deferred"):
            rotor.append(c)
        elif hasattr(c, "compressed_keys") and hasattr(c, "_seq_len"):
            tq.append(c)

    reasons: list[str] = []
    if tq:
        fallback = sum(1 for c in tq if getattr(c, "_fallback_cache", None) is not None)
        if fallback:
            reasons.append(f"turboquant: {fallback} cache(s) on the FP16 fallback (no codebook for this head_dim)")
        empty = sum(1 for c in tq if not getattr(c, "compressed_keys", None))
        if empty:
            reasons.append(f"turboquant: {empty} cache(s) never stored quantized KV")
        short = sum(1 for c in tq if getattr(c, "compressed_keys", None) and getattr(c, "offset", -1) != expected_offset)
        if short:
            reasons.append(f"turboquant: {short} cache(s) stored a different token count than expected ({expected_offset})")
    if iso:
        reads = sum(scoring_paths.get(p, 0) for p in _ISO_READ_PATHS)
        fp16 = sum(scoring_paths.get(p, 0) for p in _ISO_FP16_PATHS)
        if reads == 0:
            reasons.append("isoquant: no decode reads of the compressed cache while scoring")
        if fp16:
            reasons.append(f"isoquant: {fp16} FP16 reads while scoring ({', '.join(p for p in _ISO_FP16_PATHS if scoring_paths.get(p))})")
        still_deferred = sum(1 for c in iso if getattr(c, "_deferred", False))
        if still_deferred:
            reasons.append(f"isoquant: {still_deferred} cache(s) still deferred after scoring")
        fallback = sum(1 for c in iso if getattr(c, "_fallback_cache", None) is not None)
        if fallback:
            reasons.append(f"isoquant: {fallback} cache(s) on the FP16 fallback (no codebook for this head_dim)")
    if rotor:
        deferred = sum(1 for c in rotor if getattr(c, "is_deferred", False))
        if deferred:
            reasons.append(f"rotorquant: {deferred} cache(s) still hold deferred FP16 after scoring")
        empty = sum(1 for c in rotor if getattr(c, "keys", None) is None)
        if empty:
            reasons.append(f"rotorquant: {empty} cache(s) never stored quantized KV")
        short = sum(1 for c in rotor if getattr(c, "keys", None) is not None and getattr(c, "offset", -1) != expected_offset)
        if short:
            reasons.append(f"rotorquant: {short} cache(s) stored a different token count than expected ({expected_offset})")
    if not iso and not tq and not rotor:
        reasons.append(f"{kv_cache_type}: no compressed caches found (all layers skipped or unsupported)")

    return {
        "checked": True,
        "ok": not reasons,
        "reasons": reasons,
        "n_isoquant_caches": len(iso),
        "n_turboquant_caches": len(tq),
        "n_rotorquant_caches": len(rotor),
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly subset of a ``score_tokens`` result (no logits, no per-token list)."""
    keep = ("mode", "n_tokens", "prefix_tokens", "n_scored", "chunk", "defer_prefill", "nll", "ppl", "paths", "compressed_path")
    out = {k: result[k] for k in keep}
    out["nll"] = round(out["nll"], 6)
    out["ppl"] = round(out["ppl"], 4)
    return out
