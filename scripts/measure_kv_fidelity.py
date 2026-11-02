"""Measure KV compression fidelity: PPL + final-logit agreement.

Compares default, turboquant, and isoquant KV cache backends on the same
model and prompts. Produces a JSON artifact with PPL and final-logit
agreement metrics.

Scoring (see scripts/_kv_scoring.py): by default the window is split into a
prefix and the trailing ``--score-tokens`` positions. The prefix is prefilled
and the caches are finalized (bulk compression), then the trailing tokens are
fed through the decode path one at a time and only those are scored, so the
numbers reflect the compressed cache every backend reads during generation.
``--scoring single_pass`` restores the older single forward pass, in which
IsoQuant scores FP16 KV (deferred prefill) while TurboQuant scores its quantized
cache; it is kept for comparison with earlier artifacts. Each compressed backend records whether its compressed cache was
actually read; the run exits 1 otherwise unless ``--allow-unverified`` is set.

Usage:
    python scripts/measure_kv_fidelity.py --model <path> --output-json results/kv_fidelity.json

    # Multi-depth mode:
    python scripts/measure_kv_fidelity.py --model <path> \\
        --depths 512,2048 --score-tokens 128 --seed 42 \\
        --output-json results/qwen3_kv_ppl_depth.json
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

import _kv_paths
import _kv_scoring


def _apply_chat_template(tokenizer, text):
    """Wrap text in chat template if the tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return tokenizer.encode(text)


def _build_long_text(tokenizer, target_tokens):
    """Build a repeating natural-language passage that tokenises to at least
    `target_tokens` tokens. Used when the built-in text corpus is too short
    for the requested depth."""
    unit = (
        "The transformer architecture processes sequences through alternating "
        "self-attention and feed-forward layers. Each attention head computes "
        "scaled dot-product attention between queries, keys, and values. "
        "Residual connections and layer normalisation stabilise training. "
        "In mixture-of-experts models, the feed-forward network is replaced by "
        "a routing function that selects a sparse subset of expert modules. "
    )
    text = unit
    while len(tokenizer.encode(text)) < target_tokens:
        text += unit
    return text


def _score_text(model, tokenizer, text, kv_cache_type, max_tokens, scoring):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return _kv_scoring.score_tokens(
        model,
        tokens,
        kv_cache_type,
        mode=scoring["mode"],
        score_tokens=scoring["score_tokens"],
        chunk=scoring["chunk"],
        defer_prefill=scoring["defer_prefill"],
    )


def measure_ppl(model, tokenizer, text, kv_cache_type="default", max_tokens=512, scoring=None):
    """Perplexity of the scored positions of ``text`` (raw text, no chat template).

    Returns the ``_kv_scoring.score_tokens`` result: ``ppl``, ``nll``, ``n_scored``,
    ``prefix_tokens``, ``paths`` and the ``compressed_path`` verification.
    """
    scoring = scoring or _DEFAULT_SCORING
    return _score_text(model, tokenizer, text, kv_cache_type, max_tokens, scoring)


def measure_logit_similarity(model, tokenizer, text, kv_cache_type, max_tokens=256, scoring=None):
    """Final-logit agreement between the default and a compressed KV path.

    Both paths score the same tokens the same way (prefix, finalize, decode), and
    the logits compared are those of the last scored step.
    """
    scoring = scoring or _DEFAULT_SCORING
    tokens = _apply_chat_template(tokenizer, text)
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    kw = dict(
        mode=scoring["mode"],
        score_tokens=scoring["score_tokens"],
        chunk=scoring["chunk"],
        defer_prefill=scoring["defer_prefill"],
    )
    ref = _kv_scoring.score_tokens(model, tokens, "default", **kw)
    cand = _kv_scoring.score_tokens(model, tokens, kv_cache_type, **kw)

    a = ref["final_logits"].reshape(-1).astype(mx.float32)
    b = cand["final_logits"].reshape(-1).astype(mx.float32)
    cos_sim = float(
        (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8)).item()
    )

    k = 5
    top_default = mx.argsort(ref["final_logits"])[-k:]
    top_compressed = mx.argsort(cand["final_logits"])[-k:]
    mx.eval(top_default, top_compressed)
    topk_agreement = len(set(top_default.tolist()) & set(top_compressed.tolist())) / k
    return cos_sim, topk_agreement, cand["compressed_path"]


_DEFAULT_SCORING = {
    "mode": _kv_scoring.PREFIX_DECODE,
    "score_tokens": 128,
    "chunk": 1,
    "defer_prefill": True,
}


def main():
    parser = argparse.ArgumentParser(description="Measure KV compression fidelity")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Token count for single-depth mode (ignored when --depths is set)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="",
        help="Comma-separated token depths, e.g. '512,2048,8192'. "
        "Runs PPL at each depth for each backend.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=128)
    _kv_paths.add_argument(parser)
    _kv_scoring.add_arguments(parser)
    args = parser.parse_args()
    _kv_paths.reset()

    from mlx_lm import load

    mx.random.seed(args.seed)

    model_config = {}
    if args.expert_offload:
        model_config["expert_offload"] = True
        model_config["max_resident_experts"] = args.max_resident_experts

    model, tokenizer = load(args.model, model_config=model_config)
    provenance = collect_provenance(args.model, args)

    # Test texts for PPL measurement — longer passages for stable estimation.
    # Three domains: technical prose, narrative, and code documentation.
    texts = [
        "The transformer architecture uses self-attention to process sequences in parallel. "
        "Each layer computes query, key, and value projections, applies scaled dot-product "
        "attention, and feeds the result through a position-wise feed-forward network. "
        "The multi-head mechanism allows the model to attend to different representation "
        "subspaces at different positions simultaneously. Layer normalisation is applied "
        "before each sub-layer in the pre-norm variant, which has become the standard in "
        "modern large language models. Residual connections around each sub-layer ensure "
        "that gradient flow is preserved during backpropagation through deep networks. "
        "The position encoding provides the model with information about the relative or "
        "absolute position of tokens in the sequence. Rotary position embeddings encode "
        "position information directly into the query and key representations by applying "
        "a rotation in the complex plane. This approach allows the model to generalise to "
        "longer sequences than those seen during training, a property known as length "
        "extrapolation. The feed-forward network typically consists of two linear "
        "transformations with a nonlinear activation function between them. In mixture-of-"
        "experts architectures, this feed-forward component is replicated across multiple "
        "expert modules, with a gating network that routes each token to the most relevant "
        "subset of experts. This introduces conditional computation, where the total "
        "parameter count is large but only a fraction of parameters are activated per token.",
        "In Python, a decorator is a function that takes another function as input and "
        "extends its behavior without explicitly modifying it. Decorators are commonly "
        "used for logging, authentication, memoization, and access control. The @syntax "
        "is syntactic sugar for passing the decorated function through the decorator. "
        "Context managers provide a way to allocate and release resources precisely when "
        "needed. The with statement ensures that cleanup code runs even if an exception "
        "occurs. Generator functions use the yield keyword to produce a sequence of values "
        "lazily, consuming memory only for the current value. This is particularly useful "
        "when processing large datasets that do not fit in memory. Asynchronous programming "
        "with async and await allows the program to perform other work while waiting for "
        "IO operations to complete. The event loop manages the scheduling of coroutines "
        "and callbacks. Type hints, introduced in Python three point five, allow developers "
        "to annotate function signatures and variable types. Static type checkers like mypy "
        "use these annotations to detect type errors before runtime. Dataclasses reduce "
        "boilerplate by automatically generating init, repr, and comparison methods from "
        "class field definitions.",
    ]

    # Parse depths
    depth_list = []
    if args.depths:
        depth_list = [int(d.strip()) for d in args.depths.split(",") if d.strip()]

    if depth_list:
        _run_multi_depth(model, tokenizer, texts, depth_list, args, provenance)
    else:
        _run_single_depth(model, tokenizer, texts, args, provenance)


def _finish(results, args, provenance):
    """Attach provenance and path report, write, then enforce the run's checks."""
    attach_provenance(results, provenance)
    _kv_paths.attach(results, args.require_fused)
    _write_results(results, args.output_json)
    _kv_paths.enforce(args.require_fused)
    failed = results.get("unverified_backends", [])
    if failed and not args.allow_unverified:
        print(
            "FAIL: compressed cache was not read for the scored positions: "
            + ", ".join(failed)
            + " (see backends.<name>.compressed_path.reasons; "
            "--allow-unverified keeps the exit code 0)",
            file=sys.stderr,
        )
        sys.exit(1)


def _record_verification(results):
    failed = []
    for kv_type, backend in results["backends"].items():
        checks = backend.get("compressed_path")
        if checks and checks.get("checked") and not checks.get("ok"):
            failed.append(kv_type)
    results["unverified_backends"] = failed


def _merge_checks(checks):
    """Combine per-text verification dicts into one for the backend."""
    checks = [c for c in checks if c]
    if not checks:
        return None
    reasons = sorted({r for c in checks for r in c.get("reasons", [])})
    return {
        "checked": any(c.get("checked") for c in checks),
        "ok": all(c.get("ok", False) for c in checks),
        "reasons": reasons,
    }


def _run_single_depth(model, tokenizer, texts, args, provenance):
    """Single-depth mode."""
    scoring = _kv_scoring.settings_from_args(args)
    results = {
        "schema_version": 4,
        "mode": "single_depth",
        "metric_note": (
            "cosine/top5 metrics below are computed on final logits versus the "
            "default KV path, not directly on attention scores or latent KV vectors"
        ),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "scoring": scoring,
        "backends": {},
    }

    for kv_type in ["default", "turboquant", "isoquant"]:
        print(f"\n=== {kv_type} ===")
        backend_result = {}
        checks = []

        ppls = []
        per_text = []
        for i, text in enumerate(texts):
            try:
                r = measure_ppl(model, tokenizer, text, kv_type, args.max_tokens, scoring)
                ppls.append(r["ppl"])
                per_text.append(_kv_scoring.summarize(r))
                checks.append(r["compressed_path"])
                print(
                    f"  Text {i}: PPL={r['ppl']:.2f} NLL={r['nll']:.4f} "
                    f"({r['n_scored']} scored after a {r['prefix_tokens']}-token prefix; "
                    f"compressed path {'ok' if r['compressed_path']['ok'] else 'NOT READ'})"
                )
            except Exception as e:
                print(f"  Text {i}: PPL ERROR: {e}")

        if ppls:
            backend_result["mean_ppl"] = round(float(np.mean(ppls)), 2)
            backend_result["ppls"] = [round(p, 2) for p in ppls]
            backend_result["texts"] = per_text

        if kv_type != "default":
            try:
                cos_sim, topk, sim_checks = measure_logit_similarity(
                    model, tokenizer, texts[0], kv_type, args.max_tokens, scoring
                )
                checks.append(sim_checks)
                backend_result["logit_cosine_sim_vs_default"] = round(cos_sim, 6)
                backend_result["logit_top5_agreement_vs_default"] = topk
                # Backward-compatible aliases for older artifact readers.
                backend_result["cosine_sim_vs_default"] = round(cos_sim, 6)
                backend_result["top5_agreement_vs_default"] = topk
                print(f"  Logit cosine sim vs default: {cos_sim:.6f}")
                print(f"  Logit top-5 agreement: {topk:.1%}")
            except Exception as e:
                print(f"  Logit agreement ERROR: {e}")

        merged = _merge_checks(checks)
        if merged is not None:
            backend_result["compressed_path"] = merged
        results["backends"][kv_type] = backend_result

    _record_verification(results)
    _finish(results, args, provenance)


def _run_multi_depth(model, tokenizer, texts, depth_list, args, provenance):
    """Multi-depth mode: PPL at each depth for each backend, with deltas."""
    scoring = _kv_scoring.settings_from_args(args)
    results = {
        "schema_version": 4,
        "mode": "multi_depth",
        "model": args.model,
        "seed": args.seed,
        "depths": depth_list,
        "scoring": scoring,
        "backends": {},
    }

    for kv_type in ["default", "turboquant", "isoquant"]:
        print(f"\n=== {kv_type} ===")
        backend_result = {"by_depth": {}}
        checks = []

        for depth in depth_list:
            text = " ".join(texts)
            if len(tokenizer.encode(text)) < depth:
                text = _build_long_text(tokenizer, depth)

            try:
                r = measure_ppl(model, tokenizer, text, kv_type, depth, scoring)
                entry = _kv_scoring.summarize(r)
                checks.append(r["compressed_path"])
                backend_result["by_depth"][str(depth)] = entry
                print(
                    f"  depth={depth}: PPL={r['ppl']:.4f} NLL={r['nll']:.6f} "
                    f"({r['n_scored']} scored after a {r['prefix_tokens']}-token prefix; "
                    f"compressed path {'ok' if r['compressed_path']['ok'] else 'NOT READ'})"
                )
            except Exception as e:
                backend_result["by_depth"][str(depth)] = {"error": str(e)}
                print(f"  depth={depth}: ERROR: {e}")

        merged = _merge_checks(checks)
        if merged is not None:
            backend_result["compressed_path"] = merged
        results["backends"][kv_type] = backend_result

    default_by_depth = results["backends"].get("default", {}).get("by_depth", {})
    for kv_type in ["turboquant", "isoquant"]:
        backend = results["backends"].get(kv_type, {})
        deltas = {}
        for depth_str, data in backend.get("by_depth", {}).items():
            ref = default_by_depth.get(depth_str, {})
            if "ppl" in data and "ppl" in ref:
                deltas[depth_str] = round(data["ppl"] - ref["ppl"], 4)
        backend["delta_ppl_vs_default"] = deltas

    _record_verification(results)
    _finish(results, args, provenance)


def _write_results(results, output_json):
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
