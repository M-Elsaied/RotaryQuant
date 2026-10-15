"""Measure KV compression fidelity: PPL + final-logit agreement.

Compares default, turboquant, and isoquant KV cache backends on the same
model and prompts. Produces a JSON artifact with PPL and final-logit
agreement metrics.

Usage:
    python scripts/measure_kv_fidelity.py --model <path> --output-json results/kv_fidelity.json

    # Multi-depth mode (Phase 1 canonical):
    python scripts/measure_kv_fidelity.py --model <path> \\
        --depths 512,2048 --seed 42 \\
        --output-json results/qwen3_kv_ppl_depth.json
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

import _kv_paths


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


def measure_ppl(model, tokenizer, text, kv_cache_type="default", max_tokens=512):
    """Measure perplexity on raw text (no chat template — PPL should be measured
    on natural language continuation, not chat-formatted input)."""
    from mlx_lm.models.cache import make_prompt_cache

    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    prompt = mx.array(tokens)[None]  # (1, T)
    cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)

    # Forward pass to get logits
    logits = model(prompt, cache=cache)  # (1, T, vocab)
    mx.eval(logits)

    # Compute per-token cross-entropy
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Shift: predict token[i+1] from position i
    target = prompt[:, 1:]  # (1, T-1)
    log_probs = log_probs[:, :-1, :]  # (1, T-1, vocab)

    # Gather log probs for actual next tokens
    target_log_probs = mx.take_along_axis(
        log_probs, target[:, :, None], axis=-1
    ).squeeze(-1)  # (1, T-1)
    mx.eval(target_log_probs)

    nll = -float(mx.mean(target_log_probs).item())
    ppl = float(np.exp(nll))
    return ppl, nll, len(tokens)


def measure_logit_similarity(model, tokenizer, text, kv_cache_type, max_tokens=256):
    """Measure final-logit agreement between default and compressed KV paths."""
    from mlx_lm.models.cache import make_prompt_cache

    tokens = _apply_chat_template(tokenizer, text)
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    prompt = mx.array(tokens)[None]

    # Default (reference)
    cache_default = make_prompt_cache(model, kv_cache_type="default")
    out_default = model(prompt, cache=cache_default)
    mx.eval(out_default)

    # Compressed
    cache_compressed = make_prompt_cache(model, kv_cache_type=kv_cache_type)
    out_compressed = model(prompt, cache=cache_compressed)
    mx.eval(out_compressed)

    # Cosine similarity of flattened final logits.
    a = out_default.reshape(-1).astype(mx.float32)
    b = out_compressed.reshape(-1).astype(mx.float32)
    cos_sim = float(
        (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8)).item()
    )

    # Top-k agreement
    k = 5
    top_default = mx.argsort(out_default[0, -1])[-k:]
    top_compressed = mx.argsort(out_compressed[0, -1])[-k:]
    mx.eval(top_default, top_compressed)
    set_d = set(top_default.tolist())
    set_c = set(top_compressed.tolist())
    topk_agreement = len(set_d & set_c) / k

    return cos_sim, topk_agreement


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


def _run_single_depth(model, tokenizer, texts, args, provenance):
    """Original single-depth mode (backward compatible)."""
    results = {
        "schema_version": 2,
        "metric_note": (
            "cosine/top5 metrics below are computed on final logits versus the "
            "default KV path, not directly on attention scores or latent KV vectors"
        ),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "backends": {},
    }

    for kv_type in ["default", "turboquant", "isoquant"]:
        print(f"\n=== {kv_type} ===")
        backend_result = {}

        # PPL
        ppls = []
        for i, text in enumerate(texts):
            try:
                ppl, nll, n_tokens = measure_ppl(
                    model, tokenizer, text, kv_type, args.max_tokens
                )
                ppls.append(ppl)
                print(f"  Text {i}: PPL={ppl:.2f} NLL={nll:.4f} ({n_tokens} tokens)")
            except Exception as e:
                print(f"  Text {i}: PPL ERROR: {e}")

        if ppls:
            backend_result["mean_ppl"] = round(float(np.mean(ppls)), 2)
            backend_result["ppls"] = [round(p, 2) for p in ppls]

        # Final-logit agreement (only for non-default)
        if kv_type != "default":
            try:
                cos_sim, topk = measure_logit_similarity(
                    model, tokenizer, texts[0], kv_type, args.max_tokens
                )
                backend_result["logit_cosine_sim_vs_default"] = round(cos_sim, 6)
                backend_result["logit_top5_agreement_vs_default"] = topk
                # Backward-compatible aliases for older artifact readers.
                backend_result["cosine_sim_vs_default"] = round(cos_sim, 6)
                backend_result["top5_agreement_vs_default"] = topk
                print(f"  Logit cosine sim vs default: {cos_sim:.6f}")
                print(f"  Logit top-5 agreement: {topk:.1%}")
            except Exception as e:
                print(f"  Logit agreement ERROR: {e}")

        results["backends"][kv_type] = backend_result

    attach_provenance(results, provenance)
    _kv_paths.attach(results, args.require_fused)
    _write_results(results, args.output_json)
    _kv_paths.enforce(args.require_fused)


def _run_multi_depth(model, tokenizer, texts, depth_list, args, provenance):
    """Multi-depth mode: measure PPL at each depth for each backend, compute deltas."""
    results = {
        "schema_version": 3,
        "mode": "multi_depth",
        "model": args.model,
        "seed": args.seed,
        "depths": depth_list,
        "backends": {},
    }

    # Collect PPL per backend per depth
    for kv_type in ["default", "turboquant", "isoquant"]:
        print(f"\n=== {kv_type} ===")
        backend_result = {"by_depth": {}}

        for depth in depth_list:
            # Build or extend text to reach target depth
            text = " ".join(texts)
            if len(tokenizer.encode(text)) < depth:
                text = _build_long_text(tokenizer, depth)

            try:
                ppl, nll, n_tokens = measure_ppl(
                    model, tokenizer, text, kv_type, max_tokens=depth
                )
                backend_result["by_depth"][str(depth)] = {
                    "ppl": round(ppl, 4),
                    "nll": round(nll, 6),
                    "n_tokens": n_tokens,
                }
                print(
                    f"  depth={depth}: PPL={ppl:.4f} NLL={nll:.6f} ({n_tokens} tokens)"
                )
            except Exception as e:
                backend_result["by_depth"][str(depth)] = {"error": str(e)}
                print(f"  depth={depth}: ERROR: {e}")

        results["backends"][kv_type] = backend_result

    # Compute delta_ppl_vs_default for compressed backends
    default_by_depth = results["backends"].get("default", {}).get("by_depth", {})
    for kv_type in ["turboquant", "isoquant"]:
        backend = results["backends"].get(kv_type, {})
        deltas = {}
        for depth_str, data in backend.get("by_depth", {}).items():
            ref = default_by_depth.get(depth_str, {})
            if "ppl" in data and "ppl" in ref:
                deltas[depth_str] = round(data["ppl"] - ref["ppl"], 4)
        backend["delta_ppl_vs_default"] = deltas

    attach_provenance(results, provenance)
    _kv_paths.attach(results, args.require_fused)
    _write_results(results, args.output_json)
    _kv_paths.enforce(args.require_fused)


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
