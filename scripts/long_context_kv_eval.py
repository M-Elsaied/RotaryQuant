"""Long-context KV cache fidelity evaluator.

Measures how well KV compression (IsoQuant / TurboQuant) preserves model
quality as context length grows.  Three evaluation axes:

  1. State retention — multi-turn conversation where later answers depend
     on facts stated earlier.  Measures whether the model can retrieve
     information across a growing KV cache.

  2. PPL at depth — compute per-token perplexity at increasing context
     offsets (256, 512, 1K, 2K, 4K tokens) on a fixed passage.  KV
     compression error compounds with depth; this measures the curve.

  3. Consistency under replay — generate the same prompt twice with the
     same seed; check whether compressed-KV output diverges from the
     default-KV output.

Each test produces observable metrics with no invented thresholds.
JSON artifact per run.

Usage:
  python scripts/long_context_kv_eval.py \\
      --model <path> --expert-offload --kv-cache-type isoquant \\
      --test all --output-json results/long_ctx_isoquant.json
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

_RUNTIME = None


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        _RUNTIME = (mx, generate, load, make_sampler)
    return _RUNTIME


def build_model_config(
    expert_offload: bool,
    max_resident_experts: int | None,
    *,
    use_predictor: bool = False,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if expert_offload:
        cfg["expert_offload"] = True
        if max_resident_experts is not None:
            cfg["max_resident_experts"] = max_resident_experts
    if use_predictor:
        cfg["use_predictor"] = True
    return cfg


def render_prompt(tokenizer, text: str, system_prompt: str | None = None) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        try:
            kwargs["enable_thinking"] = False
        except Exception:
            pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer.encode(text)


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</think>\s*", "", cleaned).strip()


def detect_repetition(text: str, threshold: float = 0.4) -> bool:
    words = text.split()
    if len(words) < 10:
        return False
    top = Counter(words).most_common(1)[0][1]
    return top / len(words) > threshold


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temp: float,
    kv_cache_type: str = "default",
) -> tuple[str, float]:
    _, generate, _, make_sampler = _runtime()
    sampler = make_sampler(temp)
    prompt = render_prompt(tokenizer, prompt_text, system_prompt)
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
        kv_cache_type=kv_cache_type,
    )
    return response.strip(), time.perf_counter() - t0


def compute_nll_and_final_logits(
    model,
    token_ids: list[int],
    *,
    kv_cache_type: str,
) -> tuple[float, float, Any]:
    """Compute mean NLL, PPL, and final logits for a token prefix."""
    mx, _, _, _ = _runtime()
    from mlx_lm.models.cache import make_prompt_cache

    prompt = mx.array(token_ids)[None]
    cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)
    logits = model(prompt, cache=cache)
    mx.eval(logits)

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target = prompt[:, 1:]
    log_probs = log_probs[:, :-1, :]
    target_log_probs = mx.take_along_axis(
        log_probs, target[:, :, None], axis=-1
    ).squeeze(-1)
    mx.eval(target_log_probs)

    nll = -float(mx.mean(target_log_probs).item())
    ppl = float(math.exp(nll))
    final_logits = logits[0, -1].astype(mx.float32)
    return nll, ppl, final_logits


def compare_final_logits(default_logits, candidate_logits) -> tuple[float, float]:
    """Cosine similarity and top-5 agreement between final-logit vectors."""
    mx, _, _, _ = _runtime()

    a = default_logits.reshape(-1).astype(mx.float32)
    b = candidate_logits.reshape(-1).astype(mx.float32)
    cos_sim = float(
        (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8)).item()
    )

    k = 5
    top_default = mx.argsort(default_logits)[-k:]
    top_candidate = mx.argsort(candidate_logits)[-k:]
    mx.eval(top_default, top_candidate)
    set_default = set(top_default.tolist())
    set_candidate = set(top_candidate.tolist())
    top5_agreement = len(set_default & set_candidate) / k
    return cos_sim, top5_agreement


# ─── Test 1: State Retention ──────────────────────────────────────────────

# Each task plants facts early, then asks about them after intervening text.
# The intervening text is deliberately verbose to push the facts deeper
# into the KV cache.

_FILLER = (
    "Meanwhile, the team discussed various implementation details "
    "including error handling strategies, logging frameworks, and "
    "deployment pipelines. They considered different approaches to "
    "containerisation and eventually settled on a multi-stage build "
    "process. Testing was debated at length, covering unit tests, "
    "integration tests, and end-to-end validation suites. "
)

STATE_RETENTION_TASKS = [
    {
        "name": "Variable recall (short gap)",
        "prompt": (
            "The database server is at host 10.0.3.47 on port 5433. "
            "The credentials are user=analytics, password=Tr33house!. "
            + _FILLER * 2
            + "\nWhat host and port is the database on?"
        ),
        "check_substr": ["10.0.3.47", "5433"],
        "gap_description": "~100 tokens of filler",
    },
    {
        "name": "Variable recall (medium gap)",
        "prompt": (
            "Project codename is KINGFISHER. Release date is March 17. "
            "The lead engineer is Dr. Yuki Tanaka. Budget is $2.4M. "
            + _FILLER * 8
            + "\nWhat is the project codename and who is the lead engineer?"
        ),
        "check_substr": ["KINGFISHER", "Tanaka"],
        "gap_description": "~400 tokens of filler",
    },
    {
        "name": "Variable recall (long gap)",
        "prompt": (
            "The three target metrics are: latency under 50ms, "
            "throughput above 10K QPS, and error rate below 0.1%. "
            + _FILLER * 16
            + "\nList the three target metrics mentioned at the start."
        ),
        "check_substr": ["50", "10"],  # 50ms and 10K
        "gap_description": "~800 tokens of filler",
    },
    {
        "name": "Multi-fact cross-reference",
        "prompt": (
            "Server A runs PostgreSQL on port 5432. "
            "Server B runs Redis on port 6379. "
            "Server C runs Kafka on port 9092. "
            + _FILLER * 12
            + "\nWhich server runs Redis and on what port?"
        ),
        "check_substr": ["B", "6379"],
        "gap_description": "~600 tokens of filler",
    },
]


def run_state_retention_test(
    model,
    tokenizer,
    *,
    max_tokens: int,
    temp: float,
    kv_cache_type: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Measure fact retrieval accuracy across growing context gaps."""
    results = []

    for task in STATE_RETENTION_TASKS:
        print(f"\n  [State Retention] {task['name']}")

        response, latency = generate_response(
            model,
            tokenizer,
            task["prompt"],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temp=temp,
            kv_cache_type=kv_cache_type,
        )
        cleaned = strip_thinking(response)
        cleaned_lower = cleaned.lower()

        found = []
        missing = []
        for substr in task["check_substr"]:
            if substr.lower() in cleaned_lower:
                found.append(substr)
            else:
                missing.append(substr)

        passed = len(missing) == 0

        # Estimate prompt token count
        prompt_tokens = len(tokenizer.encode(task["prompt"]))

        entry = {
            "name": task["name"],
            "gap_description": task["gap_description"],
            "prompt_tokens": prompt_tokens,
            "passed": passed,
            "found_keywords": found,
            "missing_keywords": missing,
            "latency_s": round(latency, 2),
            "response_len_words": len(cleaned.split()),
            "repetition": detect_repetition(cleaned),
        }
        results.append(entry)
        status = "PASS" if passed else f"FAIL (missing: {missing})"
        print(f"    {status}  ({prompt_tokens} prompt tokens, {latency:.1f}s)")

    n_passed = sum(1 for r in results if r["passed"])
    summary = {
        "pass_rate": round(n_passed / max(len(results), 1), 4),
        "n_passed": n_passed,
        "n_total": len(results),
    }
    print(f"\n  Summary: {n_passed}/{len(results)} passed")

    return {"test": "state_retention", "tasks": results, "summary": summary}


# ─── Test 2: PPL at Depth ─────────────────────────────────────────────────

# A long passage for perplexity measurement.  We compute PPL on successive
# windows to see how compression error grows with depth.

_EVAL_PASSAGE = (
    "The architecture of modern large language models has evolved "
    "significantly since the introduction of the transformer in 2017. "
    "The original transformer used an encoder-decoder architecture for "
    "machine translation, but subsequent work showed that decoder-only "
    "models could achieve strong performance on a wide range of tasks. "
    "GPT-2 demonstrated that scaling up decoder-only transformers and "
    "training them on diverse internet text produced models capable of "
    "coherent long-form generation. GPT-3 pushed this further to 175 "
    "billion parameters, showing that few-shot prompting could replace "
    "task-specific fine-tuning for many applications. The mixture-of-"
    "experts approach, pioneered in the Switch Transformer and later "
    "refined in models like GShard and ST-MoE, offered a way to scale "
    "parameter counts without proportionally increasing computation. "
    "By routing each token to only a subset of expert networks, MoE "
    "models achieve the representational capacity of a much larger "
    "dense model while maintaining the inference cost of a smaller one. "
    "This sparsity property is central to making trillion-parameter "
    "models practical on consumer hardware. The key challenge is that "
    "while only a few experts are active per token, all expert weights "
    "must be accessible — either resident in memory or loadable from "
    "storage with minimal latency. This creates an offloading problem "
    "that interacts with KV cache management, attention computation, "
    "and memory allocation in complex ways. Modern solutions combine "
    "weight quantisation to reduce the per-expert memory footprint, "
    "KV cache compression to limit the attention state memory, and "
    "predictive expert loading to reduce cache miss penalties. The "
    "interaction between these components is poorly understood and "
    "rarely measured end-to-end on real hardware constraints. "
    "Quantisation of weights has become standard practice, with "
    "INT4 and even INT3 formats achieving acceptable quality on "
    "many benchmarks. However, the impact of weight quantisation "
    "on routing decisions in MoE models is not well studied. "
    "Similarly, KV cache compression techniques like TurboQuant and "
    "IsoQuant can reduce the per-token memory footprint by 5x or "
    "more, but their interaction with expert offloading and routing "
    "has not been systematically evaluated. The goal of our work is "
    "to measure these interactions on real hardware, under real "
    "memory constraints, with real models. We do not claim theoretical "
    "optimality — we claim empirical closure of a full-stack system "
    "on consumer hardware. The contribution is the composition and "
    "the measurement, not any single technique. Each component has "
    "been published independently; what has not been shown is that "
    "they compose without quality collapse under tight memory budgets."
)

# Extend passage to ~5000 tokens by repeating with section markers,
# so the depth curve can measure up to 4K context offsets.
_EVAL_PASSAGE = " ".join(f"[Section {i + 1}] {_EVAL_PASSAGE}" for i in range(15))


def run_ppl_at_depth_test(
    model,
    tokenizer,
    *,
    max_tokens: int,  # unused but kept for interface consistency
    temp: float,
    kv_cache_type: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Measure real NLL/PPL and logit agreement on deeper prefixes."""
    _runtime()

    tokens = tokenizer.encode(_EVAL_PASSAGE)
    total_tokens = len(tokens)

    # Window sizes to measure at (or the max available)
    window_targets = [64, 128, 256, 512, 1024, 2048, 4096]
    windows = [w for w in window_targets if w < total_tokens]
    if not windows:
        windows = [total_tokens // 2]

    results = []
    for window_end in windows:
        prefix_tokens = tokens[:window_end]

        t0 = time.perf_counter()
        default_nll, default_ppl, default_logits = compute_nll_and_final_logits(
            model, prefix_tokens, kv_cache_type="default"
        )
        candidate_nll, candidate_ppl, candidate_logits = compute_nll_and_final_logits(
            model, prefix_tokens, kv_cache_type=kv_cache_type
        )
        latency = time.perf_counter() - t0
        cos_sim, top5 = compare_final_logits(default_logits, candidate_logits)

        entry = {
            "context_tokens": window_end,
            "default_nll": round(default_nll, 4),
            "default_ppl": round(default_ppl, 4),
            "candidate_nll": round(candidate_nll, 4),
            "candidate_ppl": round(candidate_ppl, 4),
            "delta_nll_vs_default": round(candidate_nll - default_nll, 4),
            "delta_ppl_vs_default": round(candidate_ppl - default_ppl, 4),
            "logit_cosine_vs_default": round(cos_sim, 6),
            "logit_top5_agreement_vs_default": round(top5, 4),
            "latency_s": round(latency, 2),
        }
        results.append(entry)
        print(
            f"    depth={window_end:>5} tokens: "
            f"ΔPPL={entry['delta_ppl_vs_default']:+.4f} "
            f"cos={entry['logit_cosine_vs_default']:.6f} "
            f"top5={entry['logit_top5_agreement_vs_default']:.2f} "
            f"({latency:.1f}s)"
        )

    avg_delta_ppl = sum(r["delta_ppl_vs_default"] for r in results) / max(
        len(results), 1
    )
    avg_cosine = sum(r["logit_cosine_vs_default"] for r in results) / max(
        len(results), 1
    )
    avg_top5 = sum(r["logit_top5_agreement_vs_default"] for r in results) / max(
        len(results), 1
    )
    summary = {
        "avg_delta_ppl_vs_default": round(avg_delta_ppl, 4),
        "avg_logit_cosine_vs_default": round(avg_cosine, 6),
        "avg_logit_top5_agreement_vs_default": round(avg_top5, 4),
        "n_total": len(results),
        "total_passage_tokens": total_tokens,
        "kv_cache_type": kv_cache_type,
    }
    print(
        f"\n  Summary: avg ΔPPL={summary['avg_delta_ppl_vs_default']:+.4f}  "
        f"avg cosine={summary['avg_logit_cosine_vs_default']:.6f}  "
        f"avg top5={summary['avg_logit_top5_agreement_vs_default']:.2f}"
    )

    return {"test": "ppl_at_depth", "windows": results, "summary": summary}


# ─── Test 3: Consistency Under Replay ─────────────────────────────────────

REPLAY_PROMPTS = [
    {
        "name": "Code generation",
        "text": "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes.",
    },
    {
        "name": "Factual recall",
        "text": "Explain how a transformer's self-attention mechanism works. Be concise.",
    },
    {
        "name": "Structured output",
        "text": "List the planets of the solar system in order from the sun, with one fact about each.",
    },
]


def run_consistency_test(
    model,
    tokenizer,
    *,
    max_tokens: int,
    temp: float,
    kv_cache_type: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Compare default-KV output against the candidate KV backend."""
    mx, _, _, _ = _runtime()

    results = []
    for prompt_data in REPLAY_PROMPTS:
        print(f"\n  [Consistency] {prompt_data['name']}")

        reference_kv = "default"
        candidate_kv = kv_cache_type

        # Reference run (default KV)
        mx.random.seed(42)
        response_a, latency_a = generate_response(
            model,
            tokenizer,
            prompt_data["text"],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temp=temp,
            kv_cache_type=reference_kv,
        )
        cleaned_a = strip_thinking(response_a)

        # Candidate run (selected KV backend)
        mx.random.seed(42)
        response_b, latency_b = generate_response(
            model,
            tokenizer,
            prompt_data["text"],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temp=temp,
            kv_cache_type=candidate_kv,
        )
        cleaned_b = strip_thinking(response_b)

        # Compare
        identical = cleaned_a == cleaned_b

        # Word-level overlap for partial divergence measurement
        words_a = set(cleaned_a.lower().split())
        words_b = set(cleaned_b.lower().split())
        if words_a or words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 1.0

        entry = {
            "name": prompt_data["name"],
            "reference_kv_cache_type": reference_kv,
            "candidate_kv_cache_type": candidate_kv,
            "identical": identical,
            "jaccard_similarity": round(jaccard, 4),
            "reference_len_words": len(cleaned_a.split()),
            "candidate_len_words": len(cleaned_b.split()),
            "reference_latency_s": round(latency_a, 2),
            "candidate_latency_s": round(latency_b, 2),
        }
        results.append(entry)
        status = "IDENTICAL" if identical else f"DIVERGED (jaccard={jaccard:.3f})"
        print(f"    {status}")

    n_identical = sum(1 for r in results if r["identical"])
    avg_jaccard = sum(r["jaccard_similarity"] for r in results) / max(len(results), 1)
    summary = {
        "identical_rate": round(n_identical / max(len(results), 1), 4),
        "avg_jaccard": round(avg_jaccard, 4),
        "n_identical": n_identical,
        "n_total": len(results),
    }
    print(
        f"\n  Summary: {n_identical}/{len(results)} identical, "
        f"avg jaccard={avg_jaccard:.3f}"
    )

    return {"test": "consistency_replay", "prompts": results, "summary": summary}


# ─── Main ──────────────────────────────────────────────────────────────────

TEST_RUNNERS = {
    "state_retention": run_state_retention_test,
    "ppl_at_depth": run_ppl_at_depth_test,
    "consistency": run_consistency_test,
}


def main():
    parser = argparse.ArgumentParser(
        description="Long-context KV cache fidelity evaluator"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=list(TEST_RUNNERS.keys()) + ["all"],
    )
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--kv-cache-type", type=str, default="default")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--use-predictor", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    mx, _, load, _ = _runtime()
    mx.random.seed(args.seed)

    print(
        f"Loading model: {args.model}\n"
        f"  expert_offload={args.expert_offload}  "
        f"kv_cache_type={args.kv_cache_type}  "
        f"max_resident={args.max_resident_experts}"
    )
    model, tokenizer = load(
        args.model,
        model_config=build_model_config(
            args.expert_offload,
            args.max_resident_experts,
            use_predictor=args.use_predictor,
        ),
    )
    provenance = collect_provenance(args.model, args)

    tests_to_run = list(TEST_RUNNERS.keys()) if args.test == "all" else [args.test]

    all_results: dict[str, Any] = {
        "version": 1,
        "model_path": args.model,
        "expert_offload": args.expert_offload,
        "max_resident_experts": args.max_resident_experts,
        "kv_cache_type": args.kv_cache_type,
        "use_predictor": args.use_predictor,
        "seed": args.seed,
        "temp": args.temp,
        "max_tokens": args.max_tokens,
        "tests": {},
    }

    gen_kwargs = dict(
        max_tokens=args.max_tokens,
        temp=args.temp,
        kv_cache_type=args.kv_cache_type,
        system_prompt=args.system_prompt,
    )

    for test_name in tests_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        runner = TEST_RUNNERS[test_name]
        result = runner(model, tokenizer, **gen_kwargs)
        all_results["tests"][test_name] = result

    all_results["system_at_end"] = {
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }

    print(f"\n{'=' * 60}")
    print("All tests complete.")

    attach_provenance(all_results, provenance)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Artifact: {out}")


if __name__ == "__main__":
    main()
