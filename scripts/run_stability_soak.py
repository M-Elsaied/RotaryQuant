"""Reusable stability soak runner.

Measures tok/s, peak memory, RSS, and drift over a configurable duration.

Usage:
    python scripts/run_stability_soak.py --model <path> --duration-mins 120 \\
        --expert-offload --max-resident-experts 128 \\
        --output-dir results/soak --memory-limit-mb 12800
"""

import argparse
import json
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

import _kv_paths
from mlx_lm import generate, load


def _process_rss_mb():
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return round(int(out) / 1024, 1)
    except Exception:
        return 0


PROMPTS = [
    "Write a comprehensive Python module implementing a thread-safe LRU cache with TTL support.",
    "Explain the theory of relativity in detail with mathematical examples.",
    "Implement a complete binary search tree in Python with insert, delete, and traversal methods.",
    "Describe the process of photosynthesis at the molecular level.",
    "Write a REST API server in Python using FastAPI with authentication and rate limiting.",
]


def main():
    parser = argparse.ArgumentParser(description="Stability soak runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--duration-mins", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=128)
    parser.add_argument("--kv-cache-type", type=str, default="default")
    parser.add_argument("--memory-limit-mb", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="results/soak")
    _kv_paths.add_argument(parser)
    args = parser.parse_args()
    _kv_paths.reset()

    os.makedirs(args.output_dir, exist_ok=True)
    duration_s = args.duration_mins * 60
    model_tag = os.path.basename(args.model).replace("/", "_")

    if args.memory_limit_mb:
        mx.set_memory_limit(int(args.memory_limit_mb * 1024 * 1024))
        print(f"Memory limit: {args.memory_limit_mb:.0f} MB")

    mx.reset_peak_memory()

    model_config = {}
    if args.expert_offload:
        model_config["expert_offload"] = True
        model_config["max_resident_experts"] = args.max_resident_experts

    model, tokenizer = load(args.model, model_config=model_config)
    provenance = collect_provenance(args.model, args)

    start = time.time()
    iteration = 0
    measurements = []

    while time.time() - start < duration_s:
        prompt = PROMPTS[iteration % len(PROMPTS)]
        t0 = time.perf_counter()
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = generate(
            model,
            tokenizer,
            ids,
            max_tokens=args.max_tokens,
            verbose=False,
            kv_cache_type=args.kv_cache_type,
        )
        elapsed = time.perf_counter() - t0
        peak_mb = mx.get_peak_memory() / (1024**2)
        n_tokens = len(tokenizer.encode(response))
        tok_per_s = n_tokens / elapsed if elapsed > 0 else 0
        rss = _process_rss_mb()

        m = {
            "iteration": iteration,
            "elapsed_total_s": round(time.time() - start, 1),
            "generation_s": round(elapsed, 2),
            "tokens": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
            "peak_memory_mb": round(peak_mb, 1),
            "process_rss_mb": rss,
        }
        measurements.append(m)
        print(
            f"[{m['elapsed_total_s']:.0f}s] iter={iteration} tok/s={tok_per_s:.1f} "
            f"peak={peak_mb:.0f}MB rss={rss:.0f}MB",
            flush=True,
        )
        iteration += 1
        if iteration % 5 == 0:
            with open(f"{args.output_dir}/{model_tag}_soak_progress.json", "w") as f:
                json.dump(
                    {
                        "model": args.model,
                        "measurements": measurements,
                        "provenance": provenance,
                    },
                    f,
                    indent=2,
                )

    tok_rates = [m["tok_per_s"] for m in measurements]
    artifact = {
        "model": args.model,
        "duration_s": round(time.time() - start, 1),
        "duration_mins": args.duration_mins,
        "iterations": len(measurements),
        "kv_cache_type": args.kv_cache_type,
        "expert_offload": args.expert_offload,
        "max_resident_experts": args.max_resident_experts,
        "memory_limit_mb": args.memory_limit_mb,
        "tok_per_s_p50": round(float(np.percentile(tok_rates, 50)), 1),
        "tok_per_s_p95": round(float(np.percentile(tok_rates, 95)), 1),
        "tok_per_s_p99": round(float(np.percentile(tok_rates, 99)), 1),
        "tok_per_s_min": round(float(min(tok_rates)), 1),
        "tok_per_s_max": round(float(max(tok_rates)), 1),
        "p99_div_p50": round(
            float(np.percentile(tok_rates, 99))
            / max(float(np.percentile(tok_rates, 50)), 0.1),
            2,
        ),
        "peak_memory_mb": round(max(m["peak_memory_mb"] for m in measurements), 1),
        "initial_rss_mb": round(measurements[0]["process_rss_mb"], 1),
        "final_rss_mb": round(measurements[-1]["process_rss_mb"], 1),
        "rss_drift_ratio": round(
            measurements[-1]["process_rss_mb"]
            / max(measurements[0]["process_rss_mb"], 1),
            2,
        ),
        "measurements": measurements,
    }

    attach_provenance(artifact, provenance)
    _kv_paths.attach(artifact, args.require_fused)
    out_path = f"{args.output_dir}/{model_tag}_soak_final.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(
        f"\nSoak complete. {len(measurements)} iterations in {artifact['duration_s']:.0f}s"
    )
    print(
        f"tok/s P50={artifact['tok_per_s_p50']} P99={artifact['tok_per_s_p99']} P99/P50={artifact['p99_div_p50']}"
    )
    print(f"RSS drift: {artifact['rss_drift_ratio']}x")
    print(f"Wrote {out_path}")
    _kv_paths.enforce(args.require_fused)


if __name__ == "__main__":
    main()
