"""2-hour stability soak for Gemma4 with expert offload.

Measures tok/s, peak memory, RSS, and drift over 2 hours.
Writes intermediate progress and final artifact to results/soak/.
"""

import json
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance
from mlx_lm import generate, load

MODEL = "gemma-4-26b-a4b-it-4bit"
DURATION_S = 7200
OUT_DIR = "results/soak"

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_CONFIG = {"expert_offload": True, "max_resident_experts": 128}
model, tokenizer = load(MODEL, model_config=MODEL_CONFIG)
provenance = collect_provenance(
    MODEL,
    {
        "model": MODEL,
        "duration_s": DURATION_S,
        "kv_cache_type": "default",
        "max_tokens": 500,
        **MODEL_CONFIG,
    },
)

prompts = [
    "Write a comprehensive Python module implementing a thread-safe LRU cache with TTL support.",
    "Explain the theory of relativity in detail with mathematical examples.",
    "Implement a complete binary search tree in Python with insert, delete, and traversal methods.",
    "Describe the process of photosynthesis at the molecular level.",
    "Write a REST API server in Python using FastAPI with authentication and rate limiting.",
]

start = time.time()
iteration = 0
measurements = []

while time.time() - start < DURATION_S:
    prompt = prompts[iteration % len(prompts)]
    t0 = time.perf_counter()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    response = generate(model, tokenizer, ids, max_tokens=500, verbose=False)
    elapsed = time.perf_counter() - t0
    peak_mb = mx.get_peak_memory() / (1024**2)
    n_tokens = len(tokenizer.encode(response))
    tok_per_s = n_tokens / elapsed if elapsed > 0 else 0
    try:
        rss = (
            int(
                subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            / 1024
        )
    except Exception:
        rss = 0
    m = {
        "iteration": iteration,
        "elapsed_total_s": round(time.time() - start, 1),
        "generation_s": round(elapsed, 2),
        "tokens": n_tokens,
        "tok_per_s": round(tok_per_s, 1),
        "peak_memory_mb": round(peak_mb, 1),
        "process_rss_mb": round(rss, 1),
    }
    measurements.append(m)
    print(
        f"[{m['elapsed_total_s']:.0f}s] iter={iteration} tok/s={tok_per_s:.1f} "
        f"peak={peak_mb:.0f}MB rss={rss:.0f}MB",
        flush=True,
    )
    iteration += 1
    if iteration % 5 == 0:
        with open(f"{OUT_DIR}/gemma4_2h_soak_progress.json", "w") as f:
            json.dump(
                {"model": MODEL, "measurements": measurements, "provenance": provenance},
                f,
                indent=2,
            )

tok_rates = [m["tok_per_s"] for m in measurements]
artifact = {
    "model": MODEL,
    "duration_s": round(time.time() - start, 1),
    "iterations": len(measurements),
    "tok_per_s_p50": round(float(np.percentile(tok_rates, 50)), 1),
    "tok_per_s_p95": round(float(np.percentile(tok_rates, 95)), 1),
    "tok_per_s_p99": round(float(np.percentile(tok_rates, 99)), 1),
    "tok_per_s_min": round(float(min(tok_rates)), 1),
    "tok_per_s_max": round(float(max(tok_rates)), 1),
    "peak_memory_mb": round(max(m["peak_memory_mb"] for m in measurements), 1),
    "initial_rss_mb": round(measurements[0]["process_rss_mb"], 1),
    "final_rss_mb": round(measurements[-1]["process_rss_mb"], 1),
    "rss_drift_ratio": round(
        measurements[-1]["process_rss_mb"] / max(measurements[0]["process_rss_mb"], 1),
        2,
    ),
    "measurements": measurements,
}
attach_provenance(artifact, provenance)
with open(f"{OUT_DIR}/gemma4_2h_soak_final.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(
    f"\nSoak complete. {len(measurements)} iterations in {artifact['duration_s']:.0f}s"
)
