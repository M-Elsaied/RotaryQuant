# Build phases — from implementation to proof

**Status:** 2026-04-13. Seven phases, strict artifact gates.
**Ordering:** MLX/Apple Silicon pathway first (Phases 1–4), then llama.cpp/Metal (5–6), then optional (7).

> **Rule:** A phase is done when every artifact in its "Done when" column exists at the listed path and passes its acceptance check. No exceptions — no "mostly done", no "passes on my machine." If the artifact doesn't exist, the phase is open.

---

## Phase 1 — Canonical KV fidelity / perplexity

**Goal:** Replace the qualitative 5-prompt harness with quantitative PPL at fixed context depths.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 1a. Qwen3 KV PPL | `python scripts/measure_kv_fidelity.py --model $QWEN3_MODEL --depths 512,2048 --seed 42 --output-json results/qwen3_kv_ppl_depth.json` | `results/qwen3_kv_ppl_depth.json` | Contains `delta_ppl_vs_default` for default/turboquant/isoquant at 512/2048 tokens. Same seed (42), same text. |
| 1b. Gemma4 KV PPL | `python scripts/measure_kv_fidelity.py --model $GEMMA4_MODEL --depths 512,2048 --seed 42 --output-json results/gemma4_kv_ppl_depth.json` | `results/gemma4_kv_ppl_depth.json` | Same schema as 1a. |
| 1c. Nemotron KV PPL | `python scripts/measure_kv_fidelity.py --model $NEMOTRON_MODEL --depths 512,2048 --seed 42 --output-json results/nemotron_kv_ppl_depth.json --expert-offload --max-resident-experts 48` | `results/nemotron_kv_ppl_depth.json` | Same schema as 1a. Expert offload required for 120B. |
| 1d. Long-context depth curve (Qwen3) | `python scripts/long_context_kv_eval.py --model $QWEN3_MODEL --kv-cache-type isoquant --test ppl_at_depth --output-json results/qwen3_kv_depth_curve.json` | `results/qwen3_kv_depth_curve.json` | PPL at 256/512/1K/2K/4K context offsets. Deferred-prefill transition visible in output. |
| 1e. Long-context depth curve (Gemma4) | `python scripts/long_context_kv_eval.py --model $GEMMA4_MODEL --kv-cache-type isoquant --test ppl_at_depth --output-json results/gemma4_kv_depth_curve.json` | `results/gemma4_kv_depth_curve.json` | Same schema as 1d. |

**Pre-requisites:** Working MLX models for Qwen3-30B-A3B-4bit, Gemma 4-26B-A4B-4bit, Nemotron-H 120B.

**Script notes:**
- `measure_kv_fidelity.py` now supports `--depths` multi-depth mode and emits `delta_ppl_vs_default` per depth. Use `--seed 42` for canonical Phase 1 runs.
- `measure_kv_fidelity.py` scores the trailing `--score-tokens` positions (default 128) through the decode path after prefilling and finalizing the prefix; `schema_version: 4` adds a `scoring` block and per-backend `compressed_path` verification. Earlier artifacts (schema 2/3) were produced by a single forward pass in which IsoQuant scored FP16 KV (deferred prefill) while TurboQuant scored its quantized cache.
- `long_context_kv_eval.py` has `ppl_at_depth` test with offsets at 256/512/1K/2K/4K. Verify the script emits `delta_ppl_vs_default` per depth or add that field.
- The harness exists, but **Phase 1 is still open until the JSON artifacts are pinned**. Do not describe the repo as already having canonical perplexity coverage just because the scripts exist.

**Answers:** Section 10b.5 "Perplexity under compression."

---

## Phase 2 — End-to-end decode profiler

**Goal:** Answer "is KV attention material enough to justify further Metal work?" with a decode-step time breakdown.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 2a. Qwen3 decode breakdown | `python scripts/decode_profiler.py --model $QWEN3_MODEL --expert-offload --kv-cache-type isoquant --warm-repeat --output-json results/profile/qwen3_decode_breakdown.json` | `results/profile/qwen3_decode_breakdown.json` | Contains decode-step aggregate attribution for `kv_attention_ms`, `routed_expert_ms`, `dense_ffn_ms`, and `other_ms`. `instrumented_sum_ms` reconciles to `decode_wall_ms` within 5%. Cold and warm runs present. |
| 2b. Gemma4 decode breakdown | `python scripts/decode_profiler.py --model $GEMMA4_MODEL --expert-offload --kv-cache-type isoquant --warm-repeat --output-json results/profile/gemma4_decode_breakdown.json` | `results/profile/gemma4_decode_breakdown.json` | Same schema as 2a. |
| 2c. Nemotron decode breakdown | `python scripts/decode_profiler.py --model $NEMOTRON_MODEL --expert-offload --max-resident-experts 48 --kv-cache-type isoquant --warm-repeat --output-json results/profile/nemotron_decode_breakdown.json` | `results/profile/nemotron_decode_breakdown.json` | Same schema as 2a. |

**Script notes:**
- `benchmark_moe_offload.py` stays as the clean benchmark (no instrumentation overhead). The profiler is a **separate script** (`scripts/decode_profiler.py`) that monkey-patches layer sub-modules with `TimedWrapper` classes inserting `mx.eval()` fences around attention and MoE/FFN blocks. The fences force Metal synchronisation — the profiler measures *where* time goes, the benchmark measures *how much* time is spent.
- `decode_profiler.py` uses direct model calls (not `generate_step`) so the profiler owns the step boundary and decode attribution is aligned. Detects model type by attribute presence (`self_attn`, `linear_attn`, `mixer`, `mlp`, `router`+`experts`) and classifies: `kv_attention_ms` (full attention), `routed_expert_ms` (MoE routing + expert compute), `dense_ffn_ms` (dense feed-forward), `other_ms` (linear attention / Mamba / SSM). Qwen3 MoE vs dense MLP detected via `switch_mlp`/`gate` attributes; Gemma4 dual pathway (`mlp` dense + `router`+`experts` MoE) classified separately; Qwen3Next `linear_attn` (GatedDeltaNet) classified as `other_ms`. Output is aggregate statistics (mean/median/p95/sum), not raw per-token records. Use `--warm-repeat` for cold vs. warm comparison.

**Answers:** Section 10b.5 "End-to-end decode profiling."

---

## Phase 3 — 16GB pathway proof (Qwen3 + Gemma4)

**Goal:** Prove the full stack fits and works on a 16GB-class machine.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 3a. Qwen3 quality | `python scripts/eval_quality_gate.py --model $QWEN3_MODEL --suite all --expert-offload --max-resident-experts 4096 --kv-cache-type isoquant --output-json results/qwen3_pathway_quality.json` | `results/qwen3_pathway_quality.json` | All checks pass (exit 0). No degenerate repetition. |
| 3b. Qwen3 benchmark | `python scripts/benchmark_moe_offload.py --model $QWEN3_MODEL --profile A --expert-offload --max-resident-experts 4096 --kv-cache-type isoquant --target-envelope-mb 12800 --json-output results/qwen3_pathway_benchmark.json` | `results/qwen3_pathway_benchmark.json` | `peak_memory_mb` ≤ 12800. Decode tok/s ≥ 5. Expert cache decode hit rate ≥ 90%. |
| 3c. Qwen3 2h soak | `python scripts/run_stability_soak.py --model $QWEN3_MODEL --duration-mins 120 --expert-offload --max-resident-experts 4096 --kv-cache-type isoquant --memory-limit-mb 12800 --output-dir results/soak` | `results/soak/qwen3_2h_soak_final.json` | P99/P50 < 3.0. RSS drift ratio < 1.5×. No OOM or crash. |
| 3d. Gemma4 quality | `python scripts/eval_quality_gate.py --model $GEMMA4_MODEL --suite all --expert-offload --max-resident-experts 2048 --kv-cache-type isoquant --output-json results/gemma4_pathway_quality.json` | `results/gemma4_pathway_quality.json` | All checks pass (exit 0). |
| 3e. Gemma4 benchmark | `python scripts/benchmark_moe_offload.py --model $GEMMA4_MODEL --profile A --expert-offload --max-resident-experts 2048 --kv-cache-type isoquant --target-envelope-mb 12800 --json-output results/gemma4_pathway_benchmark.json` | `results/gemma4_pathway_benchmark.json` | `peak_memory_mb` ≤ 12800. Decode tok/s ≥ 5. Expert cache decode hit rate ≥ 90%. |
| 3f. Gemma4 2h soak | `python scripts/run_stability_soak.py --model $GEMMA4_MODEL --duration-mins 120 --expert-offload --max-resident-experts 2048 --kv-cache-type isoquant --memory-limit-mb 12800 --output-dir results/soak` | `results/soak/gemma4_2h_soak_final.json` | P99/P50 < 3.0. RSS drift ratio < 1.5×. |
| 3g. Gemma4 full-stack doc | Create `docs/GEMMA4_FULL_STACK.md` following `docs/QWEN3_FULL_STACK.md` pattern | `docs/GEMMA4_FULL_STACK.md` | Documents model path, env vars, commands, expected artifacts. |
| 3h. Update checklist | Fill Qwen3 + Gemma4 rows in `docs/PATHWAY_PROVEN_CHECKLIST.md` with artifact paths | `docs/PATHWAY_PROVEN_CHECKLIST.md` rows populated | Quality JSON + Benchmark JSON columns filled with real paths. |

**Hardware requirement:** Must run on a real 16GB-class Apple Silicon machine — not extrapolated from 32GB/64GB.

**Memory budget:** Start with `--target-envelope-mb 12800` / `--memory-limit-mb 12800` (80% of 16GB), but this is a **starting envelope, not a hard ideology**. If decode throughput or hit rate is materially better at higher residency, it is acceptable to push toward **90% of RAM** (`14400 MB` on 16GB-class hardware) provided the run stays stable and avoids memory-pressure collapse.

**Performance floor:** Any configuration that produces **sub-1 tok/s decode**, or effectively thrashes the expert cache (for example 0% decode hit rate), is **not an acceptable pathway configuration** even if it fits the nominal RAM envelope. Fit without usable throughput is a failed run, not a passing artifact.

**Expert residency:** `--max-resident-experts` must be set high enough for meaningful cache hit rates (≥90% decode). Defaults are far too low for MoE models with 128 experts across 48 layers. Tuned values: Qwen3=4096 (9.5GB peak, 96% decode hit, 9.87 tok/s), Gemma4=2048 (6.3GB peak, 99% decode hit, 13.95 tok/s). Both fit within the 12.8GB starting envelope; if a higher-residency configuration improves throughput further without instability, prefer the faster setting up to the 90% ceiling.

---

## Phase 4 — 32GB Nemotron pathway proof

**Goal:** Prove Nemotron-H 120B fits and works on a 32GB-class machine.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 4a. Nemotron quality | `bash scripts/run_nemotron_pathway_full_stack.sh` (sets `NEMOTRON_MODEL`, uses Profile B) | `results/nemotron_pathway_quality.json` | All quality checks pass. Preflight validates 120B-class checkpoint. |
| 4b. Nemotron benchmark | (produced by 4a — same script emits both) | `results/nemotron_pathway_benchmark.json` | `peak_memory_mb` ≤ 25600 (80% of 32GB). |
| 4c. Nemotron 2h soak | `python scripts/run_stability_soak.py --model $NEMOTRON_MODEL --duration-mins 120 --expert-offload --max-resident-experts 48 --kv-cache-type isoquant --memory-limit-mb 25600 --output-dir results/soak` | `results/soak/nemotron_2h_soak_final.json` | P99/P50 < 3.0. RSS drift ratio < 1.5×. No OOM or crash. |
| 4d. Update checklist | Fill Nemotron row in `docs/PATHWAY_PROVEN_CHECKLIST.md` | `docs/PATHWAY_PROVEN_CHECKLIST.md` Nemotron row populated | Quality + Benchmark JSON columns filled. |

**Hardware requirement:** Must run on a real 32GB-class Apple Silicon machine.

**Script notes:** `run_nemotron_pathway_full_stack.sh` includes a preflight check that validates `model_type=nemotron_h`, layers ≥ 90, routed_experts ∈ {64, 128}, topk ≥ 6. This prevents accidentally claiming validation on a smaller checkpoint.

---

## Phase 5 — llama.cpp/Metal isoquant3 correctness completion

**Goal:** Make non-identity SO(4) work end-to-end on Metal. Remove the identity-only guard.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 5a. Rotation tensor init | Implement non-identity SO(4) block generation in `llama-kv-cache.cpp` (from seed or config file). Set `isoquant_rot_is_identity = false` for affected layers. | Code change in `_third_party/llama-cpp-turboquant/src/llama-kv-cache.cpp` | `init_isoquant_rotation_tensor()` produces non-identity blocks. Cached flag correctly set to `false`. |
| 5b. Graph-side Q forward SO(4) | Extend `llama-graph.cpp:2104` — compose WHT + SO(4) forward rotation for Q pre-rotation when cache type is ISOQUANT3_0. | Code change in `_third_party/llama-cpp-turboquant/src/llama-graph.cpp` | Q receives composed WHT + SO(4) rotation matching the write path. |
| 5c. Graph-side output inverse SO(4) | Extend `llama-graph.cpp:1843` — compose inverse SO(4) + inverse WHT for output post-rotation when cache type is ISOQUANT3_0. | Code change in `_third_party/llama-cpp-turboquant/src/llama-graph.cpp` | Output inverse matches write-path rotation exactly. |
| 5d. Remove identity guard | After 5a–5c are implemented, the `GGML_ASSERT(isoquant_rot_is_identity)` becomes a dead guard. Keep the flag but allow `false` to pass through. | Code change in `llama-kv-cache.cpp` | Non-identity SO(4) no longer asserts. |
| 5e. Build verification | `cd _third_party/llama-cpp-turboquant && cmake --build build --target llama-server` | Clean build (0 errors) | All new code compiles on Metal. |
| 5f. Smoke test | `./bin/llama-server --model <GGUF> --cache-type-k isoquant3 --cache-type-v isoquant3 -ngl 99 -c 2048` + sample prompts | `results/llama_cpp_isoquant3_smoke.json` | Server starts, serves completions without crashes or asserts. Output coherence depends on model size and chat template — compare against turbo3 on the same model to isolate KV cache effects. |
| 5g. Correctness doc | Write up pass/fail details | `results/llama_cpp_isoquant3_correctness.md` | Documents what was tested, GGUF used, and any limitations. |

**Pre-requisites:** A GGUF model compatible with the llama.cpp fork (e.g., Qwen2.5 or similar small model for fast iteration).

---

## Phase 6 — llama.cpp/Metal comparative measurement

**Goal:** Answer "does isoquant3 beat turbo3?" with paired benchmarks on the same GGUF.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 6a. Turbo3 baseline | `./bin/llama-bench -m <GGUF> --cache-type-k turbo3 --cache-type-v turbo3 -c 2048 -n 128 -o json > results/llama_cpp_turbo3_bench.json` | `results/llama_cpp_turbo3_bench.json` | Prompt eval tok/s and generation tok/s recorded. |
| 6b. Isoquant3 benchmark | `./bin/llama-bench -m <GGUF> --cache-type-k isoquant3 --cache-type-v isoquant3 -c 2048 -n 128 -o json > results/llama_cpp_isoquant3_bench.json` | `results/llama_cpp_isoquant3_bench.json` | Same schema as 6a. Same GGUF, same context, same token count. |
| 6c. Comparative analysis | Diff 6a vs 6b: compute `delta_prompt_tps`, `delta_gen_tps`, `delta_peak_memory_mb` | `results/llama_cpp_isoquant3_vs_turbo3.json` | One of two outcomes pinned: (a) isoquant3 shows measurable benefit, or (b) clean negative result with explanation. |
| 6d. Profile writeup | Document findings | `results/llama_cpp_isoquant3_profile.md` | Explains whether SO(4) machinery is worthwhile at current stage. References exact GGUF, hardware, context length. |

**Note:** Phase 6 is only meaningful after Phase 5 delivers non-identity SO(4). Running this with identity-only rotations is benchmarking turbo3 with extra dispatch overhead — the result would be a meaningless negative.

---

## Phase 7 — Optional branches

**Gate:** Only after Phases 1–4 are green. These are net-new work, not pathway-critical.

| Step | Command | Artifact | Acceptance |
|------|---------|----------|------------|
| 7a. AttnRes predictor rehab | `python scripts/benchmark_moe_offload.py --model $GEMMA4_MODEL --profile A --expert-offload --kv-cache-type isoquant --use-predictor --split-decode-timing --json-output results/gemma4_predictor_ablation.json` | `results/gemma4_predictor_ablation.json` | Either (a) predictor shows net latency win (hit-rate benefit > overhead), or (b) documented negative result. |
| 7b. AttnRes on Qwen3 | Same as 7a with `$QWEN3_MODEL` | `results/qwen3_predictor_ablation.json` | Same acceptance as 7a. |
| 7c. QES Phase 0 | Implement QES perturbation loop (Section 11 of the paper). Run on one model. | `results/qes_phase0_report.md` | Documents: perturbation count, sigma, accuracy delta, hit-rate delta. Positive or negative result pinned. |

---

## Summary table

| Phase | What it proves | Key artifacts | Answers |
|-------|---------------|---------------|---------|
| 1 | KV compression doesn't destroy quality | `*_kv_ppl_depth.json`, `*_kv_depth_curve.json` | 10b.5 "Perplexity under compression" |
| 2 | Where decode time actually goes | `profile/*_decode_breakdown.json` | 10b.5 "End-to-end decode profiling" |
| 3 | 16GB pathway works on real hardware | `*_pathway_{quality,benchmark}.json`, `soak/*_2h_soak_final.json` | Pathway-proven gate (Qwen3, Gemma4) |
| 4 | 32GB pathway works on real hardware | `nemotron_pathway_*.json`, `soak/nemotron_2h_soak_final.json` | Pathway-proven gate (Nemotron) |
| 5 | Non-identity SO(4) works in llama.cpp | `llama_cpp_isoquant3_smoke.json`, `*_correctness.md` | llama.cpp track completeness |
| 6 | Whether SO(4) beats WHT-only in llama.cpp | `llama_cpp_isoquant3_vs_turbo3.json` | 10b.5 "Does it matter end-to-end?" (llama.cpp) |
| 7 | Optional: predictor/QES value | `*_predictor_ablation.json`, `qes_phase0_report.md` | Go/no-go on optional components |

---

## Environment variables (set before running)

```bash
# Model paths (required — no defaults)
export QWEN3_MODEL=/path/to/Qwen3-30B-A3B-4bit
export GEMMA4_MODEL=/path/to/gemma-4-26b-a4b-it-4bit
export NEMOTRON_MODEL=/path/to/nemotron-120b-checkpoint  # must be 120B-class (Phase 4 preflight validates)

# Optional overrides
export TURBOQUANT_BITS=3                              # default 3
export MAX_RESIDENT_EXPERTS=48                        # Nemotron default
```

## Script modifications required

| Script | What to add | Phase |
|--------|------------|-------|
| `scripts/measure_kv_fidelity.py` | **Done.** `--depths` flag loops over multiple context depths; emits `delta_ppl_vs_default` per depth under `schema_version: 3`. | 1 |
| `scripts/decode_profiler.py` | **Done.** Direct model calls (not generate_step) for aligned decode-step attribution. Four instrumented buckets (`kv_attention_ms`, `routed_expert_ms`, `dense_ffn_ms`, `other_ms`) with `instrumented_sum_ms` reconciled against `decode_wall_ms`. Covers Qwen3Next `linear_attn` (→ other_ms). Output is aggregate stats, not raw per-token. Cold + warm via `--warm-repeat`. | 2 |
| `_third_party/llama-cpp-turboquant/src/llama-kv-cache.cpp` | Non-identity SO(4) block init, flag management | 5 |
| `_third_party/llama-cpp-turboquant/src/llama-graph.cpp` | Composed WHT+SO(4) for Q pre-rotation and output inverse | 5 |
