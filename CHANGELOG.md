# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Fused write-path compression** (`mlx_lm/models/fused_kv_compress.py`):
  Metal-backed normalize + FWHT + SO(4) + quantize + pack in a single
  dispatch. Opt-in via `ISOQUANT_FUSED_ENCODE=1`; **graduated to default
  on 2026-05-05** after §3.4 paired-ablation evidence.
- **Pre-allocated cache mode** (`ISOQUANT_CACHE_MODE=prealloc`): O(1)
  pointer-bump append replaces O(T) `mx.concatenate`. **Graduated to
  default on 2026-05-05.**
- **NPT=8 fused attention path** (`fused_kv_decode_npt8.py`,
  `fused_kv_decode_npt8_tiled.py`): single-pass + T-tiled fused attention
  for `head_dim=256` (Gemma 4, Kimi MLA `kv_lora_rank` after split).
  Opt-in via `ISOQUANT_USE_NPT8_FUSED=1` (default OFF; on for sites that
  enable it).
- **NPT=16 fused attention path** (`fused_kv_decode_npt16.py`,
  `tests/test_fused_npt16.py`, `tests/conftest_npt16.py`): single-pass
  fused attention for `head_dim=512` (Kimi MLA `kv_lora_rank=512` latent).
  Opt-in via `ISOQUANT_USE_NPT16_FUSED=1` (default OFF). Synthetic tests
  pass; real-weight MLA logit-parity gate not run from this repo.
- §3.4 evidence: paired-repeat ablation (`profile_ablation.py`) and
  Gemma4 default-suite quality gate.
- Lane C residency-sweep tooling
  (`scripts/sweep_kimi_default_cache_residency.py`,
  `scripts/ab_kimi_layered_stack.py`) for measuring Kimi K2.x default-MLA
  decode throughput vs `max_resident_experts`.
- New IsoQuant tests: `test_fused_kv_compress.py`,
  `test_iso_incremental_pack.py`, `test_fused_npt8.py`,
  `test_fused_npt8_tiled.py` + shared `conftest_npt8.py`.
- Identity guide (RotaryQuant vs IsoQuant vs TurboQuant) in README
- MLA/DKV sub-block split constraint documentation (hard blocker for Kimi path)
- AttnRes predictor status and decision rationale
- Bootstrap install script (`scripts/bootstrap.sh`)
- 5-minute smoke test section in README
- Deployment & stability section with risk gates
- CHANGELOG.md
- New tests: `tests/test_kv_scoring.py` (scoring bookkeeping and which cache
  paths the scored positions read, on a tiny random model) and
  `tests/test_compressed_vs_fp16_attention.py` (3-bit attention against FP16
  attention on the same inputs, with documented error bounds).

### Removed
- The single-dispatch `fully_fused_attention` kernel (`fused_kv_decode_kernels.py`),
  its dispatch branch in `IsoQuantKVCache._fused_attention_metal`, its two unit
  tests, the `fused_attention_metal` benchmark variant and `scripts/sweep_threshold.py`.
  The dispatcher never selected it (`_SINGLE_KERNEL_T_THRESHOLD = 0`) and its
  reference test failed with a uniform 0.174x scale error. The decode path is
  unchanged: `fused_qk_dot`, softmax, `fused_value_accum`, inverse rotation.

### Changed
- **KV fidelity scoring goes through the decode path** (`scripts/_kv_scoring.py`,
  used by `measure_kv_fidelity.py` and `long_context_kv_eval.py`): the window is
  prefilled, the caches are finalized, and the trailing `--score-tokens`
  positions are scored one token at a time, so compressed backends are scored on
  the compressed cache they read during generation. The earlier single forward
  pass scored IsoQuant on FP16 KV (it holds multi-token updates in FP16 until
  finalize) while TurboQuant was scored on its quantized cache, so the two were
  not compared under the same cache state; it remains available as
  `--scoring single_pass`. Results carry `schema_version: 4`, a `scoring` block,
  per-backend `compressed_path` verification, and the run exits 1 when a
  compressed backend did not read its compressed cache (`--allow-unverified`).
- `IsoQuantKVCache.fused_attention` accepts multi-token queries (dense attention
  over the dequantized cache, recorded as `unfused_dense`); a non-deferred cache
  can take its first write without a prior finalize; reconstructed K/V follow
  the model dtype instead of a fixed float16. `RotorQuantKVCache.defer_prefill`
  switches prefill deferral off.
- **Runtime defaults flipped (2026-05-05, §3.4 graduation)**:
  `ISOQUANT_CACHE_MODE` default `concat_append` → `prealloc`;
  `ISOQUANT_FUSED_ENCODE` default `0` → `1`. Every IsoQuant cache
  instance constructed without explicit env vars now runs in `combined`
  mode. Opt-out: set the env vars to the old values explicitly. Defensive
  `getattr(self, "_cache_mode", "concat_append")` fallbacks elsewhere in
  `mlx_isoquant.py` are intentionally unchanged — they are safety nets
  for missing `_cache_mode`, not active defaults.

### Caveats

> **Note on evidence location:** the JSON artifacts that back the §3.4 and
> Lane C findings below were produced in the project repository
> (TurboQuantNemo) and intentionally NOT included in this RotaryQuant
> commit. The summary numbers are reproduced here for context; reproducing
> them in this repo requires running `scripts/profile_ablation.py` and
> `scripts/sweep_kimi_default_cache_residency.py` against the appropriate
> models locally.

- **FUSED_ENCODE introduces measurable numerical drift on long responses.**
  Verified on Gemma 4-26B-A4B (head_dim=256, non-fallback NPT=8 path)
  under a wall-clock end-to-end protocol (greedy + seed=42, v2 default
  prompt suite = 5 prompts × max 200 tokens):
  - All 4 conditions (`baseline_iso`, `fused_encode`, `prealloc`,
    `combined`) PASS the harness 5/5.
  - `prealloc` alone is byte-identical to baseline output.
  - `FUSED_ENCODE=1` produces measurably different outputs for 4 of 5
    prompts (the fused Metal compress/pack kernel does normalise/FWHT/
    SO(4)/quantise/pack in a different float-op order; drift accumulates
    over the response). Different markdown formatting / different
    sentence content.
  - Drift does NOT break per-task harness criteria but DOES change what
    the model says. Applications requiring bit-reproducibility vs the
    pre-§3.4 defaults must opt out: `ISOQUANT_FUSED_ENCODE=0`.
  - `combined` is 23% faster end-to-end **on this default-suite,
    wall-clock measurement** (38.6 s vs 50.0 s baseline). This is a
    different metric / protocol from the alpha-release headline numbers
    (tok/s + memory + 2 h soak). The two should not be compared
    directly; the §3.4 number is a localized write-path improvement on
    one suite, not a re-measurement of the alpha headline.
- **Kimi K2.6 default-cache residency cliff at exactly 480 experts**
  (= 60 MoE layers × 8 top-k = per-step working set). `max_resident_experts < 480`
  forces 0% hit rate (cache cannot hold one full step). Above 480 the
  throughput plateau is shallow (0.484-0.568 tok/s, ~17% spread,
  std comparable). v2 mechanical winner is 720 but margin over 480 is
  6.8% and within step-time std. **Provisional guidance:** never use
  `max_resident_experts < 480`; use 480 (RAM-safe) or 720 (v2 mechanical
  winner). Crowning a single "best" residency requires paired-repeat
  measurement (deferred).

### Fixed
- Mermaid diagram parse errors (escaped parentheses in node labels)
- Symbol reference table rendering (`\mathfrak` replaced with plain notation)
- `scripts/profile_npt8_metal.py`: pinned `ISOQUANT_FUSED_ENCODE=0` at
  the script-baseline level + restored `=0` (not `pop`) after Phase B2
  cleanup, so multi-T-loop "IsoQuant unpatched" baselines do not silently
  inherit the new `FUSED_ENCODE=1` runtime default.

## [0.1.0-alpha.1] — 2026-04-14

Initial public research checkpoint.

### Added
- IsoQuant (WHT + SO(4)) KV cache compression pipeline
- Fused Metal decode pipeline (4 kernels: fused_qk_dot, softmax, fused_value_accum, metal_rotate_inverse)
- Expert offloading with LRU eviction and `ensure_loaded()`
- Mixed-precision weight quantisation (4-bit dense, 2-bit routed experts, Q8_0 shared)
- Deferred prefill with bulk compression
- llama.cpp integration as `GGML_TYPE_ISOQUANT3_0` with fused Metal shader
- Mojo kernel benchmarks (matmul, softmax, RoPE)
- CLI wrapper (`isoquant-mlx`) with serve, validate, bench, convert subcommands
- Quality gate script (12-prompt automated pass/fail)
- 2-hour soak test automation
- Yum Cha conceptual primer with AI-generated illustrations
- Full technical paper (FROM_ATTENTION_TO_CONSUMER_HARDWARE.md)
- NotebookLM podcast and video companion media

### Validated
- Gemma 4-26B: 12.85 tok/s, 5.4 GB, 12/12 quality, 2h soak pass
- Nemotron-H 120B: 14.85 tok/s, 17.2 GB, 12/12 quality, 2h soak pass
- KV fidelity: delta PPL +0.0000 (Gemma 4), +0.0009 (Qwen3), +0.0012 (Nemotron)
- llama.cpp fused kernel: -0.5% prompt, -3.2% gen vs turbo3 baseline
