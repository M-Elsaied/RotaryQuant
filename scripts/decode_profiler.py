#!/usr/bin/env python3
"""Per-component decode time attribution via mx.eval() timing fences.

Instruments model layers to measure time spent in attention, routed MoE
expert compute (including expert I/O), dense FFN, and everything else.

Uses direct model calls (not generate_step) so the profiler owns the step
boundary and per-token attribution is aligned. The fences force Metal
synchronisation and perturb timing — this is a profiler, not a benchmark.
Use benchmark_moe_offload.py for throughput numbers.

Output: JSON with aggregate statistics (mean/median/p95/sum) computed from
per-token measurements. Cold/warm splits via --warm-repeat.

Component classification:
  kv_attention_ms  — self-attention (QKV + output projection)
  routed_expert_ms — MoE routing + expert compute (SwitchGLU/SwitchMLP)
  dense_ffn_ms     — dense feed-forward (MLP without routing)
  other_ms         — linear attention (Qwen3Next), Mamba/SSM (Nemotron), unclassified
  uninstrumented_ms — derived: total_step_ms minus instrumented component sum
                      (embedding, layer norms, residual adds, lm_head)

Usage:
    python scripts/decode_profiler.py \\
        --model <path> --expert-offload --kv-cache-type isoquant \\
        --output-json results/profile/qwen3_decode_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

_RUNTIME = None


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import (
            finalize_deferred_kv_caches,
            make_prompt_cache,
        )

        _RUNTIME = (mx, load, make_prompt_cache, finalize_deferred_kv_caches)
    return _RUNTIME


def _process_rss_mb() -> float | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return round(int(out) / 1024, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Instrumentation: monkey-patch layer __call__ to time components
# ---------------------------------------------------------------------------


class ComponentTimer:
    """Collects per-token, per-layer timing for attention vs MoE vs dense FFN."""

    COMPONENTS = ("kv_attention_ms", "routed_expert_ms", "dense_ffn_ms", "other_ms")

    def __init__(self):
        self.current_token: dict[str, float] = {}
        self.tokens: list[dict[str, float]] = []
        self._originals: list[tuple[Any, str, Any]] = []

    def _reset_token(self):
        self.current_token = {k: 0.0 for k in self.COMPONENTS}

    def finish_token(self, total_step_ms: float | None = None):
        if self.current_token:
            record = self.current_token.copy()
            if total_step_ms is not None:
                record["total_step_ms"] = total_step_ms
                instrumented = sum(record.get(k, 0.0) for k in self.COMPONENTS)
                record["uninstrumented_ms"] = max(0.0, total_step_ms - instrumented)
            self.tokens.append(record)
        self._reset_token()

    def add(self, component: str, elapsed_ms: float):
        self.current_token[component] = (
            self.current_token.get(component, 0.0) + elapsed_ms
        )

    def instrument_model(self, model):
        """Find and wrap layer components with timing hooks."""
        backbone = _find_backbone(model)
        if backbone is None:
            print("Warning: could not find model backbone for instrumentation")
            return

        layers = getattr(backbone, "layers", [])
        if not layers:
            print("Warning: no layers found in backbone")
            return

        moe_count = 0
        dense_count = 0
        attn_count = 0
        other_count = 0

        for i, layer in enumerate(layers):
            classifications = self._wrap_layer(layer, i)
            if "routed_expert_ms" in classifications:
                moe_count += 1
            if "dense_ffn_ms" in classifications:
                dense_count += 1
            if "kv_attention_ms" in classifications:
                attn_count += 1
            if "other_ms" in classifications:
                other_count += 1

        parts = [
            f"{attn_count} attention",
            f"{moe_count} MoE",
            f"{dense_count} dense FFN",
        ]
        if other_count:
            parts.append(f"{other_count} other (linear_attn/SSM)")
        print(f"Instrumented {len(layers)} layers: {', '.join(parts)}")

    def _wrap_layer(self, layer, layer_idx: int) -> set[str]:
        """Wrap layer components. Returns set of component categories used."""
        classifications = set()

        # --- Nemotron-H: block_type determines component type ---
        if hasattr(layer, "mixer") and hasattr(layer, "block_type"):
            block_type = getattr(layer, "block_type", "")
            if block_type == "*":
                self._wrap_callable(layer, "mixer", "kv_attention_ms")
                classifications.add("kv_attention_ms")
            elif block_type == "E":
                self._wrap_callable(layer, "mixer", "routed_expert_ms")
                classifications.add("routed_expert_ms")
            elif block_type == "-":
                self._wrap_callable(layer, "mixer", "dense_ffn_ms")
                classifications.add("dense_ffn_ms")
            elif block_type == "M":
                self._wrap_callable(layer, "mixer", "other_ms")
                classifications.add("other_ms")
            return classifications

        # --- Qwen3/Qwen3Next/Qwen2/Gemma4: self_attn + mlp ---
        if hasattr(layer, "self_attn"):
            self._wrap_callable(layer, "self_attn", "kv_attention_ms")
            classifications.add("kv_attention_ms")

        # Qwen3Next: linear_attn (GatedDeltaNet) on is_linear layers
        if hasattr(layer, "linear_attn"):
            self._wrap_callable(layer, "linear_attn", "other_ms")
            classifications.add("other_ms")

        # Gemma4: enable_moe flag controls dual MoE + dense pathway
        if hasattr(layer, "enable_moe") and getattr(layer, "enable_moe", False):
            if hasattr(layer, "mlp"):
                self._wrap_callable(layer, "mlp", "dense_ffn_ms")
                classifications.add("dense_ffn_ms")
            if hasattr(layer, "router"):
                self._wrap_callable(layer, "router", "routed_expert_ms")
                classifications.add("routed_expert_ms")
            if hasattr(layer, "experts"):
                self._wrap_callable(layer, "experts", "routed_expert_ms")
                classifications.add("routed_expert_ms")
        elif hasattr(layer, "mlp"):
            # Qwen3/Qwen3Next: self.mlp is MoE (has switch_mlp/gate) or dense
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp") or hasattr(mlp, "gate"):
                self._wrap_callable(layer, "mlp", "routed_expert_ms")
                classifications.add("routed_expert_ms")
            else:
                self._wrap_callable(layer, "mlp", "dense_ffn_ms")
                classifications.add("dense_ffn_ms")

        return classifications

    def _wrap_callable(self, obj, attr_name: str, component: str):
        mx = _runtime()[0]
        original = getattr(obj, attr_name)
        timer = self
        self._originals.append((obj, attr_name, original))

        class TimedWrapper:
            """Wraps a module call with mx.eval() timing fences."""

            def __init__(self, inner):
                self._inner = inner
                for a in dir(inner):
                    if not a.startswith("_") and not callable(getattr(inner, a, None)):
                        try:
                            setattr(self, a, getattr(inner, a))
                        except Exception:
                            pass

            def __call__(self, *args, **kwargs):
                mx.eval()  # drain pending work
                t0 = time.perf_counter()
                result = self._inner(*args, **kwargs)
                if isinstance(result, tuple):
                    mx.eval(*[r for r in result if hasattr(r, "shape")])
                elif hasattr(result, "shape"):
                    mx.eval(result)
                t1 = time.perf_counter()
                timer.add(component, (t1 - t0) * 1000.0)
                return result

            def __getattr__(self, name):
                return getattr(self._inner, name)

        setattr(obj, attr_name, TimedWrapper(original))

    def restore(self):
        for obj, attr_name, original in self._originals:
            setattr(obj, attr_name, original)
        self._originals.clear()


def _find_backbone(model):
    """Find the backbone/model object that contains the layer list."""
    for attr in ["model", "backbone"]:
        inner = getattr(model, attr, None)
        if inner is not None:
            if hasattr(inner, "layers"):
                return inner
            for attr2 in ["model", "backbone"]:
                inner2 = getattr(inner, attr2, None)
                if inner2 is not None and hasattr(inner2, "layers"):
                    return inner2
    if hasattr(model, "layers"):
        return model
    return None


# ---------------------------------------------------------------------------
# Direct model call helpers (no generate_step — profiler owns step boundary)
# ---------------------------------------------------------------------------


def _prefill(model, prompt_ids: list[int], cache, expert_mgr, prefill_step_size: int):
    """Run prefill in chunks via direct model calls. Returns logits from
    the final chunk (containing the last prompt token)."""
    mx = _runtime()[0]
    finalize = _runtime()[3]
    prompt = mx.array(prompt_ids)

    if expert_mgr is not None:
        expert_mgr.set_phase("prefill")

    total = len(prompt)
    offset = 0
    logits = None

    while offset < total:
        chunk_size = min(prefill_step_size, total - offset)
        chunk = prompt[offset : offset + chunk_size]
        logits = model(chunk[None], cache=cache)
        mx.eval(logits)
        offset += chunk_size

    # Transition: finalize deferred KV caches and switch to decode phase
    finalize(cache)

    if expert_mgr is not None:
        expert_mgr.set_phase("decode")

    return logits


def _greedy_sample(logits) -> int:
    """Greedy argmax from logits[:, -1, :]."""
    mx = _runtime()[0]
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return token.item()


# ---------------------------------------------------------------------------
# Profiling loop
# ---------------------------------------------------------------------------


def run_profile(
    model_path: str,
    *,
    prefill_tokens: int,
    decode_tokens: int,
    seed: int,
    kv_cache_type: str,
    expert_offload: bool,
    max_resident_experts: int | None,
    warm_repeat: bool,
) -> dict[str, Any]:
    mx, load, make_prompt_cache, _finalize = _runtime()
    mx.random.seed(seed)

    model_config: dict[str, Any] = {}
    if expert_offload:
        model_config["expert_offload"] = True
        if max_resident_experts is not None:
            model_config["max_resident_experts"] = max_resident_experts

    print(f"Loading model: {model_path}")
    mx.reset_peak_memory()
    model, tokenizer = load(model_path, model_config=model_config)
    peak_after_load_mb = mx.get_peak_memory() / (1024 * 1024)
    print(f"Peak memory after load: {peak_after_load_mb:.1f} MB")

    # Build prompt
    unit = tokenizer.encode("hello ", add_special_tokens=False)
    if not unit:
        unit = [tokenizer.eos_token_id or 0]
    prompt_ids: list[int] = []
    while len(prompt_ids) < prefill_tokens:
        prompt_ids.extend(unit)
    prompt_ids = prompt_ids[:prefill_tokens]

    timer = ComponentTimer()
    timer.instrument_model(model)

    expert_mgr = getattr(model, "expert_offload_manager", None)

    runs = []
    labels = ["cold"]
    if warm_repeat:
        labels.append("warm")

    for run_label in labels:
        print(f"\n=== {run_label} run ===")
        if run_label == "warm":
            print("  (expert LRU cache warm from cold run; fresh KV cache)")
        mx.random.seed(seed)
        mx.reset_peak_memory()
        timer.tokens.clear()

        # Fresh KV cache per run
        cache = make_prompt_cache(model, kv_cache_type=kv_cache_type)

        # --- Prefill (not timed per-component; we only profile decode) ---
        t_run_start = time.perf_counter()
        prefill_logits = _prefill(
            model,
            prompt_ids,
            cache,
            expert_mgr,
            prefill_step_size=min(512, max(1, prefill_tokens)),
        )
        t_prefill_end = time.perf_counter()
        prefill_ms = (t_prefill_end - t_run_start) * 1000.0
        print(f"  Prefill: {prefill_ms:.1f} ms ({prefill_tokens} prompt tokens)")

        # First decode token sampled from prefill logits
        current_token = _greedy_sample(prefill_logits)

        # --- Decode loop: one token at a time, profiler owns the boundary ---
        t_decode_start = time.perf_counter()

        for n in range(decode_tokens):
            timer._reset_token()

            mx.eval()  # drain before step boundary
            t_step_start = time.perf_counter()

            # Direct model call — instrumented wrappers fire per component
            logits = model(mx.array([current_token])[None], cache=cache)
            mx.eval(logits)

            t_step_end = time.perf_counter()
            step_ms = (t_step_end - t_step_start) * 1000.0
            timer.finish_token(step_ms)

            # Sample next token
            current_token = _greedy_sample(logits)

            if (n + 1) % 10 == 0:
                print(f"  Token {n + 1}/{decode_tokens}", end="\r")

        t_decode_end = time.perf_counter()
        decode_wall_ms = (t_decode_end - t_decode_start) * 1000.0

        # --- Aggregate ---
        decode_records = timer.tokens
        if not decode_records:
            print("  No tokens generated!")
            continue

        def agg(records, key):
            vals = [r.get(key, 0.0) for r in records]
            if not vals:
                return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "sum_ms": 0.0}
            vals_sorted = sorted(vals)
            p95_idx = min(int(len(vals_sorted) * 0.95), len(vals_sorted) - 1)
            return {
                "mean_ms": round(statistics.mean(vals), 3),
                "median_ms": round(statistics.median(vals), 3),
                "p95_ms": round(vals_sorted[p95_idx], 3),
                "sum_ms": round(sum(vals), 3),
            }

        kv_agg = agg(decode_records, "kv_attention_ms")
        moe_agg = agg(decode_records, "routed_expert_ms")
        ffn_agg = agg(decode_records, "dense_ffn_ms")
        other_agg = agg(decode_records, "other_ms")
        uninst_agg = agg(decode_records, "uninstrumented_ms")
        total_step_agg = agg(decode_records, "total_step_ms")
        instrumented_sum = (
            kv_agg["sum_ms"]
            + moe_agg["sum_ms"]
            + ffn_agg["sum_ms"]
            + other_agg["sum_ms"]
        )

        # Reconcile: per-token total_step sums vs decode wall clock
        # The difference is Python loop overhead + _greedy_sample time
        step_sum = total_step_agg["sum_ms"]
        recon_err = (
            round(
                abs(step_sum - decode_wall_ms) / max(decode_wall_ms, 1) * 100,
                1,
            )
            if decode_wall_ms > 0
            else 0.0
        )

        def pct(val):
            return round(val / max(step_sum, 1) * 100, 1)

        run_result = {
            "label": run_label,
            "note": (
                "expert LRU cache warm from cold run; fresh KV cache"
                if run_label == "warm"
                else "cold start"
            ),
            "prefill_tokens": prefill_tokens,
            "prefill_ms": round(prefill_ms, 1),
            "decode_tokens": len(decode_records),
            "decode_wall_ms": round(decode_wall_ms, 1),
            "kv_attention_ms": kv_agg,
            "routed_expert_ms": moe_agg,
            "dense_ffn_ms": ffn_agg,
            "other_ms": other_agg,
            "uninstrumented_ms": uninst_agg,
            "total_step_ms": total_step_agg,
            "instrumented_sum_ms": round(instrumented_sum, 1),
            "reconciliation_error_pct": recon_err,
            "kv_attention_pct": pct(kv_agg["sum_ms"]),
            "routed_expert_pct": pct(moe_agg["sum_ms"]),
            "dense_ffn_pct": pct(ffn_agg["sum_ms"]),
            "other_pct": pct(other_agg["sum_ms"]),
            "uninstrumented_pct": pct(uninst_agg["sum_ms"]),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 1),
            "rss_mb": _process_rss_mb(),
        }

        print(f"\n  {run_label} summary ({len(decode_records)} decode tokens):")
        print(
            f"    kv_attention:     {kv_agg['mean_ms']:.2f} ms/tok "
            f"({run_result['kv_attention_pct']}%)"
        )
        print(
            f"    routed_expert:    {moe_agg['mean_ms']:.2f} ms/tok "
            f"({run_result['routed_expert_pct']}%)"
        )
        print(
            f"    dense_ffn:        {ffn_agg['mean_ms']:.2f} ms/tok "
            f"({run_result['dense_ffn_pct']}%)"
        )
        print(
            f"    other (ssm):      {other_agg['mean_ms']:.2f} ms/tok "
            f"({run_result['other_pct']}%)"
        )
        print(
            f"    uninstrumented:   {uninst_agg['mean_ms']:.2f} ms/tok "
            f"({run_result['uninstrumented_pct']}%)"
        )
        print(f"    reconciliation error: {recon_err}%")

        runs.append(run_result)

    timer.restore()

    return {
        "schema_version": 3,
        "model": model_path,
        "seed": seed,
        "kv_cache_type": kv_cache_type,
        "expert_offload": expert_offload,
        "max_resident_experts": max_resident_experts,
        "component_definitions": {
            "kv_attention_ms": "Self-attention: QKV projection + attention + output projection",
            "routed_expert_ms": "MoE routing: gate/router + expert compute + expert I/O (disk/LRU)",
            "dense_ffn_ms": "Dense feedforward MLP (non-routed layers)",
            "other_ms": "Linear attention (Qwen3Next GatedDeltaNet), Mamba/SSM (Nemotron), or unclassified",
            "uninstrumented_ms": "Derived: total_step_ms minus instrumented sum (embedding, layer norms, residual adds, lm_head)",
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-component decode profiler (mx.eval fence attribution)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-cache-type", type=str, default="default")
    parser.add_argument("--expert-offload", action="store_true")
    parser.add_argument("--max-resident-experts", type=int, default=None)
    parser.add_argument(
        "--warm-repeat", action="store_true", help="Run a second pass with warm caches"
    )
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    result = run_profile(
        args.model,
        prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens,
        seed=args.seed,
        kv_cache_type=args.kv_cache_type,
        expert_offload=args.expert_offload,
        max_resident_experts=args.max_resident_experts,
        warm_repeat=args.warm_repeat,
    )
    attach_provenance(result, collect_provenance(args.model, args))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
