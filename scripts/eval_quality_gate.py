"""Quality gate for Nemotron-H MoE checkpoints.

Runs a fixed prompt suite with deterministic generation settings and
automated pass/fail criteria. Exit code 0 = all checks pass; 1 = at
least one failure.

Checks:
  - Minimum response length (not truncated / empty)
  - No degenerate repetition (the primary failure mode from pre-fix runs)
  - Expected answer present where applicable
  - Optional: malformed fenced code blocks (unbalanced ```)
  - JSON output and baseline comparison for release regression tracking

Set PYTHONHASHSEED=0 for maximum reproducibility across processes.

Fast local checks: use a **small** MLX model (e.g. 0.5B–1B), ``--quick``
(micro suite, ``max_tokens`` ≤ 48, 90s wall budget per prompt), and optional
``--max-prompt-wall-s`` to fail slow generations without killing the process.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from _provenance import attach as attach_provenance
from _provenance import collect as collect_provenance

_RUNTIME = None


def _run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _swap_used_mb() -> float | None:
    out = _run_text(["sysctl", "vm.swapusage"])
    if not out or "used =" not in out:
        return None
    try:
        value = out.split("used =", 1)[1].strip().split()[0]
        if value.endswith("M"):
            return round(float(value[:-1]), 2)
        if value.endswith("G"):
            return round(float(value[:-1]) * 1024.0, 2)
    except Exception:
        return None
    return None


def _process_rss_mb() -> float | None:
    out = _run_text(["ps", "-o", "rss=", "-p", str(os.getpid())])
    if not out:
        return None
    try:
        return round(float(out.strip()) / 1024.0, 2)
    except ValueError:
        return None


def _vm_pageouts() -> int | None:
    out = _run_text(["vm_stat"])
    if not out:
        return None
    for line in out.splitlines():
        if "Pages paged out" in line:
            try:
                return int(line.split(":")[1].strip().rstrip("."))
            except Exception:
                return None
    return None


def _system_snapshot() -> dict[str, Any]:
    return {
        "process_rss_mb": _process_rss_mb(),
        "swap_used_mb": _swap_used_mb(),
        "vm_pages_paged_out": _vm_pageouts(),
    }


def _delta(after: float | int | None, before: float | int | None):
    if after is None or before is None:
        return None
    return round(after - before, 2)


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        _RUNTIME = (mx, generate, load, make_sampler)
    return _RUNTIME


# V1 prompts: original harness (fence checking on, original prompt wording)
PROMPTS_V1_CODE_GEN = {
    "name": "Code Generation",
    "text": "Write a Python function to compute the nth Fibonacci number.",
    "min_tokens": 30,
    "expected_substr": ["def ", "fibonacci", "return"],
    "description": "Must produce a Python function definition",
}

# V2 prompts: fixed harness (fence check off for code, simpler prompt, lower min_tokens)
PROMPTS_V2_CODE_GEN = {
    "name": "Code Generation",
    "text": "Write a Python function to compute the nth Fibonacci number. Output only the function, no explanation.",
    "min_tokens": 15,
    "expected_substr": ["def ", "return"],
    "description": "Must produce a Python function definition",
    "check_fences": False,
    "skip_min_tokens": True,
}

COMMON_DEFAULT_PROMPTS = [
    {
        "name": "Instruction Following",
        "text": "Explain the theory of relativity to a 5 year old.",
        "min_tokens": 30,
        "expected_substr": [],
        "description": "Must produce a coherent multi-sentence explanation",
    },
    {
        "name": "Basic Reasoning",
        "text": "If I have 3 apples and eat 1, how many apples do I have left? Think step by step.",
        "min_tokens": 15,
        "expected_substr": ["2"],
        "description": "Must arrive at the answer 2",
    },
    {
        "name": "List Generation",
        "text": "List 5 programming languages and one strength of each.",
        "min_tokens": 30,
        "expected_substr": [],
        "description": "Must produce a structured list",
    },
    {
        "name": "Math",
        "text": "What is 17 * 24? Show your work.",
        "min_tokens": 10,
        "expected_substr": ["408"],
        "description": "Must compute 408",
    },
]

DEFAULT_PROMPTS_V1 = [PROMPTS_V1_CODE_GEN, *COMMON_DEFAULT_PROMPTS]
DEFAULT_PROMPTS_V2 = [PROMPTS_V2_CODE_GEN, *COMMON_DEFAULT_PROMPTS]

# Heavier coding-oriented tasks (use --suite coding or all).
COMMON_CODING_PROMPTS_PREFIX = [
    {
        "name": "Bugfix (off-by-one)",
        "text": (
            "Here is buggy Python:\n\n"
            "def slice_last_n(items, n):\n"
            "    return items[len(items) - n : len(items) - 1]\n\n"
            "Fix the bug so it returns the last n items. Output only the corrected function."
        ),
        "min_tokens": 25,
        "expected_substr": ["return", "items"],
        "description": "Must return a corrected function using the list tail",
        "skip_min_tokens": True,
    },
    {
        "name": "Traceback interpretation",
        "text": (
            "Explain what this Python error means and how to fix it:\n"
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        ),
        "min_tokens": 40,
        "expected_substr": ["str", "int"],
        "description": "Must mention type mismatch and conversion/casting",
        "check_fences": False,
    },
    {
        "name": "Multi-file style refactor",
        "text": (
            "You have module a.py with `def fetch(): ...` and b.py importing it. "
            "Describe concrete steps to rename `fetch` to `load_data` across both files "
            "and what to run to verify (tests or grep). Be specific."
        ),
        "min_tokens": 50,
        "expected_substr": ["rename", "import"],
        "description": "Must give actionable rename/refactor steps",
        "check_fences": False,
    },
]

CODING_PROMPT_V1_MINIMAL_TEST = {
    "name": "Write a minimal test",
    "text": (
        "Write a pytest test for a function `add(a,b)` that asserts add(2,3)==5. "
        "Output only the test code."
    ),
    "min_tokens": 15,
    "expected_substr": ["def ", "assert", "add"],
    "description": "Must output pytest-style test with assert",
    "skip_min_tokens": True,
}

CODING_PROMPT_V2_MINIMAL_TEST = {
    "name": "Write a minimal test",
    "text": (
        "Write a pytest test for a function `add(a,b)` that asserts add(2,3)==5. "
        "Show the complete test file including import."
    ),
    "min_tokens": 5,
    "expected_substr": ["def ", "assert", "add"],
    "description": "Must output pytest-style test with assert",
    "skip_min_tokens": True,
}

COMMON_CODING_PROMPTS_SUFFIX = [
    {
        "name": "Implement LRU cache get",
        "text": (
            "Implement a class method `get(self, key)` for an LRU cache with a fixed capacity. "
            "On miss return None; on hit move key to most-recent. "
            "Assume `self._order` and `self._data` exist. "
            "Output only the plain code, no markdown fences."
        ),
        "min_tokens": 5,
        "expected_substr": ["key", "return", "None"],
        "description": "Must implement method-body cache miss/hit logic",
        "skip_min_tokens": True,
        "check_fences": False,
    },
    {
        "name": "Race condition explanation",
        "text": (
            "Explain a classic race between two threads incrementing a shared counter without a lock, "
            "and fix it with one Python snippet using threading.Lock."
        ),
        "min_tokens": 45,
        "expected_substr": ["lock", "thread"],
        "description": "Must mention lock and threading",
    },
]

CODING_PROMPTS_V1 = [
    *COMMON_CODING_PROMPTS_PREFIX,
    CODING_PROMPT_V1_MINIMAL_TEST,
    *COMMON_CODING_PROMPTS_SUFFIX,
]

CODING_PROMPTS_V2 = [
    *COMMON_CODING_PROMPTS_PREFIX,
    CODING_PROMPT_V2_MINIMAL_TEST,
    *COMMON_CODING_PROMPTS_SUFFIX,
]

# Long-decode soak prompt — catches quality degradation at scale.
# Tiny suite for CI / smoke: few tokens, small models, sub-minute target per prompt.
MICRO_PROMPTS_V2 = [
    {
        "name": "Reasoning micro",
        "text": "Answer with a single character only: the digit after 8 when counting.",
        "min_tokens": 1,
        "expected_substr": ["9"],
        "description": "Must emit digit 9",
    },
    {
        "name": "Code micro",
        "text": "Output only the Python tokens def f(): pass with no other words.",
        "min_tokens": 4,
        "expected_substr": ["def", "pass"],
        "description": "Must contain def and pass",
        "check_fences": False,
    },
]

SOAK_PROMPTS = [
    {
        "name": "Long decode soak (1K+ tokens)",
        "text": (
            "Write a comprehensive Python module that implements a thread-safe LRU cache "
            "with the following features:\n"
            "1. Generic type support with TypeVar\n"
            "2. get(), put(), and delete() methods\n"
            "3. TTL (time-to-live) expiration per entry\n"
            "4. Thread safety with threading.Lock\n"
            "5. Statistics tracking (hits, misses, evictions)\n"
            "6. A complete pytest test suite at the bottom of the file\n\n"
            "Include docstrings, type hints, and at least 5 test functions."
        ),
        "min_tokens": 200,
        "expected_substr": ["class", "def get", "def put", "Lock", "def test_"],
        "description": "Must produce substantial, coherent code without repetition at 1K+ tokens",
        "check_fences": False,
        "max_tokens": 2048,
    },
]


PROMPT_SUITES = {
    "v1": {
        "default": DEFAULT_PROMPTS_V1,
        "coding": CODING_PROMPTS_V1,
        "all": DEFAULT_PROMPTS_V1 + CODING_PROMPTS_V1,
    },
    "v2": {
        "default": DEFAULT_PROMPTS_V2,
        "coding": CODING_PROMPTS_V2,
        "micro": MICRO_PROMPTS_V2,
        "soak": SOAK_PROMPTS,
        "all": DEFAULT_PROMPTS_V2 + CODING_PROMPTS_V2 + SOAK_PROMPTS,
    },
}


def prompt_suite(name: str, *, harness_version: str) -> list[dict[str, Any]]:
    version_suites = PROMPT_SUITES.get(harness_version)
    if version_suites is None:
        raise ValueError(f"Unknown harness version {harness_version!r}")

    n = name.lower().strip()
    prompts = version_suites.get(n)
    if prompts is None:
        available = ", ".join(sorted(version_suites.keys()))
        raise ValueError(
            f"Suite {name!r} is not available for harness {harness_version!r}. "
            f"Available suites: {available}"
        )
    return list(prompts)


def detect_malformed_fences(text: str) -> bool:
    """True if ``` fences look unbalanced (odd count)."""
    return text.count("```") % 2 != 0


def detect_repetition(text: str, window: int = 20, threshold: float = 0.4) -> bool:
    """Return True if the text contains degenerate repetition.

    Checks whether any sliding window of `window` words repeats more than
    `threshold` fraction of the total word count.
    """
    words = text.split()
    if len(words) < window * 2:
        return False
    for i in range(len(words) - window):
        pattern = " ".join(words[i : i + window])
        count = text.count(pattern)
        if count >= 3:
            return True
    sentences = re.split(r"[.!?\n]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 4:
        counts = Counter(sentences)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 3 and most_common_count / len(sentences) > threshold:
            return True
    return False


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output for evaluation."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"</think>\s*", "", cleaned)
    return cleaned.strip()


def build_model_config(
    expert_offload: bool,
    max_resident_experts: int | None,
    *,
    max_cached_shards: int | None = None,
    use_predictor: bool = False,
    use_dedekimi_observer: bool = False,
    task_expert_cliques: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config: dict[str, Any] = {}
    if expert_offload:
        model_config["expert_offload"] = True
        if max_resident_experts is not None:
            model_config["max_resident_experts"] = max_resident_experts
        if max_cached_shards is not None:
            model_config["max_cached_shards"] = max_cached_shards
    if use_predictor:
        model_config["use_predictor"] = True
    if use_dedekimi_observer:
        model_config["use_dedekimi_observer"] = True
    if task_expert_cliques:
        model_config["task_expert_cliques"] = task_expert_cliques
    return model_config


def render_prompt(
    tokenizer,
    prompt_text: str,
    system_prompt: str | None,
    harness_version: str = "v2",
) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        # v2: disable thinking mode to avoid budget-consuming think chains
        # v1: leave thinking mode as the tokenizer default
        if harness_version == "v2":
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    print("Warning: chat_template not found, using raw text encoding.")
    return tokenizer.encode(prompt_text)


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    *,
    system_prompt: str | None,
    max_tokens: int,
    temp: float,
    kv_cache_type: str = "default",
    harness_version: str = "v2",
) -> tuple[str, float]:
    _, generate, _, make_sampler = _runtime()
    sampler = make_sampler(temp)
    prompt = render_prompt(
        tokenizer, prompt_text, system_prompt, harness_version=harness_version
    )
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


def evaluate_response(
    prompt_data: dict,
    raw_response: str,
    *,
    default_max_word_repeat_ratio: float = 0.35,
) -> tuple[bool, list[str]]:
    """Evaluate a response against quality criteria. Returns (passed, list of failure reasons)."""
    failures = []
    response = strip_thinking(raw_response)
    response_lower = response.lower()
    words = response.split()

    min_tokens = prompt_data.get("min_tokens", 15)
    if not prompt_data.get("skip_min_tokens", False) and len(words) < min_tokens:
        failures.append(f"Too short: {len(words)} words (min {min_tokens})")

    max_words = prompt_data.get("max_words")
    if max_words is not None and len(words) > max_words:
        failures.append(f"Too long: {len(words)} words (max {max_words})")

    for substr in prompt_data.get("expected_substr", []):
        if substr.lower() not in response_lower:
            failures.append(f"Missing expected substring: {substr!r}")

    if detect_repetition(raw_response):
        failures.append("Degenerate repetition detected")

    # Stricter: single-token dominance (common degenerate mode).
    if words:
        top = Counter(words).most_common(1)[0][1]
        max_ratio = prompt_data.get(
            "max_word_repeat_ratio", default_max_word_repeat_ratio
        )
        if top / len(words) > max_ratio:
            failures.append(
                f"Word repetition ratio too high: {top / len(words):.2f} (max {max_ratio})"
            )

    if prompt_data.get("check_fences", True) and "```" in raw_response:
        if detect_malformed_fences(raw_response):
            failures.append("Malformed fenced code blocks (unbalanced ```)")

    return len(failures) == 0, failures


def run_gate(
    *,
    model_path: str,
    expert_offload: bool,
    max_resident_experts: int | None,
    max_cached_shards: int | None = None,
    system_prompt: str | None,
    max_tokens: int,
    seed: int,
    temp: float,
    prompts: list[dict[str, Any]],
    suite_name: str,
    strict: bool,
    kv_cache_type: str = "default",
    use_predictor: bool = False,
    use_dedekimi_observer: bool = False,
    task_expert_cliques: dict[str, Any] | None = None,
    harness_version: str = "v2",
    memory_limit_mb: float | None = None,
    max_prompt_wall_s: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    mx, _, load, _ = _runtime()
    mx.random.seed(seed)

    # Apply memory cap if requested — validates quality under constrained memory
    old_mem_limit = None
    if memory_limit_mb is not None:
        limit_bytes = int(memory_limit_mb * 1024 * 1024)
        old_mem_limit = mx.set_memory_limit(limit_bytes)
        print(f"Memory limit set to {memory_limit_mb:.0f} MB")
    mx.reset_peak_memory()
    system_before_load = _system_snapshot()
    repeat_ratio = 0.22 if strict else 0.35
    print(
        f"Loading model from {model_path} "
        f"(expert_offload={expert_offload}, kv_cache_type={kv_cache_type}, "
        f"use_predictor={use_predictor}, seed={seed}, temp={temp})..."
    )
    model, tokenizer = load(
        model_path,
        model_config=build_model_config(
            expert_offload,
            max_resident_experts,
            max_cached_shards=max_cached_shards,
            use_predictor=use_predictor,
            use_dedekimi_observer=use_dedekimi_observer,
            task_expert_cliques=task_expert_cliques,
        ),
    )
    system_after_load = _system_snapshot()

    print(f"\n--- Quality Gate ({len(prompts)} prompts) ---\n")
    if system_prompt:
        print(f"System prompt: {system_prompt}\n")

    all_passed = True
    results = []

    for prompt_data in prompts:
        print(f"Task: {prompt_data['name']}")
        print(f"Prompt: {prompt_data['text']}")
        print(f"Criteria: {prompt_data['description']}")
        print("-" * 40)

        prompt_max_tokens = prompt_data.get("max_tokens", max_tokens)
        response, elapsed = generate_response(
            model,
            tokenizer,
            prompt_data["text"],
            system_prompt=system_prompt,
            max_tokens=prompt_max_tokens,
            temp=temp,
            kv_cache_type=kv_cache_type,
            harness_version=harness_version,
        )
        print(f"Response ({elapsed:.2f}s):\n{response}\n")

        response_words = len(response.split())
        # Heuristic: word count ≥ 90% of token cap suggests possible truncation.
        # This is NOT a reliable token count (words ≠ tokens, especially for code).
        # It can only flag "likely near cap" — not prove EOS vs cap termination.
        likely_near_cap = response_words >= prompt_max_tokens * 0.9

        if max_prompt_wall_s is not None and elapsed > max_prompt_wall_s:
            passed = False
            failures = [
                f"Wall time {elapsed:.1f}s exceeds budget {max_prompt_wall_s:.1f}s "
                "(use smaller model, --quick, or lower --max-tokens)"
            ]
            results.append(
                (
                    prompt_data["name"],
                    passed,
                    failures,
                    elapsed,
                    response,
                    prompt_data,
                    prompt_max_tokens,
                    likely_near_cap,
                )
            )
            all_passed = False
            print("Result: FAIL")
            for f in failures:
                print(f"  - {f}")
            print("=" * 80 + "\n")
            continue

        passed, failures = evaluate_response(
            prompt_data, response, default_max_word_repeat_ratio=repeat_ratio
        )
        status = "PASS" if passed else "FAIL"
        print(f"Result: {status}")
        if failures:
            for f in failures:
                print(f"  - {f}")
        print("=" * 80 + "\n")

        results.append(
            (
                prompt_data["name"],
                passed,
                failures,
                elapsed,
                response,
                prompt_data,
                prompt_max_tokens,
                likely_near_cap,
            )
        )
        if not passed:
            all_passed = False

    print("--- Summary ---")
    for name, passed, failures, elapsed, *_ in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name} ({elapsed:.2f}s)")
    n_pass = sum(1 for _, p, *_ in results if p)
    print(f"\n{n_pass}/{len(results)} passed")

    prompt_quality_passed = all_passed
    if prompt_quality_passed:
        print("\nPrompt quality: PASSED")
    else:
        print("\nPrompt quality: FAILED")

    # Per-run memory measurement (process-local, not host-wide)
    peak_memory_mb = round(mx.get_peak_memory() / (1024 * 1024), 1)
    memory_capped = memory_limit_mb is not None
    fits_cap = peak_memory_mb <= memory_limit_mb if memory_capped else None
    system_at_end = _system_snapshot()

    if memory_capped and fits_cap is False:
        all_passed = False

    if memory_capped:
        print(
            f"\nMemory: peak={peak_memory_mb:.0f} MB / cap={memory_limit_mb:.0f} MB → {'PASS' if fits_cap else 'FAIL'}"
        )

    print(f"\nRun result: {'PASSED' if all_passed else 'FAILED'}")

    # Restore memory limit
    if old_mem_limit is not None:
        mx.set_memory_limit(old_mem_limit)

    artifact = {
        "version": 5,  # v5: added prompt_spec_hash for baseline regression safety; v4 added response persistence + likely_near_cap + prompt_spec with text/description
        "harness_version": harness_version,
        "model_path": model_path,
        "expert_offload": expert_offload,
        "kv_cache_type": kv_cache_type,
        "use_predictor": use_predictor,
        "use_dedekimi_observer": use_dedekimi_observer,
        "task_expert_cliques": bool(task_expert_cliques),
        "max_resident_experts": max_resident_experts,
        "max_cached_shards": max_cached_shards,
        "seed": seed,
        "temp": temp,
        "max_tokens": max_tokens,
        "max_prompt_wall_s": max_prompt_wall_s,
        "suite": suite_name,
        "prompt_names": [prompt["name"] for prompt in prompts],
        "strict": strict,
        "max_word_repeat_ratio_default": repeat_ratio,
        "n_pass": n_pass,
        "n_total": len(results),
        "memory": {
            "peak_mb": peak_memory_mb,
            "cap_mb": memory_limit_mb,
            "capped": memory_capped,
            "fits_cap": fits_cap,
        },
        "system_before_load": system_before_load,
        "system_after_load": system_after_load,
        "results": [
            {
                "name": name,
                "passed": passed,
                "failures": failures,
                "latency_s": round(elapsed, 4),
                "response": resp_text,
                "response_words": len(resp_text.split()),
                "effective_max_tokens": eff_max_tokens,
                "likely_near_cap": near_cap,
                "prompt_spec": (
                    ps := {
                        "text": pdata.get("text", ""),
                        "description": pdata.get("description", ""),
                        "expected_substr": pdata.get("expected_substr", []),
                        "min_tokens": pdata.get("min_tokens", 15),
                        "skip_min_tokens": pdata.get("skip_min_tokens", False),
                        "check_fences": pdata.get("check_fences", True),
                        "max_tokens": pdata.get("max_tokens"),
                        "max_words": pdata.get("max_words"),
                        "max_word_repeat_ratio": pdata.get("max_word_repeat_ratio"),
                    }
                ),
                "prompt_spec_hash": _prompt_spec_hash({"prompt_spec": ps}),
            }
            for name, passed, failures, elapsed, resp_text, pdata, eff_max_tokens, near_cap in results
        ],
        "system_at_end": system_at_end,
        "system_deltas": {
            "rss_mb_load_delta": _delta(
                system_after_load.get("process_rss_mb"),
                system_before_load.get("process_rss_mb"),
            ),
            "rss_mb_total_delta": _delta(
                system_at_end.get("process_rss_mb"),
                system_before_load.get("process_rss_mb"),
            ),
            "swap_used_mb_delta": _delta(
                system_at_end.get("swap_used_mb"),
                system_before_load.get("swap_used_mb"),
            ),
            "vm_pages_paged_out_delta": _delta(
                system_at_end.get("vm_pages_paged_out"),
                system_before_load.get("vm_pages_paged_out"),
            ),
        },
    }
    return all_passed, artifact


def load_baseline(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _prompt_spec_hash(result: dict[str, Any]) -> str:
    """Hash the prompt spec so we can detect when a prompt was edited between runs."""
    import hashlib

    spec = result.get("prompt_spec", {})
    # Hash ALL fields that affect pass/fail outcome
    key = json.dumps(
        {
            "text": spec.get("text", ""),
            "expected_substr": spec.get("expected_substr", []),
            "check_fences": spec.get("check_fences", True),
            "min_tokens": spec.get("min_tokens", 15),
            "skip_min_tokens": spec.get("skip_min_tokens", False),
            "max_tokens": spec.get("max_tokens"),
            "max_words": spec.get("max_words"),
            "max_word_repeat_ratio": spec.get("max_word_repeat_ratio"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def baseline_regressed(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return human-readable regression messages if a previously passing task now fails.

    Compares by prompt name AND prompt spec hash. If the spec changed between
    runs, the comparison is flagged as incomparable rather than reported as a
    false regression.

    Also warns on artifact-level setting mismatches (harness_version,
    repetition threshold). These are informational — they don't block the gate
    by themselves but signal that the comparison may not be apples-to-apples.

    NOTE: other artifact-level settings (suite, seed, temp, max_tokens,
    kv_cache_type, expert_offload) are NOT checked. Baselines should be
    generated with the same CLI flags for the comparison to be meaningful.
    """
    # Check artifact-level settings that affect pass/fail semantics
    curr_hv = current.get("harness_version", "?")
    base_hv = baseline.get("harness_version", "?")
    curr_strict = current.get("max_word_repeat_ratio_default", "?")
    base_strict = baseline.get("max_word_repeat_ratio_default", "?")

    by_name = {r["name"]: r for r in baseline.get("results", [])}
    bad: list[str] = []

    if curr_hv != base_hv:
        bad.append(
            f"Warning: harness_version differs ({base_hv} → {curr_hv}); "
            f"prompt sets may not be comparable"
        )
    if curr_strict != base_strict:
        bad.append(
            f"Warning: repetition threshold differs ({base_strict} → {curr_strict}); "
            f"pass/fail outcomes may not be comparable"
        )

    for r in current.get("results", []):
        name = r["name"]
        prev = by_name.get(name)
        if not prev:
            continue
        # Check if prompt spec changed between baseline and current
        curr_hash = _prompt_spec_hash(r)
        prev_hash = _prompt_spec_hash(prev)
        if curr_hash != prev_hash:
            if prev.get("passed") and not r.get("passed"):
                bad.append(
                    f"Incomparable: {name!r} passed in baseline but prompt spec "
                    f"changed (hash {prev_hash} → {curr_hash}); cannot determine regression"
                )
            continue
        # Same prompt spec — real regression check
        if prev.get("passed") and not r.get("passed"):
            bad.append(f"Regression: {name!r} passed in baseline but failed now")
    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality gate for Nemotron-H MoE checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model under evaluation.",
    )
    parser.add_argument(
        "--expert-offload",
        action="store_true",
        help="Enable expert offload.",
    )
    parser.add_argument(
        "--max-resident-experts",
        type=int,
        default=None,
        help="Resident expert capacity override (default: auto from model topology).",
    )
    parser.add_argument(
        "--max-cached-shards",
        type=int,
        default=None,
        help="Max safetensors shard files cached in memory (default: auto from max_resident; 1 for 16GB targets).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt passed through the model chat template.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max new tokens per prompt (lower = faster; use --quick for a tight cap).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast path: suite=micro (v2), max_tokens capped at 48, 90s wall budget per prompt.",
    )
    parser.add_argument(
        "--max-prompt-wall-s",
        type=float,
        default=None,
        help="Fail a prompt if generation exceeds this many seconds (wall clock).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible generation.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy/deterministic).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="default",
        choices=["default", "coding", "micro", "soak", "all"],
        help="Prompt suite: default, coding, micro (2 prompts, v2 only), soak (1K+ decode, v2 only), or all.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write run artifact (pass/fail per task) to this path.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="JSON from a prior run; fail if any task that passed then fails now.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Tighter word-level repetition gate (default max ratio 0.22 vs 0.35).",
    )
    parser.add_argument(
        "--kv-cache-type",
        type=str,
        default="default",
        choices=["default", "turboquant", "isoquant", "rotorquant"],
        help="KV cache backend for generation (rotorquant = Cl(3,0) 3D blocks, isoquant = quaternion 4D blocks).",
    )
    parser.add_argument(
        "--use-predictor",
        action="store_true",
        help="When expert offload is on, enable AttnRes-style predictor (model config).",
    )
    parser.add_argument(
        "--use-dedekimi-observer",
        action="store_true",
        help="Enable DedeKimi observer metrics (no control path by default).",
    )
    parser.add_argument(
        "--task-expert-cliques-file",
        type=str,
        default=None,
        help="Path to JSON mapping task names to {layer_idx: [expert_ids...]}.",
    )
    parser.add_argument(
        "--harness-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="v1: original harness (fence check on, thinking enabled). v2: fixed harness (fence check off for code, thinking disabled). Scores are NOT comparable across versions.",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=None,
        help="Apply mx.set_memory_limit during the run (e.g. 12800 for 16GB, 25600 for 32GB). Quality + memory validated in same process.",
    )
    args = parser.parse_args()

    task_expert_cliques: dict[str, Any] | None = None
    if args.task_expert_cliques_file:
        with open(args.task_expert_cliques_file, "r") as f:
            task_expert_cliques = json.load(f)

    suite = args.suite
    max_tokens = args.max_tokens
    max_prompt_wall_s = args.max_prompt_wall_s
    if args.quick:
        if args.harness_version != "v2":
            print("--quick requires harness v2", file=sys.stderr)
            sys.exit(2)
        if suite == "default":
            suite = "micro"
        max_tokens = min(max_tokens, 48)
        if max_prompt_wall_s is None:
            max_prompt_wall_s = 90.0

    try:
        prompts = prompt_suite(suite, harness_version=args.harness_version)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    passed, artifact = run_gate(
        model_path=args.model,
        expert_offload=args.expert_offload,
        max_resident_experts=args.max_resident_experts,
        max_cached_shards=args.max_cached_shards,
        system_prompt=args.system_prompt,
        max_tokens=max_tokens,
        seed=args.seed,
        temp=args.temp,
        prompts=prompts,
        suite_name=suite,
        strict=args.strict,
        kv_cache_type=args.kv_cache_type,
        use_predictor=args.use_predictor,
        use_dedekimi_observer=args.use_dedekimi_observer,
        task_expert_cliques=task_expert_cliques,
        harness_version=args.harness_version,
        memory_limit_mb=args.memory_limit_mb,
        max_prompt_wall_s=max_prompt_wall_s,
    )
    attach_provenance(artifact, collect_provenance(args.model, args))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"Wrote artifact: {out}")

    if args.compare_baseline:
        baseline = load_baseline(Path(args.compare_baseline))
        regs = baseline_regressed(artifact, baseline)
        warnings = [r for r in regs if r.startswith("Warning:")]
        regressions = [r for r in regs if not r.startswith("Warning:")]
        for line in warnings:
            print(line, file=sys.stderr)
        for line in regressions:
            print(line, file=sys.stderr)
        if regressions:
            sys.exit(1)

    sys.exit(0 if passed else 1)
