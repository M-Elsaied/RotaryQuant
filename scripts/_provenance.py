"""Run provenance for results JSON files.

Evaluation and benchmark scripts attach the dict returned by ``collect()``
under the ``provenance`` key of their output, so every results file records
which code, checkpoint, runtime settings and host produced it.

Paths under the user's home directory are written with a ``~`` prefix.
No hostname or user name is recorded.

CLI:
    python scripts/_provenance.py show --model <path-or-hf-id>
    python scripts/_provenance.py check results/a.json [results/b.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent

# Runtime env vars read by mlx_lm (this tree) and their in-code defaults.
# ``None`` means "unset unless provided". tests/test_provenance.py checks this
# table against the source so new env reads cannot go unrecorded.
ENV_DEFAULTS: dict[str, str | None] = {
    "ISOQUANT_BITS": None,  # falls back to TURBOQUANT_BITS
    "ISOQUANT_CACHE_MODE": "prealloc",
    "ISOQUANT_FUSED_ENCODE": "1",
    "ISOQUANT_NPT8_TILE_SIZE": "256",
    "ISOQUANT_PROFILE_METAL": "0",
    "ISOQUANT_USE_METAL": "0",
    "ISOQUANT_USE_NPT16_FUSED": "0",
    "ISOQUANT_USE_NPT8_FUSED": "0",
    "ISOQUANT_USE_STRUCTURED_MLX": "0",
    "ISOQUANT_VACCUM_TILE": "128",
    "TURBOQUANT_ASYMMETRIC_SCORE_MODULE": "",
    "TURBOQUANT_BITS": "3",
    "TURBOQUANT_CODEBOOK_DIR": "",
    "TURBOQUANT_DEBUG": "",
    "TURBOQUANT_SKIP_LAYERS": "2",
}
ENV_PREFIXES = ("ISOQUANT_", "TURBOQUANT_", "ROTORQUANT_", "MLX_", "METAL_")

CONFIG_KEYS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_global_key_value_heads",
    "head_dim",
    "global_head_dim",
    "sliding_window",
    "sliding_window_pattern",
    "attention_k_eq_v",
    "full_attention_interval",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "top_k_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "max_position_embeddings",
    "torch_dtype",
)

# Dotted paths that validate() requires to be present and non-null.
REQUIRED_FIELDS = (
    "schema_version",
    "created_utc",
    "script",
    "args",
    "git.sha",
    "git.dirty",
    "model.arg",
    "model.resolved_path",
    "model.config_sha256",
    "model.config.model_type",
    "model.config.num_hidden_layers",
    "model.config.num_attention_heads",
    "model.config.num_key_value_heads",
    "model.config.head_dim",
    "env.effective",
    "versions.python",
    "versions.mlx",
    "versions.mlx_lm",
    "versions.mlx_lm_origin",
    "host.machine",
    "host.ram_bytes",
)


class ProvenanceError(RuntimeError):
    """Raised when provenance cannot be determined for a run."""


def _home_relative(value: str | os.PathLike | None) -> str | None:
    if value is None:
        return None
    s = str(value)
    home = str(Path.home())
    candidates = [s]
    try:
        if Path(s).exists():  # expands Windows 8.3 short names, symlinks
            candidates.append(str(Path(s).resolve()))
    except OSError:
        pass
    for c in candidates:
        if c == home or c.startswith((home + os.sep, home + "/")):
            return "~" + c[len(home):].replace(os.sep, "/")
    return s


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return _home_relative(value)
    return repr(value)


def git_info(root: Path = REPO_ROOT) -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    # Tracked files differing from HEAD (staged or not); untracked files ignored.
    diff = _run(["git", "diff", "--name-only", "HEAD"], cwd=root)
    changed = diff.splitlines() if diff else []
    return {
        "sha": sha,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "describe": _run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root),
        "dirty": None if diff is None else bool(changed),
        "changed_files": changed,
    }


def _parse_hf_cache_path(path: Path) -> tuple[str | None, str | None]:
    """Return (repo_id, revision) for .../models--org--name/snapshots/<rev>."""
    parts = path.parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "snapshots" and parts[i - 1].startswith("models--"):
            repo_id = parts[i - 1][len("models--"):].replace("--", "/")
            return repo_id, parts[i + 1]
    return None, None


def resolve_model(model: str, cache_dir: str | Path | None = None) -> tuple[Path, str]:
    """Resolve a model argument to a local directory without downloading.

    A Hugging Face repo id resolves to the cached snapshot that holds its
    config.json. mlx_lm downloads only selected files, so the snapshot is
    not required to be complete.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        local, source = path.resolve(), "local"
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model, "config.json", cache_dir=cache_dir)
        except Exception as e:
            raise ProvenanceError(
                f"model {model!r} is neither a local directory nor a "
                f"Hugging Face repo id: {e}"
            ) from e
        if not isinstance(cached, str):  # None, or the cached-non-existence marker
            raise ProvenanceError(
                f"model {model!r} is neither a local directory nor in the "
                "Hugging Face cache"
            )
        local, source = Path(cached).parent, "hf_cache"
    if not (local / "config.json").is_file():
        raise ProvenanceError(f"no config.json in {local}")
    return local, source


def _weights_summary(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    return {
        "files": [{"name": f.name, "bytes": f.stat().st_size} for f in files],
        "total_bytes": sum(f.stat().st_size for f in files),
        "index_sha256": _sha256_file(index) if index.is_file() else None,
    }


def summarize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Architecture fields relevant to KV caching, from a HF/MLX config.json."""
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    merged = {**cfg, **text}
    out: dict[str, Any] = {
        "model_type": cfg.get("model_type"),
        "text_model_type": text.get("model_type"),
    }
    for key in CONFIG_KEYS:
        if key in merged:
            out[key] = merged[key]
    out["head_dim_derived"] = False
    if out.get("head_dim") is None and merged.get("hidden_size") and merged.get(
        "num_attention_heads"
    ):
        out["head_dim"] = merged["hidden_size"] // merged["num_attention_heads"]
        out["head_dim_derived"] = True
    if out.get("num_key_value_heads") is None and "num_attention_heads" in out:
        out["num_key_value_heads"] = out["num_attention_heads"]

    layer_types = merged.get("layer_types")
    if isinstance(layer_types, list):
        counts: dict[str, int] = {}
        for t in layer_types:
            counts[str(t)] = counts.get(str(t), 0) + 1
        out["layer_type_counts"] = counts
    pattern = merged.get("hybrid_override_pattern")
    if pattern is not None:
        pattern_str = "".join(pattern) if isinstance(pattern, list) else str(pattern)
        out["hybrid_override_pattern"] = pattern_str
        out["hybrid_block_counts"] = {
            c: pattern_str.count(c) for c in sorted(set(pattern_str))
        }

    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if isinstance(quant, dict):
        overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}
        bit_counts: dict[str, int] = {}
        for v in overrides.values():
            b = str(v.get("bits"))
            bit_counts[b] = bit_counts.get(b, 0) + 1
        out["quantization"] = {
            "bits": quant.get("bits"),
            "group_size": quant.get("group_size"),
            "mode": quant.get("mode"),
            "n_module_overrides": len(overrides),
            "override_bits_counts": bit_counts,
        }
    return out


def model_info(model: str) -> dict[str, Any]:
    path, source = resolve_model(model)
    repo_id, revision = _parse_hf_cache_path(path)
    if source == "hf_cache" and repo_id is None:
        repo_id = model
    config_path = path / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "arg": model,
        "resolved_path": _home_relative(path),
        "source": source,
        "hf_repo_id": repo_id,
        "hf_revision": revision,
        "config_sha256": _sha256_file(config_path),
        "config": summarize_config(cfg),
        "weights": _weights_summary(path),
    }


def env_info(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    effective = {k: env.get(k, default) for k, default in ENV_DEFAULTS.items()}
    if effective["ISOQUANT_BITS"] is None:
        effective["ISOQUANT_BITS"] = effective["TURBOQUANT_BITS"]
    return {
        "set": {k: v for k, v in sorted(env.items()) if k.startswith(ENV_PREFIXES)},
        "effective": effective,
    }


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions_info() -> dict[str, Any]:
    spec = importlib.util.find_spec("mlx_lm")
    mlx_lm_version = None
    if spec is not None and spec.origin:
        version_file = Path(spec.origin).parent / "_version.py"
        if version_file.is_file():
            m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_file.read_text())
            mlx_lm_version = m.group(1) if m else None
    return {
        "python": platform.python_version(),
        "mlx": _dist_version("mlx"),
        "mlx_metal": _dist_version("mlx-metal"),
        "mlx_lm": mlx_lm_version or _dist_version("mlx-lm"),
        # Which mlx_lm was imported: this tree's fork or a pip install.
        "mlx_lm_origin": _home_relative(spec.origin) if spec and spec.origin else None,
        "mlx_lm_in_repo": bool(
            spec and spec.origin and Path(spec.origin).resolve().is_relative_to(REPO_ROOT)
        ),
        "numpy": _dist_version("numpy"),
        "transformers": _dist_version("transformers"),
        "huggingface_hub": _dist_version("huggingface-hub"),
    }


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0] or None,
        "cpu": None,
        "ram_bytes": None,
        "metal_device": None,
    }
    if info["system"] == "Darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["ram_bytes"] = int(mem) if mem and mem.isdigit() else None
    elif hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    try:
        import mlx.core as mx

        device_info = getattr(mx, "device_info", None) or mx.metal.device_info
        info["metal_device"] = _jsonable(device_info())
    except Exception as e:  # mlx absent or no Metal device
        info["metal_device_error"] = f"{type(e).__name__}: {e}"
    return info


def collect(
    model: str,
    args: argparse.Namespace | dict[str, Any] | None = None,
    script: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance for one run. Raises ProvenanceError if the model is unresolvable."""
    if args is not None and not isinstance(args, dict):
        args = vars(args)
    prov = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script or Path(sys.argv[0]).name,
        "argv": [_home_relative(a) for a in sys.argv[1:]],
        "args": _jsonable(args or {}),
        "git": git_info(),
        "model": model_info(model),
        "env": env_info(),
        "versions": versions_info(),
        "host": host_info(),
    }
    if extra:
        prov["extra"] = _jsonable(extra)
    return prov


def attach(result: dict[str, Any], prov: dict[str, Any]) -> dict[str, Any]:
    result["provenance"] = prov
    return result


def validate(prov: Any) -> list[str]:
    """Return a list of problems; empty when all required fields are present."""
    if not isinstance(prov, dict):
        return ["provenance missing"]
    errors = []
    for dotted in REQUIRED_FIELDS:
        node: Any = prov
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            errors.append(f"missing {dotted}")
    if prov.get("schema_version") not in (None, SCHEMA_VERSION):
        errors.append(f"unknown schema_version {prov.get('schema_version')}")
    return errors


def check_file(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return ["top level is not an object"]
    return validate(data.get("provenance"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="Print provenance for a model")
    show.add_argument("--model", required=True)
    check = sub.add_parser("check", help="Validate provenance in results JSON files")
    check.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "show":
        try:
            prov = collect(args.model, script="_provenance.py show")
        except ProvenanceError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(json.dumps(prov, indent=2))
        problems = validate(prov)
        for p in problems:
            print(f"WARNING: {p}", file=sys.stderr)
        return 0

    failed = 0
    for path in args.paths:
        problems = check_file(path)
        status = "OK" if not problems else "FAIL"
        print(f"{status} {path}")
        for p in problems:
            print(f"  - {p}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
