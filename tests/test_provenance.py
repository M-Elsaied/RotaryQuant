"""Tests for scripts/_provenance.py (no MLX or Metal required)."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _provenance as prov  # noqa: E402

WIRED_SCRIPTS = [
    "measure_kv_fidelity.py",
    "long_context_kv_eval.py",
    "benchmark_moe_offload.py",
    "eval_quality_gate.py",
    "decode_profiler.py",
    "run_2h_soak.py",
    "run_stability_soak.py",
]

ENV_NAME = re.compile(r"\"((?:ISOQUANT|TURBOQUANT|ROTORQUANT)_[A-Z0-9_]+)\"")
ENV_GET = re.compile(
    r"os\.(?:environ\.get|getenv)\(\s*\"((?:ISOQUANT|TURBOQUANT)_[A-Z0-9_]+)\"\s*,\s*([^)]+?)\s*\)"
)


def _source_files():
    for root in (REPO_ROOT / "mlx-lm" / "mlx_lm", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if path.name != "_provenance.py":
                yield path


def _write_model(dirpath: Path, config: dict) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(config))
    (dirpath / "model.safetensors").write_bytes(b"\0" * 16)
    return dirpath


GEMMA_LIKE = {
    "model_type": "gemma4",
    "text_config": {
        "model_type": "gemma4_text",
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "num_experts": 128,
    },
    "quantization": {
        "bits": 4,
        "group_size": 64,
        "model.layers.0.mlp.experts": {"bits": 2, "group_size": 64},
        "model.layers.1.mlp.experts": {"bits": 2, "group_size": 64},
        "model.layers.0.mlp.shared_expert": {"bits": 8, "group_size": 32},
    },
}


def test_env_table_covers_every_env_var_in_source():
    found = set()
    for path in _source_files():
        found.update(ENV_NAME.findall(path.read_text(encoding="utf-8")))
    missing = sorted(found - set(prov.ENV_DEFAULTS))
    assert not missing, f"add to ENV_DEFAULTS: {missing}"


def test_env_table_defaults_match_source():
    mismatches = []
    for path in _source_files():
        for name, default in ENV_GET.findall(path.read_text(encoding="utf-8")):
            value = default.strip().strip("\"'")
            if prov.ENV_DEFAULTS.get(name) != value:
                mismatches.append((name, value, prov.ENV_DEFAULTS.get(name), path.name))
    assert not mismatches, mismatches


def test_env_effective_resolves_isoquant_bits_fallback():
    env = prov.env_info({"TURBOQUANT_BITS": "4", "PATH": "x", "MLX_FOO": "1"})
    assert env["effective"]["ISOQUANT_BITS"] == "4"
    assert env["effective"]["ISOQUANT_CACHE_MODE"] == "prealloc"
    assert env["set"] == {"MLX_FOO": "1", "TURBOQUANT_BITS": "4"}
    env = prov.env_info({"ISOQUANT_BITS": "2"})
    assert env["effective"]["ISOQUANT_BITS"] == "2"
    assert env["effective"]["TURBOQUANT_BITS"] == "3"


def test_summarize_config_nested_text_config():
    cfg = prov.summarize_config(GEMMA_LIKE)
    assert cfg["model_type"] == "gemma4"
    assert cfg["text_model_type"] == "gemma4_text"
    assert cfg["head_dim"] == 256 and cfg["global_head_dim"] == 512
    assert cfg["head_dim_derived"] is False
    assert cfg["layer_type_counts"] == {"sliding_attention": 5, "full_attention": 1}
    q = cfg["quantization"]
    assert (q["bits"], q["group_size"], q["n_module_overrides"]) == (4, 64, 3)
    assert q["override_bits_counts"] == {"2": 2, "8": 1}


def test_summarize_config_derives_head_dim_and_kv_heads():
    cfg = prov.summarize_config(
        {"model_type": "llama", "hidden_size": 4096, "num_attention_heads": 32}
    )
    assert cfg["head_dim"] == 128 and cfg["head_dim_derived"] is True
    assert cfg["num_key_value_heads"] == 32


def test_summarize_config_hybrid_pattern():
    cfg = prov.summarize_config(
        {"model_type": "nemotron_h", "hybrid_override_pattern": "MEM*EM*E"}
    )
    assert cfg["hybrid_block_counts"] == {"*": 2, "E": 3, "M": 3}


def test_hf_cache_path_parsing(tmp_path):
    snap = tmp_path / "hub" / "models--org--Some-Model-4bit" / "snapshots" / "abc123"
    _write_model(snap, {"model_type": "qwen3_moe", "num_hidden_layers": 2})
    assert prov._parse_hf_cache_path(snap) == ("org/Some-Model-4bit", "abc123")
    info = prov.model_info(str(snap))
    assert info["source"] == "local"
    assert info["hf_repo_id"] == "org/Some-Model-4bit"
    assert info["hf_revision"] == "abc123"
    assert info["weights"]["total_bytes"] == 16
    assert len(info["config_sha256"]) == 64


def _fake_hf_cache(root: Path, repo: str, rev: str, files: dict) -> Path:
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / rev
    snap.mkdir(parents=True)
    for name, content in files.items():
        (snap / name).write_text(content)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(rev)
    return snap


def test_hf_repo_id_resolves_from_partial_snapshot(tmp_path, monkeypatch):
    # mlx_lm downloads a subset of repo files; the snapshot is "incomplete"
    # from the hub's point of view but must still resolve.
    snap = _fake_hf_cache(
        tmp_path, "org/Partial-4bit", "abc123", {"config.json": json.dumps(GEMMA_LIKE)}
    )
    local, source = prov.resolve_model("org/Partial-4bit", cache_dir=tmp_path)
    assert (local, source) == (snap, "hf_cache")

    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    info = prov.model_info("org/Partial-4bit")
    assert info["source"] == "hf_cache"
    assert (info["hf_repo_id"], info["hf_revision"]) == ("org/Partial-4bit", "abc123")


def test_hf_repo_id_not_cached_raises(tmp_path):
    with pytest.raises(prov.ProvenanceError, match="Hugging Face"):
        prov.resolve_model("org/missing-model", cache_dir=tmp_path)


def test_unresolvable_model_raises(tmp_path):
    with pytest.raises(prov.ProvenanceError):
        prov.model_info(str(tmp_path / "does-not-exist"))
    with pytest.raises(prov.ProvenanceError):
        prov.model_info("no-such-org-xyz/no-such-model-xyz")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(prov.ProvenanceError, match="config.json"):
        prov.model_info(str(empty))


def _fake_runtime(monkeypatch):
    monkeypatch.setattr(
        prov,
        "versions_info",
        lambda: {
            "python": "3.11",
            "mlx": "0.31.0",
            "mlx_lm": "0.0.0",
            "mlx_lm_origin": "x",
        },
    )
    monkeypatch.setattr(
        prov, "host_info", lambda: {"machine": "arm64", "ram_bytes": 1 << 37}
    )


def test_collect_validates_and_redacts_home(tmp_path, monkeypatch):
    _fake_runtime(monkeypatch)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    model_dir = _write_model(tmp_path / "models" / "m", GEMMA_LIKE)
    args = argparse.Namespace(model=str(model_dir), seed=42, out=Path("r.json"))
    p = prov.collect(str(model_dir), args, script="unit")
    assert prov.validate(p) == []
    assert p["model"]["resolved_path"] == "~/models/m"
    assert p["args"]["seed"] == 42
    json.dumps(p)  # serializable


def test_validate_reports_missing_fields():
    assert prov.validate(None) == ["provenance missing"]
    errors = prov.validate({"schema_version": 1, "git": {"sha": None}})
    assert "missing git.sha" in errors
    assert "missing model.config.head_dim" in errors


def test_git_info_in_repo():
    info = prov.git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", info["sha"] or "")
    assert isinstance(info["dirty"], bool)
    for name in info["changed_files"]:
        assert (REPO_ROOT / name).exists(), name


def test_git_info_changed_files(tmp_path):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    subprocess.run(git[:-2] + ["init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("1")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    assert prov.git_info(tmp_path)["dirty"] is False
    (tmp_path / "a.txt").write_text("2")
    (tmp_path / "untracked.txt").write_text("x")
    info = prov.git_info(tmp_path)
    assert info["dirty"] is True
    assert info["changed_files"] == ["a.txt"]


def test_home_relative_resolves_existing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path.resolve()))
    target = tmp_path / "x" / "y"
    target.mkdir(parents=True)
    assert prov._home_relative(target) == "~/x/y"
    assert prov._home_relative("/elsewhere/z") == "/elsewhere/z"


def test_check_cli(tmp_path, monkeypatch, capsys):
    _fake_runtime(monkeypatch)
    model_dir = _write_model(tmp_path / "m", GEMMA_LIKE)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(prov.attach({"x": 1}, prov.collect(str(model_dir)))))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"x": 1}))
    assert prov.main(["check", str(good)]) == 0
    assert prov.main(["check", str(good), str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("script", WIRED_SCRIPTS)
def test_scripts_attach_provenance(script):
    src = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "from _provenance import" in src
    assert "collect_provenance(" in src
    assert "attach_provenance(" in src
