"""Tests for scripts/_kv_paths.py with a stub mlx_isoquant module (no MLX required)."""

import argparse
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _kv_paths  # noqa: E402

WIRED_SCRIPTS = [
    "measure_kv_fidelity.py",
    "long_context_kv_eval.py",
    "benchmark_moe_offload.py",
    "eval_quality_gate.py",
    "decode_profiler.py",
    "run_stability_soak.py",
]


@pytest.fixture
def fake_iso(monkeypatch):
    state = {"report": {"n_layers": 0, "totals": {}, "layers": {}}, "violations": [], "resets": 0}
    mod = types.ModuleType("mlx_lm.models.mlx_isoquant")
    mod.kv_path_report = lambda: dict(state["report"])
    mod.fused_path_violations = lambda report: list(state["violations"])
    mod.stats_summary = lambda: {"decode_calls": 1}

    def reset_stats():
        state["resets"] += 1

    mod.reset_stats = reset_stats
    models = types.ModuleType("mlx_lm.models")
    models.mlx_isoquant = mod
    pkg = types.ModuleType("mlx_lm")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.mlx_isoquant", mod)
    return state


def test_flag_registered():
    parser = argparse.ArgumentParser()
    _kv_paths.add_argument(parser)
    assert parser.parse_args([]).require_fused is False
    assert parser.parse_args(["--require-fused"]).require_fused is True


def test_reset_calls_runtime(fake_iso):
    _kv_paths.reset()
    assert fake_iso["resets"] == 1


def test_attach_clean_run(fake_iso):
    fake_iso["report"] = {"n_layers": 2, "totals": {"fused_metal": 4}, "layers": {}}
    result = _kv_paths.attach({}, require_fused=True)
    kv = result["kv_paths"]
    assert kv["fused_only"] is True and kv["violations"] == []
    assert kv["require_fused"] is True
    assert kv["isoquant_stats"] == {"decode_calls": 1}
    _kv_paths.enforce(True)  # no exit


def test_violations_fail_only_with_flag(fake_iso):
    fake_iso["report"] = {"n_layers": 1, "totals": {"fused_mlx": 3}, "layers": {}}
    fake_iso["violations"] = ["layer 2: {'fused_mlx': 3}"]
    result = _kv_paths.attach({}, require_fused=False)
    assert result["kv_paths"]["fused_only"] is False
    assert result["kv_paths"]["violations"] == ["layer 2: {'fused_mlx': 3}"]
    _kv_paths.enforce(False)  # reported, not enforced
    _kv_paths.attach({}, require_fused=True)
    with pytest.raises(SystemExit) as exc:
        _kv_paths.enforce(True)
    assert exc.value.code == 1


def test_no_isoquant_layers(fake_iso):
    fake_iso["violations"] = ["no IsoQuant layers recorded"]
    # default-KV run without the flag: nothing to report as a violation
    assert _kv_paths.attach({}, require_fused=False)["kv_paths"]["violations"] == []
    # with the flag, a run that never used IsoQuant fails
    assert _kv_paths.attach({}, require_fused=True)["kv_paths"]["violations"]
    with pytest.raises(SystemExit):
        _kv_paths.enforce(True)


@pytest.mark.parametrize("script", WIRED_SCRIPTS)
def test_scripts_wire_kv_paths(script):
    src = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
    for call in ("_kv_paths.add_argument(", "_kv_paths.reset(", "_kv_paths.attach(", "_kv_paths.enforce("):
        assert call in src, call
    # each enforcement point follows the artifact write, so the JSON is kept
    pos = 0
    while (pos := src.find("_kv_paths.enforce(", pos + 1)) != -1:
        window = src[max(0, pos - 500) : pos]
        assert "json.dump(" in window or "_write_results(" in window, script
