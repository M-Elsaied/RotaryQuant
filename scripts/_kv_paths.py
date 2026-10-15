"""Record which IsoQuant attention path ran; optionally require the fused Metal path.

Scripts call ``reset()`` before generating, ``attach(result, args.require_fused)``
before writing their JSON, and ``enforce(args.require_fused)`` after writing it,
so the artifact is kept even when the run is failed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

_last_violations: list[str] = []


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-fused",
        action="store_true",
        help="Exit 1 if any IsoQuant decode read did not use the fused Metal path "
        "(MLX fallback, dense reconstruct, FP16 fallback cache, raw passthrough).",
    )


def reset() -> None:
    from mlx_lm.models import mlx_isoquant

    mlx_isoquant.reset_stats()


def attach(result: dict[str, Any], require_fused: bool = False) -> dict[str, Any]:
    """Add a ``kv_paths`` block (per-layer path counts, totals, violations)."""
    global _last_violations
    from mlx_lm.models import mlx_isoquant

    report = mlx_isoquant.kv_path_report()
    if require_fused or report["n_layers"]:
        _last_violations = mlx_isoquant.fused_path_violations(report)
    else:
        _last_violations = []
    report["require_fused"] = bool(require_fused)
    report["fused_only"] = bool(report["n_layers"]) and not _last_violations
    report["violations"] = list(_last_violations)
    report["isoquant_stats"] = mlx_isoquant.stats_summary()
    result["kv_paths"] = report
    return result


def enforce(require_fused: bool) -> None:
    if require_fused and _last_violations:
        print("FAIL: --require-fused: non-fused IsoQuant paths ran:", file=sys.stderr)
        for v in _last_violations[:20]:
            print(f"  - {v}", file=sys.stderr)
        if len(_last_violations) > 20:
            print(f"  ... {len(_last_violations) - 20} more", file=sys.stderr)
        sys.exit(1)
