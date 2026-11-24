"""Parameter plateau robustness audit via single-axis and joint perturbations."""

from collections.abc import Callable, Sequence
from typing import Any


def parameter_plateau_audit(
    evaluate_sharpe: Callable[[tuple[float, ...]], float],
    base_params: Sequence[float],
    *,
    perturbation: float = 0.20,
    max_relative_drop: float = 0.40,
    integer_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Perturb each parameter by +/-`perturbation` individually and jointly to reject fragile peaks."""
    base = tuple(float(p) for p in base_params)
    if not base:
        raise ValueError("base_params must be non-empty")

    base_sharpe = float(evaluate_sharpe(base))
    masks = list(integer_mask) if integer_mask is not None else [False] * len(base)
    if len(masks) != len(base):
        raise ValueError("integer_mask length must match base_params")

    def _cast(vals: list[float]) -> tuple[float, ...]:
        out = []
        for v, is_int in zip(vals, masks):
            out.append(float(max(1, round(v))) if is_int else float(v))
        return tuple(out)

    grid: set[tuple[float, ...]] = set()
    # Single-parameter perturbations
    for i in range(len(base)):
        for mult in (1.0 - perturbation, 1.0 + perturbation):
            cand = list(base)
            cand[i] = base[i] * mult
            grid.add(_cast(cand))
    # Joint same-direction perturbations
    if len(base) > 1:
        for mult in (1.0 - perturbation, 1.0 + perturbation):
            grid.add(_cast([p * mult for p in base]))

    results = []
    for params in sorted(grid):
        sh = float(evaluate_sharpe(params))
        results.append({"params": params, "sharpe": sh})

    worst_sharpe = min(r["sharpe"] for r in results)
    rel_drop = (base_sharpe - worst_sharpe) / base_sharpe if base_sharpe > 0 else float("inf")
    passed = bool(base_sharpe > 0 and rel_drop <= max_relative_drop)

    return {
        "base_params": base,
        "base_sharpe": base_sharpe,
        "worst_sharpe": worst_sharpe,
        "relative_drop": float(rel_drop),
        "max_relative_drop": max_relative_drop,
        "points_tested": len(results),
        "passed": passed,
        "details": results,
    }
