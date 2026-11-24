"""Causality (look-ahead bias) auditing and independent naive cash-and-inventory replay."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .ledger import BacktestResult, ExecutionConfig


def is_anomalous(
    stats: dict[str, Any],
    sharpe_threshold: float = 3.5,
    min_active_days: int = 60,
    win_rate_threshold: float = 0.82,
) -> bool:
    """Flag suspiciously strong performance that warrants independent ledger reconciliation."""
    return (
        stats.get("sharpe", 0.0) > sharpe_threshold
        or (stats.get("active_days", 0) >= min_active_days and stats.get("win_rate", 0.0) > win_rate_threshold)
    )


def prefix_causality_audit(
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_data: pd.DataFrame,
    full_target: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.19, 0.48, 0.76),
    max_gross: float = 1.0,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Verify bit-level prefix invariance to catch look-ahead bias and future-bar leakage.

    Slices `raw_data` at multiple historical watermarks and asserts that `signal_fn(prefix)`
    produces identical target weights up to each watermark as `full_target`.
    """
    if not full_target.index.is_monotonic_increasing or not full_target.index.is_unique:
        raise ValueError("Signals must have strictly monotonic, unique timestamps")
    if np.isinf(full_target.to_numpy(dtype=float)).any():
        raise ValueError("Signals contain +/-inf values")
    if (full_target.abs().sum(axis=1) > max_gross + 1e-8).any():
        raise ValueError(f"Signals exceed declared max_gross={max_gross}")

    n = len(raw_data)
    if n < 10:
        raise ValueError("raw_data too short for prefix causality audit")

    checked = []
    for frac in fractions:
        cut_idx = max(1, min(n - 1, int(n * frac)))
        watermark = raw_data.index[cut_idx]
        prefix_input = raw_data.loc[raw_data.index <= watermark].copy()
        prefix_signal = signal_fn(prefix_input)
        expected = full_target.loc[full_target.index <= watermark]
        pd.testing.assert_frame_equal(
            prefix_signal,
            expected,
            check_freq=False,
            atol=atol,
            rtol=0.0,
        )
        checked.append(str(watermark))

    return {
        "causality_passed": True,
        "prefix_cuts_checked": len(checked),
        "watermarks": checked,
        "atol": atol,
    }


def reconcile_cash_replay(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    result: BacktestResult,
    *,
    tradable: pd.DataFrame | None = None,
    carry_cost: pd.DataFrame | np.ndarray | None = None,
    config: ExecutionConfig = ExecutionConfig(),
    rtol: float = 2e-10,
    atol: float = 2e-10,
) -> dict[str, Any]:
    """Independently replay cash and asset inventory in pure Python to verify every equity point."""
    cols = list(prices.columns)
    px = prices.to_numpy(dtype=float)
    tr = (
        np.isfinite(px).all(axis=1)
        if tradable is None
        else tradable.reindex(index=prices.index, columns=cols).fillna(False).to_numpy(dtype=bool).all(axis=1)
    )
    inst = targets.reindex(index=prices.index, columns=cols).to_numpy(dtype=float)
    if carry_cost is None:
        carry = np.zeros_like(px, dtype=float)
    elif isinstance(carry_cost, pd.DataFrame):
        carry = carry_cost.reindex(index=prices.index, columns=cols).fillna(0.0).to_numpy(dtype=float)
    else:
        carry = np.asarray(carry_cost, dtype=float)

    fee_mat = np.broadcast_to(config.fee, px.shape)
    inventory = np.zeros(px.shape[1], dtype=float)
    cash = 1.0
    reconciled = 0

    for i, quote in enumerate(px):
        cash -= float(np.dot(inventory, carry[i]))
        equity = cash + float(np.dot(inventory, quote))
        is_end = i == len(px) - 1
        if equity > 0.0 and (is_end or (tr[i] and np.isfinite(inst[i]).all())):
            target_qty = np.zeros_like(inventory) if is_end else equity * inst[i] / quote
            quantity = target_qty - inventory
            fill = quote * (1.0 + np.sign(quantity) * config.slippage)
            cash -= float(np.dot(quantity, fill)) + float(np.dot(np.abs(quantity) * fill, fee_mat[i]))
            inventory = target_qty
            equity = cash + float(np.dot(inventory, quote))
        np.testing.assert_allclose(
            max(0.0, equity),
            float(result.path["equity"].iat[i]),
            rtol=rtol,
            atol=atol,
            err_msg=f"Independent cash replay discrepancy at {prices.index[i]}",
        )
        reconciled = i + 1
        if equity <= 0.0:
            break

    return {"cash_inventory_replay": "passed", "reconciled_timestamps": reconciled}
