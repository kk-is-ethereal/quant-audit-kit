"""Pathwise cross-margin maintenance stress testing using intraday mark-price extremes."""

from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@njit(cache=True)
def _margin_requirement_loop(
    prices, tradable, targets, equity, trading_cost, mark_open, mark_low, mark_high, range_only, rate
):
    n, assets = prices.shape
    quantity = np.zeros(assets)
    peak = 0.0
    peak_index = 0
    for i in range(n):
        pre_fill = equity[i] + trading_cost[i]
        margin_equity = pre_fill
        notional = 0.0
        for j in range(assets):
            mark = mark_open[i, j]
            if range_only[i, j]:
                mark = mark_low[i, j] if quantity[j] >= 0.0 else mark_high[i, j]
            margin_equity += quantity[j] * (mark - prices[i, j])
            notional += abs(quantity[j]) * (mark_high[i, j] if range_only[i, j] else mark)
        if equity[i] <= 0.0 and notional > 0.0:
            return np.inf, i
        open_ratio = rate * notional / margin_equity if margin_equity > 0.0 else (np.inf if notional > 0.0 else 0.0)
        if open_ratio > peak:
            peak = open_ratio
            peak_index = i
        if i == n - 1 or equity[i] <= 0.0:
            break
        if tradable[i].all() and np.isfinite(targets[i]).all():
            quantity = targets[i] * pre_fill / prices[i]

        worst_equity = equity[i]
        upper_notional = 0.0
        for j in range(assets):
            adverse = mark_low[i, j] if quantity[j] >= 0.0 else mark_high[i, j]
            worst_equity += quantity[j] * (adverse - prices[i, j])
            upper_notional += abs(quantity[j]) * mark_high[i, j]
        range_ratio = (
            rate * upper_notional / worst_equity
            if worst_equity > 0.0
            else (np.inf if upper_notional > 0.0 else 0.0)
        )
        if range_ratio > peak:
            peak = range_ratio
            peak_index = i
    return peak, peak_index


def cross_margin_stress(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    path: pd.DataFrame,
    mark_open: pd.DataFrame,
    mark_low: pd.DataFrame,
    mark_high: pd.DataFrame,
    *,
    tradable: pd.DataFrame | None = None,
    range_only: pd.DataFrame | None = None,
    maintenance_rate: float = 0.10,
) -> dict[str, Any]:
    """Evaluate worst-case cross-margin maintenance ratio across each bar's high/low extremes."""
    index = path.index
    cols = list(prices.columns)
    px = prices.loc[index, cols].to_numpy(dtype=float)
    tr = (
        np.isfinite(px)
        if tradable is None
        else tradable.reindex(index=index, columns=cols).fillna(False).to_numpy(dtype=bool)
    )
    tg = targets.reindex(index=index, columns=cols).to_numpy(dtype=float)
    ro = (
        np.zeros_like(px, dtype=bool)
        if range_only is None
        else range_only.reindex(index=index, columns=cols).fillna(False).to_numpy(dtype=bool)
    )

    ratio, row = _margin_requirement_loop(
        px,
        tr,
        tg,
        path["equity"].to_numpy(dtype=float),
        path["trading_cost"].to_numpy(dtype=float),
        mark_open.loc[index, cols].to_numpy(dtype=float),
        mark_low.loc[index, cols].to_numpy(dtype=float),
        mark_high.loc[index, cols].to_numpy(dtype=float),
        ro,
        float(maintenance_rate),
    )
    return {
        "maintenance_rate": float(maintenance_rate),
        "peak_ratio": float(ratio) if np.isfinite(ratio) else None,
        "peak_timestamp": index[row].isoformat(),
        "fallback_range_cells": int(ro.sum()),
        "passed": bool(np.isfinite(ratio) and ratio < 1.0),
    }
