"""High-fidelity multi-asset portfolio ledger with explicit fees, slippage, and carry/funding."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit execution friction and exposure assumptions."""

    fee: float = 0.0005
    slippage: float = 0.0005
    max_gross: float = 1.0


@dataclass
class BacktestResult:
    """Simulated mark-to-market equity path and terminal asset positions."""

    path: pd.DataFrame
    final_weights: dict[str, float]


@njit(cache=True)
def _mark_inventory(position, previous, current, carry_cost):
    pnl = 0.0
    paid = 0.0
    for j in range(len(position)):
        pnl += position[j] * (current[j] - previous[j])
        paid += position[j] * carry_cost[j]
    return pnl, paid


@njit(cache=True)
def _fill_target(position, prices, weights, equity, fees, slip):
    trading_fee = 0.0
    slippage_cost = 0.0
    turnover = 0.0
    for j in range(len(position)):
        mark = prices[j]
        desired = weights[j] * equity / mark
        delta = desired - position[j]
        notional = abs(delta) * mark
        execution = mark * (1.0 + slip if delta > 0 else 1.0 - slip)
        turnover += notional / equity
        trading_fee += abs(delta) * execution * fees[j]
        slippage_cost += notional * slip
        position[j] = desired
    return turnover, trading_fee, slippage_cost


@njit(cache=True)
def _account_loop(prices, tradable, carry_cost, targets, fees, slip):
    n, m = prices.shape
    path = np.zeros((n, 6))
    position = np.zeros(m)
    flat = np.zeros(m)
    equity = 1.0
    for i in range(n):
        prev_idx = i - 1 if i > 0 else 0
        pnl, paid_carry = _mark_inventory(position, prices[prev_idx], prices[i], carry_cost[i])
        equity += pnl - paid_carry
        trading_fee = 0.0
        slippage_cost = 0.0
        turnover = 0.0
        if equity > 0.0 and (i == n - 1 or (np.isfinite(targets[i]).all() and tradable[i].all())):
            intended = flat if i == n - 1 else targets[i]
            turnover, trading_fee, slippage_cost = _fill_target(
                position, prices[i], intended, equity, fees[i], slip
            )
        equity -= trading_fee + slippage_cost
        gross = np.sum(np.abs(position * prices[i])) / max(equity, 1e-12)
        path[i, 0] = max(0.0, equity)
        path[i, 1] = pnl
        path[i, 2] = paid_carry
        path[i, 3] = trading_fee + slippage_cost
        path[i, 4] = turnover
        path[i, 5] = gross
        if equity <= 0.0:
            return path, position
    return path, position


def simulate_portfolio(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
    carry_cost: pd.DataFrame | np.ndarray | None = None,
    config: ExecutionConfig = ExecutionConfig(),
) -> BacktestResult:
    """Simulate a multi-asset portfolio over a fixed timestamp grid.

    Parameters
    ----------
    prices:
        DataFrame of execution/mark prices (rows = timestamps, cols = assets).
    targets:
        DataFrame of desired portfolio weights indexed on the execution grid.
        Rows containing NaN hold existing inventory without rebalancing.
    tradable:
        Optional boolean DataFrame indicating whether each asset can fill at each bar.
    carry_cost:
        Optional per-unit carry/funding cashflow per bar (positive means longs pay).
    config:
        ExecutionConfig specifying fee rate, slippage rate, and max gross exposure.
    """
    if prices.empty:
        raise ValueError("prices DataFrame cannot be empty")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices index must be strictly monotonic and unique")

    cols = list(prices.columns)
    aligned_targets = targets.reindex(index=prices.index, columns=cols).to_numpy(dtype=float)
    if np.any(np.nansum(np.abs(aligned_targets), axis=1) > config.max_gross + 1e-8):
        raise ValueError("Position targets exceed declared ExecutionConfig.max_gross")

    px = prices.to_numpy(dtype=float)
    if tradable is None:
        tr = np.isfinite(px)
    else:
        tr = tradable.reindex(index=prices.index, columns=cols).fillna(False).to_numpy(dtype=bool)

    if carry_cost is None:
        carry = np.zeros_like(px, dtype=float)
    elif isinstance(carry_cost, pd.DataFrame):
        carry = carry_cost.reindex(index=prices.index, columns=cols).fillna(0.0).to_numpy(dtype=float)
    else:
        carry = np.asarray(carry_cost, dtype=float)
        if carry.shape != px.shape:
            raise ValueError(f"carry_cost shape {carry.shape} does not match prices shape {px.shape}")

    fee_matrix = np.broadcast_to(config.fee, px.shape).astype(float).copy()
    values, positions = _account_loop(px, tr, carry, aligned_targets, fee_matrix, float(config.slippage))
    path = pd.DataFrame(
        values,
        index=prices.index,
        columns=["equity", "price_pnl", "carry_cost", "trading_cost", "turnover", "gross"],
    )
    return BacktestResult(path=path, final_weights=dict(zip(cols, positions.tolist())))


def equal_weight_benchmark(
    prices: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
    carry_cost: pd.DataFrame | np.ndarray | None = None,
    config: ExecutionConfig = ExecutionConfig(),
) -> BacktestResult:
    """Fixed-quantity, equal-initial-weight buy-and-hold benchmark with identical execution friction."""
    tr = prices.notna() if tradable is None else tradable.reindex_like(prices).fillna(False).astype(bool)
    feasible = tr.all(axis=1)
    if not feasible.any():
        raise ValueError("No fully tradable bar available to initialize benchmark")
    schedule = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    first_bar = feasible.idxmax()
    schedule.loc[first_bar] = 1.0 / len(prices.columns)
    return simulate_portfolio(prices=prices, targets=schedule, tradable=tr, carry_cost=carry_cost, config=config)
