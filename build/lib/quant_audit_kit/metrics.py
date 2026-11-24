"""Performance metrics, Circular Block Bootstrap benchmark significance, and failure attribution."""

import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DAYS_PER_YEAR = 365.2425


def daily_returns(path: pd.DataFrame) -> pd.Series:
    """Compute daily net simple returns from an intraday or daily equity path."""
    daily = path["equity"].iloc[1:].resample("1D", closed="right", label="right").last().dropna()
    previous = daily.shift(fill_value=1.0)
    return daily.div(previous.where(previous > 0)).sub(1.0).fillna(0.0)


def compute_metrics(path: pd.DataFrame, days_per_year: float = DEFAULT_DAYS_PER_YEAR) -> dict[str, Any]:
    """Compute core return, risk, turnover, and cost attribution metrics."""
    returns = daily_returns(path)
    years = (path.index[-1] - path.index[0]).total_seconds() / (86400.0 * days_per_year)
    if years <= 0:
        raise ValueError("Path duration must be strictly positive")

    equity = path["equity"].to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = 1.0 - equity / running_peak
    trough = int(drawdown.argmax())
    peak = max(0, int(np.r_[1.0, equity[: trough + 1]].argmax()) - 1)
    cagr = (equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    stdev = float(returns.std())
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    active = len(positive) + len(negative)
    gross_daily = path["gross"].resample("1D").max()

    carry_col = "carry_cost" if "carry_cost" in path.columns else "funding_cost"
    return {
        "cagr": float(cagr),
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdown[trough]),
        "sharpe": float(returns.mean() / stdev * math.sqrt(days_per_year)) if stdev > 0 else 0.0,
        "calmar": float(cagr / drawdown[trough]) if drawdown[trough] > 0 else 0.0,
        "win_rate": len(positive) / active if active else 0.0,
        "payoff": float(positive.mean() / -negative.mean()) if len(positive) and len(negative) else None,
        "annual_turnover": float(path["turnover"].sum() / years),
        "day_count": len(returns),
        "active_days": active,
        "exposed_days": int((gross_daily > 1e-5).sum()),
        "max_gross_after_drift": float(path["gross"].max()),
        "price_pnl": float(path["price_pnl"].sum()),
        "trading_cost": float(path["trading_cost"].sum()),
        "carry_cost": float(path[carry_col].sum()) if carry_col in path.columns else 0.0,
        "drawdown_peak": path.index[peak].isoformat(),
        "drawdown_trough": path.index[trough].isoformat(),
    }


def block_bootstrap_benchmark(
    path: pd.DataFrame,
    benchmark_path: pd.DataFrame,
    *,
    alpha: float = 0.05,
    repeats: int = 5000,
    block_size: int = 14,
    days_per_year: float = DEFAULT_DAYS_PER_YEAR,
    seed: int = 71043,
) -> dict[str, Any]:
    """Paired Circular Block Bootstrap test for statistically significant excess return over a benchmark.

    Automatically degrades from log-excess to paired arithmetic-excess if the benchmark hits ruin (-100%),
    while additionally requiring a positive terminal wealth lead.
    """
    a, b = daily_returns(path).align(daily_returns(benchmark_path), join="inner")
    if (a <= -1.0).any() or len(a) < 2 * block_size:
        return {"annual_log_excess": None, "lower": None, "p_one_sided": None, "alpha": alpha, "wins": False}

    b_ruined = bool((b <= -1.0).any())
    excess = (a - b).to_numpy() if b_ruined else np.log1p(a.to_numpy()) - np.log1p(b.to_numpy())
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(excess) / block_size)
    starts = rng.integers(0, len(excess), size=(repeats, blocks_needed))
    windows = (starts[:, :, None] + np.arange(block_size)) % len(excess)
    samples = excess[windows.reshape(repeats, -1)[:, : len(excess)]].mean(axis=1) * days_per_year

    lower = float(np.quantile(samples, alpha))
    p_value = (1 + int(np.count_nonzero(samples <= 0.0))) / (repeats + 1)
    annual_mean = float(excess.mean() * days_per_year)
    net_lead = bool(path["equity"].iat[-1] > benchmark_path["equity"].iat[-1])

    return {
        "annual_log_excess": None if b_ruined else annual_mean,
        "annual_simple_excess": annual_mean if b_ruined else None,
        "comparison": "arithmetic_after_benchmark_ruin" if b_ruined else "log_excess",
        "terminal_net_lead": net_lead,
        "lower": lower,
        "p_one_sided": float(p_value),
        "alpha": alpha,
        "wins": bool(lower > 0.0 and net_lead),
    }


def diagnose_drawdown(
    path: pd.DataFrame,
    prices: pd.DataFrame,
    stats: dict[str, Any],
    *,
    max_drawdown_limit: float = 0.20,
    cost_share_limit: float = 0.25,
    regime_threshold: float = 0.10,
) -> dict[str, Any]:
    """Assign an automated post-mortem failure diagnosis for research scheduling."""
    peak = pd.Timestamp(stats["drawdown_peak"])
    trough = pd.Timestamp(stats["drawdown_trough"])
    segment = path.loc[peak:trough]
    seg_prices = prices.loc[peak:trough]
    reference = seg_prices.div(seg_prices.iloc[0]).mean(axis=1)
    market_move = float(reference.iloc[-1] - 1.0)
    move_sum = float(reference.pct_change().abs().sum())
    efficiency = abs(market_move) / max(move_sum, 1e-12)

    regime = "downtrend" if market_move < -regime_threshold else (
        "rally" if market_move > regime_threshold else "range"
    )
    denom = max(0.01, abs(stats["price_pnl"]))
    carry_val = stats.get("carry_cost", stats.get("funding_cost", 0.0))
    if stats["trading_cost"] >= cost_share_limit * denom:
        cause = "trading friction"
    elif carry_val > cost_share_limit * denom:
        cause = "adverse carry"
    elif stats["max_drawdown"] > max_drawdown_limit:
        cause = "tail drawdown"
    else:
        cause = "insufficient edge"

    carry_col = "carry_cost" if "carry_cost" in segment.columns else "funding_cost"
    return {
        "regime": regime,
        "cause": cause,
        "equal_weight_price_move": market_move,
        "market_directionality": float(efficiency),
        "daily_bars": len(reference.index.normalize().unique()),
        "drawdown_price_pnl": float(segment["price_pnl"].sum()),
        "drawdown_trading_cost": float(segment["trading_cost"].sum()),
        "drawdown_carry_cost": float(segment[carry_col].sum()) if carry_col in segment.columns else 0.0,
        "drawdown_average_gross": float(segment["gross"].mean()),
    }
