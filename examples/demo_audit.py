"""End-to-end demonstration of quant-audit-kit catching look-ahead bias and auditing a causal strategy."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant_audit_kit import (
    ExecutionConfig,
    OOSLockbox,
    block_bootstrap_benchmark,
    compute_metrics,
    cross_margin_stress,
    diagnose_drawdown,
    equal_weight_benchmark,
    parameter_plateau_audit,
    prefix_causality_audit,
    reconcile_cash_replay,
    simulate_portfolio,
)


def make_synthetic_market(days: int = 365, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    rets = rng.normal(loc=0.0008, scale=0.03, size=(days, 2))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=["ASSET_A", "ASSET_B"])
    low = prices * (1.0 - rng.uniform(0.005, 0.03, size=prices.shape))
    high = prices * (1.0 + rng.uniform(0.005, 0.03, size=prices.shape))
    return prices, low, high


def leaky_strategy(prices: pd.DataFrame) -> pd.DataFrame:
    z = (prices - prices.mean()) / prices.std()
    return (z > 0).astype(float).div(prices.shape[1])


def causal_trend_strategy(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = prices.rolling(window, min_periods=window).mean().shift(1)
    sig = (prices.shift(1) > ma).astype(float).div(prices.shape[1])
    return sig.fillna(0.0)


def main() -> None:
    prices, mark_low, mark_high = make_synthetic_market()
    cfg = ExecutionConfig(fee=0.0005, slippage=0.0005, max_gross=1.0)

    print("=== 1. Prefix Causality Audit (Look-Ahead Detection) ===")
    try:
        prefix_causality_audit(leaky_strategy, prices, leaky_strategy(prices), max_gross=cfg.max_gross)
    except AssertionError:
        print("[PASS] Leaky full-sample z-score strategy CAUGHT immediately by prefix truncation!")

    targets = causal_trend_strategy(prices, window=20)
    causality = prefix_causality_audit(
        lambda df: causal_trend_strategy(df, window=20), prices, targets, max_gross=cfg.max_gross
    )
    print(f"[PASS] Causal strategy verified across {causality['prefix_cuts_checked']} watermarks.")

    print("\n=== 2. Backtest + Independent Cash/Inventory Replay ===")
    res = simulate_portfolio(prices, targets, config=cfg)
    replay = reconcile_cash_replay(prices, targets, res, config=cfg)
    print(f"[PASS] Reconciled {replay['reconciled_timestamps']} timestamps within 2e-10.")

    print("\n=== 3. Metrics + 14-Day Block Bootstrap vs Buy & Hold ===")
    bench = equal_weight_benchmark(prices, config=cfg)
    stats = compute_metrics(res.path)
    boot = block_bootstrap_benchmark(res.path, bench.path, alpha=0.05, repeats=1000, block_size=14)
    diag = diagnose_drawdown(res.path, prices, stats)
    print(
        f"CAGR={stats['cagr']:.2%}, Sharpe={stats['sharpe']:.2f}, MaxDD={stats['max_drawdown']:.2%}, "
        f"Bootstrap p={boot['p_one_sided']:.4f}, Wins={boot['wins']}, Diagnosis='{diag['cause']}'"
    )

    print("\n=== 4. Intraday Cross-Margin Stress Test ===")
    margin = cross_margin_stress(prices, targets, res.path, prices, mark_low, mark_high, maintenance_rate=0.10)
    print(f"Peak maintenance ratio={margin['peak_ratio']:.4f}, Passed={margin['passed']}")

    print("\n=== 5. +/-20% Parameter Plateau Audit ===")
    plateau = parameter_plateau_audit(
        lambda p: compute_metrics(
            simulate_portfolio(prices, causal_trend_strategy(prices, int(p[0])), config=cfg).path
        )["sharpe"],
        base_params=(20,),
        integer_mask=(True,),
    )
    print(f"Base Sharpe={plateau['base_sharpe']:.2f}, Worst={plateau['worst_sharpe']:.2f}, Passed={plateau['passed']}")

    print("\n=== 6. OOS Lockbox Sequential Bonferroni Budget ===")
    with tempfile.TemporaryDirectory() as tmp:
        lockbox = OOSLockbox(Path(tmp) / "oos_ledger.json", total_alpha=0.05)
        rec = lockbox.register_unlock(
            candidate_id="DEMO_TREND_20",
            source_code="window=20",
            is_summary={"sharpe": stats["sharpe"]},
            oos_p_value=0.04,
        )
        print(
            f"Unlock #{rec['unlock_index']}: alpha_budget={rec['alpha_budget']:.4f}, "
            f"oos_p={rec['oos_p_value']:.4f}, Passed={rec['passed']}, Next alpha={lockbox.next_alpha():.4f}"
        )


if __name__ == "__main__":
    main()
