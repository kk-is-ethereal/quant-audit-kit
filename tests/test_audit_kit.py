"""Pure synthetic unit tests for quant-audit-kit (zero private strategies or market data)."""

import tempfile
import unittest
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
    is_anomalous,
    parameter_plateau_audit,
    prefix_causality_audit,
    reconcile_cash_replay,
    simulate_portfolio,
)


class QuantAuditKitTests(unittest.TestCase):
    def _synthetic_prices(self, periods: int = 120, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=periods, freq="1D", tz="UTC")
        rets = rng.normal(0.001, 0.02, size=(periods, 2))
        return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=["A", "B"])

    def test_buy_hold_exact_fees_and_replay(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
        prices = pd.DataFrame({"A": [100.0, 120.0, 120.0, 120.0], "B": [100.0, 80.0, 80.0, 80.0]}, index=idx)
        cfg = ExecutionConfig(fee=0.0005, slippage=0.0005, max_gross=1.0)
        bench = equal_weight_benchmark(prices, config=cfg)
        targets = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
        targets.iloc[0] = 0.5
        replay = reconcile_cash_replay(prices, targets, bench, config=cfg)
        self.assertEqual(replay["cash_inventory_replay"], "passed")
        self.assertEqual(replay["reconciled_timestamps"], 4)
        self.assertLess(bench.path["equity"].iloc[-1], 1.0)

    def test_carry_cost_deducted_before_rebalance(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
        targets = pd.DataFrame({"A": [1.0, np.nan, 0.0]}, index=idx)
        carry = pd.DataFrame({"A": [0.0, 2.0, 0.0]}, index=idx)
        cfg = ExecutionConfig(fee=0.0, slippage=0.0, max_gross=1.0)
        res = simulate_portfolio(prices, targets, carry_cost=carry, config=cfg)
        reconcile_cash_replay(prices, targets, res, carry_cost=carry, config=cfg)
        self.assertAlmostEqual(float(res.path["equity"].iloc[-1]), 0.98, places=10)

    def test_prefix_causality_catches_lookahead_and_accepts_causal(self) -> None:
        prices = self._synthetic_prices(100)

        def leaky(df: pd.DataFrame) -> pd.DataFrame:
            return ((df - df.mean()) > 0).astype(float).div(df.shape[1])

        def causal(df: pd.DataFrame) -> pd.DataFrame:
            return (df.shift(1) > df.rolling(10, min_periods=10).mean().shift(1)).astype(float).div(df.shape[1]).fillna(0.0)

        with self.assertRaises(AssertionError):
            prefix_causality_audit(leaky, prices, leaky(prices))

        audit = prefix_causality_audit(causal, prices, causal(prices))
        self.assertTrue(audit["causality_passed"])
        self.assertEqual(audit["prefix_cuts_checked"], 3)

    def test_block_bootstrap_handles_benchmark_ruin(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="1D", tz="UTC")
        strat_eq = np.linspace(1.0, 1.5, len(idx))
        bench_eq = np.r_[np.linspace(1.0, 0.2, 20), np.zeros(len(idx) - 20)]
        path = pd.DataFrame({"equity": strat_eq}, index=idx)
        bench = pd.DataFrame({"equity": bench_eq}, index=idx)
        res = block_bootstrap_benchmark(path, bench, alpha=0.05, repeats=500, block_size=7)
        self.assertEqual(res["comparison"], "arithmetic_after_benchmark_ruin")
        self.assertTrue(res["terminal_net_lead"])
        self.assertTrue(res["wins"])

    def test_cross_margin_stress_flags_intraday_wick_breach(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
        targets = pd.DataFrame({"A": [3.0, np.nan, 0.0]}, index=idx)
        cfg = ExecutionConfig(fee=0.0, slippage=0.0, max_gross=3.0)
        res = simulate_portfolio(prices, targets, config=cfg)
        mark_low = pd.DataFrame({"A": [65.0, 100.0, 100.0]}, index=idx)
        mark_high = pd.DataFrame({"A": [101.0, 101.0, 101.0]}, index=idx)
        stress = cross_margin_stress(prices, targets, res.path, prices, mark_low, mark_high, maintenance_rate=0.10)
        self.assertFalse(stress["passed"])

    def test_parameter_plateau_and_anomalies(self) -> None:
        self.assertTrue(is_anomalous({"sharpe": 3.8, "active_days": 10, "win_rate": 0.5}))
        self.assertFalse(is_anomalous({"sharpe": 1.9, "active_days": 100, "win_rate": 0.55}))

        smooth = parameter_plateau_audit(lambda p: 2.0 - 0.01 * abs(p[0] - 20.0), (20.0,))
        self.assertTrue(smooth["passed"])

        spike = parameter_plateau_audit(lambda p: 2.5 if abs(p[0] - 20.0) < 0.5 else 0.3, (20.0,))
        self.assertFalse(spike["passed"])

    def test_oos_lockbox_bonferroni_and_duplicate_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            box = OOSLockbox(Path(tmp) / "ledger.json", total_alpha=0.05)
            self.assertAlmostEqual(box.next_alpha(), 0.025)
            r1 = box.register_unlock("C1", "def sig(): return 1", {"sharpe": 2.1}, oos_p_value=0.03)
            self.assertFalse(r1["passed"])
            self.assertAlmostEqual(box.next_alpha(), 0.05 / 6.0)
            with self.assertRaises(RuntimeError):
                box.register_unlock("C1_DUP", "def sig(): return 1", {"sharpe": 2.1}, oos_p_value=0.001)

    def test_diagnose_drawdown(self) -> None:
        prices = self._synthetic_prices(60)
        targets = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
        cfg = ExecutionConfig(fee=0.0005, slippage=0.0005, max_gross=1.0)
        res = simulate_portfolio(prices, targets, config=cfg)
        stats = compute_metrics(res.path)
        diag = diagnose_drawdown(res.path, prices, stats)
        self.assertIn(diag["cause"], {"trading friction", "adverse carry", "tail drawdown", "insufficient edge"})


if __name__ == "__main__":
    unittest.main()
