# `quant-audit-kit`

**Anti-cheating causality, block-bootstrap benchmark, parameter-plateau, margin-stress, and OOS-lockbox auditing for systematic & AI-driven trading research.**

Born out of **3,857 automated crypto quant experiments** where **914 strategies** cleared `CAGR ≥ 40%`, `Sharpe ≥ 1.8`, `MaxDD ≤ 20%`, and `Calmar ≥ 2.0` simultaneously — yet **0** survived paired block-bootstrap benchmark significance with **0 out-of-sample (OOS) peeks**.

> **Disclaimer:** Research and engineering audit software only. Not financial advice. Contains no trading signals, alpha formulas, or return promises.

---

## Why This Exists

When you let an LLM or coding agent iterate on trading strategies (`"keep tuning until Sharpe > 2"`), it does not automatically find alpha — it finds **look-ahead bias, backtester bugs, fee omissions, and multiple-testing luck**.

`quant-audit-kit` gives you **6 lightweight, strategy-agnostic guardrails** your agent cannot fool:

| Module | What It Catches |
| :--- | :--- |
| `prefix_causality_audit` | Look-ahead bias, full-sample normalization (`z-score`), and unclosed higher-timeframe bar leakage via multi-watermark prefix truncation (`atol = 1e-12`). |
| `reconcile_cash_replay` | Silent state-machine bugs in fast vectorized/Numba backtesters by replaying cash and asset inventory bar-by-bar in pure Python. |
| `block_bootstrap_benchmark` | Strategies that look high-Sharpe only by riding beta. Uses 14-day Circular Block Bootstrap on paired log-excess returns (with automatic arithmetic fallback if the benchmark hits ruin). |
| `cross_margin_stress` | Intraday liquidation risk hidden by close-to-close equity curves, testing worst-case high/low mark extremes per leg. |
| `parameter_plateau_audit` | Fragile parameter spikes via ±20% single-axis and joint same-direction perturbations. |
| `OOSLockbox` | Silent OOS p-hacking. SHA-256 binds candidate source code and enforces a strict infinite-horizon Bonferroni alpha budget: $\alpha_n = \frac{0.05}{n(n+1)}$. |

---

## Installation

```bash
git clone https://github.com/kk-is-ethereal/quant-audit-kit.git
cd quant-audit-kit
pip install -e .
# Optional Numba JIT acceleration:
pip install -e ".[fast]"
```

Requires only `numpy` and `pandas` (`numba` is optional and auto-detected).

---

## Quick Start (30-Second Look-Ahead Trap Demo)

```python
import pandas as pd
from quant_audit_kit import (
    ExecutionConfig,
    block_bootstrap_benchmark,
    compute_metrics,
    equal_weight_benchmark,
    prefix_causality_audit,
    reconcile_cash_replay,
    simulate_portfolio,
)

# 1. Catch sneaky full-sample look-ahead bias instantly
def leaky_signal(prices: pd.DataFrame) -> pd.DataFrame:
    z = (prices - prices.mean()) / prices.std()  # BUG: future rows leaked!
    return (z > 0).astype(float).div(prices.shape[1])

def causal_signal(prices: pd.DataFrame) -> pd.DataFrame:
    rolling_mean = prices.rolling(20, min_periods=20).mean().shift(1)
    return (prices.shift(1) > rolling_mean).astype(float).div(prices.shape[1]).fillna(0.0)

# prefix_causality_audit(leaky_signal, prices, leaky_signal(prices))  -> RAISES AssertionError!
audit = prefix_causality_audit(causal_signal, prices, causal_signal(prices))

# 2. Simulate with explicit 20 bps round-trip friction & verify cash replay
cfg = ExecutionConfig(fee=0.0005, slippage=0.0005, max_gross=1.0)
targets = causal_signal(prices)
res = simulate_portfolio(prices, targets, config=cfg)
reconcile_cash_replay(prices, targets, res, config=cfg)

# 3. Test statistical significance against Buy & Hold via 14-day Circular Block Bootstrap
bench = equal_weight_benchmark(prices, config=cfg)
stats = compute_metrics(res.path, days_per_year=365.2425)
boot = block_bootstrap_benchmark(res.path, bench.path, alpha=0.05, repeats=5000, block_size=14)
print(stats["sharpe"], boot["p_one_sided"], boot["wins"])
```

Run the full runnable demo:

```bash
python examples/demo_audit.py
```

---

## Need the Full AI Quant Research Lab Harness?

`quant-audit-kit` (MIT) covers the core anti-cheating audit primitives. If you want the complete production templates behind our 3,857-run research loop:
- **Zero-Leakage AI Research Agent Harness** (automated pivot-after-3-failures, attribution routing, SHA-256 source-locked experiment ledger)
- **Perpetual Futures Multi-Source Lag-Alignment Pipeline** (Binance OHLCV + Funding + 15m-lagged OI + Deribit DVOL)
- **Stratified Matched-Control Factor Pilot Templates** (kill 90% of fake factors before full backtests)

Check out the **Pro Research Lab Kit** and **Weekly Honest Quant Notes** linked in the repository profile.

---

## License

MIT License — see [LICENSE](./LICENSE).
