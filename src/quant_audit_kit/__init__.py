"""quant-audit-kit: Anti-cheating audit primitives for systematic & AI-driven quant research."""

from .audit import is_anomalous, prefix_causality_audit, reconcile_cash_replay
from .ledger import BacktestResult, ExecutionConfig, equal_weight_benchmark, simulate_portfolio
from .margin import cross_margin_stress
from .metrics import block_bootstrap_benchmark, compute_metrics, daily_returns, diagnose_drawdown
from .oos_lock import OOSLockbox
from .plateau import parameter_plateau_audit

__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "OOSLockbox",
    "block_bootstrap_benchmark",
    "compute_metrics",
    "cross_margin_stress",
    "daily_returns",
    "diagnose_drawdown",
    "equal_weight_benchmark",
    "is_anomalous",
    "parameter_plateau_audit",
    "prefix_causality_audit",
    "reconcile_cash_replay",
    "simulate_portfolio",
]
__version__ = "0.1.0"
