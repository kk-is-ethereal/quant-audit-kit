"""Physical Out-of-Sample (OOS) lockbox with cross-version Bonferroni alpha budgeting."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OOSLockbox:
    """Append-only audit ledger enforcing sequential Bonferroni alpha spending on OOS unlocks.

    Every OOS evaluation must register a frozen candidate's source hash before touching OOS.
    The n-th unlock receives alpha_n = total_alpha / (n * (n + 1)), ensuring the infinite sum
    of false-positive budgets across all future versions never exceeds `total_alpha`.
    """

    def __init__(self, ledger_path: str | Path, total_alpha: float = 0.05) -> None:
        self.ledger_path = Path(ledger_path)
        self.total_alpha = float(total_alpha)
        if not self.ledger_path.exists():
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._save({"total_alpha": self.total_alpha, "unlocks": []})

    def _load(self) -> dict[str, Any]:
        with self.ledger_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, payload: dict[str, Any]) -> None:
        with self.ledger_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def fingerprint_source(source_text: str) -> str:
        return hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    def next_alpha(self) -> float:
        data = self._load()
        n = len(data["unlocks"]) + 1
        return self.total_alpha / (n * (n + 1))

    def register_unlock(
        self,
        candidate_id: str,
        source_code: str,
        is_summary: dict[str, Any],
        oos_p_value: float,
        oos_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Permanently record an OOS evaluation and judge pass/fail against the remaining alpha budget."""
        data = self._load()
        source_hash = self.fingerprint_source(source_code)
        for existing in data["unlocks"]:
            if existing["source_hash"] == source_hash:
                raise RuntimeError(f"Duplicate OOS evaluation blocked for source_hash={source_hash[:12]}")

        n = len(data["unlocks"]) + 1
        alpha_n = self.total_alpha / (n * (n + 1))
        passed = bool(oos_p_value < alpha_n)
        record = {
            "unlock_index": n,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate_id,
            "source_hash": source_hash,
            "alpha_budget": alpha_n,
            "oos_p_value": float(oos_p_value),
            "passed": passed,
            "is_summary": is_summary,
            "oos_summary": oos_summary or {},
        }
        data["unlocks"].append(record)
        self._save(data)
        return record

    def status(self) -> dict[str, Any]:
        data = self._load()
        spent = sum(u["alpha_budget"] for u in data["unlocks"])
        return {
            "unlock_count": len(data["unlocks"]),
            "total_alpha": self.total_alpha,
            "alpha_spent": spent,
            "next_unlock_alpha": self.next_alpha(),
            "passed_count": sum(1 for u in data["unlocks"] if u["passed"]),
        }
