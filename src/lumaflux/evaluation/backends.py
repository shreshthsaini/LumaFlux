"""Shared failures for optional evaluation backends."""

from __future__ import annotations


class EvalBackendUnavailable(RuntimeError):
    """An optional metric cannot run in the current environment."""

    def __init__(self, metric: str, reason: str) -> None:
        self.metric = metric
        self.reason = reason.strip()
        super().__init__(f"{metric} unavailable: {self.reason}")
