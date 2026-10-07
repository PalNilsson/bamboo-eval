"""Metrics.

One module per question the framework can answer.  Each declares the
production entry points it calls in ``PRODUCTION_ENTRY_POINTS`` and the name
its rows are stored under in ``METRIC_NAME``; a test asserts that every
declared entry point is one :mod:`bamboo_eval.production` knows about, so a
metric cannot quietly reach into Bamboo for something undeclared.
"""
from __future__ import annotations

from . import tool_retrieval

__all__ = ["tool_retrieval"]
