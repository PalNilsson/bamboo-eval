"""Bamboo Evaluation Framework.

Measures retrieval, selection and relevance against labelled corpora, on the
production code path, with every number fingerprinted.  It is not a model
benchmark and it is not a dashboard.

The package core is pure standard library.  Metrics that need an embedding
model or an LLM gateway degrade to a recorded skip with a stated reason, never
to a wrong number, so the deterministic metrics run on a bare checkout.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
