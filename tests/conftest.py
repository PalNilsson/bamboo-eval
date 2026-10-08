"""Shared fixtures.

The catalogue fixtures reach into Bamboo through
:mod:`bamboo_eval.production`, which is the only module allowed to import it.
Where Bamboo is not installed they skip with the reason, so a bare checkout
runs the pure-stdlib tests rather than erroring out of the whole suite — the
same degradation rule the metrics follow.
"""
from __future__ import annotations

# Each fixture below either returns a value or skips. ``pytest.skip()`` is
# typed NoReturn and raises, but a static analyser only knows that if it can
# resolve pytest — and the pre-commit pyright hook runs in its own isolated
# environment where it often cannot, which turns every skip branch into
# "function must return value on all code paths". The bare ``raise`` after each
# skip is unreachable at runtime and makes the control flow explicit to any
# analyser, in any environment, which is cheaper than requiring every
# environment to be configured correctly.

from typing import Any

import pytest

from bamboo_eval import production
from bamboo_eval.corpus import Corpus, ToolSelectionCase, bundled_corpus_path, load_corpus
from bamboo_eval.errors import MetricSkipped, ProductionContractError


@pytest.fixture(scope="session")
def corpus() -> Corpus[ToolSelectionCase]:
    """Return the bundled tool-selection corpus.

    Returns:
        Corpus[ToolSelectionCase]: The parsed corpus.
    """
    return load_corpus(bundled_corpus_path("tool_selection_corpus"), ToolSelectionCase)


@pytest.fixture(scope="session")
def catalogue() -> list[dict[str, Any]]:
    """Return the live ATLAS planner catalogue.

    Returns:
        List[Dict[str, Any]]: Catalogue entries.
    """
    try:
        return list(production.collect_catalogue("atlas"))
    except (ProductionContractError, MetricSkipped) as exc:
        pytest.skip(f"bamboo-core not importable: {exc}")
        raise


@pytest.fixture(scope="session")
def routing_rules() -> tuple[Any, ...]:
    """Return the live ATLAS routing guidance.

    Returns:
        Tuple[Any, ...]: Routing rules in precedence order.
    """
    try:
        return production.routing_rules("atlas")
    except (ProductionContractError, MetricSkipped) as exc:
        pytest.skip(f"bamboo-core not importable: {exc}")
        raise
