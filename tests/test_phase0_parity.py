"""Phase 0's acceptance criterion, as a test.

The move from ``core/bamboo/evaluation/`` to this package was supposed to
change nothing about what is measured.  "Supposed to" is not a measurement, so
this test runs the ported harness against the catalogue and compares every
aggregate with the numbers the pre-move harness produced on the same
catalogue, recorded in ``tests/data/reference_numbers.json``.

If this fails, nothing else in the repository is trustworthy: every later
comparison is against a baseline that moved during a refactor nobody thought
changed behaviour.

A catalogue whose fingerprint is not in the reference file skips with
instructions rather than failing.  A different catalogue is a different
measurement — three states appeared in one week during the retrieval work, and
a fourth turned up the first time this package ran on another checkout — so
there is nothing to compare it against until someone records the old harness's
numbers for it.
"""
from __future__ import annotations

# pylint: disable=redefined-outer-name
# A test that consumes a fixture takes it as a parameter of the same name,
# which shadows the fixture function. That is how pytest works, not a defect.
#
# The unreachable `raise` after each pytest.skip() is deliberate: see the note
# at the top of conftest.py. It keeps this file analysable by a pyright that
# cannot resolve pytest, which is the state the pre-commit hook runs in.

import json
from pathlib import Path
from typing import Any

import pytest

from bamboo_eval import production
from bamboo_eval.corpus import Corpus, ToolSelectionCase
from bamboo_eval.errors import MetricSkipped
from bamboo_eval.metrics.tool_retrieval import evaluate

#: Mirrors the CLI's ``PINNED_TOOLS``; the reference numbers were produced with
#: these pinned, so the parity run must pin the same pair.
PINNED_TOOLS = frozenset({"panda_doc_search", "panda_doc_bm25"})

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_numbers.json"

#: Tolerance for the comparison.  The reference file stores four decimal
#: places, which is the precision the old harness printed; anything tighter
#: would be comparing rounding, anything looser would let a real change pass.
TOLERANCE = 5e-5


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """Return the recorded pre-move numbers.

    Returns:
        Dict[str, Any]: The parsed reference file.
    """
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def expected_block(
    reference: dict[str, Any], catalogue: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the reference block for the catalogue under test.

    Args:
        reference: The parsed reference file.
        catalogue: The live catalogue.

    Returns:
        Dict[str, Any]: The matching block.
    """
    fingerprint = production.catalogue_fingerprint(catalogue)[:12]
    block = reference["fingerprints"].get(fingerprint)
    if block is None:
        known = sorted(reference["fingerprints"])
        pytest.skip(
            f"no recorded pre-move numbers for catalogue {fingerprint} "
            f"({len(catalogue)} tools); recorded fingerprints are {known}. "
            f"See 'how_to_add_a_fingerprint' in {REFERENCE_PATH.name}."
        )
        raise AssertionError("unreachable: pytest.skip raises")
    return dict(block)


class TestParityWithThePreMoveHarness:
    """Every aggregate reproduces, on the same catalogue."""

    def test_the_catalogue_matches_the_reference(
        self, expected_block: dict[str, Any], catalogue: list[dict[str, Any]]
    ) -> None:
        """The fingerprint matched, so the payload denominator must match too.

        A guard against a fingerprint collision across genuinely different
        catalogues: the digest covers indexed terms only, so two catalogues
        could in principle agree on it while differing in serialised size.
        """
        corpus: Corpus[ToolSelectionCase] = Corpus(cases=())
        report = evaluate(production.retriever("null"), corpus, catalogue, k=10)
        assert report.full_payload_bytes == expected_block["full_payload_chars"]

    def test_corpus_size_matches_the_reference(
        self, expected_block: dict[str, Any], corpus: Corpus[ToolSelectionCase]
    ) -> None:
        """The corpus moved verbatim; a different case count invalidates the rest."""
        assert len(corpus.cases) == expected_block["cases"]

    def test_every_recorded_report_reproduces(
        self,
        expected_block: dict[str, Any],
        corpus: Corpus[ToolSelectionCase],
        catalogue: list[dict[str, Any]],
        routing_rules: tuple[Any, ...],
    ) -> None:
        """Recall, hard recall, guidance coverage, payload and budget all match.

        One test over every recorded report rather than a parametrised case per
        aggregate: a partial reproduction is not a partial success, and the
        failure message should name every number that moved at once.
        """
        mismatches: list[str] = []
        for expected in expected_block["reports"]:
            try:
                retriever = production.retriever(expected["retriever"])
            except MetricSkipped as exc:
                pytest.skip(f"{expected['retriever']} unavailable: {exc}")
                raise
            report = evaluate(
                retriever,
                corpus,
                catalogue,
                k=expected["k"],
                pinned=PINNED_TOOLS,
                routing_rules=routing_rules,
            )
            actual = {
                "recall_at_k": report.recall(),
                "recall_at_k_hard": report.recall(hard_only=True),
                "guidance_coverage": report.guidance_coverage(),
                "payload_fraction": report.mean_payload_fraction(),
            }
            label = f"{expected['retriever']}@k={expected['k']}"
            for key, value in actual.items():
                if abs(value - expected[key]) > TOLERANCE:
                    mismatches.append(
                        f"{label} {key}: {value:.4f} now, {expected[key]:.4f} before"
                    )
            if report.over_budget_cases() != expected["over_budget_cases"]:
                mismatches.append(
                    f"{label} over_budget_cases: {report.over_budget_cases()} now, "
                    f"{expected['over_budget_cases']} before"
                )
        assert not mismatches, (
            "bamboo-eval does not reproduce the pre-move harness on this "
            "catalogue:\n  " + "\n  ".join(mismatches)
        )


class TestTheReferenceFileItself:
    """A weak block must announce itself as one."""

    def test_a_block_without_the_lexical_sweep_says_so(
        self, reference: dict[str, Any]
    ) -> None:
        """The null baseline reproduces by construction — recall is 1.0 however
        the retriever behaves — so a block holding only that one checks the
        payload denominator and little else.  It may be recorded, because a
        catalogue's denominator is worth pinning on its own, but it must carry
        a note saying what is missing.  Otherwise a green parity test reads as
        a reproduction when nothing about the retriever was reproduced.
        """
        incomplete = [
            fingerprint
            for fingerprint, block in reference["fingerprints"].items()
            if not any(r["retriever"] != "null" for r in block["reports"])
            and not block.get("note")
        ]
        assert not incomplete, (
            f"these blocks record only the null baseline and do not say so: "
            f"{incomplete}"
        )
