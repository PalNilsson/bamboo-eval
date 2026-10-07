"""Every metric names the production entry point it calls (decision E-12).

Five diagnostic instruments misreported during the retrieval work, each
believed because it looked like evidence, and every one reported on something
*adjacent to* the code under test.  The rule that came out of it is that a
metric must call the function the server calls, and that a test must assert it.

These tests are the assertion.  The ones that need Bamboo installed skip
without it; the ones that pin the declaration itself do not, because a metric
reaching for an undeclared symbol is a defect whether or not Bamboo is present.
"""
from __future__ import annotations

import pytest

from bamboo_eval import production
from bamboo_eval.errors import MetricSkipped, ProductionContractError
from bamboo_eval.metrics import tool_retrieval

#: Every metric module, with the attribute names it declares.
METRIC_MODULES = [tool_retrieval]


class TestDeclarations:
    """What the metrics claim to call, checked without importing Bamboo."""

    @pytest.mark.parametrize("module", METRIC_MODULES, ids=lambda m: m.METRIC_NAME)
    def test_every_metric_declares_entry_points(self, module: object) -> None:
        """A metric with no declared entry point is measuring something unstated."""
        assert getattr(module, "PRODUCTION_ENTRY_POINTS", ())

    @pytest.mark.parametrize("module", METRIC_MODULES, ids=lambda m: m.METRIC_NAME)
    def test_declared_entry_points_are_known(self, module: object) -> None:
        """A metric cannot name a symbol production.py does not govern."""
        declared = {e.attribute for e in production.ENTRY_POINTS}
        unknown = set(module.PRODUCTION_ENTRY_POINTS) - declared  # type: ignore[attr-defined]
        assert not unknown, (
            f"{module.METRIC_NAME} declares entry points absent from "  # type: ignore[attr-defined]
            f"ENTRY_POINTS: {sorted(unknown)}"
        )

    def test_every_entry_point_explains_itself(self) -> None:
        """A contract failure must tell its reader why the symbol was needed."""
        for entry in production.ENTRY_POINTS:
            assert entry.why.strip(), f"{entry.dotted} is declared with no reason"

    def test_the_path_under_test_is_never_optional(self) -> None:
        """Only backends may be optional.

        A missing backend is a skip; a missing production path is a metric that
        has started measuring something else, and must fail the run.
        """
        required = {e.attribute for e in production.ENTRY_POINTS if not e.optional}
        assert {"_collect_tool_catalog", "routing_rules_for_plugin"} <= required

    def test_reaching_for_an_undeclared_symbol_raises(self) -> None:
        """The guard against bypassing the declaration table."""
        with pytest.raises(ProductionContractError, match="not a declared production"):
            production._entry("_collect_everything_somehow")


class TestAgainstLiveBamboo:
    """The declarations resolve against the installed Bamboo."""

    def test_contract_holds(self) -> None:
        """Nothing required has moved."""
        if not production.bamboo_available():
            pytest.skip("bamboo-core is not installed; the contract was not tested")
        try:
            production.collect_catalogue("atlas")
        except ProductionContractError as exc:
            pytest.fail(f"production contract broken: {exc}")
        except MetricSkipped as exc:
            pytest.skip(f"bamboo-core not importable: {exc}")
        broken = [p for p in production.check_contract() if p.startswith("BROKEN")]
        assert not broken, broken

    def test_an_absent_bamboo_is_a_skip_not_a_breach(self) -> None:
        """A bare checkout and a broken refactor must not look alike.

        Constraint 4.3 says a metric that cannot run degrades to a stated skip;
        decision E-12 says a metric whose entry point has moved fails the run.
        Both arrive as ImportError, so the distinction is made explicitly.
        """
        problems = production.check_contract()
        if production.bamboo_available():
            assert not any(p.startswith("NOT INSTALLED") for p in problems)
        else:
            assert problems == [p for p in problems if p.startswith("NOT INSTALLED")]
            assert len(problems) == 1, "one environment note, not one per symbol"

    def test_the_catalogue_fingerprint_is_stable(
        self, catalogue: list[dict[str, object]]
    ) -> None:
        """The same catalogue hashes the same way twice.

        Trivial, and the reason every stored comparison is trustworthy.
        """
        first = production.catalogue_fingerprint(catalogue)
        assert first == production.catalogue_fingerprint(catalogue)
        assert len(first) >= 12

    def test_the_guidance_fingerprint_follows_the_guidance(
        self, routing_rules: tuple[object, ...]
    ) -> None:
        """Changing a clause changes the digest.

        Guidance drift caused a production misroute before retrieval existed,
        so the guidance is fingerprinted rather than assumed to track the
        catalogue.
        """

        class _Rule:
            def __init__(self, tools: frozenset[str], text: str) -> None:
                self.tools = tools
                self.text = text

        before = production.guidance_fingerprint(routing_rules)
        assert before
        assert before != production.guidance_fingerprint(
            list(routing_rules) + [_Rule(frozenset({"x"}), "- use x.")]
        )

    def test_no_guidance_fingerprints_as_empty(self) -> None:
        """Absent guidance is recorded as absent, not as a digest of nothing."""
        assert production.guidance_fingerprint(()) == ""
