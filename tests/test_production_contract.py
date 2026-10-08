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

# pylint: disable=protected-access,too-few-public-methods,unused-argument
# production._entry is private and is tested directly on purpose: it is the
# guard against a metric reaching into Bamboo for an undeclared symbol, which
# is the failure this whole module exists to prevent. The local _Rule stub is
# a two-attribute stand-in, not an object with an interface. The stand-in
# planner tools ignore their ``arguments`` but must keep the signature the
# wrapper calls them with, which is the thing under test.

import importlib.machinery
import os
import sys
import types

import pytest

from bamboo_eval import production
from bamboo_eval.errors import MetricSkipped, PlanParseError, ProductionContractError
from bamboo_eval.metrics import selection_accuracy, tool_retrieval

#: Every metric module, with the attribute names it declares.
METRIC_MODULES = [tool_retrieval, selection_accuracy]


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
        assert {
            "_collect_tool_catalog",
            "routing_rules_for_plugin",
            "bamboo_plan_tool",
        } <= required

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


class TestPlanParsing:
    """What comes back from the planner, and what counts as a plan."""

    def test_a_plan_parses(self) -> None:
        """The happy path, including the fields the metric reads."""
        parsed = production.parse_plan(
            '{"route": "PLAN", "confidence": 0.8, '
            '"tool_calls": [{"tool": "atlas.job_stats", "arguments": {}}]}'
        )
        assert parsed["tool_calls"][0]["tool"] == "atlas.job_stats"

    def test_a_fenced_plan_parses(self) -> None:
        """Models fence JSON; that is not a defect in the plan."""
        parsed = production.parse_plan(
            '```json\n{"route": "RETRIEVE", "confidence": 0.1, "tool_calls": []}\n```'
        )
        assert parsed["route"] == "RETRIEVE"

    @pytest.mark.parametrize(
        "payload",
        [
            "I am sorry, I cannot help with that.",
            "[1, 2, 3]",
            '{"confidence": 0.5, "tool_calls": []}',
            '{"route": "PLAN", "tool_calls": {"tool": "x"}}',
            '{"route": "PLAN", "tool_calls": [{"arguments": {}}]}',
        ],
    )
    def test_what_is_not_a_plan_is_rejected(self, payload: str) -> None:
        """Unparseable is its own outcome, so it needs its own exception.

        Args:
            payload: Text the planner might return instead of a plan.
        """
        with pytest.raises(PlanParseError):
            production.parse_plan(payload)

    def test_the_offending_payload_travels_with_the_error(self) -> None:
        """A failure nobody can read afterwards gets rediscovered."""
        with pytest.raises(PlanParseError) as caught:
            production.parse_plan("not json at all")
        assert "not json at all" in caught.value.payload


class TestPlannerWrapper:
    """``plan()`` drives the entry point decision E-25 names.

    Exercised against a stand-in module rather than the real Bamboo, because
    what is under test here is the wrapper's own shape — that it drives a
    coroutine, reads the first content block and rejects an entry point whose
    signature has changed — and that must hold on a bare checkout too.
    """

    @staticmethod
    def _install(monkeypatch: pytest.MonkeyPatch, tool: object) -> None:
        """Put a stand-in ``bamboo.tools.planner`` in front of the resolver.

        Args:
            monkeypatch: The fixture that restores ``sys.modules`` afterwards.
            tool: The object to publish as ``bamboo_plan_tool``.
        """
        for name in ("bamboo", "bamboo.tools", "bamboo.tools.planner"):
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(
            sys.modules["bamboo.tools.planner"],
            "bamboo_plan_tool",
            tool,
            raising=False,  # the stand-in module starts empty
        )

    def test_the_coroutine_is_driven_and_the_text_block_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``execute=False`` is passed explicitly: it is too important to imply."""
        seen: dict[str, object] = {}

        class _Tool:
            async def call(self, arguments: dict[str, object]) -> list[object]:
                """Return a canned plan.

                Args:
                    arguments: What the wrapper passed.

                Returns:
                    List[object]: One text content block.
                """
                seen.update(arguments)
                return [types.SimpleNamespace(text='{"route": "PLAN", "tool_calls": []}')]

        self._install(monkeypatch, _Tool())
        assert production.plan("why did the job fail?")["route"] == "PLAN"
        assert seen["execute"] is False
        assert seen["question"] == "why did the job fail?"

    def test_a_mapping_content_block_is_read_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which shape MCP content takes depends on the SDK version in use."""

        class _Tool:
            async def call(self, arguments: dict[str, object]) -> list[object]:
                """Return a canned plan as a mapping.

                Args:
                    arguments: Ignored.

                Returns:
                    List[object]: One mapping-shaped content block.
                """
                return [{"type": "text", "text": '{"route": "PLAN", "tool_calls": []}'}]

        self._install(monkeypatch, _Tool())
        assert production.plan("q")["tool_calls"] == []

    def test_an_entry_point_that_is_no_longer_a_coroutine_fails_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A change of shape here changes what the metric measures, so it is a
        contract breach rather than a per-case error."""

        class _Tool:
            def call(self, arguments: dict[str, object]) -> list[object]:
                """Return nothing of interest.

                Args:
                    arguments: Ignored.

                Returns:
                    List[object]: Empty.
                """
                return []

        self._install(monkeypatch, _Tool())
        with pytest.raises(ProductionContractError, match="coroutine"):
            production.plan("q")

    def test_an_empty_response_is_not_a_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returning nothing is a failure to answer, not an empty plan."""

        class _Tool:
            async def call(self, arguments: dict[str, object]) -> list[object]:
                """Return no content.

                Args:
                    arguments: Ignored.

                Returns:
                    List[object]: Empty.
                """
                return []

        self._install(monkeypatch, _Tool())
        with pytest.raises(PlanParseError):
            production.plan("q")


class TestModelSelection:
    """The lever a run pulls to choose the planner's model."""

    def test_the_variable_is_set_and_restored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A measurement must not leave the process configured differently."""
        monkeypatch.setenv(production.MODEL_ENV_VAR, "before")
        with production.model_selected("gpt-oss-20b") as lever:
            assert os.environ[production.MODEL_ENV_VAR] == "gpt-oss-20b"
            assert lever == production.MODEL_ENV_VAR
        assert os.environ[production.MODEL_ENV_VAR] == "before"

    def test_an_empty_model_changes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leaving the deployment's own selection in force is a configuration,
        not an omission, and is recorded as one."""
        monkeypatch.delenv(production.MODEL_ENV_VAR, raising=False)
        with production.model_selected("") as lever:
            assert lever == ""
            assert production.MODEL_ENV_VAR not in os.environ

    def test_the_recorded_retrieval_settings_name_every_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset is recorded rather than omitted: "the default was in force"
        and "nobody looked" are different runs.  ``BAMBOO_FAST_PATH`` is absent
        because the planner never reads it (decision E-26)."""
        monkeypatch.setenv("BAMBOO_TOOL_RETRIEVAL", "0")
        settings = production.retrieval_settings()
        assert settings["BAMBOO_TOOL_RETRIEVAL"] == "0"
        assert set(settings) == set(production.RETRIEVAL_ENV_VARS)
        assert "BAMBOO_FAST_PATH" not in settings
