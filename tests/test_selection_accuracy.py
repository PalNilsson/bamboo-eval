"""Phase 1, exercised end to end with no gateway anywhere.

Every one of the six outcomes, the whole name-resolution table, repeats,
resume and both budget guards are driven by a stub planner returning canned
plans.  That is deliberate: the parts of this metric that can be wrong in a
quiet, number-moving way — a suffix rule that resolves too much, a declined
plan pooled with a wrong tool, a budget that stops a run and then aggregates
what finished anyway — are exactly the parts that need no model to test.  Only
the first real measurement needs CERN.
"""
from __future__ import annotations

# pylint: disable=too-few-public-methods
# The test classes are namespaces grouping related assertions, not objects.

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bamboo_eval import ledger, production, store
from bamboo_eval.budget import Budget
from bamboo_eval.cli import main
from bamboo_eval.corpus import Corpus, ToolSelectionCase
from bamboo_eval.errors import BudgetExceeded, MetricSkipped, PlanParseError
from bamboo_eval.metrics import selection_accuracy as sa
from bamboo_eval.record import SCHEMA_VERSION, RunContext

#: A catalogue with both naming conventions and one genuinely ambiguous
#: suffix, which is the live ATLAS catalogue's shape in miniature.
CATALOGUE = (
    "atlas.job_stats",
    "panda_task_status",
    "panda_log_analysis",
    "code_query",
    "atlas.core_dump_analysis",
    "epic.job_stats",
)


def _case(
    case_id: str,
    tools: Sequence[str],
    hard: bool = False,
    question: str | None = None,
) -> ToolSelectionCase:
    """Build a labelled case.

    Args:
        case_id: Identifier.
        tools: Expected tools.
        hard: Whether the case is deliberately confusable.
        question: The question; defaults to the identifier.

    Returns:
        ToolSelectionCase: The case.
    """
    return ToolSelectionCase(
        case_id=case_id,
        question=question or f"question {case_id}",
        hard=hard,
        expected_tools=frozenset(tools),
    )


def _corpus(*cases: ToolSelectionCase) -> Corpus[ToolSelectionCase]:
    """Build an in-memory corpus.

    Args:
        *cases: The cases it holds.

    Returns:
        Corpus[ToolSelectionCase]: The corpus.
    """
    return Corpus(cases=cases, name="test-corpus", version=1, sha256="deadbeef")


def _plan(tools: Sequence[str], route: str = "PLAN", confidence: float = 0.9) -> dict[str, Any]:
    """Build a plan as the planner's entry point returns it.

    Args:
        tools: Tool names, in either naming convention.
        route: The plan's route.
        confidence: The plan's stated confidence.

    Returns:
        Dict[str, Any]: The plan.
    """
    return {
        "route": route,
        "confidence": confidence,
        "tool_calls": [{"tool": name, "arguments": {}} for name in tools],
        "explain": "because",
    }


class StubPlanner:
    """Returns canned plans, or raises, per question.

    Attributes:
        script: Question to plan, exception, or a list consumed one entry per
            call so that a case can behave differently across repeats.
        calls: Questions asked, in order.
    """

    def __init__(self, script: Mapping[str, Any]) -> None:
        """Initialise the stub.

        Args:
            script: Per-question responses.
        """
        self.script = {k: list(v) if isinstance(v, list) else v for k, v in script.items()}
        self.calls: list[str] = []

    def __call__(self, question: str) -> Mapping[str, Any]:
        """Answer one question.

        Args:
            question: The question asked.

        Returns:
            Mapping[str, Any]: The canned plan.

        Raises:
            Exception: Whatever the script says this question raises.
        """
        self.calls.append(question)
        response = self.script[question]
        if isinstance(response, list):
            response = response.pop(0) if len(response) > 1 else response[0]
        if isinstance(response, BaseException):
            raise response
        return response


class TestNameResolution:
    """Decision E-27: the resolution rule, and what it refuses to do."""

    @pytest.mark.parametrize(
        ("proposed", "resolved", "how"),
        [
            ("atlas.job_stats", "atlas.job_stats", "exact"),
            ("panda_task_status", "panda_task_status", "exact"),
            ("task_status", None, "unknown"),
            ("atlas.core_dump_analysis", "atlas.core_dump_analysis", "exact"),
            ("core_dump_analysis", "atlas.core_dump_analysis", "suffix"),
            ("atlas.code_query", "code_query", "suffix"),
            ("job_stats", None, "ambiguous"),
            ("cgsim.job_stats", None, "ambiguous"),
            ("invented_tool", None, "unknown"),
        ],
    )
    def test_the_table(self, proposed: str, resolved: str | None, how: str) -> None:
        """A bare name and its namespaced form are the same answer; a suffix
        matching two catalogue entries is not an answer at all."""
        result = sa.NameResolver(CATALOGUE).resolve(proposed)
        assert (result.resolved, result.how) == (resolved, how)

    def test_an_ambiguous_suffix_is_never_guessed(self) -> None:
        """Guessing would make the headline number depend on catalogue order."""
        resolver = sa.NameResolver(CATALOGUE)
        assert resolver.resolve("job_stats").resolved is None
        assert sa.NameResolver(("atlas.job_stats",)).resolve("job_stats").resolved == (
            "atlas.job_stats"
        )

    def test_whitespace_does_not_make_a_name_unknown(self) -> None:
        """Models pad names; that is not a grounding defect."""
        assert sa.NameResolver(CATALOGUE).resolve(" code_query ").how == "exact"


class TestOutcomes:
    """Decision E-28: six buckets, and the precedence between them."""

    @pytest.mark.parametrize(
        ("response", "outcome"),
        [
            (_plan(["atlas.job_stats"]), "correct"),
            (_plan(["job_stats"]), "unknown_tool"),
            (_plan(["panda_log_analysis"]), "wrong_tool"),
            (_plan([], route="RETRIEVE"), "declined"),
            (_plan(["invented_tool"]), "unknown_tool"),
            (PlanParseError("not JSON", "sorry, I cannot"), "unparseable"),
            (TimeoutError("gateway timed out"), "error"),
        ],
    )
    def test_each_bucket_is_reachable(self, response: Any, outcome: str) -> None:
        """Every outcome the classification admits is produced by some plan."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": response}), corpus, CATALOGUE, repeats=1
        )
        assert report.observations[0].outcome == outcome

    def test_a_declined_plan_is_not_a_wrong_tool(self) -> None:
        """Pooling them would say the planner chose badly when it chose nothing."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan([], route="RETRIEVE")}),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        counts = report.counts()
        assert counts["declined"] == 1
        assert counts["wrong_tool"] == 0

    def test_containment_wins_over_a_stray_invented_name(self) -> None:
        """The plan contains what a correct plan needs; the stray name is
        over-proposal, and the precision figure is where it belongs."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["atlas.job_stats", "invented_tool"])}),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        assert report.observations[0].outcome == "correct"
        assert report.over_proposal() == 1.0  # the invented name resolves to nothing
        assert report.resolution_counts()["unknown"] == 1

    def test_half_a_co_occurrence_pair_is_a_miss(self) -> None:
        """Scoring is strict: half of a required pair is a broken plan."""
        corpus = _corpus(_case("c1", ["atlas.job_stats", "panda_task_status"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["atlas.job_stats"])}),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        assert report.accuracy() == 0.0

    def test_a_skip_is_not_an_outcome(self) -> None:
        """An absent runtime stops the metric; it does not score every case 0."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        with pytest.raises(MetricSkipped):
            sa.evaluate(
                StubPlanner({"question c1": MetricSkipped("no gateway")}),
                corpus,
                CATALOGUE,
                repeats=1,
            )


class TestAggregates:
    """The numbers, including the two that are never fused into one."""

    def test_accuracy_precision_and_slices(self) -> None:
        """Right tool plus two others, and the wrong tool, are different rows."""
        corpus = _corpus(
            _case("c1", ["atlas.job_stats"]),
            _case("c2", ["panda_task_status"], hard=True),
        )
        report = sa.evaluate(
            StubPlanner(
                {
                    "question c1": _plan(["atlas.job_stats", "code_query"]),
                    "question c2": _plan(["panda_log_analysis"]),
                }
            ),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        assert report.accuracy() == 0.5
        assert report.accuracy(hard_only=True) == 0.0
        assert report.over_proposal() == pytest.approx(0.25)  # 0.5 and 0.0

    def test_confidence_is_recorded(self) -> None:
        """Decision E-31: whether the model knew it was guessing is free."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["atlas.job_stats"], confidence=0.4)}),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        assert report.mean_confidence() == pytest.approx(0.4)

    def test_a_failed_call_contributes_no_confidence(self) -> None:
        """Averaging a missing confidence as zero would invent an opinion."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": TimeoutError("down")}), corpus, CATALOGUE, repeats=1
        )
        assert report.mean_confidence() is None

    def test_unanimity_names_the_flaky_corpus_share(self) -> None:
        """A variance figure says flakiness exists; this says how much."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]), _case("c2", ["code_query"]))
        planner = StubPlanner(
            {
                "question c1": _plan(["atlas.job_stats"]),
                "question c2": [_plan(["code_query"]), _plan(["panda_log_analysis"])],
            }
        )
        report = sa.evaluate(planner, corpus, CATALOGUE, repeats=2)
        assert report.unanimity() == 0.5
        assert report.per_repeat_accuracy() == [1.0, 0.5]
        assert report.stddev() == pytest.approx(0.353553, abs=1e-5)

    def test_a_single_repeat_has_no_unanimity_or_stddev(self) -> None:
        """Both are undefined rather than perfect with one observation."""
        corpus = _corpus(_case("c1", ["atlas.job_stats"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["atlas.job_stats"])}),
            corpus,
            CATALOGUE,
            repeats=1,
        )
        assert report.unanimity() is None
        assert report.stddev() is None

    def test_labels_outside_the_catalogue_are_reported(self) -> None:
        """Otherwise a catalogue that lost a tool reads as a model failure."""
        corpus = _corpus(_case("c1", ["atlas.gone_missing"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["code_query"])}), corpus, CATALOGUE, repeats=1
        )
        assert report.expected_outside_catalogue == ("atlas.gone_missing",)


class TestBudget:
    """Decision E-30: a run that hits a limit stops; it does not report."""

    def test_the_call_budget_stops_the_run(self, tmp_path: Path) -> None:
        """And raises rather than returning a report, so nothing can aggregate it."""
        corpus = _corpus(*[_case(f"c{i}", ["code_query"]) for i in range(4)])
        planner = StubPlanner({f"question c{i}": _plan(["code_query"]) for i in range(4)})
        with pytest.raises(BudgetExceeded, match="call budget"):
            sa.evaluate(
                planner,
                corpus,
                CATALOGUE,
                repeats=1,
                ledger_file=tmp_path / "l.jsonl",
                budget=Budget(max_calls=2, max_seconds=None),
            )
        assert len(planner.calls) == 2

    def test_what_was_paid_for_is_kept(self, tmp_path: Path) -> None:
        """The budget bounds the cost; it does not discard it."""
        path = tmp_path / "l.jsonl"
        corpus = _corpus(*[_case(f"c{i}", ["code_query"]) for i in range(4)])
        planner = StubPlanner({f"question c{i}": _plan(["code_query"]) for i in range(4)})
        with pytest.raises(BudgetExceeded):
            sa.evaluate(
                planner,
                corpus,
                CATALOGUE,
                repeats=1,
                ledger_file=path,
                budget=Budget(max_calls=2, max_seconds=None),
            )
        assert len(ledger.read(path)) == 2

    def test_the_time_budget_stops_the_run(self) -> None:
        """Checked before a call, so the limit bounds spending rather than
        recording that it was overspent."""
        corpus = _corpus(_case("c1", ["code_query"]))
        with pytest.raises(BudgetExceeded, match="time budget"):
            sa.evaluate(
                StubPlanner({"question c1": _plan(["code_query"])}),
                corpus,
                CATALOGUE,
                repeats=1,
                budget=Budget(max_calls=None, max_seconds=0.0),
            )

    def test_a_dead_gateway_stops_the_run_rather_than_scoring_zero(self) -> None:
        """Five errors in a row is a gateway, not a model, and a run that keeps
        going reports a confident 0.000 for it."""
        corpus = _corpus(*[_case(f"c{i}", ["code_query"]) for i in range(10)])
        planner = StubPlanner(
            {f"question c{i}": ConnectionError("refused") for i in range(10)}
        )
        with pytest.raises(BudgetExceeded, match="consecutive"):
            sa.evaluate(
                planner, corpus, CATALOGUE, repeats=1, max_consecutive_errors=5
            )
        assert len(planner.calls) == 5

    def test_scattered_errors_do_not_stop_a_run(self) -> None:
        """One call in fifty dropping is a fact about the measurement."""
        corpus = _corpus(*[_case(f"c{i}", ["code_query"]) for i in range(4)])
        script: dict[str, Any] = {
            f"question c{i}": _plan(["code_query"]) for i in range(4)
        }
        script["question c1"] = ConnectionError("blip")
        report = sa.evaluate(
            StubPlanner(script), corpus, CATALOGUE, repeats=1, max_consecutive_errors=2
        )
        assert report.counts()["error"] == 1
        assert report.accuracy() == 0.75


class TestResume:
    """A 1,800-call run must survive a dropped VPN."""

    def test_a_resumed_run_makes_only_the_missing_calls(self, tmp_path: Path) -> None:
        """The ledger is what makes the budget guard affordable."""
        path = tmp_path / "l.jsonl"
        corpus = _corpus(*[_case(f"c{i}", ["code_query"]) for i in range(4)])
        script = {f"question c{i}": _plan(["code_query"]) for i in range(4)}
        with pytest.raises(BudgetExceeded):
            sa.evaluate(
                StubPlanner(script),
                corpus,
                CATALOGUE,
                repeats=1,
                ledger_file=path,
                budget=Budget(max_calls=2, max_seconds=None),
            )
        second = StubPlanner(script)
        report = sa.evaluate(
            second, corpus, CATALOGUE, repeats=1, ledger_file=path, resume=True
        )
        assert len(second.calls) == 2
        assert report.resumed == 2
        assert report.accuracy() == 1.0

    def test_without_the_flag_nothing_is_reused(self, tmp_path: Path) -> None:
        """A rerun that answered itself out of a file would measure the file."""
        path = tmp_path / "l.jsonl"
        corpus = _corpus(_case("c1", ["code_query"]))
        script = {"question c1": _plan(["code_query"])}
        sa.evaluate(StubPlanner(script), corpus, CATALOGUE, repeats=1, ledger_file=path)
        second = StubPlanner(script)
        sa.evaluate(second, corpus, CATALOGUE, repeats=1, ledger_file=path)
        assert len(second.calls) == 1

    def test_errors_can_be_asked_again(self, tmp_path: Path) -> None:
        """What stopped the first run decides, so the caller chooses."""
        path = tmp_path / "l.jsonl"
        corpus = _corpus(_case("c1", ["code_query"]))
        sa.evaluate(
            StubPlanner({"question c1": ConnectionError("blip")}),
            corpus,
            CATALOGUE,
            repeats=1,
            ledger_file=path,
        )
        second = StubPlanner({"question c1": _plan(["code_query"])})
        report = sa.evaluate(
            second,
            corpus,
            CATALOGUE,
            repeats=1,
            ledger_file=path,
            resume=True,
            retry_outcomes=frozenset({"error"}),
        )
        assert len(second.calls) == 1
        assert report.accuracy() == 1.0

    def test_a_resumed_call_is_reclassified_not_trusted(self, tmp_path: Path) -> None:
        """The resolution rule is part of the measurement, so a resumed run
        must not carry half its classifications from an older rule."""
        path = tmp_path / "l.jsonl"
        ledger.append(
            path,
            [
                ledger.LedgerEntry(
                    case_id="c1",
                    repeat=0,
                    model="",
                    outcome="wrong_tool",
                    proposed=("core_dump_analysis",),
                    resolved=(),
                )
            ],
        )
        corpus = _corpus(_case("c1", ["atlas.core_dump_analysis"]))
        report = sa.evaluate(
            StubPlanner({}), corpus, CATALOGUE, repeats=1, ledger_file=path, resume=True
        )
        assert report.observations[0].outcome == "correct"


class TestRecords:
    """What reaches the store."""

    def _records(self, model: str = "gpt-oss-20b") -> list[Any]:
        """Measure a small corpus and convert it to rows.

        Args:
            model: Model identifier to record.

        Returns:
            List[Any]: The rows.
        """
        corpus = _corpus(
            _case("c1", ["atlas.job_stats"]),
            _case("c2", ["panda_task_status"], hard=True),
            _case("c3", ["code_query"]),
        )
        planner = StubPlanner(
            {
                "question c1": _plan(["job_stats"]),  # ambiguous suffix
                "question c2": _plan([], route="RETRIEVE"),
                "question c3": _plan(["code_query", "panda_log_analysis"]),
            }
        )
        report = sa.evaluate(planner, corpus, CATALOGUE, model=model, repeats=2)
        return sa.report_to_records(
            report,
            corpus,
            RunContext.capture("0.1.0"),
            "7e3891f672ac",
            "guidance12",
            {"namespace": "atlas"},
        )

    def test_the_counters_partition_the_observations(self) -> None:
        """Six buckets, every call in exactly one, and a reader can check it."""
        row = next(r for r in self._records() if r.metric == sa.METRIC_NAME and r.slice == "all")
        total = (
            row.n_pass
            + row.n_fail
            + row.n_declined
            + row.n_unknown_tool
            + row.n_unparseable
            + row.n_error
        )
        assert total == row.n_cases * row.repeats == 6

    def test_the_new_counters_are_schema_two(self) -> None:
        """A count that lives in a text field is a count nobody queries."""
        row = next(r for r in self._records() if r.metric == sa.METRIC_NAME)
        assert row.schema_version == SCHEMA_VERSION == 2
        assert row.n_declined == 2 and row.n_unknown_tool == 2

    def test_the_slices_emitted(self) -> None:
        """Accuracy and precision over all, hard and the model."""
        emitted = {(r.metric, r.slice) for r in self._records()}
        assert ("selection_accuracy", "all") in emitted
        assert ("selection_accuracy", "hard") in emitted
        assert ("selection_accuracy", "model:gpt-oss-20b") in emitted
        assert ("selection_over_proposal", "hard") in emitted
        assert ("selection_accuracy_confidence", "all") in emitted
        assert ("selection_name_resolution", "all") in emitted

    def test_no_model_slice_when_no_model_was_selected(self) -> None:
        """A row named after nothing would be a slice nobody can query."""
        emitted = {r.slice for r in self._records(model="")}
        assert not any(s.startswith("model:") for s in emitted)

    def test_the_resolution_row_counts_proposals_not_cases(self) -> None:
        """It bounds how much the headline number owed to the suffix rule."""
        row = next(r for r in self._records() if r.metric == sa.NAME_RESOLUTION_METRIC)
        # Three names per repeat: c1 proposes one, c2 none, c3 two.
        assert row.n_cases == 6
        assert row.n_skipped == 2  # the ambiguous 'job_stats', once per repeat
        assert row.n_pass + row.n_fail == row.n_cases

    def test_every_row_carries_its_fingerprints(self) -> None:
        """A number without them is not comparable to anything."""
        for row in self._records():
            assert row.catalogue_fingerprint == "7e3891f672ac"
            assert row.corpus_sha256 == "deadbeef"
            assert row.repeats == 2


class TestCommandLine:
    """The two paths that decide what a run reports when it cannot measure."""

    def test_a_bare_checkout_skips_and_says_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constraint 4.3: no Bamboo is a stated skip at exit 0, never a zero."""

        def _absent(namespace: str = "atlas") -> list[dict[str, Any]]:
            """Stand in for an uninstalled Bamboo.

            Args:
                namespace: Ignored.

            Returns:
                List[Dict[str, Any]]: Never; this always raises.

            Raises:
                MetricSkipped: Always.
            """
            raise MetricSkipped("bamboo-core is not importable")

        monkeypatch.setattr(production, "collect_catalogue", _absent)
        status = main(["selection-accuracy", "--record", "--results-dir", str(tmp_path)])
        assert status == 0
        rows = store.read(sa.METRIC_NAME, tmp_path)
        assert [r.status for r in rows] == ["skipped"]
        assert rows[0].value is None and rows[0].skip_reason

    def test_a_spent_budget_records_a_failure_and_no_aggregate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decision E-30: a partial measurement presented as a measurement is
        the failure mode this framework exists to prevent.  A budget of zero
        calls is the smallest run that reaches the guard."""
        monkeypatch.setattr(
            production,
            "collect_catalogue",
            lambda namespace="atlas": [{"name": name} for name in CATALOGUE],
        )
        monkeypatch.setattr(production, "catalogue_fingerprint", lambda catalogue: "f" * 12)
        monkeypatch.setattr(production, "routing_rules", lambda plugin_id="atlas": ())
        status = main(
            [
                "selection-accuracy",
                "--record",
                "--results-dir",
                str(tmp_path),
                "--limit",
                "1",
                "--repeats",
                "1",
                "--max-calls",
                "0",
            ]
        )
        assert status == 1
        rows = store.read(sa.METRIC_NAME, tmp_path)
        assert [r.status for r in rows] == ["failed"]
        assert all(r.value is None for r in rows)


class TestEnvironmentSelection:
    """What a run applies before it measures, and what it records."""

    def test_a_malformed_override_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run reporting a configuration it did not have is the whole failure
        mode; silently dropping the argument would produce one.  Checked before
        anything else, so a usage error is never reported as a skip."""
        assert main(["selection-accuracy", "--set-env", "BAMBOO_TOOL_RETRIEVAL"]) == 1
        assert "NAME=VALUE" in capsys.readouterr().err

    def test_the_applied_environment_reaches_the_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The baseline and the narrowed run differ only in the environment, so
        a row that does not carry it cannot tell them apart."""
        monkeypatch.setattr(
            production,
            "collect_catalogue",
            lambda namespace="atlas": [{"name": name} for name in CATALOGUE],
        )
        monkeypatch.setattr(production, "catalogue_fingerprint", lambda catalogue: "f" * 12)
        monkeypatch.setattr(production, "routing_rules", lambda plugin_id="atlas": ())
        monkeypatch.setattr(
            production, "plan", lambda question, **kwargs: _plan(["code_query"])
        )
        status = main(
            [
                "selection-accuracy",
                "--record",
                "--results-dir",
                str(tmp_path),
                "--limit",
                "1",
                "--repeats",
                "1",
                "--model",
                "gpt-oss-20b",
                "--set-env",
                "BAMBOO_TOOL_RETRIEVAL=0",
            ]
        )
        assert status == 0
        row = store.read(sa.METRIC_NAME, tmp_path)[0]
        assert row.planner_model == "gpt-oss-20b"
        assert '"BAMBOO_TOOL_RETRIEVAL":"0"' in row.config
        assert '"LLM_DEFAULT_MODEL":"gpt-oss-20b"' in row.config

    def test_the_environment_is_restored_afterwards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A measurement must not leave the process configured differently."""
        monkeypatch.delenv("BAMBOO_TOOL_RETRIEVAL", raising=False)
        with production.env_overrides({"BAMBOO_TOOL_RETRIEVAL": "0"}) as applied:
            assert applied == {"BAMBOO_TOOL_RETRIEVAL": "0"}
        assert "BAMBOO_TOOL_RETRIEVAL" not in os.environ


class TestEmptySlices:
    """1.0 over no cases is the right answer and the wrong thing to store."""

    def test_no_hard_row_when_no_case_is_hard(self) -> None:
        """A row reading 1.000 over zero cases is indistinguishable from a
        perfect score until someone reads ``n_cases``."""
        corpus = _corpus(_case("c1", ["code_query"]))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["code_query"])}), corpus, CATALOGUE, repeats=1
        )
        records = sa.report_to_records(
            report, corpus, RunContext.capture("0.1.0"), "f" * 12, "g", {}
        )
        assert not [r for r in records if r.slice == "hard"]

    def test_the_hard_row_returns_with_a_hard_case(self) -> None:
        """The slice is omitted when empty, not dropped from the metric."""
        corpus = _corpus(_case("c1", ["code_query"], hard=True))
        report = sa.evaluate(
            StubPlanner({"question c1": _plan(["code_query"])}), corpus, CATALOGUE, repeats=1
        )
        records = sa.report_to_records(
            report, corpus, RunContext.capture("0.1.0"), "f" * 12, "g", {}
        )
        assert [r.metric for r in records if r.slice == "hard"] == [
            sa.METRIC_NAME,
            sa.OVER_PROPOSAL_METRIC,
            sa.CONFIDENCE_METRIC,
        ]
