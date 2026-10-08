"""The stored record and the result store (decisions E-13, E-17).

The store's job is not to hold numbers; it is to refuse to compare numbers
that are not comparable.  Most of what is tested here is that refusal.
"""
from __future__ import annotations

# pylint: disable=use-implicit-booleaness-not-comparison
# `== []` rather than `not ...` is deliberate here: these assertions are about
# a function returning an empty *list*, and the explicit comparison fails
# loudly if it ever starts returning None or a generator.

from pathlib import Path

import pytest

from bamboo_eval import store
from bamboo_eval.record import (
    COMPARABILITY_VERSION,
    SCHEMA_VERSION,
    EvalRecord,
    RunContext,
    canonical_config,
    skipped_record,
)


def _record(**overrides: object) -> EvalRecord:
    """Build a record with sensible defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        EvalRecord: The record.
    """
    base = {
        "run_id": "run-1",
        "timestamp": "2026-10-07T12:00:00+00:00",
        "metric": "tool_retrieval_recall",
        "slice": "all",
        "value": 0.992,
        "n_cases": 120,
        "corpus_name": "tool_selection_corpus",
        "corpus_version": 1,
        "corpus_sha256": "c" * 64,
        "catalogue_fingerprint": "7e3891f672ac",
        "git_commit": "abc1234",
        "host": "aipanda033",
        "framework_version": "0.1.0",
    }
    base.update(overrides)
    return EvalRecord(**base)  # type: ignore[arg-type]


class TestRecordValidation:
    """A row that is not a measurement must say what stopped it."""

    def test_a_skip_without_a_reason_is_rejected(self) -> None:
        """A silent skip leaves a gap in the series that reads as 'no change'."""
        with pytest.raises(ValueError, match="skip_reason"):
            _record(status="skipped", value=None)

    def test_an_ok_row_without_a_value_is_rejected(self) -> None:
        """A row that claims success and carries nothing is not a measurement."""
        with pytest.raises(ValueError, match="ok with no value"):
            _record(value=None)

    def test_a_skipped_record_carries_its_reason(self) -> None:
        """The helper produces a storable row with no value and a stated cause."""
        context = RunContext.capture("0.1.0")
        row = skipped_record(context, "tool_retrieval_recall", "no embedding model")
        assert row.status == "skipped"
        assert row.value is None
        assert "embedding" in row.skip_reason


class TestSerialisation:
    """Rows survive a round trip, and tolerate a newer writer."""

    def test_round_trip(self) -> None:
        """A record reconstructs from its own dict."""
        row = _record()
        assert EvalRecord.from_dict(row.to_dict()) == row

    def test_unknown_fields_are_dropped_not_rejected(self) -> None:
        """An older reader keeps working against a newer writer's rows."""
        data = dict(_record().to_dict(), some_future_field=1)
        assert EvalRecord.from_dict(data).metric == "tool_retrieval_recall"

    def test_schema_version_travels_with_every_row(self) -> None:
        """A reader that does not recognise a version must be able to see it."""
        assert _record().schema_version == SCHEMA_VERSION

    def test_config_is_canonical(self) -> None:
        """Identical settings in different key order compare equal."""
        assert canonical_config({"k": 10, "a": 1}) == canonical_config({"a": 1, "k": 10})


class TestStore:
    """Append, read, and the comparability rules."""

    def test_append_and_read(self, tmp_path: Path) -> None:
        """Rows land in a per-metric file and come back in order."""
        store.append([_record(value=0.9), _record(value=0.95, run_id="run-2")], tmp_path)
        rows = store.read("tool_retrieval_recall", tmp_path)
        assert [r.value for r in rows] == [0.9, 0.95]

    def test_reading_a_metric_that_never_ran_is_empty_not_an_error(
        self, tmp_path: Path
    ) -> None:
        """No history is information, not a failure."""
        assert store.read("selection_accuracy", tmp_path) == []

    def test_rows_for_other_slices_are_not_mixed_in(self, tmp_path: Path) -> None:
        """``all`` and ``hard`` are separate series in one file."""
        store.append([_record(), _record(slice="hard", value=0.983)], tmp_path)
        assert len(store.history("tool_retrieval_recall", "hard", tmp_path)) == 1

    def test_a_different_catalogue_is_not_comparable(self) -> None:
        """The headline case: 0.983 and 0.992 were two catalogues, not noise."""
        old = _record(catalogue_fingerprint="251039736f13", value=0.983)
        new = _record(run_id="run-2", catalogue_fingerprint="8f67488c6d95", value=0.992)
        assert list(store.comparable([old], new)) == []

    def test_a_different_corpus_is_not_comparable(self) -> None:
        """Relabelling the corpus changes what the number means."""
        old = _record(corpus_sha256="d" * 64)
        new = _record(run_id="run-2")
        assert list(store.comparable([old], new)) == []

    def test_a_different_config_is_not_comparable(self) -> None:
        """k=10 and k=12 are different measurements wearing one metric name."""
        old = _record(config=canonical_config({"k": 10}))
        new = _record(run_id="run-2", config=canonical_config({"k": 12}))
        assert list(store.comparable([old], new)) == []

    def test_skipped_rows_are_never_compared_against(self) -> None:
        """A skip is not a worse score."""
        skipped = _record(status="skipped", value=None, skip_reason="no model")
        assert list(store.comparable([skipped], _record(run_id="run-2"))) == []

    def test_describe_change_reports_the_delta(self, tmp_path: Path) -> None:
        """The version that catches problems: a number with its predecessor."""
        store.append([_record(value=1.0)], tmp_path)
        line = store.describe_change(_record(run_id="run-2", value=0.992), tmp_path)
        assert "down from 1.0000" in line
        assert "-0.0080" in line

    def test_describe_change_distinguishes_no_history_from_no_match(
        self, tmp_path: Path
    ) -> None:
        """'Never measured' and 'measured under other conditions' differ.

        Collapsing the two is how a change of catalogue gets read as a fresh
        start rather than as the reason the comparison is missing.
        """
        assert "first run" in store.describe_change(_record(), tmp_path)
        store.append([_record(catalogue_fingerprint="251039736f13")], tmp_path)
        line = store.describe_change(_record(run_id="run-2"), tmp_path)
        assert "no comparable earlier run" in line

    def test_describe_change_on_a_skip_states_the_reason(self, tmp_path: Path) -> None:
        """A skipped row renders as its reason, not as a missing number."""
        row = _record(status="skipped", value=None, skip_reason="no embedding model")
        assert "no embedding model" in store.describe_change(row, tmp_path)

    def test_a_rows_own_presence_is_not_counted_as_history(self, tmp_path: Path) -> None:
        """describe_change is called after append, so the row sees itself.

        Found by running the CLI: the first recorded run reported "no
        comparable earlier run; 1 rows differ", where the one row was itself.
        """
        row = _record()
        store.append([row], tmp_path)
        assert "first run" in store.describe_change(row, tmp_path)


class TestComparabilityVersion:
    """A row gaining a field is not a measurement changing."""

    def test_a_pre_existing_row_still_compares(self, tmp_path: Path) -> None:
        """Schema 1 rows carry no comparability version and read back as 1.

        Phase 1 added two counters no earlier row populated and changed nothing
        about what recall means, so retiring the phase 0 series over it would
        have thrown away history that is still valid.
        """
        old = _record(schema_version=1)
        stored = old.to_dict()
        del stored["comparability_version"]
        store.append([EvalRecord.from_dict(stored)], tmp_path)
        line = store.describe_change(_record(run_id="run-2", value=0.95), tmp_path)
        assert "down from" in line

    def test_a_changed_measurement_refuses_to_compare(self, tmp_path: Path) -> None:
        """Which is the point: the alternative is two incomparable numbers on
        one axis, with nothing in the row to show it."""
        store.append([_record(comparability_version=1)], tmp_path)
        line = store.describe_change(
            _record(run_id="run-2", comparability_version=2), tmp_path
        )
        assert "no comparable earlier run" in line

    def test_the_version_travels_with_every_row(self) -> None:
        """A row that does not carry it cannot be refused later."""
        assert _record().to_dict()["comparability_version"] == COMPARABILITY_VERSION
