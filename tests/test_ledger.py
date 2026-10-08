"""The ledger exists to survive a run not finishing, so it is tested that way.

Decision E-29.  Every test here is about a run that stopped: killed mid-write,
repeated without being told to resume, or resumed after a gateway dropped a
call.  Round-tripping a well-formed file is the easy half.
"""
from __future__ import annotations

# pylint: disable=too-few-public-methods
# The test classes are namespaces grouping related assertions, not objects.

from pathlib import Path

import pytest

from bamboo_eval import ledger
from bamboo_eval.errors import BambooEvalError


def _entry(case_id: str, repeat: int = 0, outcome: str = "correct") -> ledger.LedgerEntry:
    """Build an entry with the fields a test cares about.

    Args:
        case_id: Case identifier.
        repeat: Repeat index.
        outcome: Recorded outcome.

    Returns:
        ledger.LedgerEntry: The entry.
    """
    return ledger.LedgerEntry(
        case_id=case_id,
        repeat=repeat,
        model="m1",
        outcome=outcome,
        proposed=("atlas.job_stats",),
        resolved=("atlas.job_stats",),
        confidence=0.8,
    )


class TestPath:
    """Where a ledger lives."""

    def test_the_catalogue_is_in_the_file_name(self, tmp_path: Path) -> None:
        """Resuming across catalogues would merge two measurements."""
        path = ledger.ledger_path("selection_accuracy", "7e3891f672ac", tmp_path)
        assert path.name == "selection_accuracy-7e3891f672ac.jsonl"
        assert path.parent == tmp_path / "ledger"

    def test_an_unknown_fingerprint_still_names_a_file(self, tmp_path: Path) -> None:
        """A missing fingerprint must not produce a path ending in a dash."""
        assert ledger.ledger_path("m", "", tmp_path).name == "m-unknown.jsonl"


class TestRoundTrip:
    """What is written is what is read."""

    def test_entries_survive_the_file(self, tmp_path: Path) -> None:
        """Including the tuple fields, which JSON has no type for."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1"), _entry("c2")])
        back = ledger.read(path)
        assert [e.case_id for e in back] == ["c1", "c2"]
        assert back[0].proposed == ("atlas.job_stats",)
        assert back[0].confidence == 0.8

    def test_reading_a_ledger_that_does_not_exist_is_not_an_error(
        self, tmp_path: Path
    ) -> None:
        """A run that has not started has no ledger, not a broken one."""
        assert not ledger.read(tmp_path / "absent.jsonl")

    def test_unknown_fields_are_dropped_not_rejected(self, tmp_path: Path) -> None:
        """A ledger from a newer version stays resumable for the shared fields."""
        path = tmp_path / "l.jsonl"
        path.write_text(
            '{"case_id": "c1", "repeat": 0, "model": "m1", "outcome": "correct", '
            '"something_new": 1}\n',
            encoding="utf-8",
        )
        assert ledger.read(path)[0].case_id == "c1"


class TestInterruption:
    """A killed run and a corrupted file must not look alike."""

    def test_a_truncated_last_line_is_dropped(self, tmp_path: Path) -> None:
        """That is what a process killed mid-write leaves behind."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1")])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"case_id": "c2", "repeat": 0, "mod')
        entries = ledger.read(path)
        assert [e.case_id for e in entries] == ["c1"]

    def test_a_broken_earlier_line_is_refused(self, tmp_path: Path) -> None:
        """Nothing this program does produces one, so resuming would be guessing."""
        path = tmp_path / "l.jsonl"
        path.write_text('{"case_id": "c1"\n{"case_id": "c2", "repeat": 0, '
                        '"model": "m", "outcome": "correct"}\n', encoding="utf-8")
        with pytest.raises(BambooEvalError, match="corrupt at line 1"):
            ledger.read(path)


class TestResume:
    """Which calls a resumed run may skip."""

    def test_the_last_write_for_a_key_wins(self, tmp_path: Path) -> None:
        """Append-only semantics: the newest classification of a call."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1", outcome="error")])
        ledger.append(path, [_entry("c1", outcome="correct")])
        latest = ledger.latest_by_key(ledger.read(path))
        assert latest[("c1", 0, "m1")].outcome == "correct"

    def test_completed_keys_covers_every_recorded_call(self, tmp_path: Path) -> None:
        """By default nothing is retried: an outcome retried enough is selected for."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1"), _entry("c1", repeat=1, outcome="error")])
        assert ledger.completed_keys(path) == {("c1", 0, "m1"), ("c1", 1, "m1")}

    def test_requested_outcomes_are_left_to_be_remade(self, tmp_path: Path) -> None:
        """A dropped VPN leaves errors that are worth asking again."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1"), _entry("c1", repeat=1, outcome="error")])
        keys = ledger.completed_keys(path, frozenset({"error"}))
        assert keys == {("c1", 0, "m1")}

    def test_a_key_retried_into_success_is_complete_again(self, tmp_path: Path) -> None:
        """The last row decides, so a successful retry is not retried forever."""
        path = tmp_path / "l.jsonl"
        ledger.append(path, [_entry("c1", outcome="error"), _entry("c1")])
        assert ledger.completed_keys(path, frozenset({"error"})) == {("c1", 0, "m1")}
