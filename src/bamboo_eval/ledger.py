"""Per-call working state for metrics that drive a model (decision E-29).

The result store holds *measurements*: one row per run, metric and slice, and
it is committed because the history is the point.  This is the other thing a
model-driven run produces — one row per call, carrying the plan that came back
and how it was classified.  It is an order of magnitude larger, it is of no
interest once the aggregate exists, and it is what makes a 1,800-call run
resumable after a dropped VPN rather than something to start again.

So it lives beside the store and not in it:
``results/ledger/<metric>-<catalogue_fingerprint>.jsonl``, gitignored, with the
fingerprint in the name because rows measured against a different catalogue are
not rows the same run may resume from.

Append-only, one JSON object per line, flushed as each call returns.  Two
properties follow from writing it that way and both are relied on:

* A run killed mid-write leaves a truncated last line.  :func:`read` drops a
  trailing unparseable line and refuses the file on any other, because the one
  is a crash and the other is corruption.
* A key may appear more than once — a resumed run that was not told to resume,
  say.  The last occurrence wins, which matches append-only semantics and
  keeps the newest classification of a call.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import BambooEvalError
from .store import DEFAULT_RESULTS_DIR

#: Identifies one call within a run: the case asked, which repeat it was, and
#: which model answered.  A resumed run skips the keys it already holds.
LedgerKey = tuple[str, int, str]


@dataclass(frozen=True)
class LedgerEntry:
    """One call to a model, and what came back.

    Attributes:
        case_id: The corpus case asked.
        repeat: Zero-based repeat index, since a non-deterministic metric asks
            the same question more than once (decision E-2).
        model: Model identifier, as recorded in ``planner_model``.
        outcome: The classification, e.g. ``"correct"``, ``"declined"``.
        proposed: Tool names exactly as the model gave them, before any
            normalisation.  Kept raw so that a resolution rule can be reviewed
            after the fact against what was actually said.
        resolved: Catalogue names the proposals resolved to, in order.
        unresolved: Proposals that resolved to nothing, with why — an invented
            name or an ambiguous suffix.
        confidence: The plan's own confidence, or ``None`` when there was no
            plan to read it from.
        duration_s: Wall-clock seconds for this call.
        timestamp: UTC, ISO 8601, second precision.
        detail: The returned plan as text, or the error, truncated by the
            caller.  A failure nobody can read afterwards gets rediscovered.
    """

    case_id: str
    repeat: int
    model: str
    outcome: str
    proposed: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    confidence: float | None = None
    duration_s: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    detail: str = ""

    @property
    def key(self) -> LedgerKey:
        """Return the key identifying this call.

        Returns:
            LedgerKey: ``(case_id, repeat, model)``.
        """
        return (self.case_id, self.repeat, self.model)

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a JSON-serialisable mapping.

        Returns:
            Dict[str, Any]: One JSONL line's content, tuples flattened to
            lists.
        """
        data = asdict(self)
        for name in ("proposed", "resolved", "unresolved"):
            data[name] = list(data[name])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LedgerEntry":
        """Rebuild an entry from a stored mapping.

        Unknown fields are dropped rather than rejected, so a ledger written by
        a newer version stays resumable for the fields both versions share.

        Args:
            data: A stored entry.

        Returns:
            LedgerEntry: The rebuilt entry.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        for name in ("proposed", "resolved", "unresolved"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)


def ledger_path(
    metric: str,
    catalogue_fingerprint: str,
    results_dir: Path | None = None,
) -> Path:
    """Return the ledger file for one metric against one catalogue.

    Args:
        metric: Metric name, e.g. ``"selection_accuracy"``.
        catalogue_fingerprint: Digest of the catalogue measured, abbreviated by
            the caller as it is elsewhere.  In the file name rather than in the
            rows because resuming across catalogues would mix two measurements,
            and a path is harder to ignore than a field.
        results_dir: Directory holding the result store; defaults to
            :data:`~bamboo_eval.store.DEFAULT_RESULTS_DIR`.

    Returns:
        Path: The ledger file, which need not exist yet.
    """
    base = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    return base / "ledger" / f"{metric}-{catalogue_fingerprint or 'unknown'}.jsonl"


def append(path: Path, entries: Iterable[LedgerEntry]) -> None:
    """Append entries to a ledger, creating it and its directory if absent.

    Written and flushed per call rather than at the end of a run: the file
    exists to survive the run not finishing.

    Args:
        path: The ledger file.
        entries: Entries to append, in call order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
            handle.flush()


def read(path: Path) -> list[LedgerEntry]:
    """Read every entry from a ledger, oldest first.

    Args:
        path: The ledger file.

    Returns:
        List[LedgerEntry]: Entries in file order; empty when the file does not
        exist, because a run that has not started has no ledger rather than a
        broken one.

    Raises:
        BambooEvalError: If a line other than the last is unreadable.  A
            truncated final line is a run that was killed mid-write and is
            dropped; anything earlier means the file was corrupted by something
            other than this program, and silently skipping it would resume a
            run from a state nobody can account for.
    """
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[LedgerEntry] = []
    for index, line in enumerate(lines):
        try:
            entries.append(LedgerEntry.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if index == len(lines) - 1:
                break  # killed mid-write; the call it describes is simply re-made
            raise BambooEvalError(
                f"{path} is corrupt at line {index + 1} of {len(lines)}: {exc}. "
                f"Only a truncated final line is tolerated, because only that "
                f"one is explained by a run being killed."
            ) from exc
    return entries


def latest_by_key(entries: Iterable[LedgerEntry]) -> dict[LedgerKey, LedgerEntry]:
    """Return one entry per key, keeping the last occurrence.

    Args:
        entries: Entries in file order.

    Returns:
        Dict[LedgerKey, LedgerEntry]: The newest entry for each key.
    """
    return {entry.key: entry for entry in entries}


def completed_keys(
    path: Path,
    retry_outcomes: frozenset[str] = frozenset(),
) -> set[LedgerKey]:
    """Return the calls a resumed run may skip.

    Args:
        path: The ledger file.
        retry_outcomes: Outcomes that do not count as complete, so a resumed
            run makes the call again.  The caller decides rather than this
            module, because whether a transport failure is worth retrying
            depends on why the earlier run stopped — but note that an outcome
            retried enough times is an outcome selected for, so the default is
            to retry nothing.

    Returns:
        Set[LedgerKey]: Keys already answered.
    """
    return {
        key
        for key, entry in latest_by_key(read(path)).items()
        if entry.outcome not in retry_outcomes
    }
