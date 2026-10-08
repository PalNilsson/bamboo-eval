"""Append-only JSONL result store.

Decision E-17: JSONL in the repository is the system of record; OpenSearch is
an optional mirror of the same flat row.  The reason is the bare-checkout
constraint — OpenSearch needs the CERN VPN, and a result store that cannot
record a run from a laptop is the same kind of fragility the framework exists
to measure.  Git supplies provenance and diffs for nothing, and the volume is
hundreds of rows a year.

Append-only is deliberate.  A result is a historical fact about a commit, a
catalogue and a corpus; rewriting one destroys the only thing that turns
``0.992`` into ``0.992, down from 1.000``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .record import COMPARABLE_SCHEMA_VERSIONS, EvalRecord

#: Where results live when no path is given.  Inside the repository, because
#: the history is meant to be reviewed in a diff alongside the change that
#: moved a number.
DEFAULT_RESULTS_DIR = Path("results")


def store_path(metric: str, results_dir: Path | None = None) -> Path:
    """Return the file a metric's rows are appended to.

    One file per metric rather than one per run, so that a series is a file
    and not a directory listing to be sorted.

    Args:
        metric: Metric name, e.g. ``"tool_retrieval_recall"``.
        results_dir: Directory to use; defaults to :data:`DEFAULT_RESULTS_DIR`.

    Returns:
        Path: The JSONL file for this metric.
    """
    base = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    return base / f"{metric}.jsonl"


def append(records: Iterable[EvalRecord], results_dir: Path | None = None) -> list[Path]:
    """Append records to their metrics' files, creating them if absent.

    Args:
        records: Records to store; they may span several metrics.
        results_dir: Directory to write into; defaults to
            :data:`DEFAULT_RESULTS_DIR`.

    Returns:
        List[Path]: The files written, in first-written order, without
        duplicates.
    """
    written: list[Path] = []
    for record in records:
        path = store_path(record.metric, results_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        if path not in written:
            written.append(path)
    return written


def read(metric: str, results_dir: Path | None = None) -> list[EvalRecord]:
    """Read every stored record for a metric, oldest first.

    Args:
        metric: Metric name.
        results_dir: Directory to read from; defaults to
            :data:`DEFAULT_RESULTS_DIR`.

    Returns:
        List[EvalRecord]: Stored records in file order; empty when the file
        does not exist, since a metric that has never run has no history
        rather than an error.
    """
    path = store_path(metric, results_dir)
    if not path.is_file():
        return []
    records: list[EvalRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(EvalRecord.from_dict(json.loads(line)))
    return records


def history(
    metric: str,
    slice_name: str = "all",
    results_dir: Path | None = None,
) -> list[EvalRecord]:
    """Return one metric's rows for one slice, oldest first.

    Args:
        metric: Metric name.
        slice_name: Which slice to follow.
        results_dir: Directory to read from.

    Returns:
        List[EvalRecord]: Matching records in file order.
    """
    return [r for r in read(metric, results_dir) if r.slice == slice_name]


def comparable(records: Sequence[EvalRecord], against: EvalRecord) -> Iterator[EvalRecord]:
    """Yield the records that can honestly be compared with *against*.

    Comparability is not "same metric and slice".  A score measured on a
    different catalogue, a different corpus or a different model is a different
    measurement wearing the same name, and the whole reason every row carries
    its fingerprints is so that this function can say no.

    Args:
        records: Candidate rows, typically a metric's whole history.
        against: The row being explained.

    Schema versions are checked against
    :data:`~bamboo_eval.record.COMPARABLE_SCHEMA_VERSIONS` rather than for
    equality, so that a purely additive schema change does not silently retire
    the history it did not invalidate.  Anything outside that set is not
    compared.

    Yields:
        EvalRecord: Rows measured under the same conditions, in input order,
        excluding *against* itself and anything that was skipped or failed.
    """
    for record in records:
        if record.run_id == against.run_id or record.status != "ok":
            continue
        same = (
            record.metric == against.metric
            and record.slice == against.slice
            and record.catalogue_fingerprint == against.catalogue_fingerprint
            and record.corpus_sha256 == against.corpus_sha256
            and record.config == against.config
            and record.planner_model == against.planner_model
            and record.judge_model == against.judge_model
            and {record.schema_version, against.schema_version}
            <= COMPARABLE_SCHEMA_VERSIONS
        )
        if same:
            yield record


def describe_change(record: EvalRecord, results_dir: Path | None = None) -> str:
    """Render a record against the most recent comparable one.

    Args:
        record: The row just measured.
        results_dir: Directory holding the history.

    Returns:
        str: A one-line summary — the value alone when nothing comparable
        exists, otherwise the value with its predecessor and the delta.  Says
        so explicitly when earlier rows exist but none is comparable, because
        "no history" and "history that does not apply" mean different things.
    """
    if record.value is None:
        return f"{record.metric}/{record.slice}: {record.status} ({record.skip_reason})"
    stored = read(record.metric, results_dir)
    previous = list(comparable(stored, record))
    if not previous:
        # The record may already be in the store: describe_change is normally
        # called just after append, so its own row would otherwise be counted
        # as an earlier run that happens not to be comparable with itself.
        existing = [
            r
            for r in stored
            if r.slice == record.slice
            and r.status == "ok"
            and r.run_id != record.run_id
        ]
        if existing:
            return (
                f"{record.metric}/{record.slice}: {record.value:.4f} "
                f"(no comparable earlier run; {len(existing)} rows differ in "
                f"catalogue, corpus or configuration)"
            )
        return f"{record.metric}/{record.slice}: {record.value:.4f} (first run)"
    last = previous[-1]
    assert last.value is not None  # comparable() excludes rows without a value
    delta = record.value - last.value
    direction = "up from" if delta > 0 else "down from" if delta < 0 else "unchanged from"
    return (
        f"{record.metric}/{record.slice}: {record.value:.4f} "
        f"({direction} {last.value:.4f} on {last.timestamp}, {delta:+.4f})"
    )
