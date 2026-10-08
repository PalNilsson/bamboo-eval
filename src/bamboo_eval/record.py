"""The stored record: one flat row per measurement, fully fingerprinted.

Decision E-13: a number without its fingerprint is not comparable to anything.
Three catalogue states appeared in one week — 29,527, 29,750 and 30,065
characters — and the same command gave recall 0.983 and 0.992 across two of
them.  That is not retriever noise; it is two different catalogues.  A fourth
state, 29,842 characters, turned up the first time this package was exercised
on a different checkout, which is the rule rather than the exception.

Decision E-17: this record is also the OpenSearch document.  The fields below
are flat and typed so they can be mapped into an index without a dynamic
template, and so that shipping them later is an emitter rather than a
redesign.  Nested objects are deliberately absent; ``config`` is a canonical
JSON *string* for exactly that reason.

The ``status``/``skip_reason`` pair is the bare-checkout constraint expressed
as a field.  A metric that cannot run records why it could not, and never a
number.
"""
from __future__ import annotations

import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

#: Bumped whenever a field is added, removed or re-typed.  A reader that does
#: not recognise a schema version must refuse to compare across it rather than
#: guess, so the version travels with every row.
#:
#: 2 — adds ``n_declined`` and ``n_unknown_tool`` (decision E-28).
SCHEMA_VERSION = 2

#: Schema versions whose rows may be compared with each other.
#:
#: The default rule is equality: a row written under a different schema is a
#: different measurement until someone says otherwise.  This set is that
#: someone, and it is correct only while every version in it differs from the
#: others by *added fields with defaults* — a reader of a version-1 row gets
#: the same answer to the same question from a version-2 row, because the
#: fields version 2 adds are counters version 1 never populated and no
#: existing field changed meaning.
#:
#: Re-typing a field, changing what one counts, or removing one means the new
#: version starts a set of its own.  Widening this set to preserve a series is
#: how two incomparable numbers end up on the same axis, so it is a decision
#: with a comment, not a default.
COMPARABLE_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2})

Status = Literal["ok", "skipped", "failed"]


def canonical_config(config: Mapping[str, Any]) -> str:
    """Serialise a metric's configuration to a stable string.

    Sorted keys and fixed separators, so that two runs with the same settings
    produce byte-identical strings and an index can treat the field as a
    keyword.  A dict would invite a dynamic mapping explosion in OpenSearch,
    and an unsorted dump would make identical configurations compare unequal.

    Args:
        config: The metric's settings.

    Returns:
        str: Canonical JSON.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def _git_commit() -> str:
    """Return the current commit of the checkout this package runs from.

    Returns:
        str: Abbreviated commit hash, suffixed ``+dirty`` when the working tree
        has uncommitted changes, or ``"unknown"`` outside a git checkout.  A
        dirty tree is recorded rather than rejected, because a measurement
        taken mid-change is still worth storing as long as nobody later mistakes
        it for a measurement of the commit.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


@dataclass(frozen=True)
class RunContext:
    """Facts shared by every record a single invocation produces.

    Attributes:
        run_id: Identifies one CLI invocation, so the rows it wrote can be
            recovered as a set.
        timestamp: UTC, ISO 8601, second precision.
        git_commit: Commit of the bamboo-eval checkout, see :func:`_git_commit`.
        host: Where the measurement ran.  Two hosts producing different numbers
            from the same command is a thing that has happened.
        framework_version: Version of this package.
    """

    run_id: str
    timestamp: str
    git_commit: str
    host: str
    framework_version: str

    @classmethod
    def capture(cls, framework_version: str) -> "RunContext":
        """Build a context for the current invocation.

        Args:
            framework_version: Version of this package.

        Returns:
            RunContext: The captured context.
        """
        return cls(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=_git_commit(),
            host=platform.node() or "unknown",
            framework_version=framework_version,
        )


@dataclass(frozen=True)
class EvalRecord:  # pylint: disable=too-many-instance-attributes
    """One measurement of one metric over one slice of one corpus.

    The field count is the schema, not sprawl: every one of them is a
    fingerprint, a count or a provenance fact that a later reader needs in
    order to decide whether this row may be compared with another.

    A record is written per slice rather than per run, so that ``all`` and
    ``hard`` are separate rows and a query for the hard subset does not have to
    know it lives inside another row's payload.

    Attributes:
        run_id: See :class:`RunContext`.
        timestamp: See :class:`RunContext`.
        metric: What was measured, e.g. ``"tool_retrieval_recall"``.
        slice: Which subset, e.g. ``"all"``, ``"hard"``, ``"topic:rucio"``,
            ``"model:gpt-oss-20b"``.
        value: The metric's headline value, or ``None`` when skipped or failed.
        n_cases: Cases in the slice.
        n_pass: Cases that met the metric's criterion.
        n_fail: Cases that did not.
        n_skipped: Cases the metric declined to score.
        n_unparseable: Results rejected by a schema rather than scored wrong.
            Distinct from ``n_fail`` because choosing the wrong tool and
            emitting something unparseable are different defects with different
            fixes, and pooling them hides both.
        n_error: Transport, timeout and other infrastructure failures.
        n_declined: Results that were valid and proposed nothing — a planner
            that returned a well-formed plan with no tool calls declined to
            answer, which is neither a wrong choice nor a malformed one.
            Counted apart because pooling it with either misattributes the
            failure (decision E-28).
        n_unknown_tool: Results naming a tool that is not in the catalogue.
            Inventing a tool and choosing the wrong real one are different
            defects: the first is a prompt or model-grounding problem, the
            second a discrimination problem.
        repeats: Evaluations per case; 1 for a deterministic metric.  The
            counters above count *observations*, so with ``repeats`` above 1
            they sum to ``n_cases * repeats`` rather than to ``n_cases``, and
            ``value`` is the mean over observations rather than
            ``n_pass / n_cases``.
        stddev: Standard deviation of the per-repeat values, or ``None`` when
            deterministic.
        unanimity: Fraction of cases where every repeat agreed, or ``None``
            when deterministic.  More legible than a variance figure: it names
            how much of the corpus is flaky rather than only that flakiness
            exists.
        catalogue_fingerprint: Digest of the indexed tool catalogue.
        corpus_name: Which corpus.
        corpus_version: Its declared version.
        corpus_sha256: Digest of its bytes.
        guidance_version: Digest of the routing guidance in force.
        planner_model: Model under test, or empty for a deterministic metric.
        judge_model: Judge model, empty before phase 5.  Changing it
            invalidates the series, see decision E-20.
        config: Canonical JSON of the metric's settings.
        git_commit: See :class:`RunContext`.
        host: See :class:`RunContext`.
        framework_version: See :class:`RunContext`.
        status: ``ok``, ``skipped`` or ``failed``.
        skip_reason: Required when ``status`` is not ``ok``.
        duration_s: Wall-clock seconds.
        schema_version: See :data:`SCHEMA_VERSION`.
    """

    run_id: str
    timestamp: str
    metric: str
    slice: str
    value: float | None
    n_cases: int
    corpus_name: str
    corpus_version: int
    corpus_sha256: str
    catalogue_fingerprint: str
    git_commit: str
    host: str
    framework_version: str
    status: Status = "ok"
    n_pass: int = 0
    n_fail: int = 0
    n_skipped: int = 0
    n_unparseable: int = 0
    n_error: int = 0
    n_declined: int = 0
    n_unknown_tool: int = 0
    repeats: int = 1
    stddev: float | None = None
    unanimity: float | None = None
    guidance_version: str = ""
    planner_model: str = ""
    judge_model: str = ""
    config: str = "{}"
    skip_reason: str = ""
    duration_s: float = 0.0
    schema_version: int = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Reject records that would be unreadable later.

        Raises:
            ValueError: If a non-``ok`` record carries no reason, or an ``ok``
                record carries no value.  Both produce rows that look like
                measurements and are not.
        """
        if self.status != "ok" and not self.skip_reason:
            raise ValueError(
                f"record for {self.metric}/{self.slice} has status {self.status!r} "
                f"and no skip_reason; a result that is not a measurement must "
                f"say what stopped it"
            )
        if self.status == "ok" and self.value is None:
            raise ValueError(
                f"record for {self.metric}/{self.slice} is ok with no value"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a flat JSON-serialisable mapping.

        Returns:
            Dict[str, Any]: One OpenSearch document, one JSONL line.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRecord":
        """Rebuild a record from a stored mapping.

        Unknown fields are dropped rather than rejected, so a newer writer's
        rows remain readable by an older reader for the fields they share.

        Args:
            data: A stored record.

        Returns:
            EvalRecord: The rebuilt record.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _non_measurement_record(
    context: RunContext,
    metric: str,
    reason: str,
    status: Status,
    corpus_name: str = "",
    slice_name: str = "all",
) -> EvalRecord:
    """Build a record that stands in for a measurement that did not happen.

    Args:
        context: The invocation's shared facts.
        metric: Which metric did not produce a number.
        reason: Why.
        status: ``"skipped"`` or ``"failed"``.
        corpus_name: The corpus it would have used, where known.
        slice_name: The slice it would have reported.

    Returns:
        EvalRecord: A record with no value and a stated reason.
    """
    return EvalRecord(
        run_id=context.run_id,
        timestamp=context.timestamp,
        metric=metric,
        slice=slice_name,
        value=None,
        n_cases=0,
        corpus_name=corpus_name,
        corpus_version=0,
        corpus_sha256="",
        catalogue_fingerprint="",
        git_commit=context.git_commit,
        host=context.host,
        framework_version=context.framework_version,
        status=status,
        skip_reason=reason,
    )


def skipped_record(
    context: RunContext,
    metric: str,
    reason: str,
    corpus_name: str = "",
    slice_name: str = "all",
) -> EvalRecord:
    """Build the record a metric writes when it cannot run.

    Recording a skip is not bookkeeping.  A metric that silently stops running
    leaves a gap in the series that reads as "nothing changed", which is how an
    embedding backend can be broken for a fortnight without anyone noticing.

    Args:
        context: The invocation's shared facts.
        metric: Which metric was skipped.
        reason: Why, in terms a later reader will understand.
        corpus_name: The corpus it would have used, where known.
        slice_name: The slice it would have reported.

    Returns:
        EvalRecord: A record with ``status="skipped"`` and no value.
    """
    return _non_measurement_record(
        context, metric, reason, "skipped", corpus_name, slice_name
    )


def failed_record(
    context: RunContext,
    metric: str,
    reason: str,
    corpus_name: str = "",
    slice_name: str = "all",
) -> EvalRecord:
    """Build the record a metric writes when a run stopped part-way.

    A skip says the measurement could not be attempted; a failure says it was
    attempted and abandoned.  Decision E-30 keeps them apart because a run that
    spent its budget, or whose planner errored on every call, has partial data
    — and the one thing that must not happen is for that partial data to be
    aggregated and stored as though it were the corpus.

    Args:
        context: The invocation's shared facts.
        metric: Which metric was abandoned.
        reason: What stopped it, naming the limit or the error.
        corpus_name: The corpus it was measuring.
        slice_name: The slice it would have reported.

    Returns:
        EvalRecord: A record with ``status="failed"`` and no value.
    """
    return _non_measurement_record(
        context, metric, reason, "failed", corpus_name, slice_name
    )
