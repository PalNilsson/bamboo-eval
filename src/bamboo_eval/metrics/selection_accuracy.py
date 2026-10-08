"""End-task selection accuracy: shown the catalogue, did the planner choose?

Retrieval shipped on a measured claim about *shortlisting* — recall 1.000 over
120 labelled questions at 39% of the former prompt size.  That claim says
nothing about whether the planner, given the narrowed catalogue, then proposed
the tool a correct plan needs.  This metric is the conversion of "we
shortlisted correctly" into "narrowing did not degrade the task", which is the
claim a reader wants and the first thing a referee asks about.

The corpus is already labelled for it; no new corpus exists or is needed.

Four things here are deliberate and each exists because the obvious
alternative loses information:

* **Six outcomes, not four** (decision E-28).  A plan that validates with
  ``route=RETRIEVE`` and no tool calls is a planner that *declined*; a plan
  naming something absent from the catalogue is a planner that *invented*.
  Neither is a wrong choice and neither is malformed output, and pooling them
  with either hides a different defect with a different fix.
* **Names are resolved, and the resolution is counted** (decision E-27).  The
  live catalogue carries both ``atlas.job_stats`` and ``panda_task_status``,
  and the plan schema says a tool call may use either form.  Under naive
  string equality a substantively correct answer scores as a miss, which
  deflates the headline number silently — the worst direction for an error to
  go.  So a suffix match is accepted, a suffix matching two catalogue entries
  is refused as ambiguous, and the counts of both are reported next to the
  number they could have moved.
* **Two numbers, never one** (decision E-1).  Containment and precision are
  reported separately.  An F1 would make "right tool plus two others" and
  "wrong tool" the same figure, and they are different failures with different
  costs.
* **Unanimity, not only a standard deviation** (decision E-2).  A variance
  figure says flakiness exists; unanimity says how much of the corpus is flaky,
  and the per-case ledger says which cases.
"""
from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .. import ledger as ledger_mod
from ..budget import Budget
from ..corpus import Corpus, ToolSelectionCase
from ..errors import BudgetExceeded, MetricSkipped, PlanParseError
from ..record import EvalRecord, RunContext, canonical_config

#: Name under which this metric's headline rows are stored.
METRIC_NAME = "selection_accuracy"

#: Precision over the tools proposed; 1.0 means nothing superfluous was named.
OVER_PROPOSAL_METRIC = "selection_over_proposal"

#: Mean of the planner's own stated confidence (decision E-31).
CONFIDENCE_METRIC = "selection_accuracy_confidence"

#: How often a proposed name needed normalising before it could be scored
#: (decision E-27).  A row rather than a log line, because a count that lives
#: in prose is a count nobody queries — and this one bounds how much the
#: headline number depends on a resolution rule.
NAME_RESOLUTION_METRIC = "selection_name_resolution"

#: Production symbols this metric calls; a test asserts each is declared in
#: :data:`bamboo_eval.production.ENTRY_POINTS` (decision E-12).
PRODUCTION_ENTRY_POINTS = (
    "bamboo_plan_tool",
    "_collect_tool_catalog",
    "catalog_fingerprint",
)

Outcome = Literal[
    "correct", "wrong_tool", "declined", "unknown_tool", "unparseable", "error"
]

#: The six buckets, in reporting order.  They partition the observations: every
#: call lands in exactly one, so the stored counters sum to
#: ``n_cases * repeats`` and a reader can check that they do.
OUTCOMES: tuple[Outcome, ...] = (
    "correct",
    "wrong_tool",
    "declined",
    "unknown_tool",
    "unparseable",
    "error",
)

#: How a proposed name was matched to the catalogue.
How = Literal["exact", "suffix", "ambiguous", "unknown"]

#: A planner, as this metric needs it: question in, parsed plan out.  Injected
#: rather than imported so that every outcome, the resolution table, resume and
#: the budget guard are exercised by tests with no gateway in sight; the CLI
#: passes a closure over :func:`bamboo_eval.production.plan`, which is the
#: production entry point decision E-25 names.
PlannerFn = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class Resolution:
    """One proposed tool name, matched against the catalogue.

    Attributes:
        proposed: The name as the planner gave it, unmodified.
        resolved: The catalogue name it stands for, or ``None`` when it stands
            for nothing the catalogue offers.
        how: ``exact``, ``suffix`` for a bare/namespaced pair, ``ambiguous``
            when a suffix matched more than one catalogue entry, ``unknown``
            when nothing matched.
    """

    proposed: str
    resolved: str | None
    how: How


class NameResolver:
    """Matches proposed tool names to catalogue names (decision E-27).

    Exact match first.  Failing that, the segment after the last ``.`` is
    compared with the same segment of each catalogue name, which is what makes
    ``task_status`` and ``atlas.task_status`` the same answer.  A suffix that
    matches two catalogue entries is refused rather than guessed: resolving it
    would mean the headline number depended on which of two tools the resolver
    happened to prefer.
    """

    def __init__(self, catalogue_names: Sequence[str]) -> None:
        """Build a resolver for one catalogue.

        Args:
            catalogue_names: Every name in the catalogue the planner was shown.
        """
        self.names: tuple[str, ...] = tuple(catalogue_names)
        self._exact = set(self.names)
        self._by_suffix: dict[str, list[str]] = {}
        for name in self.names:
            self._by_suffix.setdefault(name.rsplit(".", 1)[-1], []).append(name)

    def resolve(self, proposed: str) -> Resolution:
        """Match one proposed name.

        Args:
            proposed: The name a plan used.

        Returns:
            Resolution: What it resolved to, and how.
        """
        name = proposed.strip()
        if name in self._exact:
            return Resolution(proposed, name, "exact")
        candidates = self._by_suffix.get(name.rsplit(".", 1)[-1], [])
        if len(candidates) == 1:
            return Resolution(proposed, candidates[0], "suffix")
        if len(candidates) > 1:
            return Resolution(proposed, None, "ambiguous")
        return Resolution(proposed, None, "unknown")


@dataclass(frozen=True)
class Observation:
    """One call: one case, one repeat, one model.

    Attributes:
        case: The case asked.
        repeat: Zero-based repeat index.
        model: Model identifier, empty when the deployment's own selection was
            left in force.
        outcome: Which of :data:`OUTCOMES` this call landed in.
        resolutions: Every proposed name with its resolution, in plan order.
        confidence: The plan's stated confidence, or ``None``.
        duration_s: Wall-clock seconds for the call.
        detail: The plan as text, or the error, for the ledger.
    """

    case: ToolSelectionCase
    repeat: int
    model: str
    outcome: Outcome
    resolutions: tuple[Resolution, ...] = ()
    confidence: float | None = None
    duration_s: float = 0.0
    detail: str = ""

    @property
    def proposed(self) -> tuple[str, ...]:
        """Return the names the planner gave, unmodified.

        Returns:
            Tuple[str, ...]: Proposed names in plan order.
        """
        return tuple(r.proposed for r in self.resolutions)

    @property
    def resolved(self) -> tuple[str, ...]:
        """Return the catalogue names the proposals stand for.

        Returns:
            Tuple[str, ...]: Resolved names in plan order, duplicates kept.
        """
        return tuple(r.resolved for r in self.resolutions if r.resolved is not None)

    @property
    def unresolved(self) -> tuple[str, ...]:
        """Return the proposals that matched nothing usable.

        Returns:
            Tuple[str, ...]: Each unresolved proposal, suffixed with why.
        """
        return tuple(
            f"{r.proposed} ({r.how})" for r in self.resolutions if r.resolved is None
        )

    def precision(self) -> float | None:
        """Return the fraction of resolved proposals the case asked for.

        Returns:
            Optional[float]: Precision in [0, 1], or ``None`` when nothing was
            proposed — a call that proposed nothing has no precision, and
            scoring it as 0.0 would charge a declining planner twice while
            scoring it as 1.0 would reward it.
        """
        resolved = set(self.resolved)
        if not resolved:
            return None
        return len(resolved & set(self.case.expected_tools)) / len(resolved)


def classify(
    case: ToolSelectionCase, resolutions: Sequence[Resolution]
) -> Outcome:
    """Classify a parsed plan's tool calls (decision E-28).

    Precedence matters and is stated rather than left to the order of a chain
    of ``if``s:

    1. No tool calls at all is ``declined``, whatever the route said.
    2. Containment of every expected tool is ``correct``, *even when the plan
       also named something unresolvable*.  The planner produced what a correct
       plan needs; the stray name is over-proposal, which the precision figure
       and the resolution counters both carry.  Calling this case
       ``unknown_tool`` would move a success into a failure bucket on the
       strength of an extra token.
    3. Otherwise an unresolvable name is ``unknown_tool`` — the planner
       invented a tool, which is a grounding defect.
    4. Otherwise ``wrong_tool`` — real tools, wrong ones, a discrimination
       defect.

    Args:
        case: The case asked, carrying the expected tools.
        resolutions: The plan's proposed names, resolved.

    Returns:
        Outcome: The bucket this call belongs in.
    """
    if not resolutions:
        return "declined"
    if case.expected_tools <= set(r.resolved for r in resolutions if r.resolved):
        return "correct"
    if any(r.resolved is None for r in resolutions):
        return "unknown_tool"
    return "wrong_tool"


@dataclass(frozen=True)
class Report:
    """Every observation of one model over one corpus, with its aggregates.

    Attributes:
        model: Model identifier, empty when the deployment's own selection was
            left in force.
        repeats: Evaluations per case.
        observations: Every call, in the order they were made or restored.
        n_cases: Cases in the corpus evaluated.
        expected_outside_catalogue: Corpus labels absent from the catalogue the
            planner was shown.  Non-empty means some cases cannot be scored
            correct however the planner behaves, and the run says so rather
            than reporting the resulting dip as a model failure.
        resumed: Observations restored from the ledger rather than measured in
            this run.
    """

    model: str
    repeats: int
    observations: tuple[Observation, ...]
    n_cases: int
    expected_outside_catalogue: tuple[str, ...] = ()
    resumed: int = 0

    def select(self, hard_only: bool = False) -> tuple[Observation, ...]:
        """Return the observations in a slice.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Tuple[Observation, ...]: Matching observations, in order.
        """
        return tuple(o for o in self.observations if o.case.hard or not hard_only)

    def cases_in(self, hard_only: bool = False) -> int:
        """Return how many distinct cases a slice covers.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            int: Distinct case identifiers in the slice.
        """
        return len({o.case.case_id for o in self.select(hard_only)})

    def counts(self, hard_only: bool = False) -> Counter[str]:
        """Return the six outcome counts for a slice.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Counter[str]: One entry per outcome in :data:`OUTCOMES`, zeros
            included, so a reader never has to decide whether a missing key is
            a zero or an omission.
        """
        counts: Counter[str] = Counter({outcome: 0 for outcome in OUTCOMES})
        counts.update(o.outcome for o in self.select(hard_only))
        return counts

    def accuracy(self, hard_only: bool = False) -> float:
        """Return the fraction of observations that proposed every expected tool.

        Strict containment, directly comparable with ``tool_retrieval_recall``:
        a question routed to two tools scores zero when only one is proposed,
        because half of a co-occurrence pair is a broken plan.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            float: Accuracy in [0, 1]; 1.0 over an empty slice, as a planner
            cannot be blamed for a subset that does not exist.
        """
        chosen = self.select(hard_only)
        if not chosen:
            return 1.0
        return sum(1 for o in chosen if o.outcome == "correct") / len(chosen)

    def over_proposal(self, hard_only: bool = False) -> float:
        """Return mean precision over the tools proposed.

        Reported beside accuracy and never fused with it (decision E-1):
        proposing the right tool plus two others and proposing the wrong tool
        are different failures with different costs, and one number destroys
        exactly that distinction.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            float: Mean precision in [0, 1], 1.0 meaning nothing superfluous;
            1.0 when nothing was proposed anywhere in the slice.
        """
        values = [p for p in (o.precision() for o in self.select(hard_only)) if p is not None]
        return statistics.fmean(values) if values else 1.0

    def mean_confidence(self, hard_only: bool = False) -> float | None:
        """Return the planner's mean stated confidence (decision E-31).

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Optional[float]: Mean confidence, or ``None`` when no observation
            carried one — which is itself worth seeing, since it means every
            call failed before a plan came back.
        """
        values = [o.confidence for o in self.select(hard_only) if o.confidence is not None]
        return statistics.fmean(values) if values else None

    def per_repeat_accuracy(self, hard_only: bool = False) -> list[float]:
        """Return accuracy computed separately for each repeat.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            List[float]: One accuracy per repeat index present, in index order.
        """
        chosen = self.select(hard_only)
        by_repeat: dict[int, list[Observation]] = {}
        for observation in chosen:
            by_repeat.setdefault(observation.repeat, []).append(observation)
        return [
            sum(1 for o in group if o.outcome == "correct") / len(group)
            for _, group in sorted(by_repeat.items())
        ]

    def stddev(self, hard_only: bool = False) -> float | None:
        """Return the standard deviation of the per-repeat accuracies.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Optional[float]: Sample standard deviation, or ``None`` with fewer
            than two repeats, where it is undefined rather than zero.
        """
        values = self.per_repeat_accuracy(hard_only)
        return statistics.stdev(values) if len(values) > 1 else None

    def unanimity(self, hard_only: bool = False) -> float | None:
        """Return the fraction of cases where every repeat gave one outcome.

        Agreement is on the outcome, not merely on pass or fail: a case that
        was ``wrong_tool`` once and ``declined`` twice is not a stable failure,
        and treating it as one would hide that the planner is unstable on it.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Optional[float]: Fraction in [0, 1], or ``None`` with a single
            repeat, where unanimity is vacuous.
        """
        if self.repeats < 2:
            return None
        by_case: dict[str, set[str]] = {}
        for observation in self.select(hard_only):
            by_case.setdefault(observation.case.case_id, set()).add(observation.outcome)
        if not by_case:
            return 1.0
        return sum(1 for outcomes in by_case.values() if len(outcomes) == 1) / len(by_case)

    def resolution_counts(self) -> Counter[str]:
        """Return how every proposed name was matched (decision E-27).

        Returns:
            Counter[str]: Counts keyed by ``exact``, ``suffix``, ``ambiguous``
            and ``unknown``, zeros included.
        """
        counts: Counter[str] = Counter(
            {"exact": 0, "suffix": 0, "ambiguous": 0, "unknown": 0}
        )
        counts.update(r.how for o in self.observations for r in o.resolutions)
        return counts

    def failures(self, hard_only: bool = False) -> tuple[Observation, ...]:
        """Return observations that did not propose every expected tool.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Tuple[Observation, ...]: Failing observations, in order.
        """
        return tuple(o for o in self.select(hard_only) if o.outcome != "correct")


def _observe(
    planner: PlannerFn,
    case: ToolSelectionCase,
    repeat: int,
    model: str,
    resolver: NameResolver,
) -> Observation:
    """Make one call and classify what came back.

    Every exception other than :class:`~bamboo_eval.errors.MetricSkipped` is an
    ``error`` observation rather than an aborted run: a gateway that drops one
    call in fifty is a fact about the measurement, not a reason to discard the
    other forty-nine.  A systemic failure is caught by the consecutive-error
    guard in :func:`evaluate`, which stops the run rather than letting it
    aggregate zeros.

    Args:
        planner: The planner to call.
        case: The case to ask.
        repeat: Zero-based repeat index.
        model: Model identifier, for the record.
        resolver: Name resolver built from the catalogue.

    Returns:
        Observation: The classified call.

    Raises:
        MetricSkipped: If the planner reports the runtime is unavailable, which
            is a skip for the whole metric and not a per-case outcome.
    """
    started = time.monotonic()
    try:
        plan = planner(case.question)
    except MetricSkipped:
        raise
    except PlanParseError as exc:
        return Observation(
            case=case,
            repeat=repeat,
            model=model,
            outcome="unparseable",
            duration_s=time.monotonic() - started,
            detail=f"{exc} :: {exc.payload[:1000]}",
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Deliberately broad: a planner reaches a gateway over a network, and
        # every exception that is not a skip or a parse failure is one call
        # that did not come back. Narrowing this to the exception types seen so
        # far would turn the next unseen one into a crashed run instead of a
        # counted error.
        return Observation(
            case=case,
            repeat=repeat,
            model=model,
            outcome="error",
            duration_s=time.monotonic() - started,
            detail=f"{type(exc).__name__}: {exc}",
        )

    calls = plan.get("tool_calls") or []
    resolutions = tuple(
        resolver.resolve(str(call["tool"])) for call in calls if isinstance(call, Mapping)
    )
    raw_confidence = plan.get("confidence")
    confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None
    return Observation(
        case=case,
        repeat=repeat,
        model=model,
        outcome=classify(case, resolutions),
        resolutions=resolutions,
        confidence=confidence,
        duration_s=time.monotonic() - started,
        detail=str(plan)[:1000],
    )


def _restore(
    entry: ledger_mod.LedgerEntry,
    case: ToolSelectionCase,
    resolver: NameResolver,
) -> Observation:
    """Rebuild an observation from a ledger row.

    The proposals are re-resolved rather than trusted: the resolution rule is
    part of the measurement, so a resumed run must not carry half its
    classifications from the rule as it stood before an edit.

    Args:
        entry: The stored call.
        case: The case it asked.
        resolver: Name resolver built from the catalogue.

    Returns:
        Observation: The restored observation.
    """
    resolutions = tuple(resolver.resolve(name) for name in entry.proposed)
    outcome: Outcome = (
        entry.outcome  # type: ignore[assignment]
        if entry.outcome in ("unparseable", "error")
        else classify(case, resolutions)
    )
    return Observation(
        case=case,
        repeat=entry.repeat,
        model=entry.model,
        outcome=outcome,
        resolutions=resolutions,
        confidence=entry.confidence,
        duration_s=entry.duration_s,
        detail=entry.detail,
    )


def _to_ledger(observation: Observation) -> ledger_mod.LedgerEntry:
    """Convert an observation into the row that is appended per call.

    Args:
        observation: The call to record.

    Returns:
        LedgerEntry: The ledger row.
    """
    return ledger_mod.LedgerEntry(
        case_id=observation.case.case_id,
        repeat=observation.repeat,
        model=observation.model,
        outcome=observation.outcome,
        proposed=observation.proposed,
        resolved=observation.resolved,
        unresolved=observation.unresolved,
        confidence=observation.confidence,
        duration_s=round(observation.duration_s, 3),
        detail=observation.detail,
    )


def evaluate(
    planner: PlannerFn,
    corpus: Corpus[ToolSelectionCase],
    catalogue_names: Sequence[str],
    *,
    model: str = "",
    repeats: int = 5,
    ledger_file: Path | None = None,
    budget: Budget | None = None,
    resume: bool = False,
    retry_outcomes: frozenset[str] = frozenset(),
    max_consecutive_errors: int = 5,
) -> Report:
    """Measure how often the planner proposes the tools a case needs.

    Repeats are the outer loop, cases the inner one, so that a run which stops
    early has covered the whole corpus at a lower repeat count rather than
    part of the corpus at the full one.  The first is a weaker measurement; the
    second is not a measurement of the corpus at all.

    Args:
        planner: Question in, parsed plan out.  The CLI passes a closure over
            the production entry point; tests pass a stub.
        corpus: Labelled questions.
        catalogue_names: Every name in the catalogue the planner is shown, for
            name resolution.
        model: Model identifier to record, empty to leave the deployment's own
            selection in force.
        repeats: Evaluations per case (decision E-2; five, floor of three).
        ledger_file: Where per-call rows are appended, and where a resumed run
            reads completed calls from.  ``None`` keeps the run in memory,
            which is what the offline tests do.
        budget: Call and wall-clock allowance; ``None`` means unlimited, which
            is appropriate only for a stub planner.
        resume: Whether to reuse calls the ledger already holds.  Off by
            default and on only when asked: a rerun that silently answered
            itself out of a file would be a measurement of the file.
        retry_outcomes: Recorded outcomes a resumed run should make again.
        max_consecutive_errors: Consecutive failed calls after which the run
            stops rather than aggregating zeros.  A gateway that is down
            produces a long run of errors and then a confident 0.000.

    Returns:
        Report: Every observation, with the aggregates computed from them.

    Raises:
        BudgetExceeded: If a declared limit is reached.  Nothing is aggregated;
            the ledger holds what was paid for and ``--resume`` continues it.
        MetricSkipped: If the planner reports its runtime is unavailable.
    """
    resolver = NameResolver(catalogue_names)
    outside = tuple(sorted(corpus.expected_labels() - set(catalogue_names)))
    reusable = resume and ledger_file is not None
    done = (
        ledger_mod.latest_by_key(ledger_mod.read(ledger_file))
        if reusable and ledger_file
        else {}
    )
    skippable = (
        ledger_mod.completed_keys(ledger_file, retry_outcomes)
        if reusable and ledger_file
        else set()
    )

    observations: list[Observation] = []
    resumed = 0
    consecutive_errors = 0
    for repeat in range(repeats):
        for case in corpus.cases:
            key = (case.case_id, repeat, model)
            if key in skippable:
                observations.append(_restore(done[key], case, resolver))
                resumed += 1
                continue
            if budget is not None:
                budget.check()
            observation = _observe(planner, case, repeat, model, resolver)
            if budget is not None:
                budget.spend()
            observations.append(observation)
            if ledger_file is not None:
                ledger_mod.append(ledger_file, [_to_ledger(observation)])
            consecutive_errors = consecutive_errors + 1 if observation.outcome == "error" else 0
            if consecutive_errors >= max_consecutive_errors:
                raise BudgetExceeded(
                    f"stopped after {consecutive_errors} consecutive failed calls; "
                    f"last was {observation.detail[:200]}. A run that keeps going "
                    f"here reports a confident 0.000 for a gateway that is down."
                )

    return Report(
        model=model,
        repeats=repeats,
        observations=tuple(observations),
        n_cases=len(corpus.cases),
        expected_outside_catalogue=outside,
        resumed=resumed,
    )


def _outcome_counters(report: Report, hard_only: bool) -> dict[str, int]:
    """Map a slice's six outcome counts onto the record's counter fields.

    Args:
        report: The evaluated report.
        hard_only: Restrict to cases flagged *hard*.

    Returns:
        Dict[str, int]: ``EvalRecord`` field names to counts.  They partition
        the slice's observations, so they sum to ``n_cases * repeats``.
    """
    counts = report.counts(hard_only)
    return {
        "n_pass": counts["correct"],
        "n_fail": counts["wrong_tool"],
        "n_declined": counts["declined"],
        "n_unknown_tool": counts["unknown_tool"],
        "n_unparseable": counts["unparseable"],
        "n_error": counts["error"],
    }


def report_to_records(
    report: Report,
    corpus: Corpus[ToolSelectionCase],
    context: RunContext,
    catalogue_fingerprint: str,
    guidance_version: str,
    config: Mapping[str, Any],
    duration_s: float = 0.0,
) -> list[EvalRecord]:
    """Convert a report into the rows that get stored.

    Up to nine rows: accuracy and precision over ``all``, ``hard`` and
    ``model:<id>``, mean confidence over ``all`` and ``hard``, and one row of
    name-resolution counts.  Each answers a different question, and burying
    eight of them inside one row's payload would mean a later query has to know
    where to dig.

    On an accuracy or precision row the counters partition the observations:
    ``n_pass`` is ``correct``, ``n_fail`` is ``wrong_tool``, and
    ``n_declined``, ``n_unknown_tool``, ``n_unparseable`` and ``n_error`` carry
    the rest.  The name-resolution row counts *proposals* rather than
    observations, and says so in its own terms: ``n_cases`` is how many names
    were proposed, ``n_pass`` how many matched the catalogue exactly,
    ``n_fail`` how many did not, of which ``n_unknown_tool`` matched nothing
    and ``n_skipped`` were suffixes matching more than one catalogue entry.
    The remainder — ``n_fail - n_unknown_tool - n_skipped`` — is how often the
    headline number depended on the suffix rule in decision E-27.

    Args:
        report: The evaluated report.
        corpus: The corpus measured, for the citation fields.
        context: Facts shared by every row this invocation writes.
        catalogue_fingerprint: Digest of the catalogue the planner was shown.
        guidance_version: Digest of the routing guidance in force.
        config: The metric's settings, stored canonically.
        duration_s: Wall-clock seconds the measurement took.

    Returns:
        List[EvalRecord]: Rows ready for the store, in reporting order.
    """
    canonical = canonical_config({**config, "repeats": report.repeats})

    def row(
        metric: str,
        slice_name: str,
        value: float,
        hard_only: bool,
        n_cases: int | None = None,
        counters: Mapping[str, int] | None = None,
    ) -> EvalRecord:
        """Build one row with this report's shared citation fields.

        Args:
            metric: Metric name.
            slice_name: Slice name.
            value: The measured value.
            hard_only: Which slice the aggregates are taken over.
            n_cases: Override for the case count, where the row counts
                something other than cases.
            counters: Override for the counter fields.

        Returns:
            EvalRecord: The row.
        """
        tally = _outcome_counters(report, hard_only) if counters is None else dict(counters)
        return EvalRecord(
            run_id=context.run_id,
            timestamp=context.timestamp,
            metric=metric,
            slice=slice_name,
            value=round(value, 6),
            n_cases=report.cases_in(hard_only) if n_cases is None else n_cases,
            n_pass=tally.get("n_pass", 0),
            n_fail=tally.get("n_fail", 0),
            n_declined=tally.get("n_declined", 0),
            n_unknown_tool=tally.get("n_unknown_tool", 0),
            n_unparseable=tally.get("n_unparseable", 0),
            n_error=tally.get("n_error", 0),
            n_skipped=tally.get("n_skipped", 0),
            repeats=report.repeats,
            stddev=report.stddev(hard_only),
            unanimity=report.unanimity(hard_only),
            corpus_name=corpus.name,
            corpus_version=corpus.version,
            corpus_sha256=corpus.sha256,
            catalogue_fingerprint=catalogue_fingerprint,
            guidance_version=guidance_version,
            planner_model=report.model,
            config=canonical,
            git_commit=context.git_commit,
            host=context.host,
            framework_version=context.framework_version,
            duration_s=round(duration_s, 3),
        )

    records = [
        row(METRIC_NAME, "all", report.accuracy(), False),
        row(OVER_PROPOSAL_METRIC, "all", report.over_proposal(), False),
    ]
    # The hard slice is 1.0 over an empty subset, which is the right answer to
    # "did it fail on a confusable case" and the wrong thing to store: a row
    # reading 1.000 over no cases is indistinguishable from a perfect score
    # until someone reads n_cases. A --limit run that happens to exclude every
    # hard case therefore emits no hard row at all.
    if report.cases_in(True):
        records.append(row(METRIC_NAME, "hard", report.accuracy(True), True))
        records.append(
            row(OVER_PROPOSAL_METRIC, "hard", report.over_proposal(True), True)
        )
    if report.model:
        slice_name = f"model:{report.model}"
        records.append(row(METRIC_NAME, slice_name, report.accuracy(), False))
        records.append(row(OVER_PROPOSAL_METRIC, slice_name, report.over_proposal(), False))

    for name, hard_only in (("all", False), ("hard", True)):
        if hard_only and not report.cases_in(True):
            continue
        confidence = report.mean_confidence(hard_only)
        if confidence is not None:
            counters = _outcome_counters(report, hard_only)
            counters["n_skipped"] = sum(
                1 for o in report.select(hard_only) if o.confidence is None
            )
            records.append(
                row(CONFIDENCE_METRIC, name, confidence, hard_only, counters=counters)
            )

    resolution = report.resolution_counts()
    proposals = sum(resolution.values())
    records.append(
        row(
            NAME_RESOLUTION_METRIC,
            "all",
            resolution["exact"] / proposals if proposals else 1.0,
            False,
            n_cases=proposals,
            counters={
                "n_pass": resolution["exact"],
                "n_fail": proposals - resolution["exact"],
                "n_unknown_tool": resolution["unknown"],
                "n_skipped": resolution["ambiguous"],
            },
        )
    )
    return records
