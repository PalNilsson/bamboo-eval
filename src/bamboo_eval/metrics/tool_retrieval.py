"""Tool-retrieval quality: did shortlisting keep the tools a plan needs?

Ported unchanged in behaviour from ``core/bamboo/evaluation/tool_retrieval.py``
in bamboo-mcp.  The scoring, the budget handling and the guidance metric are
deliberately identical: phase 0's acceptance criterion is that this package
reproduces the numbers the old harness produced on the same catalogue, and a
"small improvement" made during the move would have made that check
meaningless.

The planner is shown the whole tool catalogue on every question.  Retrieval
narrows that, trading a smaller prompt for the risk of withholding the tool the
planner actually required.  That trade cannot be argued, only measured, and it
has to be measured against the behaviour it replaces — so this module evaluates
*any* retriever, including the null one that returns everything and reproduces
the pre-retrieval planner exactly.

Three numbers, because one is not enough:

* **Recall@k.** Did the retrieved set contain every tool a correct plan needs?
  Strict by design: a question routed to two tools scores zero when only one
  survives, since half of a co-occurrence pair is a broken plan, not a partial
  one.
* **Guidance coverage.** Did the routing guidance that survives filtering still
  name those tools?  Retrieval can keep the right tool in the catalogue while
  dropping the clause that says when to use it — a silent, retrieval-specific
  regression no recall figure reveals.
* **Payload bytes.** What the narrowing actually bought, as a fraction of the
  unfiltered catalogue: the only reason to accept any recall loss at all.

Every metric is additionally reported over the *hard* subset, questions paired
with a deliberately confusable neighbour.  Aggregate recall is dominated by easy
cases and will look fine while the confusable ones rot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..corpus import Corpus, ToolSelectionCase
from ..record import EvalRecord, RunContext, canonical_config

#: Name under which this metric's rows are stored.
METRIC_NAME = "tool_retrieval_recall"

#: Production symbols this metric depends on, by attribute name.  Decision
#: E-12 requires every metric to name the production entry point it calls; a
#: test asserts that each name here is declared in
#: :data:`bamboo_eval.production.ENTRY_POINTS`.
PRODUCTION_ENTRY_POINTS = (
    "_collect_tool_catalog",
    "routing_rules_for_plugin",
    "catalog_fingerprint",
)


@runtime_checkable
class ToolRetriever(Protocol):
    """A scorer that narrows a tool catalogue to the tools a question needs."""

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return up to *k* tool names, best match first.

        Args:
            question: The user's question, verbatim.
            catalog: Catalogue entries as the planner would receive them; each
                has at least ``name``, ``description`` and ``inputSchema``.
            k: Maximum number of names to return.  A retriever may return
                fewer, never more.

        Returns:
            List[str]: Tool names drawn from *catalog*, ordered by descending
            relevance.
        """
        ...


class NullRetriever:
    """Returns the whole catalogue, reproducing the pre-retrieval planner.

    The baseline every candidate is measured against.  Recall is 1.0 by
    construction and payload cost is 100%, which is the point: it fixes both
    ends of the trade so a candidate's numbers mean something.

    *k* is deliberately ignored rather than honoured, because truncating the
    catalogue to its first *k* entries in catalogue order would be a retriever
    — an arbitrarily bad one — and not the behaviour being used as a reference.
    An earlier version did truncate, and produced a plausible-looking 0.383
    that would have become the figure every later measurement was judged
    against.
    """

    name = "null"

    def retrieve(
        self,
        question: str,
        catalog: Sequence[Mapping[str, Any]],
        k: int,
    ) -> list[str]:
        """Return every catalogue name, in catalogue order.

        Args:
            question: Ignored.
            catalog: Catalogue entries.
            k: Ignored; see the class docstring.

        Returns:
            List[str]: Every name in *catalog*.
        """
        return [str(entry["name"]) for entry in catalog]


@dataclass(frozen=True)
class CaseResult:
    """What a retriever did on one case.

    Attributes:
        case: The case evaluated.
        retrieved: Names returned, best match first.
        hit: Whether every expected tool was retrieved.
        missing: Expected tools that were not retrieved.
        worst_rank: Zero-based rank of the last-placed expected tool, or
            ``None`` when one is missing.  The *worst* rank rather than the
            best, because a plan needs all of them: a pair at ranks 0 and 9 is
            only as safe as its rank-9 member.
        guidance_covered: Whether the surviving routing guidance still names
            every expected tool, or ``None`` when no clause names them and the
            question is therefore outside the guidance's scope.
        payload_bytes: Serialised size of the retrieved catalogue subset.
        over_budget: How many tools beyond *k* were returned, pins included;
            0 when the budget was respected.
    """

    case: ToolSelectionCase
    retrieved: tuple[str, ...]
    hit: bool
    missing: frozenset[str]
    worst_rank: int | None
    guidance_covered: bool | None
    payload_bytes: int
    over_budget: int = 0


@dataclass(frozen=True)
class Report:
    """Aggregate results for one retriever at one *k*.

    Attributes:
        retriever: Name of the retriever evaluated.
        k: The *k* it was asked for.
        results: Per-case results, in corpus order.
        full_payload_bytes: Serialised size of the unfiltered catalogue, the
            denominator for payload savings.
    """

    retriever: str
    k: int
    results: tuple[CaseResult, ...]
    full_payload_bytes: int

    def recall(self, hard_only: bool = False) -> float:
        """Return the fraction of cases where every expected tool was retrieved.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            float: Recall in [0, 1]; 1.0 when no case qualifies, since a
            retriever cannot be blamed for a subset that does not exist.
        """
        chosen = [r for r in self.results if r.case.hard or not hard_only]
        if not chosen:
            return 1.0
        return sum(1 for r in chosen if r.hit) / len(chosen)

    def guidance_coverage(self) -> float:
        """Return the fraction of in-scope cases whose guidance survived.

        Cases no routing clause names are excluded rather than counted as
        passes, which would dilute the figure towards 1.0 with questions the
        metric says nothing about.

        Returns:
            float: Coverage in [0, 1]; 1.0 when no case is in scope.
        """
        chosen = [r for r in self.results if r.guidance_covered is not None]
        if not chosen:
            return 1.0
        return sum(1 for r in chosen if r.guidance_covered) / len(chosen)

    def mean_payload_fraction(self) -> float:
        """Return mean retrieved payload size as a fraction of the full catalogue.

        Returns:
            float: Mean fraction; 1.0 when the catalogue is empty or no case
            ran.
        """
        if not self.results or self.full_payload_bytes <= 0:
            return 1.0
        return sum(r.payload_bytes for r in self.results) / (
            len(self.results) * self.full_payload_bytes
        )

    def over_budget_cases(self) -> int:
        """Return how many cases returned more tools than *k* allowed.

        The harness measures the budget rather than enforcing it.  Truncating a
        retriever's output here would hide a retriever that ignores its budget,
        and would reduce :class:`NullRetriever` to "the first *k* entries in
        catalogue order".  The baseline is therefore expected to be over budget
        on every case; a candidate retriever is not.

        Returns:
            int: Number of cases that exceeded *k*.
        """
        return sum(1 for r in self.results if r.over_budget)

    def failures(self, hard_only: bool = False) -> tuple[CaseResult, ...]:
        """Return cases that missed at least one expected tool.

        Args:
            hard_only: Restrict to cases flagged *hard*.

        Returns:
            Tuple[CaseResult, ...]: Failing cases, in corpus order.
        """
        return tuple(
            r for r in self.results if not r.hit and (r.case.hard or not hard_only)
        )


def _payload_bytes(catalog: Sequence[Mapping[str, Any]], names: Sequence[str]) -> int:
    """Return the serialised size of the named subset of a catalogue.

    Args:
        catalog: Full catalogue entries.
        names: Names to include.

    Returns:
        int: Length of the JSON serialisation, in characters.  A proxy for
        prompt cost that avoids depending on a tokeniser, and one that moves in
        step with tokens closely enough for a ratio.
    """
    wanted = set(names)
    subset = [entry for entry in catalog if str(entry["name"]) in wanted]
    return len(json.dumps(subset, ensure_ascii=False))


def _guidance_names(rendered: str, tools: frozenset[str]) -> bool | None:
    """Report whether rendered guidance names every tool in *tools*.

    Args:
        rendered: Rendered routing guidance.
        tools: Tools the case expects.

    Returns:
        Optional[bool]: Whether every tool appears in *rendered*.
    """
    return all(tool in rendered for tool in tools)


def evaluate(
    retriever: ToolRetriever,
    corpus: Corpus[ToolSelectionCase],
    catalog: Sequence[Mapping[str, Any]],
    k: int,
    pinned: frozenset[str] = frozenset(),
    routing_rules: Sequence[Any] = (),
) -> Report:
    """Evaluate a retriever over a corpus against a catalogue.

    Args:
        retriever: The retriever under test.
        corpus: Labelled questions.
        catalog: Catalogue entries the planner would otherwise receive whole.
        k: Maximum tools the planner should end up with, *including* pinned
            ones.  Pins that did not count against *k* would let a caller claim
            a budget it does not keep.
        pinned: Tools always present regardless of score — the universal
            fallback route, which has no graceful degradation if dropped.
        routing_rules: ``RoutingRule``-shaped objects with ``tools`` and
            ``text``, used for the guidance-coverage metric.  Empty disables it.

    Returns:
        Report: Per-case results and aggregates.
    """
    catalog_names = [str(entry["name"]) for entry in catalog]
    available_pins = frozenset(pinned) & frozenset(catalog_names)
    full_bytes = _payload_bytes(catalog, catalog_names)
    budget = max(0, k - len(available_pins))

    unfiltered_guidance = "\n".join(str(rule.text) for rule in routing_rules)

    results: list[CaseResult] = []
    for case in corpus.cases:
        scored = retriever.retrieve(case.question, catalog, budget)
        ordered = [name for name in scored if name not in available_pins]
        retrieved = tuple(sorted(available_pins)) + tuple(ordered)
        retrieved_set = frozenset(retrieved)

        missing = case.expected_tools - retrieved_set
        if missing:
            worst_rank: int | None = None
        else:
            worst_rank = max(retrieved.index(tool) for tool in case.expected_tools)

        guidance_covered: bool | None = None
        if routing_rules and any(
            tool in unfiltered_guidance for tool in case.expected_tools
        ):
            kept = [
                str(rule.text)
                for rule in routing_rules
                if frozenset(rule.tools) <= retrieved_set
            ]
            guidance_covered = _guidance_names("\n".join(kept), case.expected_tools)

        results.append(
            CaseResult(
                case=case,
                retrieved=retrieved,
                hit=not missing,
                missing=missing,
                worst_rank=worst_rank,
                guidance_covered=guidance_covered,
                payload_bytes=_payload_bytes(catalog, retrieved),
                over_budget=max(0, len(retrieved) - k),
            )
        )

    return Report(
        retriever=getattr(retriever, "name", type(retriever).__name__),
        k=k,
        results=tuple(results),
        full_payload_bytes=full_bytes,
    )


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

    Four rows per report — recall over all cases, recall over the hard subset,
    guidance coverage and payload fraction — because each answers a different
    question and burying three of them inside one row's payload would mean a
    later query has to know where to dig.

    Args:
        report: The evaluated report.
        corpus: The corpus it was measured against, for the citation fields.
        context: Facts shared by every row this invocation writes.
        catalogue_fingerprint: Digest of the catalogue measured.
        guidance_version: Digest of the routing guidance in force.
        config: The metric's settings, stored canonically.
        duration_s: Wall-clock seconds the measurement took.

    Returns:
        List[EvalRecord]: Rows ready for the store, in reporting order.
    """
    canonical = canonical_config({**config, "k": report.k, "retriever": report.retriever})
    hard = [r for r in report.results if r.case.hard]
    in_scope = [r for r in report.results if r.guidance_covered is not None]

    def row(
        metric: str,
        slice_name: str,
        value: float,
        n_cases: int,
        n_pass: int,
    ) -> EvalRecord:
        """Build one row with this report's shared citation fields.

        Args:
            metric: Metric name.
            slice_name: Slice name.
            value: The measured value.
            n_cases: Cases in the slice.
            n_pass: Cases that met the criterion.

        Returns:
            EvalRecord: The row.
        """
        return EvalRecord(
            run_id=context.run_id,
            timestamp=context.timestamp,
            metric=metric,
            slice=slice_name,
            value=round(value, 6),
            n_cases=n_cases,
            n_pass=n_pass,
            n_fail=n_cases - n_pass,
            corpus_name=corpus.name,
            corpus_version=corpus.version,
            corpus_sha256=corpus.sha256,
            catalogue_fingerprint=catalogue_fingerprint,
            guidance_version=guidance_version,
            config=canonical,
            git_commit=context.git_commit,
            host=context.host,
            framework_version=context.framework_version,
            duration_s=round(duration_s, 3),
        )

    return [
        row(
            METRIC_NAME,
            "all",
            report.recall(),
            len(report.results),
            sum(1 for r in report.results if r.hit),
        ),
        row(
            METRIC_NAME,
            "hard",
            report.recall(hard_only=True),
            len(hard),
            sum(1 for r in hard if r.hit),
        ),
        row(
            "tool_retrieval_guidance_coverage",
            "all",
            report.guidance_coverage(),
            len(in_scope),
            sum(1 for r in in_scope if r.guidance_covered),
        ),
        row(
            "tool_retrieval_payload_fraction",
            "all",
            report.mean_payload_fraction(),
            len(report.results),
            len(report.results),
        ),
    ]
