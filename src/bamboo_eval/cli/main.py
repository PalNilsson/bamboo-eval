"""``bamboo-eval`` — run a metric, check the contract, read the history.

Subcommands exit non-zero on failure so that the same command gates a pull
request and reports a number: ``tool-retrieval --min-recall 0.99`` is the gate,
``tool-retrieval`` alone is the report (decision E-11).

Every text report leads with the catalogue fingerprint.  Two hosts running the
identical command produced 29,527 and 29,750 characters and recalls of 0.983
and 0.992, because their plugin descriptions differed; without that line the
discrepancy reads as noise in the retriever rather than a difference in what
was measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import __version__
from ..corpus import ToolSelectionCase, bundled_corpus_path, load_corpus
from ..errors import BambooEvalError, MetricSkipped, ProductionContractError
from ..metrics import tool_retrieval as tr
from ..record import RunContext, skipped_record
from .. import production, store

#: Tools exempt from retrieval because the universal fallback route has no
#: graceful degradation without them.  Mirrors decision T-4 in bamboo-mcp: a
#: retriever never sees them, and they count against the *k* budget.
PINNED_TOOLS = frozenset({"panda_doc_search", "panda_doc_bm25"})


def _format_report(report: tr.Report, show_failures: int) -> str:
    """Render a report as plain text.

    Args:
        report: The report to render.
        show_failures: Maximum failing cases to list; 0 lists none.

    Returns:
        str: The rendered report.
    """
    lines = [
        f"retriever={report.retriever}  k={report.k}  cases={len(report.results)}",
        f"  recall@k            {report.recall():.3f}",
        f"  recall@k (hard)     {report.recall(hard_only=True):.3f}",
        f"  guidance coverage   {report.guidance_coverage():.3f}",
        f"  payload fraction    {report.mean_payload_fraction():.3f}"
        f"  (full catalogue = {report.full_payload_bytes:,} chars)",
        f"  over budget         {report.over_budget_cases()}/{len(report.results)} cases",
    ]
    failures = report.failures()
    if failures and show_failures:
        lines.append(f"  failures ({len(failures)}):")
        for result in failures[:show_failures]:
            flag = " [hard]" if result.case.hard else ""
            lines.append(
                f"    {result.case.case_id}{flag} missing="
                f"{sorted(result.missing)}  {result.case.question[:64]!r}"
            )
        if len(failures) > show_failures:
            lines.append(f"    ... and {len(failures) - show_failures} more")
    return "\n".join(lines)


def _cmd_tool_retrieval(args: argparse.Namespace) -> int:
    """Run the tool-retrieval metric.

    Args:
        args: Parsed arguments.

    Returns:
        int: Process exit status; 1 when ``--min-recall`` is not met.
    """
    started = time.monotonic()
    context = RunContext.capture(__version__)
    corpus_path = args.corpus or bundled_corpus_path("tool_selection_corpus")
    corpus = load_corpus(corpus_path, ToolSelectionCase)

    try:
        retriever = production.retriever(args.retriever)
    except MetricSkipped as exc:
        print(f"skipped: {exc.reason}", file=sys.stderr)
        if args.record:
            store.append(
                [skipped_record(context, tr.METRIC_NAME, exc.reason, corpus.name)],
                args.results_dir,
            )
        return 0

    catalogue = production.collect_catalogue(args.namespace)
    fingerprint = production.catalogue_fingerprint(catalogue)
    rules = production.routing_rules(args.plugin_id)
    guidance = production.guidance_fingerprint(rules)
    pinned = frozenset() if args.no_pins else PINNED_TOOLS

    try:
        reports = [
            tr.evaluate(
                retriever, corpus, catalogue, k=k, pinned=pinned, routing_rules=rules
            )
            for k in (args.k or [10])
        ]
    except MetricSkipped as exc:
        # The encoder resolves on first use, not at construction, so a missing
        # model can surface here rather than above.
        print(f"skipped: {exc.reason}", file=sys.stderr)
        if args.record:
            store.append(
                [skipped_record(context, tr.METRIC_NAME, exc.reason, corpus.name)],
                args.results_dir,
            )
        return 0

    config = {
        "namespace": args.namespace,
        "plugin_id": args.plugin_id,
        "pinned": sorted(pinned),
    }
    elapsed = time.monotonic() - started
    records = [
        record
        for report in reports
        for record in tr.report_to_records(
            report, corpus, context, fingerprint[:12], guidance, config, elapsed
        )
    ]

    if args.json:
        print(json.dumps({"records": [r.to_dict() for r in records]}, indent=2))
    else:
        print(
            f"catalogue: {len(catalogue)} tools, namespace={args.namespace!r}, "
            f"fingerprint={fingerprint[:12]}"
        )
        print(f"corpus:    {len(corpus.cases)} cases, {corpus.name} v{corpus.version}")
        print(f"           sha256={corpus.sha256[:12]} from {corpus_path}")
        print(f"guidance:  {len(rules)} clauses, fingerprint={guidance or 'none'}")
        print(f"pinned:    {sorted(pinned) or 'none'}")
        print()
        for report in reports:
            print(_format_report(report, args.show_failures))
            print()

    if args.record:
        written = store.append(records, args.results_dir)
        for record in records:
            print(store.describe_change(record, args.results_dir), file=sys.stderr)
        for path in written:
            print(f"recorded in {path}", file=sys.stderr)

    if args.min_recall is not None:
        worst = min(r.recall() for r in reports)
        if worst < args.min_recall:
            print(
                f"FAIL: recall@k {worst:.3f} below threshold {args.min_recall:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_check_contract(args: argparse.Namespace) -> int:
    """Resolve every declared production entry point.

    Args:
        args: Parsed arguments.

    Returns:
        int: 0 when the contract holds, 1 when a required entry point has
        moved, 3 when Bamboo is not installed so the contract was never
        tested.  Three statuses rather than two, because a CI job that forgot
        to install the system under test must not report the same green as one
        that checked it.  Unavailable optional backends are printed but do not
        fail the check.
    """
    problems = production.check_contract()
    broken = [p for p in problems if p.startswith("BROKEN")]
    missing = [p for p in problems if p.startswith("NOT INSTALLED")]
    for line in problems:
        stream = sys.stderr if line.startswith(("BROKEN", "NOT INSTALLED")) else sys.stdout
        print(line, file=stream)
    if not problems:
        print(
            f"contract ok: {len(production.ENTRY_POINTS)} production entry points "
            f"resolved"
        )
    if missing:
        return 3
    return 1 if broken else 0


def _cmd_history(args: argparse.Namespace) -> int:
    """Print a metric's stored history for one slice.

    Args:
        args: Parsed arguments.

    Returns:
        int: Always 0; an empty history is information, not an error.
    """
    rows = store.history(args.metric, args.slice, args.results_dir)
    if not rows:
        print(f"no stored rows for {args.metric}/{args.slice}")
        return 0
    for row in rows[-args.limit:]:
        value = "—" if row.value is None else f"{row.value:.4f}"
        note = f"  {row.skip_reason}" if row.status != "ok" else ""
        print(
            f"{row.timestamp}  {value}  n={row.n_cases:<4} "
            f"cat={row.catalogue_fingerprint or '—':<12} "
            f"corpus={row.corpus_sha256[:12] or '—':<12} "
            f"{row.git_commit}{note}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="bamboo-eval", description="Bamboo Evaluation Framework"
    )
    parser.add_argument("--version", action="version", version=f"bamboo-eval {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Declared per subcommand rather than on the top-level parser, so that
    # `bamboo-eval tool-retrieval --results-dir X` works. argparse accepts a
    # top-level option only before the subcommand, which is the opposite of
    # where anyone types it.
    store_args = argparse.ArgumentParser(add_help=False)
    store_args.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory holding the JSONL result store (default: ./results).",
    )

    retrieval = subparsers.add_parser(
        "tool-retrieval",
        parents=[store_args],
        help="Measure tool-retrieval recall, guidance and payload.",
    )
    retrieval.add_argument("--corpus", type=Path, default=None)
    retrieval.add_argument(
        "--retriever",
        default="null",
        choices=["null", "lexical", "embedding", "hybrid"],
        help="Retriever to evaluate (default: the no-retrieval baseline).",
    )
    retrieval.add_argument(
        "--k",
        type=int,
        action="append",
        help="Tool budget, including pinned tools. Repeatable. Default: 10.",
    )
    retrieval.add_argument("--namespace", default="atlas")
    retrieval.add_argument("--plugin-id", default="atlas")
    retrieval.add_argument(
        "--no-pins",
        action="store_true",
        help="Do not pin the fallback documentation tools (measures their value).",
    )
    retrieval.add_argument("--show-failures", type=int, default=10)
    retrieval.add_argument("--json", action="store_true", help="Emit the stored rows as JSON.")
    retrieval.add_argument(
        "--record",
        action="store_true",
        help="Append the rows to the result store and report the change.",
    )
    retrieval.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Exit non-zero if recall@k falls below this at any k.",
    )
    retrieval.set_defaults(func=_cmd_tool_retrieval)

    contract = subparsers.add_parser(
        "check-contract",
        help="Resolve every production entry point the metrics depend on.",
    )
    contract.set_defaults(func=_cmd_check_contract)

    hist = subparsers.add_parser(
        "history", parents=[store_args], help="Print a metric's stored history."
    )
    hist.add_argument("metric")
    hist.add_argument("--slice", default="all")
    hist.add_argument("--limit", type=int, default=20)
    hist.set_defaults(func=_cmd_history)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        int: Process exit status.  A broken production contract exits 2, to
        distinguish "the measurement failed its threshold" from "the thing
        being measured is no longer where this package looks for it".
    """
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ProductionContractError as exc:
        print(f"production contract broken: {exc}", file=sys.stderr)
        return 2
    except BambooEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
