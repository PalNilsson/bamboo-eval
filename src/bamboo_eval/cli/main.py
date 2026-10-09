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
from ..budget import DEFAULT_MAX_CALLS, DEFAULT_MAX_SECONDS, Budget
from ..corpus import Corpus, ToolSelectionCase, bundled_corpus_path, load_corpus
from ..errors import BambooEvalError, BudgetExceeded, MetricSkipped, ProductionContractError
from ..metrics import selection_accuracy as sa
from ..metrics import tool_retrieval as tr
from ..record import RunContext, failed_record, skipped_record
from .. import ledger, production, store

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


def _record_skip(
    args: argparse.Namespace,
    context: RunContext,
    metric: str,
    corpus_name: str,
    reason: str,
) -> int:
    """Report a skipped metric and, when recording, store it.

    A skip is stored rather than omitted: a metric that stops running leaves a
    gap in the series that reads as "nothing changed", which is how a broken
    backend survives a fortnight unnoticed.

    Args:
        args: Parsed arguments, for ``record`` and ``results_dir``.
        context: The invocation's shared facts.
        metric: The metric that could not run.
        corpus_name: The corpus it would have used.
        reason: Why it could not run.

    Returns:
        int: 0 — a skip is not a failure of the run.
    """
    print(f"skipped: {reason}", file=sys.stderr)
    if args.record:
        store.append(
            [skipped_record(context, metric, reason, corpus_name)], args.results_dir
        )
    return 0


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
        return _record_skip(args, context, tr.METRIC_NAME, corpus.name, exc.reason)

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
        return _record_skip(args, context, tr.METRIC_NAME, corpus.name, exc.reason)

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


def _format_selection_report(report: sa.Report, show_failures: int) -> str:
    """Render a selection-accuracy report as plain text.

    Args:
        report: The report to render.
        show_failures: Maximum failing observations to list; 0 lists none.

    Returns:
        str: The rendered report.
    """
    counts = report.counts()
    resolution = report.resolution_counts()
    unanimity = report.unanimity()
    stddev = report.stddev()
    confidence = report.mean_confidence()
    lines = [
        f"model={report.model or '<deployment default>'}  repeats={report.repeats}  "
        f"cases={report.n_cases}  observations={len(report.observations)}"
        + (f"  ({report.resumed} resumed)" if report.resumed else ""),
        f"  selection accuracy  {report.accuracy():.3f}"
        + (f"  ± {stddev:.3f}" if stddev is not None else "")
        + (f"  unanimity {unanimity:.3f}" if unanimity is not None else ""),
        "  accuracy (hard)     "
        + (f"{report.accuracy(True):.3f}" if report.cases_in(True) else "— (no hard cases)"),
        f"  over-proposal       {report.over_proposal():.3f}  (precision over proposed)",
        "  confidence          "
        + ("—" if confidence is None else f"{confidence:.3f}"),
        "  outcomes            "
        + "  ".join(f"{name}={counts[name]}" for name in sa.OUTCOMES),
        "  names resolved      "
        + "  ".join(f"{how}={resolution[how]}" for how in ("exact", "suffix", "ambiguous", "unknown")),
    ]
    if report.expected_outside_catalogue:
        lines.append(
            f"  WARNING: {len(report.expected_outside_catalogue)} expected tools are "
            f"absent from this catalogue and can never be proposed: "
            f"{list(report.expected_outside_catalogue)}"
        )
    failures = report.failures()
    if failures and show_failures:
        lines.append(f"  failures ({len(failures)}):")
        for observation in failures[:show_failures]:
            flag = " [hard]" if observation.case.hard else ""
            lines.append(
                f"    {observation.case.case_id}{flag} r{observation.repeat} "
                f"{observation.outcome}: expected="
                f"{sorted(observation.case.expected_tools)} proposed="
                f"{list(observation.proposed)}"
            )
        if len(failures) > show_failures:
            lines.append(f"    ... and {len(failures) - show_failures} more")
    return "\n".join(lines)


def _selection_corpus(args: argparse.Namespace) -> tuple[Corpus[ToolSelectionCase], Path]:
    """Load the corpus a selection run measures, honouring ``--limit``.

    Args:
        args: Parsed arguments.

    Returns:
        Tuple[Corpus[ToolSelectionCase], Path]: The corpus and where it came
        from.  A limited corpus keeps its name, version and digest — it is the
        same labels — and the limit is recorded in the configuration, so the
        store refuses to compare a five-case smoke run with a full one.
    """
    path = args.corpus or bundled_corpus_path("tool_selection_corpus")
    corpus = load_corpus(path, ToolSelectionCase)
    if args.limit:
        corpus = Corpus(
            cases=corpus.cases[: args.limit],
            name=corpus.name,
            version=corpus.version,
            sha256=corpus.sha256,
            coverage_exempt=corpus.coverage_exempt,
            description=corpus.description,
            path=corpus.path,
        )
    return corpus, path


def _env_overrides(args: argparse.Namespace, model: str) -> dict[str, str]:
    """Return the environment a run applies before measuring.

    Bamboo resolves its LLM profiles from the environment, so this *is* the
    selection of what gets measured, and it is recorded in the row rather than
    left to whatever the shell happened to carry.

    Args:
        args: Parsed arguments.
        model: Model identifier, empty for the deployment's own selection.

    Returns:
        Dict[str, str]: Variables to set.

    Raises:
        BambooEvalError: If a ``--set-env`` argument is not ``NAME=VALUE``.
            Silently ignoring it would leave a run reporting a configuration it
            did not have.
    """
    overrides = {args.model_env: model} if model else {}
    for item in args.set_env or []:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise BambooEvalError(f"--set-env expects NAME=VALUE, got {item!r}")
        overrides[name] = value
    if "BAMBOO_TOOL_RETRIEVAL" in overrides:
        production.check_retrieval_setting(overrides["BAMBOO_TOOL_RETRIEVAL"])
    if len(args.model or []) > 1:
        raise BambooEvalError(
            "pass one --model per invocation. Bamboo's LLM selector is a "
            "process-global populated from the environment when the server "
            "runtime starts, so a second model measured in the same process "
            "would answer under the first one's selection while the rows named "
            "the second. The ledger and the store make a shell loop cheap."
        )
    return overrides


def _run_one_model(
    args: argparse.Namespace,
    model: str,
    corpus: Corpus[ToolSelectionCase],
    names: Sequence[str],
    fingerprint: str,
) -> sa.Report:
    """Measure one model, appending every call to the ledger as it returns.

    Args:
        args: Parsed arguments.
        model: Model identifier, empty for the deployment's own selection.
        corpus: The labelled questions.
        names: Catalogue names, for resolution.
        fingerprint: Abbreviated catalogue fingerprint, which names the ledger.

    Returns:
        sa.Report: The model's observations and aggregates.
    """
    ledger_file = ledger.ledger_path(sa.METRIC_NAME, fingerprint, args.results_dir)
    budget = Budget(max_calls=args.max_calls, max_seconds=args.max_seconds)

    def planner(question: str) -> dict[str, object]:
        """Ask the production planner.

        Args:
            question: The corpus question.

        Returns:
            Dict[str, object]: The parsed plan.
        """
        return production.plan(
            question,
            namespaces=[args.namespace],
            temperature=args.temperature,
            plugin_id=args.plugin_id,
            runtime_init=args.runtime_init,
        )

    with production.env_overrides(_env_overrides(args, model)):
        return sa.evaluate(
            planner,
            corpus,
            names,
            model=model,
            repeats=args.repeats,
            ledger_file=ledger_file,
            budget=budget,
            resume=args.resume,
            retry_outcomes=frozenset({"error"}) if args.resume else frozenset(),
            max_consecutive_errors=args.max_consecutive_errors,
        )


def _cmd_selection_accuracy(args: argparse.Namespace) -> int:
    """Run the end-task selection-accuracy metric (phase 1).

    Args:
        args: Parsed arguments.

    Returns:
        int: 0 on a completed measurement or a stated skip, 1 when a declared
        limit stopped the run.  No threshold option: everything LLM-dependent
        gates nothing (decision E-11), so this command reports and records, and
        a regression is caught by comparing rows rather than by a red build.
    """
    started = time.monotonic()
    context = RunContext.capture(__version__)
    _env_overrides(args, "")  # validated here: a usage error must precede a skip
    corpus, corpus_path = _selection_corpus(args)

    try:
        catalogue = production.collect_catalogue(args.namespace)
    except MetricSkipped as exc:
        return _record_skip(args, context, sa.METRIC_NAME, corpus.name, exc.reason)
    names = [str(entry["name"]) for entry in catalogue]
    fingerprint = production.catalogue_fingerprint(catalogue)[:12]
    guidance = production.guidance_fingerprint(production.routing_rules(args.plugin_id))
    config = {
        "namespace": args.namespace,
        "plugin_id": args.plugin_id,
        "temperature": args.temperature,
        "limit": args.limit or 0,
        "model_env": args.model_env,
        "runtime_init": args.runtime_init,
        "env": production.retrieval_settings(),
    }

    print(f"catalogue: {len(catalogue)} tools, fingerprint={fingerprint}")
    print(f"corpus:    {len(corpus.cases)} cases, {corpus.name} v{corpus.version}")
    print(f"           sha256={corpus.sha256[:12]} from {corpus_path}")
    print(f"ledger:    {ledger.ledger_path(sa.METRIC_NAME, fingerprint, args.results_dir)}")
    print(f"retrieval: {production.retrieval_settings()}  (before overrides)")
    print()

    records = []
    for model in args.model or [""]:
        applied = _env_overrides(args, model)
        print(f"applying:  {applied or 'nothing; the deployment selects'}")
        try:
            report = _run_one_model(args, model, corpus, names, fingerprint)
        except MetricSkipped as exc:
            return _record_skip(args, context, sa.METRIC_NAME, corpus.name, exc.reason)
        except BudgetExceeded as exc:
            print(f"stopped: {exc.reason}", file=sys.stderr)
            if args.record:
                store.append(
                    [failed_record(context, sa.METRIC_NAME, exc.reason, corpus.name)],
                    args.results_dir,
                )
            return 1
        print(_format_selection_report(report, args.show_failures))
        print()
        records.extend(
            sa.report_to_records(
                report,
                corpus,
                context,
                fingerprint,
                guidance,
                {**config, "env_overrides": applied},
                time.monotonic() - started,
            )
        )

    if args.json:
        print(json.dumps({"records": [r.to_dict() for r in records]}, indent=2))
    if args.record:
        written = store.append(records, args.results_dir)
        for record in records:
            print(store.describe_change(record, args.results_dir), file=sys.stderr)
        for path in written:
            print(f"recorded in {path}", file=sys.stderr)
    return 0


def _cmd_check_contract(args: argparse.Namespace) -> int:  # pylint: disable=unused-argument
    """Resolve every declared production entry point.

    Args:
        args: Parsed arguments. Unused — the check takes no options, but the
            signature is fixed by the subcommand dispatch table.

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

    selection = subparsers.add_parser(
        "selection-accuracy",
        parents=[store_args],
        help="Measure whether the planner proposes the tools a case needs.",
    )
    selection.add_argument("--corpus", type=Path, default=None)
    selection.add_argument(
        "--model",
        action="append",
        help="Model identifier, repeatable. Omit to measure whatever the "
        "deployment selects, which is recorded as such.",
    )
    selection.add_argument(
        "--model-env",
        default=production.MODEL_ENV_VAR,
        help=f"Environment variable --model is applied through "
        f"(default: {production.MODEL_ENV_VAR}). Recorded in the row, so a run "
        f"cannot hide which lever it pulled.",
    )
    selection.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Evaluations per case (default: 5, floor of 3 for a usable stddev).",
    )
    selection.add_argument(
        "--runtime-init",
        default="",
        metavar="MODULE:FUNCTION",
        help="Initialiser for the server runtime whose startup populates the "
        "planner's LLM selector. Found automatically when it is where it is "
        "expected; recorded in the row either way.",
    )
    selection.add_argument(
        "--set-env",
        action="append",
        metavar="NAME=VALUE",
        help="Set an environment variable for the run, repeatable. Recorded in "
        "the row. Use it for anything --model does not cover — a provider "
        "switch, BAMBOO_TOOL_RETRIEVAL=0 for the baseline, BAMBOO_MODEL_PRICES.",
    )
    selection.add_argument("--temperature", type=float, default=0.0)
    selection.add_argument("--namespace", default="atlas")
    selection.add_argument("--plugin-id", default="atlas")
    selection.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Measure only the first N cases, for a smoke run. Recorded in the "
        "configuration, so a limited run is never compared with a full one.",
    )
    selection.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Stop after this many calls (default: {DEFAULT_MAX_CALLS}).",
    )
    selection.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help=f"Stop after this many seconds (default: {DEFAULT_MAX_SECONDS:.0f}).",
    )
    selection.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop after this many failed calls in a row (default: 5), rather "
        "than aggregating zeros for a gateway that is down.",
    )
    selection.add_argument(
        "--resume",
        action="store_true",
        help="Reuse calls already in the ledger for this catalogue, remaking "
        "only the ones that errored.",
    )
    selection.add_argument("--show-failures", type=int, default=10)
    selection.add_argument("--json", action="store_true", help="Emit the stored rows as JSON.")
    selection.add_argument(
        "--record",
        action="store_true",
        help="Append the rows to the result store and report the change.",
    )
    selection.set_defaults(func=_cmd_selection_accuracy)

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
