# Changelog

All notable changes to bamboo-eval are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning.

## [Unreleased]

### Added — phase 1, end-task selection accuracy (E-25 – E-31)

- `bamboo_eval.metrics.selection_accuracy`: does the planner, shown the
  catalogue, propose the tools a correct plan needs? Strict containment
  (`expected ⊆ proposed`), directly comparable with `tool_retrieval_recall`,
  reported beside precision over the proposed tools and never fused into an F1
  (decision E-1). Six outcomes, not four — `correct`, `wrong_tool`, `declined`,
  `unknown_tool`, `unparseable`, `error` — which partition the observations, so
  the stored counters sum to `n_cases * repeats` (E-28). Name resolution is
  explicit and counted: exact match, then the segment after the last `.`, with
  a suffix matching two catalogue entries refused rather than guessed (E-27).
  Repeats carry mean, standard deviation and per-case unanimity (E-2), and the
  planner's own confidence is stored as its own metric row (E-31).
- `bamboo_eval.ledger`: append-only per-call working state at
  `results/ledger/<metric>-<fingerprint>.jsonl`, written as each call returns,
  so a 1,800-call run survives a dropped VPN (E-29). The last write for a key
  wins; a truncated final line is a killed run and is dropped; any earlier
  broken line is refused rather than skipped. `--resume` reuses it and nothing
  else does, so a rerun cannot quietly answer itself out of a file.
- `bamboo_eval.budget`: call and wall-clock limits, checked before each call
  (E-30). Reaching one stops the run, records `status="failed"` naming the
  limit, and writes no aggregate. A sixth consecutive failed call stops it too
  — a gateway that is down otherwise produces a confident 0.000.
- `production.plan()` and the eighth declared entry point
  `bamboo.tools.planner.bamboo_plan_tool` (E-25), called with `execute=False`
  and driven by `asyncio.run`, with a check that `call` is still a coroutine
  function. `parse_plan()` rejects what is not a plan as its own error type, so
  `unparseable` stays distinct from a transport failure.
- `production.retrieval_settings()` and `production.model_selected()`:
  `BAMBOO_TOOL_RETRIEVAL` and its tuning variables are recorded;
  `BAMBOO_FAST_PATH` is neither set nor recorded, because the planner never
  reads it (E-26, superseding the phrasing of E-24).
- CLI `bamboo-eval selection-accuracy` with `--model` (repeatable),
  `--model-env`, `--repeats`, `--limit`, `--max-calls`, `--max-seconds`,
  `--max-consecutive-errors`, `--resume`, `--record`. No threshold option:
  LLM-dependent metrics gate nothing (E-11).
- 73 new tests (143 in total), all offline. A stub planner drives every
  outcome, the whole resolution table, resume, both budget guards and the
  consecutive-error guard; a stand-in module exercises the planner wrapper on a
  bare checkout.

### Changed — phase 1

- `record.py`: `SCHEMA_VERSION` 2, adding `n_declined` and `n_unknown_tool`
  (E-28). A count that lives in a text field is a count nobody queries.
- `record.py`: `failed_record()` beside `skipped_record()`. A skip says the
  measurement could not be attempted; a failure says it was attempted and
  abandoned, and only the second needs a reason naming a limit.
- `store.comparable()` now checks `COMPARABLE_SCHEMA_VERSIONS` rather than
  equality, so the schema bump does not silently retire the phase 0 retrieval
  history it did not invalidate. Valid only while versions differ by added
  fields with defaults; the constant says so.
- `pyproject.toml`: pylint `max-args` raised to 10 for `evaluate()`, whose
  parameters after the first three are keyword-only.

### Added

- Package skeleton `bamboo_eval` (src layout), pure standard library in the
  core, with optional extras `bamboo`, `embedding`, `llm`, `opensearch`, `dev`.
- `bamboo_eval.corpus`: one loader over several case types (decision E-15).
  `BaseCase` carries the shared fields, `ToolSelectionCase` and
  `RagRelevanceCase` carry their own labels. Rejects unlabelled cases,
  duplicate identifiers, missing questions and malformed files. A corpus
  carries its own name, declared version and SHA-256 so it can be cited in a
  stored row.
- `bamboo_eval.record`: the flat, fully fingerprinted `EvalRecord` (decision
  E-13), doubling as the OpenSearch document (decision E-17). A row with
  `status != "ok"` must carry a `skip_reason`; a row with `status == "ok"` must
  carry a value.
- `bamboo_eval.store`: append-only JSONL result store with comparability rules
  that refuse to compare across a different catalogue, corpus, configuration or
  model, and `describe_change` to render a value against its predecessor.
- `bamboo_eval.production`: the only module that imports Bamboo (decision
  E-12). Declares all seven production entry points with the reason each is
  called, and distinguishes "Bamboo not installed" (skip) from "declared symbol
  has moved" (fail).
- `bamboo_eval.metrics.tool_retrieval`: the retrieval harness, ported from
  `core/bamboo/evaluation/tool_retrieval.py` in bamboo-mcp with behaviour
  unchanged, plus `report_to_records` emitting four rows per report.
- `tool_selection_corpus.json` (120 cases) moved here as package data
  (decision E-23).
- CLI `bamboo-eval` with `tool-retrieval`, `check-contract` and `history`.
  `check-contract` exits 0/1/3 so that a CI job which forgot to install the
  system under test cannot report the same green as one that checked it.
- `tests/test_phase0_parity.py` — phase 0's acceptance criterion as a test:
  every aggregate is compared against the numbers the pre-move harness produced
  on the same catalogue, recorded per fingerprint in
  `tests/data/reference_numbers.json`. Verified passing on catalogue
  `7e3891f672ac` (22 tools, 29,842 chars).
- 70 tests; the 59 that need no Bamboo install pass on a bare checkout.

### Changed

- `EvalRecord.from_dict` uses `dataclasses.fields()` rather than the private
  `__dataclass_fields__`, which is the public API and which type checkers can
  actually see.
- `production.py` imports `hashlib` and `NullRetriever` at module level. Both
  were function-level imports with no cycle to justify them; verified acyclic.
- `cli/main.py` grew `_record_skip`, replacing two copies of the
  report-and-store-a-skip block.
- Four tests in `test_record_store.py` no longer take an unused `tmp_path`.

### Notes

- `results/` is intended to be committed. The history is the point.
- pylint is clean at 10.00/10. Its `R09xx` size thresholds are raised in
  `pyproject.toml` to fit frozen dataclasses and explicit parameter lists;
  `duplicate-code` is disabled globally because it is cross-file and cannot be
  suppressed from inside either file. Everything else is suppressed per file
  with a written reason, so `unused-argument` and
  `inconsistent-return-statements` keep working on `src/`.
- README carries a "Development setup" section covering the node/pyright
  bootstrap, the pytest-unresolved false errors, the pre-commit interpreter
  trap and pylint's lack of default targets.
