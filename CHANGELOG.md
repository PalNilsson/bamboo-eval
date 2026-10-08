# Changelog

All notable changes to bamboo-eval are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning.

## [Unreleased]

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
