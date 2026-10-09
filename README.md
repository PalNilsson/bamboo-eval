# bamboo-eval

**Bamboo Evaluation Framework** measures retrieval, selection and relevance
against labelled corpora, on the production code path, with every number
fingerprinted. It is not a model benchmark and it is not a dashboard.

That sentence is the scope fence. "Evaluation framework" is broad enough to
invite anything; the charter above is what keeps it finite.

## Why it exists

The tool-retrieval work shipped on a measured claim — recall 1.000 over 120
labelled questions at 39% of the former prompt size. That claim is about
**shortlisting**. It says nothing about whether the planner then selected the
right tool, whether answers improved, or whether anything regressed.

Separately, five diagnostic instruments misreported in sequence during the same
period, each believed because it looked like evidence, while the actual bug
produced no error at all. Every one of them reported on something *adjacent to*
the code under test.

Both facts shape the design.

## Two constraints, neither negotiable

**Every metric names the production entry point it calls, and a test asserts
it.** All of them are declared in `bamboo_eval/production.py`, the only module
in the package that imports Bamboo. A metric that calls a helper rather than
the function the server calls is measuring a path that merely resembles
production. A measurement harness that is wrong is worse than none, because it
is believed.

**Everything is fingerprinted.** Every stored row carries the catalogue
fingerprint, the corpus name, version and SHA-256, the guidance fingerprint,
the model identifiers and the canonical configuration. Three catalogue states
appeared in one week during the retrieval work — 29,527, 29,750 and 30,065
characters — and the same command gave recall 0.983 and 0.992 across two of
them. A fourth, 29,842, turned up the first time this package ran on another
checkout. A number without its fingerprint is not comparable to anything, and
the result store refuses to compare across one.

## Degradation

The package core is pure standard library and runs on a bare checkout. A metric
that needs an embedding model or an LLM gateway records `status="skipped"` with
a stated `skip_reason` — never a wrong number. Two failures that both arrive as
`ImportError` are deliberately kept apart:

| Situation | Result |
|---|---|
| Bamboo is not installed | skip, with the install command |
| Bamboo is installed, a declared symbol has moved | **fail the run** |
| An optional backend is unavailable | skip, with the reason |

## Install

```bash
# the framework
pip install -e .

# the system under test, from a bamboo-mcp checkout beside this one
pip install -e ../bamboo-mcp/core -e ../bamboo-mcp/packages/askpanda_atlas
```

## Use

```bash
bamboo-eval check-contract                 # 0 ok, 1 a symbol moved, 3 Bamboo absent
bamboo-eval tool-retrieval                 # the null baseline
bamboo-eval tool-retrieval --retriever lexical --k 8 --k 10 --k 12
bamboo-eval tool-retrieval --retriever lexical --k 10 --record
bamboo-eval history tool_retrieval_recall --slice hard
bamboo-eval tool-retrieval --retriever lexical --k 10 --min-recall 0.99   # the gate
```

```bash
# phase 1: shown the catalogue, did the planner choose the right tool?
bamboo-eval selection-accuracy --limit 5 --repeats 1            # smoke run
bamboo-eval selection-accuracy --model gpt-oss-20b --repeats 5 --record
bamboo-eval selection-accuracy --model gpt-oss-20b --resume     # after a drop
bamboo-eval history selection_accuracy --slice hard
```

The baseline is the same run with `BAMBOO_TOOL_RETRIEVAL=off` — the planner shown
the whole catalogue, the direct analogue of `NullRetriever`, and the thing a
narrowed run must not be worse than (decision E-26). `BAMBOO_FAST_PATH` is
deliberately neither set nor recorded: the planner never reads it, and calling
`bamboo_plan_tool` already bypasses the fast path.

`--model` is applied through `LLM_DEFAULT_MODEL`, the *default* profile's
model, which is the profile the planner resolves through —
`scripts/probe_llm.py` in bamboo-mcp prints the resolved profiles and is how to
confirm it. Anything that lever does not cover goes through `--set-env
NAME=VALUE`: the provider variable when the model is on another provider
(`claude-haiku-4-5` as the commercial reference point), `BAMBOO_TOOL_RETRIEVAL=off`
for the baseline, `BAMBOO_MODEL_PRICES` so `cost_guard` prices the CERN models
rather than only counting their tokens. Every variable applied is recorded in
the row's `config`; omit `--model` to measure whatever the deployment selects,
which is recorded as such rather than guessed at.

One `--model` per invocation. Bamboo's LLM selector is a process-global
populated from the environment when the server runtime starts, so a second
model measured in the same process would answer under the first one's
selection while the rows named the second. The runtime is started once, inside
the overrides, and `--runtime-init module:function` names its initialiser if it
is not where this package expects.

Every call is appended to `results/ledger/selection_accuracy-<fingerprint>.jsonl`
as it returns, which is what `--resume` reads. **Add `results/ledger/` to
`.gitignore`**: the aggregates are committed, the per-call working state is
not. A run that hits `--max-calls`, `--max-seconds` or five consecutive failed
calls stops, records `status="failed"` with the limit named, and writes no
aggregate — a partial measurement presented as a measurement is the failure
this package exists to prevent (decision E-30).

`--record` appends to `results/<metric>.jsonl` and prints the change against
the most recent *comparable* row, which is what turns `0.992` into
`0.992, down from 1.000 on 2026-10-06`.

## Gating

Deterministic metrics gate pull requests; LLM-dependent metrics run on a
schedule, gate nothing, and raise an issue when they fall outside a declared
band (decision E-11). In bamboo-mcp's CI:

```bash
pip install -e ../bamboo-eval
bamboo-eval check-contract
bamboo-eval tool-retrieval --retriever lexical --k 10 --min-recall 0.99
```

## Results, and OpenSearch

`results/*.jsonl` is the system of record. The row is flat and typed so it can
also be an OpenSearch document without a dynamic mapping; `config` is a
canonical JSON *string* for that reason. Shipping rows to OpenSearch is an
emitter over the same schema, not a second format — see
`EvalRecord` in `bamboo_eval/record.py` for the field list.

JSONL comes first because OpenSearch needs the CERN VPN, and a result store
that cannot record a run from a laptop is the same fragility this package
exists to measure.

## Corpora

One file per metric, sharing one loader (decision E-15). Corpora ship as
package data, so an installed wheel can be evaluated from any working
directory and the artefact released with a DOI is the same file the
measurements used.

`tool_selection_corpus.json` — 120 labelled questions, 87 verbatim from the
project question cheat sheet, 33 authored, 59 flagged `hard`, every
user-facing tool covered with at least four cases, five interface-invoked tools
carrying written exemptions.

The loader rejects rather than skips: an unlabelled case would score as a free
pass against every candidate, and a duplicate identifier makes two results
indistinguishable in a stored row.

## Development setup

```bash
pip install -e '.[dev]'                                   # quote in zsh
pip install -e ../bamboo-mcp/core -e ../bamboo-mcp/packages/askpanda_atlas
pre-commit install
```

Four things bite people setting this up, in roughly this order.

**zsh eats square brackets.** `pip install pyright[nodejs]` gives
`zsh: no matches found`. Quote the whole argument.

**pyright needs node, and will try to download one.** With no `node` on PATH,
`pyright-python` bootstraps one via `nodeenv` from nodejs.org — which fails
behind a TLS-inspecting proxy, and on a fresh python.org framework install that
has no root certificates (pip is unaffected; it carries its own). Either
install node once (`brew install node`), or `pip install 'pyright[nodejs]'`,
which takes node from PyPI as a wheel and never touches nodejs.org. The wheel
route is the one that works on `aipanda033`, where outbound HTTPS to
nodejs.org is unlikely to be permitted at all.

**pyright without pytest reports four errors that are not errors.**
`pytest.skip()` is typed `NoReturn`, which is what tells pyright that a fixture
ending in a skip does not fall off the end. With pytest unresolved it reports
`must return value on all code paths` in `conftest.py`, an `Any | None` return
in `test_phase0_parity.py`, and a possibly-unbound `retriever`. All four are
downstream of the `Import "pytest" could not be resolved` warnings — install
the `dev` extra and they disappear.

That matters for the pre-commit hook in particular: it runs pyright in its own
isolated environment and locates Python via `PATH`, so committing from an
activated virtualenv passes and committing from a shell where it is not active
fails, with no obvious connection to what changed. Pin the interpreter rather
than relying on `PATH` — keep the virtualenv inside the repository as `.venv`
(gitignored) and add to `pyrightconfig.json`:

```json
"venvPath": ".",
"venv": ".venv"
```

An absolute path to a virtualenv elsewhere works on one laptop and breaks in CI
and on `aipanda033`.

**pylint takes explicit targets.** Unlike flake8 and pydocstyle it does not
default to the working directory, and bare `pylint` just prints
`No files to lint: exiting.`

```bash
pylint src tests
```

## Quality gate

```bash
pytest -q
flake8 src tests
pydocstyle --convention=google src tests
pyright
pylint src tests
```

All five are expected to be clean, pylint at 10.00/10. Its `R09xx` size
thresholds are raised in `pyproject.toml` to fit a package of frozen
dataclasses and explicit parameter lists; the remaining suppressions are
per-file, with the reason written at the top of each file, so that
`unused-argument` and `inconsistent-return-statements` keep working on `src/`
where they catch real defects.

## Status

Phase 0 complete: the package, the loader, the record, the store, the contract,
and the tool-retrieval metric ported from bamboo-mcp with its numbers verified
unchanged (`tests/test_phase0_parity.py`).

Next: end-task selection accuracy — did the planner, shown the right tool,
choose it. That is the one missing number that changes what can be claimed.
