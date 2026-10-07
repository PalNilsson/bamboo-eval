"""The generic loader (decision E-15).

The claim being tested is that one loader serves several case types without
weakening what it validates for any of them.  A loader that generalised by
dropping checks would be worse than the per-metric duplication it replaced,
so the second case type is exercised here rather than only declared.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bamboo_eval.corpus import (
    Corpus,
    RagRelevanceCase,
    ToolSelectionCase,
    bundled_corpus_path,
    load_corpus,
)
from bamboo_eval.errors import CorpusError


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Write a throwaway corpus file.

    Args:
        tmp_path: Pytest temporary directory.
        payload: The whole corpus object.

    Returns:
        Path: The written file.
    """
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestGenerality:
    """A second case type loads with the same validation."""

    def test_rag_cases_load(self, tmp_path: Path) -> None:
        """A RAG-relevance corpus parses into its own case type."""
        path = _write(
            tmp_path,
            {
                "version": 2,
                "cases": [
                    {
                        "id": "r1",
                        "question": "How do I set a Rucio quota?",
                        "expected_sources": ["rucio/admin.md"],
                        "topic": "rucio",
                    }
                ],
            },
        )
        corpus = load_corpus(path, RagRelevanceCase)
        assert corpus.cases[0].topic == "rucio"
        assert corpus.expected_labels() == frozenset({"rucio/admin.md"})
        assert corpus.version == 2

    def test_an_unlabelled_rag_case_is_rejected_too(self, tmp_path: Path) -> None:
        """Generality does not come at the cost of the emptiness check.

        The check runs against whatever ``expected_labels`` the chosen type
        reads, so it holds for a case type that did not exist when it was
        written.
        """
        path = _write(tmp_path, {"cases": [{"id": "r1", "question": "q"}]})
        with pytest.raises(CorpusError, match="carries no label"):
            load_corpus(path, RagRelevanceCase)

    def test_loading_with_the_wrong_case_type_fails_loudly(self, tmp_path: Path) -> None:
        """A tool corpus read as a RAG corpus raises instead of loading empty labels.

        The failure mode this prevents: every case silently unlabelled, and a
        metric reporting a perfect score against nothing.
        """
        path = _write(
            tmp_path, {"cases": [{"id": "c1", "question": "q", "expected_tools": ["a"]}]}
        )
        with pytest.raises(CorpusError, match="carries no label"):
            load_corpus(path, RagRelevanceCase)


class TestIdentity:
    """A corpus can cite itself."""

    def test_digest_is_of_the_bytes_on_disk(self, tmp_path: Path) -> None:
        """Reformatting the file changes the digest even when the cases do not.

        The version counter records intent and the digest records fact; the
        point of storing both is that they can disagree.
        """
        cases = [{"id": "c1", "question": "q", "expected_tools": ["a"]}]
        compact = _write(tmp_path, {"version": 1, "cases": cases})
        first = load_corpus(compact, ToolSelectionCase)
        compact.write_text(
            json.dumps({"version": 1, "cases": cases}, indent=4), encoding="utf-8"
        )
        second = load_corpus(compact, ToolSelectionCase)
        assert first.sha256 != second.sha256
        assert first.version == second.version

    def test_name_defaults_to_the_file_stem(self, tmp_path: Path) -> None:
        """A corpus with no declared name is cited by its file name."""
        path = _write(tmp_path, {"cases": [{"id": "c1", "question": "q", "expected_tools": ["a"]}]})
        assert load_corpus(path, ToolSelectionCase).name == "corpus"


class TestMalformedFiles:
    """Structural problems fail at load rather than mid-measurement."""

    def test_invalid_json_is_rejected(self, tmp_path: Path) -> None:
        """A truncated file names itself in the error."""
        path = tmp_path / "corpus.json"
        path.write_text('{"cases": [', encoding="utf-8")
        with pytest.raises(CorpusError, match="not valid JSON"):
            load_corpus(path, ToolSelectionCase)

    def test_a_json_list_is_rejected(self, tmp_path: Path) -> None:
        """The top level must be an object, since the metadata lives beside the cases."""
        path = tmp_path / "corpus.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(CorpusError, match="must hold a JSON object"):
            load_corpus(path, ToolSelectionCase)

    def test_a_case_without_an_id_is_rejected(self, tmp_path: Path) -> None:
        """An anonymous case cannot be joined to its result across runs."""
        path = _write(tmp_path, {"cases": [{"question": "q", "expected_tools": ["a"]}]})
        with pytest.raises(CorpusError, match="without an id"):
            load_corpus(path, ToolSelectionCase)


class TestBundledCorpora:
    """Corpora ship inside the package, not beside it."""

    def test_the_tool_corpus_is_bundled(self) -> None:
        """An installed wheel can be evaluated from any working directory.

        It also means the artefact released with a DOI is the same file the
        measurements used.
        """
        assert bundled_corpus_path("tool_selection_corpus").is_file()
        assert bundled_corpus_path("tool_selection_corpus.json").is_file()

    def test_an_unknown_corpus_lists_what_exists(self) -> None:
        """The error names the available corpora rather than only the missing one."""
        with pytest.raises(CorpusError, match="tool_selection_corpus"):
            bundled_corpus_path("no_such_corpus")


class TestInMemoryCorpora:
    """A corpus built in code needs no file."""

    def test_defaults_make_a_corpus_constructible_from_cases_alone(self) -> None:
        """Metric tests construct corpora directly; the citation fields default."""
        corpus: Corpus[ToolSelectionCase] = Corpus(
            cases=(
                ToolSelectionCase(
                    case_id="c1", question="q", expected_tools=frozenset({"a"})
                ),
            )
        )
        assert corpus.name == "in-memory"
        assert corpus.sha256 == ""
        assert corpus.expected_labels() == frozenset({"a"})
