"""Labelled corpora, and the one loader they all share.

Decision E-15: each metric gets its own corpus file rather than one file with
a polymorphic ``expected`` field.  Tool selection labels tools, RAG relevance
labels source documents, and answer quality will label assertions; forcing
those into one schema would mean a loader that cannot validate any of them
properly, and a validator that cannot tell a mislabelled case from a case of
another kind.

What they share is everything except the label: an identifier, a question,
where the question came from, whether it is deliberately confusable, and the
coverage exemptions that make an absent label a recorded decision rather than
a blind spot.  That shared part lives here, and a new metric adds a case class
rather than a loader.

The loader rejects rather than skips.  A case with no label would score as a
free pass on every candidate, and a duplicate identifier makes two different
results indistinguishable in a stored record; both are corpus bugs that get
quieter the longer they survive, so they fail at load.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar, cast

from .errors import CorpusError


@dataclass(frozen=True)
class BaseCase:
    """Fields every labelled case has, whatever it is labelled with.

    Attributes:
        case_id: Stable identifier, used to name a case in a report and to
            join a result to its case across runs.
        question: The question, as a user would type it.
        source: Where the question came from — ``"cheatsheet"`` for one lifted
            verbatim from project documentation, ``"authored"`` for one written
            for the corpus, ``"harvested"`` for one taken from a real user.
            Recorded because self-authorship is a threat to validity that a
            reader must be able to quantify.
        hard: Whether the case is deliberately paired with a confusable
            neighbour.  Reported as its own slice, because aggregate scores are
            dominated by easy cases and stay healthy-looking while the
            confusable ones rot.
        notes: Why the case exists, where that is not obvious.
    """

    case_id: str
    question: str
    source: str = "authored"
    hard: bool = False
    notes: str = ""

    def expected_labels(self) -> frozenset[str]:
        """Return the labels a correct result must produce.

        Returns:
            FrozenSet[str]: The case's labels, whatever kind they are.

        Raises:
            NotImplementedError: Always; a concrete case type defines this.
        """
        raise NotImplementedError

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], base: Mapping[str, Any]) -> "BaseCase":
        """Build a case from one raw corpus entry.

        Args:
            entry: The raw JSON object for this case.
            base: Validated values for the fields declared on :class:`BaseCase`,
                ready to splat into the constructor.

        Returns:
            BaseCase: The parsed case.

        Raises:
            NotImplementedError: Always; a concrete case type defines this.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ToolSelectionCase(BaseCase):
    """A question labelled with the tools a correct plan must name.

    Attributes:
        expected_tools: Every tool a correct result must produce.  More than
            one is a co-occurrence requirement, not a choice: half of a pair is
            a broken plan, not a partial one, so scoring is strict.
    """

    expected_tools: frozenset[str] = frozenset()

    def expected_labels(self) -> frozenset[str]:
        """Return the expected tool names.

        Returns:
            FrozenSet[str]: The tools a correct result must produce.
        """
        return self.expected_tools

    @classmethod
    def from_entry(
        cls, entry: Mapping[str, Any], base: Mapping[str, Any]
    ) -> "ToolSelectionCase":
        """Build a tool-selection case from one raw corpus entry.

        Args:
            entry: The raw JSON object for this case.
            base: Validated :class:`BaseCase` field values.

        Returns:
            ToolSelectionCase: The parsed case.
        """
        expected = frozenset(str(tool) for tool in entry.get("expected_tools", []))
        return cls(**base, expected_tools=expected)


@dataclass(frozen=True)
class RagRelevanceCase(BaseCase):
    """A question labelled with the source documents an answer should use.

    Defined now, unused until phase 4, so that the loader's generality is
    exercised by a second case type rather than asserted by a comment.

    Attributes:
        expected_sources: Source document identifiers, as the RAG probe reports
            them in ``top_source``.
        topic: The collection the question belongs to, so a per-topic query set
            can be selected.  The single PanDA-specific suite applied to every
            collection is why a routing bug survived two sessions.
    """

    expected_sources: frozenset[str] = frozenset()
    topic: str = ""

    def expected_labels(self) -> frozenset[str]:
        """Return the expected source document identifiers.

        Returns:
            FrozenSet[str]: The sources a correct answer should draw on.
        """
        return self.expected_sources

    @classmethod
    def from_entry(
        cls, entry: Mapping[str, Any], base: Mapping[str, Any]
    ) -> "RagRelevanceCase":
        """Build a RAG-relevance case from one raw corpus entry.

        Args:
            entry: The raw JSON object for this case.
            base: Validated :class:`BaseCase` field values.

        Returns:
            RagRelevanceCase: The parsed case.
        """
        sources = frozenset(str(src) for src in entry.get("expected_sources", []))
        return cls(**base, expected_sources=sources, topic=str(entry.get("topic", "")))


CaseT = TypeVar("CaseT", bound=BaseCase)


@dataclass(frozen=True)
class Corpus(Generic[CaseT]):
    """A labelled corpus together with everything a record needs to cite it.

    Attributes:
        cases: The labelled questions, in file order.
        name: Short identifier, used in stored records.  Defaults to the file
            stem, so a corpus cannot be cited under a name it does not carry.
        version: The corpus file's own version counter.  Bumped whenever cases
            change, so a score can be attributed to the labels it was measured
            against.
        sha256: Digest of the corpus file's bytes.  The version counter records
            intent; the digest records fact, and the two disagreeing is itself
            worth knowing.
        cases: The labelled questions, in file order.
        coverage_exempt: Label to the reason it needs no cases — a tool invoked
            by the interface rather than asked for in words, say.  Recorded with
            the data rather than in a test, so that adding a tool forces a
            decision: write cases, or write down why not.
        description: What the corpus is for.
        path: Where it was loaded from, or ``None`` when built in memory.
    """

    cases: tuple[CaseT, ...]
    name: str = "in-memory"
    version: int = 0
    sha256: str = ""
    coverage_exempt: Mapping[str, str] = field(default_factory=dict)
    description: str = ""
    path: Path | None = None

    def expected_labels(self) -> frozenset[str]:
        """Return every label named by any case.

        Returns:
            FrozenSet[str]: Union of all cases' labels; empty for an empty
            corpus.
        """
        if not self.cases:
            return frozenset()
        return frozenset().union(*(case.expected_labels() for case in self.cases))

    def hard_cases(self) -> tuple[CaseT, ...]:
        """Return the cases flagged *hard*.

        Returns:
            Tuple[CaseT, ...]: Confusable cases, in file order.
        """
        return tuple(case for case in self.cases if case.hard)


def _base_kwargs(entry: Mapping[str, Any], seen: set[str]) -> dict[str, Any]:
    """Validate and extract the fields every case type shares.

    Args:
        entry: The raw JSON object for one case.
        seen: Identifiers already used, mutated with this case's identifier.

    Returns:
        Dict[str, Any]: Constructor keyword arguments for :class:`BaseCase`.

    Raises:
        CorpusError: If the case has no identifier, reuses one, or has no
            question.
    """
    case_id = str(entry.get("id", "")).strip()
    if not case_id:
        raise CorpusError(f"corpus case without an id: {entry!r}")
    if case_id in seen:
        raise CorpusError(f"duplicate corpus case id: {case_id}")
    question = str(entry.get("question", "")).strip()
    if not question:
        raise CorpusError(f"corpus case {case_id} has no question")
    seen.add(case_id)
    return {
        "case_id": case_id,
        "question": question,
        "source": str(entry.get("source", "authored")),
        "hard": bool(entry.get("hard", False)),
        "notes": str(entry.get("notes", "")),
    }


def load_corpus(path: Path, case_type: type[CaseT]) -> Corpus[CaseT]:
    """Load a labelled corpus from JSON.

    Args:
        path: Path to the corpus file.
        case_type: The concrete case class this file holds.  Passing the wrong
            one surfaces as a case with no labels rather than silently
            producing unlabelled cases, because the emptiness check below runs
            against whatever ``expected_labels`` the chosen type reads.

    Returns:
        Corpus[CaseT]: The parsed corpus, with its digest computed from the
        bytes on disk rather than from the re-serialised structure.

    Raises:
        CorpusError: If the file is not a JSON object, if a case lacks an
            identifier, a question or a label, or if two cases share an
            identifier.
    """
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CorpusError(f"{path} must hold a JSON object, found {type(raw).__name__}")

    cases: list[CaseT] = []
    seen: set[str] = set()
    for entry in raw.get("cases", []):
        base = _base_kwargs(entry, seen)
        # ``from_entry`` is declared on BaseCase and overridden covariantly by
        # each case type; the cast records that the caller's choice of
        # *case_type* is what fixes the element type, which the base signature
        # cannot express before typing.Self (3.11).
        case = cast(CaseT, case_type.from_entry(entry, base))
        if not case.expected_labels():
            raise CorpusError(
                f"corpus case {base['case_id']} carries no label; an unlabelled "
                f"case scores as a free pass on every candidate"
            )
        cases.append(case)

    return Corpus(
        cases=tuple(cases),
        name=str(raw.get("name", path.stem)),
        version=int(raw.get("version", 0)),
        sha256=digest,
        coverage_exempt=dict(raw.get("coverage_exempt", {})),
        description=str(raw.get("description", "")),
        path=path,
    )


def bundled_corpus_path(name: str) -> Path:
    """Return the path of a corpus shipped inside this package.

    Shipping the corpora as package data rather than as repository fixtures
    means an installed wheel can be evaluated against them from any working
    directory, and that the artefact released with a DOI is the same file the
    measurements used.

    Args:
        name: File name, with or without the ``.json`` suffix.

    Returns:
        Path: Absolute path to the bundled corpus file.

    Raises:
        CorpusError: If no such corpus is bundled.
    """
    stem = name[:-5] if name.endswith(".json") else name
    path = Path(__file__).resolve().parent / "data" / f"{stem}.json"
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.json"))
        raise CorpusError(f"no bundled corpus {stem!r}; available: {available}")
    return path
