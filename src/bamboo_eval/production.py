"""The only module in this package that imports Bamboo.

Decision E-12, and the reason the framework exists at all: five diagnostic
instruments misreported in sequence during the retrieval work, each believed
because it looked like evidence, and every one of them reported on something
*adjacent to* the code under test.  A probe opened collections by name and
bypassed the routing it was meant to test; a scorer computed cosine on an L2
index and called a healthy index dead.  A measurement harness that is wrong is
worse than none, because it is believed.

So every production symbol a metric depends on is declared in
:data:`ENTRY_POINTS`, resolved through this module, and nowhere else.  One
import site means one place to audit, one place a refactor in bamboo-mcp
breaks, and a contract check that can enumerate what the framework claims to
measure rather than discovering it by running everything.

A missing entry point raises :class:`~bamboo_eval.errors.ProductionContractError`
and fails the run.  It is not a skip: a metric whose entry point has moved is
not a metric that cannot run today, it is a metric that is now measuring
something else.  A missing *optional* dependency — an embedding model, an LLM
gateway — is a skip, because the production path is intact and only the
backend is absent.
"""
from __future__ import annotations

import importlib
import importlib.util  # noqa: F401 - `import importlib` alone does not bind .util
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import MetricSkipped, ProductionContractError

#: Distribution that must be installed for anything here to resolve.  Named in
#: the error so that a fresh checkout's first failure explains itself.
BAMBOO_DISTRIBUTION = "bamboo-core"

_INSTALL_HINT = (
    f"{BAMBOO_DISTRIBUTION} is not importable. bamboo-eval measures Bamboo's "
    f"production code path and cannot substitute for it. Install it with "
    f"'pip install -e ../bamboo-mcp/core' and the ATLAS plugin with "
    f"'pip install -e ../bamboo-mcp/packages/askpanda_atlas'."
)


@dataclass(frozen=True)
class EntryPoint:
    """One production symbol a metric calls.

    Attributes:
        module: Dotted module path inside Bamboo.
        attribute: Symbol name within that module.
        why: What a metric uses it for.  Written for the person reading a
            contract failure months from now, who needs to know whether the
            symbol moved or the measurement did.
        optional: Whether absence is a skip rather than a contract breach.
            True only for backends, never for the path under test.
    """

    module: str
    attribute: str
    why: str
    optional: bool = False

    @property
    def dotted(self) -> str:
        """Return the fully qualified name.

        Returns:
            str: ``module.attribute``.
        """
        return f"{self.module}.{self.attribute}"


#: Every production symbol this package calls.  The contract check walks this
#: table; bamboo-mcp keeps a standalone test asserting the same symbols, so a
#: refactor there fails in its own CI rather than silently in a scheduled run
#: on this repository two weeks later (decision E-22).
ENTRY_POINTS: tuple[EntryPoint, ...] = (
    EntryPoint(
        "bamboo.tools.planner",
        "_collect_tool_catalog",
        "assembles the catalogue the planner is given; the retrieval metrics "
        "measure what this returns, not a reconstruction of it",
    ),
    EntryPoint(
        "bamboo.tools.planner",
        "routing_rules_for_plugin",
        "supplies the routing guidance whose survival the guidance-coverage "
        "metric checks",
    ),
    EntryPoint(
        "bamboo.tools.planner",
        "RoutingRule",
        "the clause/tool pairing the guidance-coverage metric relies on",
    ),
    EntryPoint(
        "bamboo.tools.tool_retrieval",
        "LexicalRetriever",
        "the shipped BM25 retriever under test",
    ),
    EntryPoint(
        "bamboo.tools._tool_retrieval_embedding",
        "catalog_fingerprint",
        "hashes the indexed catalogue text; every stored record cites it",
    ),
    EntryPoint(
        "bamboo.tools._tool_retrieval_embedding",
        "EmbeddingRetriever",
        "embedding backend, for the backend comparison",
        optional=True,
    ),
    EntryPoint(
        "bamboo.tools._tool_retrieval_embedding",
        "HybridRetriever",
        "RRF fusion of the two backends, for the RRF constant sweep",
        optional=True,
    ),
)


def bamboo_available() -> bool:
    """Report whether the system under test is installed at all.

    Returns:
        bool: True when the ``bamboo`` package can be found.
    """
    try:
        return importlib.util.find_spec("bamboo") is not None
    except (ImportError, ValueError):
        return False


def resolve(entry: EntryPoint) -> Any:
    """Resolve one declared entry point.

    Two different failures hide behind "cannot import", and conflating them is
    how a bare checkout and a broken refactor end up looking alike:

    * Bamboo is not installed.  That is a bare checkout, which constraint 4.3
      says must degrade to a stated skip — nothing has moved, there is simply
      nothing to measure here.
    * Bamboo is installed and the symbol is not where this package looks.  That
      is the contract breach, and it fails the run.

    Args:
        entry: The entry point to resolve.

    Returns:
        Any: The production symbol.

    Raises:
        ProductionContractError: If Bamboo is installed but a required symbol
            or its module is missing.
        MetricSkipped: If Bamboo is not installed, or an optional symbol is
            missing.
    """
    if not bamboo_available():
        raise MetricSkipped(_INSTALL_HINT)
    try:
        module = importlib.import_module(entry.module)
    except ImportError as exc:
        message = f"cannot import {entry.module} ({exc}); {entry.why}. {_INSTALL_HINT}"
        if entry.optional:
            raise MetricSkipped(message) from exc
        raise ProductionContractError(message) from exc
    try:
        return getattr(module, entry.attribute)
    except AttributeError as exc:
        message = (
            f"{entry.dotted} is gone; bamboo-eval calls it because it {entry.why}. "
            f"Either restore the symbol or update ENTRY_POINTS in "
            f"bamboo_eval/production.py and the metric that depends on it."
        )
        if entry.optional:
            raise MetricSkipped(message) from exc
        raise ProductionContractError(message) from exc


def check_contract() -> list[str]:
    """Resolve every declared entry point and report what is wrong.

    Returns:
        List[str]: One line per problem, empty when the contract holds.
        Optional entry points that are absent are reported as notes rather
        than omitted, so that a run which silently lost its embedding backend
        is visible in the check rather than only in a missing row.  When
        Bamboo is not installed at all, a single ``NOT INSTALLED`` line is
        returned instead of one breach per symbol, because the contract has
        not been tested rather than failed.
    """
    if not bamboo_available():
        return [f"NOT INSTALLED {BAMBOO_DISTRIBUTION}: {_INSTALL_HINT}"]
    problems: list[str] = []
    for entry in ENTRY_POINTS:
        try:
            resolve(entry)
        except ProductionContractError as exc:
            problems.append(f"BROKEN   {entry.dotted}: {exc}")
        except MetricSkipped as exc:
            problems.append(f"OPTIONAL {entry.dotted} unavailable: {exc}")
    return problems


def _entry(attribute: str) -> EntryPoint:
    """Return the declared entry point for a symbol name.

    Args:
        attribute: The symbol's name.

    Returns:
        EntryPoint: Its declaration.

    Raises:
        ProductionContractError: If the symbol is not declared.  Reaching into
            Bamboo for something undeclared is the exact failure this module
            exists to prevent, so it is an error rather than a lookup miss.
    """
    for entry in ENTRY_POINTS:
        if entry.attribute == attribute:
            return entry
    raise ProductionContractError(
        f"{attribute!r} is not a declared production entry point; add it to "
        f"ENTRY_POINTS with a reason before calling it"
    )


def collect_catalogue(namespace: str = "atlas") -> list[dict[str, Any]]:
    """Return the tool catalogue the planner would be given.

    Args:
        namespace: Plugin namespace to include, or empty for no filter.

    Returns:
        List[Dict[str, Any]]: Catalogue entries, unnarrowed.
    """
    collect = resolve(_entry("_collect_tool_catalog"))
    return collect(namespaces=[namespace] if namespace else None)


def routing_rules(plugin_id: str = "atlas") -> tuple[Any, ...]:
    """Return the routing-guidance clauses for a plugin.

    Args:
        plugin_id: Active plugin identifier.

    Returns:
        Tuple[Any, ...]: ``RoutingRule``-shaped objects in precedence order.
    """
    rules_for = resolve(_entry("routing_rules_for_plugin"))
    return tuple(rules_for(plugin_id))


def catalogue_fingerprint(catalogue: Sequence[Mapping[str, Any]]) -> str:
    """Return the digest of a catalogue's indexed text.

    Args:
        catalogue: Catalogue entries.

    Returns:
        str: Hex digest, full length; callers abbreviate for display.
    """
    fingerprint = resolve(_entry("catalog_fingerprint"))
    return str(fingerprint(catalogue))


def guidance_fingerprint(rules: Sequence[Any]) -> str:
    """Return a digest of the routing guidance in force.

    Guidance drift caused a production misroute before retrieval existed — the
    catalogue advertised ``core_dump_analysis`` while the guidance said
    ``atlas.core_dump_analysis`` — so the guidance is fingerprinted alongside
    the catalogue rather than assumed to follow it.

    Args:
        rules: ``RoutingRule``-shaped objects.

    Returns:
        str: Twelve hex characters, or ``""`` when there is no guidance.
    """
    import hashlib

    if not rules:
        return ""
    digest = hashlib.sha256()
    for rule in rules:
        digest.update(str(getattr(rule, "text", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(",".join(sorted(getattr(rule, "tools", ()))).encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:12]


def retriever(name: str) -> Any:
    """Instantiate a production retriever by name.

    Args:
        name: ``"null"``, ``"lexical"``, ``"embedding"`` or ``"hybrid"``.

    Returns:
        Any: A retriever satisfying the harness protocol.

    Raises:
        ProductionContractError: If *name* is unknown.
        MetricSkipped: If the backend needs a model that is not available.  The
            encoder resolves on first use rather than at construction, so this
            can also surface later, during evaluation.
    """
    from .metrics.tool_retrieval import NullRetriever

    if name == "null":
        return NullRetriever()
    attribute = {
        "lexical": "LexicalRetriever",
        "embedding": "EmbeddingRetriever",
        "hybrid": "HybridRetriever",
    }.get(name)
    if attribute is None:
        raise ProductionContractError(
            f"unknown retriever {name!r}; available: null, lexical, embedding, hybrid"
        )
    factory = resolve(_entry(attribute))
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - backend failures are skips, not crashes
        if type(exc).__name__ == "EncoderUnavailable":
            raise MetricSkipped(f"{name} retriever needs an embedding model: {exc}") from exc
        raise
