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

import asyncio
import contextlib
import hashlib
import importlib
import importlib.util  # noqa: F401 - `import importlib` alone does not bind .util
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .errors import BambooEvalError, MetricSkipped, PlanParseError, ProductionContractError
from .metrics.tool_retrieval import NullRetriever

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
        "bamboo.tools.planner",
        "bamboo_plan_tool",
        "is the planner the server itself calls; the selection-accuracy metric "
        "measures what this returns rather than a reconstruction of the "
        "planning path (decision E-25)",
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


#: Environment variable the planner's model is selected through.
#:
#: The planner resolves through the *default* profile — ``route`` takes the
#: fast profile and ``log_analysis``/``rag_answer`` the reasoning one, so the
#: default profile is the planner's.  ``scripts/probe_llm.py`` in bamboo-mcp
#: prints the resolved profiles, which is how this was established rather than
#: assumed.
#:
#: Named here rather than inside a metric because it is a fact about Bamboo's
#: configuration, and because a run that pulled a lever nobody checked is a run
#: that measured the default model while reporting another one.  Every run
#: records the variables it set, so the lever is visible in the row rather than
#: only in this comment.
#:
#: Switching to a model on another provider — ``claude-haiku-4-5`` as the
#: commercial reference point of decision E-4 — needs the provider variable
#: too, which is what :func:`env_overrides` is for.
MODEL_ENV_VAR = "LLM_DEFAULT_MODEL"

#: Environment variables that change what the planner is shown, and therefore
#: what a selection measurement means (decision E-26).  ``BAMBOO_FAST_PATH`` is
#: deliberately absent: the planner never reads it, and recording a variable
#: that had no effect would misdescribe the run.
RETRIEVAL_ENV_VARS: tuple[str, ...] = (
    "BAMBOO_TOOL_RETRIEVAL",
    "BAMBOO_TOOL_RETRIEVAL_K",
    "BAMBOO_TOOL_RETRIEVAL_RRF_K",
    "BAMBOO_TOOL_RETRIEVAL_MIN_CATALOG",
    "BAMBOO_TOOL_RETRIEVAL_LOG",
)


def retrieval_settings() -> dict[str, str | None]:
    """Return the retrieval environment as it stands for this process.

    Returns:
        Dict[str, Optional[str]]: Each variable in :data:`RETRIEVAL_ENV_VARS`
        with its value, or ``None`` where it is unset.  Unset is recorded
        rather than omitted, because "the default was in force" and "nobody
        looked" are different runs.
    """
    return {name: os.environ.get(name) for name in RETRIEVAL_ENV_VARS}


@contextlib.contextmanager
def env_overrides(overrides: Mapping[str, str]) -> Iterator[dict[str, str]]:
    """Apply environment overrides for the duration of the block.

    Bamboo resolves its LLM profiles from the environment, so this is how a
    measurement selects what it measures.  Restored afterwards, including the
    difference between "was set to something else" and "was not set": leaving
    a process configured differently is how the second run of a session
    measures something the first one chose.

    Args:
        overrides: Variables to set.

    Yields:
        Dict[str, str]: What was applied, for the record.  The values are
        model and provider identifiers, never credentials — the gateway's key
        is not something a run sets or stores.
    """
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield dict(overrides)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def model_selected(model: str, env_var: str = MODEL_ENV_VAR) -> Iterator[str]:
    """Select the planner's model for the duration of the block.

    Args:
        model: Model identifier, or empty to leave the deployment's own
            selection alone — which is itself a measurable configuration and is
            recorded as such rather than silently substituted.
        env_var: Variable to set; see :data:`MODEL_ENV_VAR`.

    Yields:
        str: The variable that was set, or ``""`` when nothing was changed, so
        a caller can record which lever it pulled.
    """
    if not model:
        yield ""
        return
    with env_overrides({env_var: model}):
        yield env_var


def _first_text_block(blocks: Any) -> str:
    """Return the text of the first content block a tool returned.

    Args:
        blocks: Whatever the tool's ``call`` returned — a sequence of MCP
            content objects, which may be attribute-style or mapping-style
            depending on the SDK version in use.

    Returns:
        str: The first block's text.

    Raises:
        PlanParseError: If there is no block, or the first one carries no text.
    """
    if not isinstance(blocks, Sequence) or not blocks:
        raise PlanParseError(f"planner returned no content ({type(blocks).__name__})")
    first = blocks[0]
    text = first.get("text") if isinstance(first, Mapping) else getattr(first, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("planner returned a content block with no text")
    return text


def _strip_code_fence(text: str) -> str:
    """Return *text* with a surrounding Markdown code fence removed.

    Args:
        text: The returned text.

    Returns:
        str: The text inside the fence, or the text unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return body.rsplit("```", 1)[0].strip()


def parse_plan(text: str) -> dict[str, Any]:
    """Parse the planner's text block into a plan.

    The planner validates its own output against the ``Plan`` model before
    returning it, so text arriving here that is not a plan means the tool
    returned something else — an error payload, or prose where JSON was
    expected.  That is a distinct defect from choosing the wrong tool, which is
    why it gets its own exception and its own outcome bucket.

    Args:
        text: The text block the planner returned.

    Returns:
        Dict[str, Any]: The plan, as the fields a metric reads: ``route``,
        ``confidence``, ``tool_calls``.

    Raises:
        PlanParseError: If the text is not a JSON object carrying a ``route``
            and a list of ``tool_calls`` whose entries name a tool.
    """
    payload = _strip_code_fence(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"planner output is not JSON: {exc}", payload) from exc
    if not isinstance(parsed, dict):
        raise PlanParseError(
            f"planner output is a JSON {type(parsed).__name__}, not a plan object",
            payload,
        )
    if not isinstance(parsed.get("route"), str):
        raise PlanParseError("plan has no route", payload)
    calls = parsed.get("tool_calls")
    if not isinstance(calls, list):
        raise PlanParseError("plan has no tool_calls list", payload)
    for call in calls:
        if not isinstance(call, Mapping) or not isinstance(call.get("tool"), str):
            raise PlanParseError(f"plan has a tool call without a tool name: {call!r}", payload)
    return parsed


def plan(
    question: str,
    namespaces: Sequence[str] = ("atlas",),
    temperature: float = 0.0,
    max_tokens: int = 900,
    plugin_id: str | None = None,
    runtime_init: str = "",
) -> dict[str, Any]:
    """Ask the production planner for a plan, without executing it.

    Decision E-25.  This calls ``bamboo_plan_tool.call()`` — the object the MCP
    server itself dispatches to — rather than the LLM client underneath it, so
    the catalogue narrowing, the prompt assembly and the schema validation that
    happen on the way are all part of what is measured.  Reconstructing any of
    that here would produce a metric for a path that merely resembles
    production, which is the failure mode the framework exists to rule out.

    ``execute`` is passed explicitly as ``False`` although that is also the
    default: whether a plan was executed is too important to leave implied.

    Args:
        question: The corpus question, verbatim.
        namespaces: Plugin namespaces the planner may draw tools from.
        temperature: Sampling temperature; 0 where the gateway honours it.
        max_tokens: Generation limit, the tool's own default.
        plugin_id: Active plugin, or ``None`` for the server's default.
        runtime_init: ``module:function`` initialising the server runtime, or
            empty to find it.  See :func:`ensure_runtime`.

    Returns:
        Dict[str, Any]: The parsed plan.

    Raises:
        ProductionContractError: If the entry point has moved, or no longer has
            the asynchronous ``call(arguments)`` shape this depends on.
        MetricSkipped: If Bamboo is not installed.
        PlanParseError: If what came back is not a plan.
        BambooEvalError: If called from inside a running event loop, which this
            wrapper cannot drive a coroutine from.
    """
    ensure_runtime(runtime_init)
    tool = resolve(_entry("bamboo_plan_tool"))
    call = getattr(tool, "call", None)
    if not inspect.iscoroutinefunction(call):
        raise ProductionContractError(
            f"{_entry('bamboo_plan_tool').dotted}.call is not a coroutine function "
            f"(found {type(call).__name__}); bamboo-eval drives it with asyncio.run, "
            f"so a change of shape here changes what the metric measures"
        )
    arguments: dict[str, Any] = {
        "question": question,
        "namespaces": list(namespaces),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "execute": False,
    }
    if plugin_id:
        arguments["plugin_id"] = plugin_id
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise BambooEvalError(
            "bamboo_eval.production.plan() was called from inside a running event "
            "loop; drive the planner's coroutine directly in that case"
        )
    return parse_plan(_first_text_block(asyncio.run(call(arguments))))


#: Values ``BAMBOO_TOOL_RETRIEVAL`` accepts.  An unrecognised value does not
#: fail there — it warns and falls back to ``lexical`` — so a run asking for a
#: baseline with ``0`` measures the narrowed catalogue and records ``0`` in its
#: configuration.  That is a row that lies about what it measured, which is
#: why :func:`check_retrieval_setting` refuses the value here instead.
RETRIEVAL_BACKENDS: tuple[str, ...] = ("off", "lexical", "embedding", "hybrid")

#: Where the server runtime's initialiser might live.  The planner resolves its
#: model through an LLM selector that ``create_server()`` populates; calling the
#: planner without it raises ``RuntimeError: LLM selector is not initialized``.
#: Tried in order, and overridable with ``--runtime-init module:function``, so
#: a move does not need a release of this package.
RUNTIME_INIT_CANDIDATES: tuple[str, ...] = (
    "bamboo.core:create_server",
    "bamboo:create_server",
    "bamboo.server:create_server",
    "bamboo.core.server:create_server",
)

#: Process state for :func:`ensure_runtime`.  ``server`` holds the returned
#: object so it is not garbage-collected under the selector it populated.
_RUNTIME: dict[str, Any] = {"spec": "", "environment": None, "server": None}


def check_retrieval_setting(value: str) -> None:
    """Reject a ``BAMBOO_TOOL_RETRIEVAL`` value Bamboo would silently replace.

    Args:
        value: The value a run is about to set.

    Raises:
        BambooEvalError: If it is not one of :data:`RETRIEVAL_BACKENDS`.  The
            baseline is ``off``, not ``0``: ``0`` warns, falls back to
            ``lexical``, and leaves a row claiming a baseline it did not run.
    """
    if value not in RETRIEVAL_BACKENDS:
        raise BambooEvalError(
            f"BAMBOO_TOOL_RETRIEVAL={value!r} is not a backend; Bamboo would warn "
            f"and fall back to 'lexical', so the run would measure retrieval while "
            f"the row said {value!r}. Expected one of {list(RETRIEVAL_BACKENDS)} — "
            f"the baseline is 'off'."
        )


def ensure_runtime(spec: str = "") -> str:
    """Initialise the server runtime the planner needs, once per process.

    ``bamboo_plan_tool`` resolves its model through an LLM selector that the
    server's own startup populates.  Called without it, the planner raises
    ``RuntimeError: LLM selector is not initialized`` on every case — which the
    consecutive-error guard turns into a stopped run rather than a 0.000, but
    stopping is not the same as working.

    The selector is process-global and is populated from the environment as it
    stands at initialisation, so this is called *inside* the environment
    overrides a run applies, and a second initialisation under a different
    environment is refused rather than attempted: the planner would otherwise
    answer under the first model's selection while the rows named the second.

    Args:
        spec: ``module:function`` to call, or empty to try
            :data:`RUNTIME_INIT_CANDIDATES` in order.

    Returns:
        str: The spec that was used, or the one already in force.

    Raises:
        MetricSkipped: If no initialiser can be found.  Not a contract breach:
            a checkout without a usable runtime cannot be measured, and saying
            so is constraint 4.3.
        ProductionContractError: If the named initialiser cannot be called, or
            if the environment has changed since the runtime started.
    """
    environment = tuple(sorted(retrieval_settings().items())) + (
        (MODEL_ENV_VAR, os.environ.get(MODEL_ENV_VAR)),
    )
    if _RUNTIME["server"] is not None:
        if _RUNTIME["environment"] != environment:
            raise ProductionContractError(
                "the server runtime was started under a different environment and "
                "the LLM selector it populated is process-global; measuring a "
                "second configuration here would answer under the first one. Run "
                "one configuration per invocation."
            )
        return str(_RUNTIME["spec"])

    candidates = (spec,) if spec else RUNTIME_INIT_CANDIDATES
    for candidate in candidates:
        module_name, _, attribute = candidate.partition(":")
        try:
            initialiser = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError, ValueError):
            continue
        try:
            server = initialiser()
            if inspect.isawaitable(server):
                server = asyncio.run(server)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ProductionContractError(
                f"{candidate} could not be called with no arguments ({exc}); pass "
                f"the right initialiser with --runtime-init module:function"
            ) from exc
        _RUNTIME.update(spec=candidate, environment=environment, server=server)
        return candidate

    raise MetricSkipped(
        f"no server runtime initialiser found (tried {list(candidates)}); the "
        f"planner's LLM selector is populated by the server's startup, so the "
        f"metric cannot run without it. Name it with --runtime-init "
        f"module:function."
    )
