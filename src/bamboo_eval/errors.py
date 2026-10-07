"""Typed errors.

The framework's first design constraint is that a measurement which cannot be
made must say so, loudly and with a reason, rather than degrade to a number
that looks like a measurement.  Every failure mode below therefore has a name,
and nothing in this package raises a bare :class:`Exception` or returns a
sentinel value a caller might mistake for data.
"""
from __future__ import annotations


class BambooEvalError(Exception):
    """Base class for every error this package raises."""


class ProductionContractError(BambooEvalError):
    """A production entry point a metric depends on could not be resolved.

    Raised rather than skipped.  A metric whose production entry point has
    moved is not a metric that cannot run today — it is a metric that is now
    measuring nothing, and the only safe response is to fail the run.  See
    :mod:`bamboo_eval.production` for the declared entry points and
    ``docs`` for decision E-12.
    """


class MetricSkipped(BambooEvalError):
    """A metric could not run for a stated, benign reason.

    The reason is carried so that it reaches the stored record's
    ``skip_reason`` field.  A missing embedding model or an unreachable LLM
    gateway is a skip; a missing production entry point is not.

    Attributes:
        reason: Why the metric could not run, in a form a reader of a stored
            record will understand without the surrounding context.
    """

    def __init__(self, reason: str) -> None:
        """Initialise the error.

        Args:
            reason: Why the metric could not run.
        """
        super().__init__(reason)
        self.reason = reason


class CorpusError(BambooEvalError, ValueError):
    """A corpus file is malformed, mislabelled or internally inconsistent.

    Also a :class:`ValueError`, because that is what the harness raised before
    the move and callers that caught it should keep working.
    """
