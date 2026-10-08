"""Call and wall-clock limits for metrics that spend a gateway.

Decision E-30.  A metric that drives a model can fail in a way the
deterministic ones cannot: not by producing a wrong number, but by producing a
number over the part of the corpus that finished before the run was killed,
the gateway rate-limited it, or somebody's patience ran out.  Such a number is
indistinguishable from a real one once it is in the store.

So the limits are declared up front, the spend is counted, and reaching a limit
raises :class:`~bamboo_eval.errors.BudgetExceeded`, which stops the run before
anything is aggregated.  The ledger keeps what was already paid for, so a
resumed run continues rather than starting again — the budget bounds the cost,
it does not discard it.

The module is deliberately tiny and free of metric knowledge: phase 1 is the
first metric to need it and will not be the last.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .errors import BudgetExceeded

#: Defaults, chosen to be reached rather than theoretical: 120 cases times 5
#: repeats times three models is 1,800 calls, so 2,000 stops a run that has
#: started repeating itself without stopping the run that was intended.
DEFAULT_MAX_CALLS = 2000
DEFAULT_MAX_SECONDS = 3600.0


@dataclass
class Budget:
    """A call and wall-clock allowance for one run.

    Attributes:
        max_calls: Calls the run may make, or ``None`` for no limit.
        max_seconds: Wall-clock seconds the run may take, or ``None`` for no
            limit.  Measured from construction, so a resumed run gets a fresh
            allowance — the limit is on the sitting, not on the measurement.
        calls: Calls spent so far.
        started: Monotonic clock reading at construction.
    """

    max_calls: int | None = DEFAULT_MAX_CALLS
    max_seconds: float | None = DEFAULT_MAX_SECONDS
    calls: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        """Return wall-clock seconds since the budget was created.

        Returns:
            float: Elapsed seconds.
        """
        return time.monotonic() - self.started

    def check(self) -> None:
        """Raise if a limit has already been reached.

        Called before a call rather than after it, so that the limit bounds
        what is spent rather than recording that it was overspent.

        Raises:
            BudgetExceeded: If either limit has been reached.
        """
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"call budget exhausted: {self.calls} calls, limit {self.max_calls} "
                f"(--max-calls); the ledger is resumable with --resume"
            )
        if self.max_seconds is not None and self.elapsed_s >= self.max_seconds:
            raise BudgetExceeded(
                f"time budget exhausted: {self.elapsed_s:.0f}s elapsed, limit "
                f"{self.max_seconds:.0f}s (--max-seconds); the ledger is "
                f"resumable with --resume"
            )

    def spend(self, calls: int = 1) -> None:
        """Record calls made.

        Args:
            calls: How many calls were made.
        """
        self.calls += calls
