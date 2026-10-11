"""Discrete editing gates inspired by PID control.

Each gate decides *when* to invoke an LLM editor based on observed error signals; the
content of the edit is chosen by the editor itself. All three gates share an eligibility
check (unused call budget, at least one observed error since the last reset) that the
Scheduler enforces separately.

See the companion paper for the formulation and empirical evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field


class Gate(ABC):
    """Abstract base class for a scheduling gate."""

    name: str = "gate"

    @abstractmethod
    def observe(self, error: float) -> None:
        """Called every step with the latest observed error (non-negative)."""

    @abstractmethod
    def should_fire(self) -> bool:
        """Return True if the editor should be invoked right now (ignoring budget / cooldown)."""

    @abstractmethod
    def on_edit(self) -> None:
        """Called immediately after the editor has been invoked (success or failure)."""

    def should_rollback(self) -> bool:
        """Return True if the gate has detected a post-edit regression and wants a rollback.

        Default: no gate wants a rollback. The Derivative gate overrides this.
        """
        return False

    def on_rollback(self) -> None:  # noqa: B027  (intentional no-op default)
        """Called when the Scheduler restores a previous prompt in response to should_rollback()."""


@dataclass
class ProportionalGate(Gate):
    """Fire whenever the current error >= tau_P.

    The most responsive schedule, and the most vulnerable to isolated label-noise.
    """

    tau_p: float = 1.0

    _last_error: float = field(default=0.0, init=False, repr=False)
    _seen_error: bool = field(default=False, init=False, repr=False)
    name: str = field(default="P", init=False)

    def observe(self, error: float) -> None:
        self._last_error = error
        if error >= 1:
            self._seen_error = True

    def should_fire(self) -> bool:
        return self._seen_error and self._last_error >= self.tau_p

    def on_edit(self) -> None:
        self._last_error = 0.0
        self._seen_error = False


@dataclass
class IntegralGate(Gate):
    """Accumulate a decayed error sum and fire when it crosses tau_I.

    A_t = alpha * A_{t-1} + e_t,  fire if A_t >= tau_I.
    """

    tau_i: float = 1.5
    alpha: float = 0.9

    _accumulator: float = field(default=0.0, init=False, repr=False)
    _seen_error: bool = field(default=False, init=False, repr=False)
    name: str = field(default="I", init=False)

    def observe(self, error: float) -> None:
        self._accumulator = self.alpha * self._accumulator + error
        if error >= 1:
            self._seen_error = True

    def should_fire(self) -> bool:
        return self._seen_error and self._accumulator >= self.tau_i

    def on_edit(self) -> None:
        self._accumulator = 0.0
        self._seen_error = False


@dataclass
class DerivativeGate(Gate):
    """Integral trigger + a pre/post regression check that can roll back an edit.

    After firing, the gate watches the next ``window`` observations and asks the Scheduler to
    roll back if the post-edit error mean exceeds the pre-edit mean by at least tau_D.
    """

    tau_i: float = 1.5
    tau_d: float = 0.15
    alpha: float = 0.9
    window: int = 5

    _accumulator: float = field(default=0.0, init=False, repr=False)
    _seen_error: bool = field(default=False, init=False, repr=False)
    _recent: deque[float] = field(default_factory=deque, init=False, repr=False)
    _pre_mean: float | None = field(default=None, init=False, repr=False)
    _post_window: list[float] = field(default_factory=list, init=False, repr=False)
    _cooldown: int = field(default=0, init=False, repr=False)
    name: str = field(default="D", init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def observe(self, error: float) -> None:
        if self._pre_mean is not None:
            self._post_window.append(error)
        self._recent.append(error)
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        self._accumulator = self.alpha * self._accumulator + error
        if error >= 1:
            self._seen_error = True

    def should_fire(self) -> bool:
        return (
            self._cooldown == 0
            and self._pre_mean is None  # not currently watching a post-edit window
            and self._seen_error
            and self._accumulator >= self.tau_i
        )

    def should_rollback(self) -> bool:
        if self._pre_mean is None or len(self._post_window) < self.window:
            return False
        post_mean = sum(self._post_window) / len(self._post_window)
        return post_mean >= self._pre_mean + self.tau_d

    def on_edit(self) -> None:
        if len(self._recent) == self.window:
            self._pre_mean = sum(self._recent) / self.window
            self._post_window = []
        self._accumulator = 0.0
        self._seen_error = False
        self._cooldown = self.window

    def on_rollback(self) -> None:
        self._pre_mean = None
        self._post_window = []
        self._accumulator = 0.0
        self._seen_error = False
