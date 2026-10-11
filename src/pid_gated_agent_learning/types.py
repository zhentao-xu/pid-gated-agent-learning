"""Data types used across the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """One step of agent-environment interaction, recorded for logging and the editor."""

    step: int
    inputs: Any
    prediction: Any
    truth: Any
    observed: Any  # possibly noisy label from the sensor
    error: float  # 0/1 for categorical, bucketed distance for numeric
    verbalization: str = ""  # natural-language description of the error


@dataclass
class EditEvent:
    """Record of a prompt edit (or rollback) performed by the LLM editor."""

    step: int
    prompt_before: str
    prompt_after: str
    reason: str  # e.g., "integral A=1.80 >= tau_I=1.50" or "rollback: post-edit error rose by 0.20"
    rolled_back: bool = False


@dataclass
class Trajectory:
    """Result of running a Scheduler over a task."""

    task_name: str
    gate_name: str
    noise: float
    seed: int
    steps: int
    observations: list[Observation] = field(default_factory=list)
    edits: list[EditEvent] = field(default_factory=list)
    final_prompt: str = ""

    @property
    def num_calls(self) -> int:
        """Number of LLM editor invocations (one per non-rollback edit)."""
        return sum(1 for e in self.edits if not e.rolled_back)

    @property
    def mean_error(self) -> float:
        return sum(o.error for o in self.observations) / max(len(self.observations), 1)

    @property
    def accuracy(self) -> float:
        """Fraction of observations with error == 0 (useful for categorical tasks)."""
        if not self.observations:
            return 0.0
        return sum(1 for o in self.observations if o.error == 0) / len(self.observations)
