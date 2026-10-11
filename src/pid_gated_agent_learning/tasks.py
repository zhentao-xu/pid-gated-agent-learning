"""Three drifting benchmark tasks (ported from the paper's parent repo).

Each task owns:
    * a structured prompt dataclass (``ParityPrompt`` / ``NumberGuessPrompt``) that
      encodes the agent's belief;
    * a deterministic ``predict(prompt, x)`` that reads the structured prompt and
      returns the agent's answer (no LLM call);
    * an ``editor_system_prompt()`` + ``editor_user_prompt(prompt, feedback)`` pair that
      drives the LLM editor;
    * a ``parse_editor_output(text)`` that converts the editor's response back into a
      structured prompt (returns ``None`` when parsing fails);
    * a ``resolve_step(x, rule, prediction, rng, noise_prob)`` that produces (clean truth,
      observed label, reward, observed-error flag) with sensor noise applied.

See ``src/prompt_adaptation/tasks/`` in the parent repo for the original implementations.
"""

from __future__ import annotations

import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ---- Feedback + Task base --------------------------------------------------


@dataclass
class Feedback:
    """One past (input, prediction, observed) triple, rendered into the editor prompt."""

    x: Any
    predicted: str
    observed_correct: str

    def render(self) -> str:
        return (
            f"For input {self.x!r}, you predicted {self.predicted}, "
            f"but the correct answer was {self.observed_correct}."
        )


class Task(ABC):
    """Environment, deterministic interpreter, and editor adapter.

    ``resolve_step`` returns clean truth, observed feedback, clean reward, and the observed
    binary error. Only observed feedback and error may be passed to the controller.
    """

    name: str = ""
    labels: tuple[str, ...] = ()
    total_steps: int = 120

    @property
    def length(self) -> int:
        """Alias used by the Scheduler and plotting code."""
        return self.total_steps

    def rule_at(self, t: int) -> int:
        """Return the active rule; a continuously moving task uses rule zero."""
        del t
        return 0

    def shift_steps(self) -> list[int]:
        """Return discrete shift boundaries, when the task has any."""
        return []

    @abstractmethod
    def sample_input(self, rng: random.Random, t: int = 0) -> Any:
        """Draw the input at prediction step ``t``."""

    @abstractmethod
    def label(self, x: Any, rule: int) -> str:
        """Return the clean label or target for an input."""

    @abstractmethod
    def optimal_prompt(self, rule: int) -> Any:
        """Construct the initial structured prompt for a rule."""

    @abstractmethod
    def predict(self, prompt: Any, x: Any) -> str:
        """Deterministic predictor: read the structured prompt and return the answer."""

    @abstractmethod
    def editor_system_prompt(self) -> str:
        """Return the task instructions for the shared LLM editor."""

    @abstractmethod
    def editor_user_prompt(self, prompt: Any, verbalizations: list[Feedback]) -> str:
        """Render ordered feedback into an editor request."""

    @abstractmethod
    def parse_editor_output(self, text: str) -> Any | None:
        """Return a structured prompt, or ``None`` when parsing fails."""

    def resolve_step(
        self,
        x: Any,
        rule: int,
        prediction: str,
        rng: random.Random,
        noise_prob: float,
    ) -> tuple[str, str, int, int]:
        """Apply categorical feedback corruption without changing the reward."""
        truth = self.label(x, rule)
        observed = truth
        if rng.random() < noise_prob:
            other = [label for label in self.labels if label != truth]
            if other:
                observed = rng.choice(other)
        return truth, observed, int(prediction == truth), int(prediction != observed)


# ---- Parity (categorical, sudden rule flips) -------------------------------


COLORS: tuple[str, ...] = ("Red", "Blue")


@dataclass
class ParityPrompt:
    """Belief about which color names even/odd integers."""

    even_color: str = "Red"
    odd_color: str = "Blue"

    def render(self) -> str:
        return f"EVEN: {self.even_color}\nODD: {self.odd_color}"

    def copy(self) -> ParityPrompt:
        return ParityPrompt(self.even_color, self.odd_color)


class Parity(Task):
    """Classify a random integer in [1, 10] as Red or Blue by parity.

    Rule 0: even -> Red, odd -> Blue.
    Rule 1: even -> Blue, odd -> Red (mapping reversed).
    """

    name = "parity"
    labels = COLORS
    RULES: dict[int, dict[int, str]] = {
        0: {0: "Red", 1: "Blue"},
        1: {0: "Blue", 1: "Red"},
    }

    def __init__(self, seed: int = 0, length: int = 120) -> None:
        self.total_steps = length
        self._seed = seed

    def rule_at(self, t: int) -> int:
        n = self.total_steps
        return sum(1 for b in (n // 4, n // 2, 3 * n // 4) if t >= b) % 2

    def shift_steps(self) -> list[int]:
        n = self.total_steps
        return [n // 4, n // 2, 3 * n // 4]

    def sample_input(self, rng: random.Random, t: int = 0) -> int:
        del t
        return rng.randint(1, 10)

    def label(self, x: int, rule: int) -> str:
        return self.RULES[rule][x % 2]

    def optimal_prompt(self, rule: int) -> ParityPrompt:
        return ParityPrompt(even_color=self.RULES[rule][0], odd_color=self.RULES[rule][1])

    def predict(self, prompt: ParityPrompt, x: int) -> str:
        return prompt.even_color if x % 2 == 0 else prompt.odd_color

    def editor_system_prompt(self) -> str:
        return (
            "An integer is classified Red or Blue by parity. The two parity classes "
            "have different colors. The mapping may reverse over time. "
            "Output exactly:\nEVEN: <Red or Blue>\nODD: <Red or Blue>"
        )

    def editor_user_prompt(self, prompt: ParityPrompt, verbalizations: list[Feedback]) -> str:
        lines = [v.render() for v in verbalizations]
        return (
            "Current structured prompt:\n" + prompt.render()
            + "\n\nPast observations (oldest to newest):\n" + "\n".join(lines)
            + "\n\nPropose updated instructions in the required format."
        )

    def parse_editor_output(self, text: str) -> ParityPrompt | None:
        even = re.search(r"EVEN\s*:\s*(Red|Blue)\b", text, re.I)
        odd = re.search(r"ODD\s*:\s*(Red|Blue)\b", text, re.I)
        if not even or not odd:
            return None
        e, o = even.group(1).capitalize(), odd.group(1).capitalize()
        if e == o:
            return None
        return ParityPrompt(even_color=e, odd_color=o)


# ---- Numeric: shared prompt + feedback helpers -----------------------------


CERTAINTY_LEVELS = ("for sure", "maybe", "unknown")

MAGNITUDE_BUCKETS: list[tuple[int, str, int]] = [
    (10, "close", 5),
    (20, "moderate", 15),
    (40, "far", 30),
    (101, "very_far", 60),
]


def _bucket_for(distance: int) -> tuple[str, int]:
    for ub, name, rep in MAGNITUDE_BUCKETS:
        if distance < ub:
            return name, rep
    return MAGNITUDE_BUCKETS[-1][1], MAGNITUDE_BUCKETS[-1][2]


def _magnitude_label(target: int, guess: int) -> str:
    if guess == target:
        return "correct"
    direction = "higher" if guess < target else "lower"
    adj, _rep = _bucket_for(abs(target - guess))
    return f"{adj}_{direction}"


NUMBER_GUESS_LABELS: tuple[str, ...] = ("correct",) + tuple(
    f"{adj}_{d}" for _ub, adj, _rep in MAGNITUDE_BUCKETS for d in ("higher", "lower")
)


@dataclass
class NumberGuessPrompt:
    """Belief about a hidden integer target in [1, 100], with per-bound certainty."""

    best_guess: int = 50
    lower: int | None = 1
    lower_certainty: str = "for sure"
    upper: int | None = 100
    upper_certainty: str = "for sure"

    def render(self) -> str:
        lo = "?" if self.lower is None or self.lower_certainty == "unknown" else str(self.lower)
        up = "?" if self.upper is None or self.upper_certainty == "unknown" else str(self.upper)
        return f"target in [{lo}, {up}]"

    def copy(self) -> NumberGuessPrompt:
        return NumberGuessPrompt(
            self.best_guess,
            self.lower, self.lower_certainty,
            self.upper, self.upper_certainty,
        )


def _bucket_ranges() -> dict[str, tuple[int, int]]:
    return {
        name: (0 if i == 0 else MAGNITUDE_BUCKETS[i - 1][0], ub - 1)
        for i, (ub, name, _rep) in enumerate(MAGNITUDE_BUCKETS)
    }


def _parse_number_guess_output(text: str) -> NumberGuessPrompt | None:
    """Parse ``LOWER: N\\nUPPER: M`` from the editor; derive ``best_guess`` as midpoint."""
    cert_alt = r"for\s*sure|maybe|unknown"
    bound_pat = (
        r"{}\s*:\s*(?:"
        r"(UNKNOWN)"
        r"|"
        r"(\d+)\s*(?:[\(\[]?\s*(" + cert_alt + r")\s*[\)\]]?)?"
        r")"
    )

    def _parse_bound(name: str) -> tuple[int | None, str] | None:
        m = re.search(bound_pat.format(name), text, re.IGNORECASE)
        if not m:
            return None
        unknown_hit, num_hit, cert_hit = m.group(1), m.group(2), m.group(3)
        if unknown_hit is not None:
            return None, "unknown"
        try:
            num = int(num_hit) if num_hit is not None else None
        except (TypeError, ValueError):
            return None
        cert = (cert_hit or "for sure").lower().replace("  ", " ")
        if cert not in CERTAINTY_LEVELS:
            cert = "for sure"
        if num is not None:
            num = max(1, min(100, num))
        return num, cert

    lo = _parse_bound("LOWER")
    up = _parse_bound("UPPER")
    if lo is None or up is None:
        return None
    lower, lower_cert = lo
    upper, upper_cert = up
    # Compute best_guess = midpoint (substituting domain limits when a bound is UNKNOWN).
    lo_val = 1 if lower is None else lower
    up_val = 100 if upper is None else upper
    if lo_val > up_val:
        lo_val, up_val = up_val, lo_val
    guess = int(round((lo_val + up_val) / 2))
    return NumberGuessPrompt(
        best_guess=guess,
        lower=lower, lower_certainty=lower_cert,
        upper=upper, upper_certainty=upper_cert,
    )


# ---- Guess number (numeric, abrupt state shifts) ---------------------------


class GuessNumber(Task):
    """Guess a hidden integer in [1, 100] using magnitude-bucketed feedback.

    Non-stationary: the hidden target flips periodically inside the task (25 <-> 75), so a
    fixed guess drifts as soon as the target changes.
    """

    name = "number_guess"
    labels = NUMBER_GUESS_LABELS
    RULES: dict[int, int] = {0: 25, 1: 75}

    def __init__(self, seed: int = 0, length: int = 120) -> None:
        del seed
        self.total_steps = length

    def rule_at(self, t: int) -> int:
        n = self.total_steps
        return sum(1 for b in (n // 4, n // 2, 3 * n // 4) if t >= b) % 2

    def shift_steps(self) -> list[int]:
        n = self.total_steps
        return [n // 4, n // 2, 3 * n // 4]

    def sample_input(self, rng: random.Random, t: int = 0) -> int:
        del rng
        return t

    def label(self, x: int, rule: int) -> str:
        del x
        return str(self.RULES[rule])

    def target_at(self, t: int, rule: int | None = None) -> int:
        return self.RULES[self.rule_at(t) if rule is None else rule]

    def optimal_prompt(self, rule: int) -> NumberGuessPrompt:
        t = self.RULES[rule]
        return NumberGuessPrompt(best_guess=t, lower=t, upper=t)

    def predict(self, prompt: NumberGuessPrompt, x: int) -> str:
        del x
        return str(prompt.best_guess)

    def resolve_step(
        self,
        x: int,
        rule: int,
        prediction: str,
        rng: random.Random,
        noise_prob: float,
    ) -> tuple[str, str, int, int]:
        """Magnitude-bucketed feedback based on the agent's guess."""
        del x
        target = self.RULES[rule]
        try:
            guess = int(prediction)
        except (ValueError, TypeError):
            guess = -1
        truth = _magnitude_label(target, guess)
        observed = truth
        if rng.random() < noise_prob:
            other = [c for c in self.labels if c != truth]
            if other:
                observed = rng.choice(other)
        reward = int(guess == target)
        error = int(observed != "correct")
        return truth, observed, reward, error

    def editor_system_prompt(self) -> str:
        return (
            "You estimate the plausible RANGE of a hidden integer in\n"
            "[1, 100]. You do NOT pick the next guess.\n\n"
            "The user message shows recent observations of the form:\n"
            "  'target lies in [A, B].'\n"
            "meaning the true target was between A and B at that step.\n\n"
            "The target may CHANGE between steps. If the LATEST\n"
            "observation contradicts earlier observations (their ranges\n"
            "don't overlap), the target has moved -- IGNORE the earlier\n"
            "observations and output the range from the LATEST one.\n"
            "Only tighten across multiple observations when their ranges\n"
            "OVERLAP.\n\n"
            "Output exactly two lines (nothing else):\n"
            "LOWER: <int 1-100>\n"
            "UPPER: <int 1-100>\n"
        )

    def editor_user_prompt(self, prompt: NumberGuessPrompt, verbalizations: list[Feedback]) -> str:
        """User prompt with per-guess magnitude-bucketed history, rendered as absolute ranges."""
        del prompt
        br = _bucket_ranges()
        lines: list[str] = []
        for c in verbalizations:
            lbl = c.observed_correct
            if lbl == "correct":
                lines.append(f"- target was exactly {c.predicted}.")
                continue
            parts = lbl.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("higher", "lower"):
                adj, direction = parts
                lo, hi = br.get(adj, (0, 100))
                try:
                    g = int(c.predicted)
                except (TypeError, ValueError):
                    g = 0
                if direction == "higher":
                    a, b = max(1, g + lo), min(100, g + hi)
                else:
                    a, b = max(1, g - hi), min(100, g - lo)
                lines.append(f"- target lies in [{a}, {b}].")
        return (
            "Observations (oldest to newest):\n"
            + "\n".join(lines)
            + "\n\nPropose the new range:"
        )

    def parse_editor_output(self, text: str) -> NumberGuessPrompt | None:
        return _parse_number_guess_output(text)


# ---- Moving target (numeric, smooth continuous drift) ----------------------


@dataclass(frozen=True)
class SignalConfig:
    pattern: str = "sine"
    rate: float = 1.0 / 45.0
    amplitude: float = 25.0
    baseline: float = 50.0


def _signal_at(t: int, cfg: SignalConfig) -> int:
    if cfg.pattern != "sine":
        raise ValueError("Only the paper's sine signal is supported")
    value = cfg.baseline + cfg.amplitude * math.sin(2 * math.pi * cfg.rate * t)
    return int(round(max(1.0, min(100.0, value))))


class MovingTarget(Task):
    """Track a sinusoidal target with nine-category numeric feedback."""

    name = "moving_target"
    labels = NUMBER_GUESS_LABELS

    def __init__(
        self,
        seed: int = 0,
        length: int = 120,
        signal: SignalConfig | None = None,
    ) -> None:
        del seed
        self.total_steps = length
        self.signal = signal if signal is not None else SignalConfig()

    def target_at(self, t: int, rule: int = 0) -> int:
        del rule
        return _signal_at(t, self.signal)

    def sample_input(self, rng: random.Random, t: int = 0) -> int:
        del rng
        return t

    def label(self, x: int, rule: int) -> str:
        return str(self.target_at(x, rule))

    def optimal_prompt(self, rule: int = 0) -> NumberGuessPrompt:
        del rule
        cfg = self.signal
        return NumberGuessPrompt(
            best_guess=int(round(cfg.baseline)),
            lower=int(max(1, round(cfg.baseline - cfg.amplitude))),
            upper=int(min(100, round(cfg.baseline + cfg.amplitude))),
        )

    def predict(self, prompt: NumberGuessPrompt, x: int) -> str:
        del x
        return str(prompt.best_guess)

    def resolve_step(
        self,
        x: int,
        rule: int,
        prediction: str,
        rng: random.Random,
        noise_prob: float,
    ) -> tuple[str, str, int, int]:
        target = self.target_at(x, rule)
        try:
            guess = int(prediction)
        except (ValueError, TypeError):
            guess = -1
        truth = _magnitude_label(target, guess)
        observed = truth
        if rng.random() < noise_prob:
            other = [c for c in self.labels if c != truth]
            if other:
                observed = rng.choice(other)
        reward = int(guess == target)
        error = int(observed != "correct")
        return truth, observed, reward, error

    def editor_system_prompt(self) -> str:
        return (
            "You estimate the plausible RANGE of a MOVING integer\n"
            "y(t) in [1, 100]. The target CHANGES every step, so any\n"
            "previously-estimated range is stale by now.\n\n"
            "The user message shows recent observations of the form:\n"
            "  'At step t, target lay in [A, B].'\n"
            "Output the range from the LATEST (most recent) observation.\n"
            "Ignore older observations -- the target has moved since.\n\n"
            "Output exactly two lines (nothing else):\n"
            "LOWER: <int 1-100>\n"
            "UPPER: <int 1-100>\n"
        )

    def editor_user_prompt(self, prompt: NumberGuessPrompt, verbalizations: list[Feedback]) -> str:
        del prompt
        br = _bucket_ranges()
        lines: list[str] = []
        for c in verbalizations:
            lbl = c.observed_correct
            step_str = f"step {c.x}" if hasattr(c, "x") else "at some step"
            if lbl == "correct":
                lines.append(f"- At {step_str}, target was exactly {c.predicted}.")
                continue
            parts = lbl.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("higher", "lower"):
                adj, direction = parts
                lo, hi = br.get(adj, (0, 100))
                try:
                    g = int(c.predicted)
                except (TypeError, ValueError):
                    g = 0
                if direction == "higher":
                    a, b = max(1, g + lo), min(100, g + hi)
                else:
                    a, b = max(1, g - hi), min(100, g - lo)
                lines.append(f"- At {step_str}, target lay in [{a}, {b}].")
        return (
            "Observations (oldest to newest):\n"
            + "\n".join(lines)
            + "\n\nOutput the range from the LATEST observation:"
        )

    def parse_editor_output(self, text: str) -> NumberGuessPrompt | None:
        return _parse_number_guess_output(text)
