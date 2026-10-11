"""Smoke tests for Scheduler on each task with a mock provider."""

from __future__ import annotations

import pytest

import pid_gated_agent_learning as p
from pid_gated_agent_learning import (
    GuessNumber,
    IntegralGate,
    MockProvider,
    MovingTarget,
    Parity,
    Scheduler,
    Task,
)


def _run(task: Task, mock_response: str) -> None:
    editor = MockProvider(response=mock_response)
    traj = Scheduler(
        task=task,
        gate=IntegralGate(tau_i=1.5, alpha=0.9),
        editor=editor,
        noise=0.1,
        seed=0,
        call_budget=180,
        history_size=8,
    ).run()
    assert len(traj.observations) == task.length
    assert traj.task_name == task.name
    assert traj.num_calls <= 180
    assert 0.0 <= traj.accuracy <= 1.0
    assert traj.mean_error >= 0.0


def test_parity_runs_end_to_end() -> None:
    _run(Parity(seed=0, length=24), "EVEN: Red\nODD: Blue")


def test_guess_number_runs_end_to_end() -> None:
    _run(GuessNumber(seed=0, length=24), "LOWER: 20\nUPPER: 30")


def test_moving_target_runs_end_to_end() -> None:
    _run(MovingTarget(seed=0, length=24), "LOWER: 40\nUPPER: 60")


def test_no_calls_when_budget_zero() -> None:
    editor = MockProvider(response="EVEN: Red\nODD: Blue")
    traj = Scheduler(
        task=Parity(seed=0, length=24),
        gate=IntegralGate(),
        editor=editor,
        noise=0.0,
        seed=0,
        call_budget=0,
    ).run()
    assert traj.num_calls == 0
    assert len(traj.edits) == 0


def test_all_optional_providers_import_cleanly() -> None:
    try:
        from google import genai  # noqa: F401

        inst_g = p.GeminiProvider(api_key="dummy")
        assert inst_g.model.startswith("gemini")
    except ImportError:
        with pytest.raises(ImportError, match="google-genai"):
            p.GeminiProvider(api_key="dummy")
    try:
        import openai  # noqa: F401

        inst = p.DeepSeekProvider(api_key="dummy")
        assert inst.model == "deepseek-chat"
    except ImportError:
        with pytest.raises(ImportError, match="openai"):
            p.DeepSeekProvider(api_key="dummy")
