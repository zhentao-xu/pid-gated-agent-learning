"""Unit tests for the three PID-inspired gates."""

from __future__ import annotations

from pid_gated_agent_learning import DerivativeGate, IntegralGate, ProportionalGate


def test_proportional_requires_an_error_above_threshold() -> None:
    gate = ProportionalGate(tau_p=1.0)
    gate.observe(0.0)
    assert not gate.should_fire()
    gate.observe(1.0)
    assert gate.should_fire()
    gate.on_edit()
    assert not gate.should_fire()


def test_integral_accumulates_and_resets() -> None:
    gate = IntegralGate(tau_i=1.5, alpha=0.9)
    for e in (1.0, 0.0, 1.0):
        gate.observe(e)
    # A = 0.9*(0.9*(0.9*0 + 1) + 0) + 1 = 1.81 -> fires
    assert gate.should_fire()
    gate.on_edit()
    assert not gate.should_fire()


def test_integral_needs_at_least_one_error_bit() -> None:
    gate = IntegralGate(tau_i=0.1, alpha=0.9)
    gate.observe(0.5)  # non-binary error below the "seen_error" threshold
    assert not gate.should_fire(), "should not fire until a full-unit error has been observed"
    gate.observe(1.0)
    assert gate.should_fire()


def test_derivative_rolls_back_when_post_edit_errors_rise() -> None:
    gate = DerivativeGate(tau_i=1.0, tau_d=0.1, alpha=0.9, window=5)
    # Pre-edit: mostly clean. _recent fills with a mix tilted toward 0.
    for _ in range(5):
        gate.observe(0.0)
    # One isolated error pushes the accumulator just past tau_i.
    gate.observe(1.0)
    gate.observe(1.0)
    assert gate.should_fire(), "integral trigger should fire after two unit errors"
    gate.on_edit()
    # Post-edit: five observations with full-unit errors (worse than pre).
    for _ in range(5):
        gate.observe(1.0)
    assert gate.should_rollback()
    gate.on_rollback()
    assert not gate.should_rollback()


def test_derivative_does_not_rollback_when_post_edit_errors_fall() -> None:
    gate = DerivativeGate(tau_i=1.0, tau_d=0.1, alpha=0.9, window=3)
    # Build up pre-edit history with errors, then fire
    for _ in range(3):
        gate.observe(1.0)
    assert gate.should_fire()
    gate.on_edit()
    # Post-edit: errors drop to zero. No rollback.
    for _ in range(3):
        gate.observe(0.0)
    assert not gate.should_rollback()
