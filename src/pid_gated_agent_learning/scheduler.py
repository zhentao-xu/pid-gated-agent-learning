"""The scheduling loop: given a task, a gate, and an LLM editor, run T steps.

The task now owns prediction and the editor prompts (ported from the paper's parent repo).
The Scheduler just coordinates: at each step it asks the task to predict, resolve, and
when the gate fires, calls the editor with the task's own system/user prompts, then asks
the task to parse the response back into a structured prompt.

Example::

    traj = Scheduler(
        task=GuessNumber(seed=0, length=120),
        gate=IntegralGate(),
        editor=OpenAIProvider(model="gpt-4o-mini"),
        noise=0.1,
        seed=0,
        call_budget=120,
    ).run()
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pid_gated_agent_learning.gates import Gate
from pid_gated_agent_learning.providers import LLMProvider
from pid_gated_agent_learning.tasks import Feedback, Task
from pid_gated_agent_learning.types import EditEvent, Observation, Trajectory


@dataclass
class Scheduler:
    """Run one task-trajectory under a gate, calling the editor only when the gate fires."""

    task: Task
    gate: Gate
    editor: LLMProvider
    noise: float = 0.0
    seed: int = 0
    call_budget: int = 180
    history_size: int = 8
    verbose: bool = False
    progress: bool = False  # show a tqdm progress bar if tqdm is installed

    def run(self) -> Trajectory:
        """Execute the full trajectory and return a :class:`Trajectory`."""
        traj: Trajectory | None = None
        for traj in self.run_iter():  # noqa: B007
            pass
        assert traj is not None
        return traj

    def run_iter(self) -> Iterator[Trajectory]:
        """Generator version: yield the growing trajectory after every step."""
        rng = random.Random(self.seed)
        prompt = self.task.optimal_prompt(self.task.rule_at(0))
        feedbacks: list[Feedback] = []
        trajectory = Trajectory(
            task_name=self.task.name,
            gate_name=self.gate.name,
            noise=self.noise,
            seed=self.seed,
            steps=self.task.length,
        )
        calls_used = 0
        snapshots: list[Any] = []  # structured-prompt stack for potential rollback

        iterator: Any = range(self.task.length)
        pbar = None
        if self.progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                tqdm = None
            if tqdm is not None:
                pbar = tqdm(
                    iterator,
                    total=self.task.length,
                    desc=f"{self.task.name}/{self.gate.name}",
                    leave=False,
                )
                iterator = pbar

        for t in iterator:
            if pbar is not None:
                pbar.set_postfix(edits=len(trajectory.edits), calls=calls_used, refresh=False)

            x = self.task.sample_input(rng, t)
            rule = self.task.rule_at(t)
            prediction = self.task.predict(prompt, x)
            _truth_label, observed, _reward, obs_error = self.task.resolve_step(
                x, rule, prediction, rng, self.noise
            )
            # For numeric tasks, log the ground-truth integer; for categorical, the clean label.
            if hasattr(self.task, "target_at"):
                truth_value: Any = self.task.target_at(t)
            else:
                truth_value = self.task.label(x, rule)

            obs = Observation(
                step=t,
                inputs=x,
                prediction=prediction,
                truth=truth_value,
                observed=observed,
                error=float(obs_error),
                verbalization=observed,
            )
            trajectory.observations.append(obs)
            feedbacks.append(Feedback(x=x, predicted=prediction, observed_correct=observed))
            if len(feedbacks) > self.history_size:
                feedbacks = feedbacks[-self.history_size :]

            self.gate.observe(float(obs_error))

            rolled_back = False
            if self.gate.should_rollback() and snapshots:
                restored = snapshots.pop()
                trajectory.edits.append(
                    EditEvent(
                        step=t,
                        prompt_before=self._render(prompt),
                        prompt_after=self._render(restored),
                        reason="rollback (post-edit regression)",
                        rolled_back=True,
                    )
                )
                prompt = restored
                self.gate.on_rollback()
                if self.verbose:
                    print(f"[t={t}] rollback -> restored prior prompt")
                rolled_back = True

            if (
                not rolled_back
                and calls_used < self.call_budget
                and self.gate.should_fire()
            ):
                system_msg = self.task.editor_system_prompt()
                user_msg = self.task.editor_user_prompt(prompt, feedbacks)
                response = self.editor.generate(user_msg, system=system_msg).strip()
                calls_used += 1
                self.gate.on_edit()
                new_prompt = self.task.parse_editor_output(response)
                if new_prompt is None:
                    trajectory.edits.append(
                        EditEvent(
                            step=t,
                            prompt_before=self._render(prompt),
                            prompt_after=self._render(prompt),
                            reason="editor response did not parse",
                        )
                    )
                    if self.verbose:
                        print(f"[t={t}] parse failed (calls={calls_used}): {response!r}")
                elif self._render(new_prompt) == self._render(prompt):
                    trajectory.edits.append(
                        EditEvent(
                            step=t,
                            prompt_before=self._render(prompt),
                            prompt_after=self._render(prompt),
                            reason="editor returned unchanged prompt",
                        )
                    )
                    if self.verbose:
                        print(f"[t={t}] editor returned no change (calls={calls_used})")
                else:
                    trajectory.edits.append(
                        EditEvent(
                            step=t,
                            prompt_before=self._render(prompt),
                            prompt_after=self._render(new_prompt),
                            reason=f"{self.gate.name} fired at error={obs_error}",
                        )
                    )
                    snapshots.append(prompt)
                    prompt = new_prompt
                    if self.verbose:
                        rendered = self._render(new_prompt)[:120]
                        print(f"[t={t}] edit (calls={calls_used}) new_prompt={rendered!r}")

            trajectory.final_prompt = self._render(prompt)
            yield trajectory

        if pbar is not None:
            pbar.set_postfix(edits=len(trajectory.edits), calls=calls_used)
            pbar.close()

    @staticmethod
    def _render(prompt: Any) -> str:
        """Render a structured prompt to text (dataclasses expose ``.render()``)."""
        return prompt.render() if hasattr(prompt, "render") else str(prompt)
