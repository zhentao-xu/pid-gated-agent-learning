# pid-gated-agent-learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Control-theoretic **scheduling** for continual LLM prompt adaptation under noisy feedback.
When should an agent revise its instructions in response to feedback that may itself be
wrong? This library offers three lightweight **gates** — Proportional, Integral, and
Derivative — that decide *when* to invoke an LLM-based prompt editor, with any provider
(OpenAI, Anthropic, Ollama, or your own).

Defaults to OpenAI; swap in Anthropic, Gemini, DeepSeek, Ollama, or any custom provider via a one-method interface.

## Install

Not published to PyPI — install directly from GitHub:

```bash
# Default: OpenAI backend
pip install "pid-gated-agent-learning[openai] @ git+https://github.com/zhentao-xu/pid-gated-agent-learning.git"

# Other providers: anthropic, gemini, deepseek, ollama, all
# Example:
pip install "pid-gated-agent-learning[anthropic] @ git+https://github.com/zhentao-xu/pid-gated-agent-learning.git"
```

Equivalent with [uv](https://docs.astral.sh/uv/):

```bash
uv add "git+https://github.com/zhentao-xu/pid-gated-agent-learning.git#egg=pid-gated-agent-learning[openai]"
```

Or clone and install locally for development — see [Development](#development) below.

## Quickstart

**👉 See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for the full end-to-end walkthrough.**

Set your OpenAI key first:

```bash
cp .env.template .env
# edit .env and set OPENAI_API_KEY
```

Minimal example:

```python
from pid_gated_agent_learning import IntegralGate, OpenAIProvider, Parity, Scheduler

trajectory = Scheduler(
    task=Parity(seed=0, length=120),
    gate=IntegralGate(tau_i=1.5, alpha=0.9),
    predictor=lambda prompt, x: "red" if x % 2 == 1 else "blue",
    editor=OpenAIProvider(),              # reads OPENAI_API_KEY + OPENAI_MODEL from env
    noise=0.1,
    seed=0,
).run()

print(f"accuracy = {trajectory.accuracy:.2%}, editor calls = {trajectory.num_calls}")
```

Prefer a different backend? Swap `OpenAIProvider()` for `AnthropicProvider()`, `GeminiProvider()`, `DeepSeekProvider()`, `OllamaProvider()`, or `MockProvider(response=...)` for a dry run with no API key. Any object with `generate(prompt, system=None) -> str` works as a provider.

## Gates

| Gate | Fires when… | When to use |
|------|-------------|-------------|
| `ProportionalGate(tau_p)` | current error exceeds `tau_p` | Low-noise regimes, fast adaptation |
| `IntegralGate(tau_i, alpha)` | decayed error accumulator `A ≥ tau_i` | **Default.** Robust to isolated noise |
| `DerivativeGate(tau_i, tau_d, window)` | Integral trigger + rollback if post-edit mean rises by `tau_d` over pre-edit mean | Highest robustness; fewest calls |

See `src/pid_gated_agent_learning/gates.py` for implementations (short enough to read in one sitting).

## Tasks

Three drifting benchmarks are included out of the box:

- **`Parity`** — categorical (red/blue), sudden rule flips at steps 30/60/90
- **`GuessNumber`** — numeric, hidden target shifts every 40 steps
- **`MovingTarget`** — numeric, smooth sinusoidal drift

Add your own by subclassing `Task` and implementing `input_at`, `observe`, and `initial_prompt`.

## Development

```bash
git clone https://github.com/zhentao-xu/pid-gated-agent-learning.git
cd pid-gated-agent-learning
uv sync                 # installs core + dev + notebook deps (one command)
source .venv/bin/activate
pytest
jupyter notebook examples/quickstart.ipynb
```

If you prefer pip, use the `notebook` extra instead:

```bash
pip install -e ".[notebook]"      # core + notebook deps
pip install -e ".[notebook,all]"  # also pull every provider SDK
```

### API keys for provider examples

Copy `.env.template` to `.env` and fill in your keys:

```bash
cp .env.template .env
# edit .env and set OPENAI_API_KEY / ANTHROPIC_API_KEY as needed
set -a; source .env; set +a       # or use direnv / python-dotenv
```

`.env` is gitignored; `.env.template` is the committed documentation of which
environment variables the providers read.

Formatting / linting:

```bash
ruff check .
ruff format .
mypy src
```

## Why PID?

Agents in non-stationary environments have to decide not just **what** to change about
their instructions, but **when** to make a change — and whether to react to a single
mistake that may have been mislabeled. Classical proportional–integral–derivative
control gives three complementary answers:

- **P**: react to the current error
- **I**: accumulate evidence over time (noise-robust)
- **D**: watch for post-edit deterioration and roll back

This library packages those three heuristics as drop-in schedulers for any LLM-based
prompt editor. In our paper, the Integral gate matches TextGrad quality on three
drifting tasks while issuing roughly 2.7–3.5× fewer LLM calls.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{xu2026pidgated,
  title   = {PID-Gated Continual Prompt Adaptation Under Noisy Feedback},
  author  = {Zhentao Xu},
  year    = {2026},
  eprint  = {TBD},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
