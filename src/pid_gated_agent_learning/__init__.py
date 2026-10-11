"""PID-gated scheduling for continual LLM prompt adaptation under noisy feedback."""

from pid_gated_agent_learning.gates import (
    DerivativeGate,
    Gate,
    IntegralGate,
    ProportionalGate,
)
from pid_gated_agent_learning.providers import (
    AnthropicProvider,
    CachedProvider,
    DeepSeekProvider,
    GeminiProvider,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)
from pid_gated_agent_learning.scheduler import Scheduler
from pid_gated_agent_learning.tasks import (
    Feedback,
    GuessNumber,
    MovingTarget,
    NumberGuessPrompt,
    Parity,
    ParityPrompt,
    SignalConfig,
    Task,
)
from pid_gated_agent_learning.types import EditEvent, Observation, Trajectory

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "CachedProvider",
    "DeepSeekProvider",
    "DerivativeGate",
    "EditEvent",
    "Feedback",
    "Gate",
    "GeminiProvider",
    "GuessNumber",
    "IntegralGate",
    "LLMProvider",
    "MockProvider",
    "MovingTarget",
    "NumberGuessPrompt",
    "Observation",
    "OllamaProvider",
    "OpenAIProvider",
    "Parity",
    "ParityPrompt",
    "ProportionalGate",
    "Scheduler",
    "SignalConfig",
    "Task",
    "Trajectory",
]
