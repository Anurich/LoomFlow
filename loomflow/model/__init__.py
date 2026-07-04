"""Model adapters.

* :class:`EchoModel` — zero-key, echoes the prompt; default
* :class:`ScriptedModel` — replays canned turns for tests
* :class:`AnthropicModel` — Claude via the ``anthropic`` SDK
* :class:`OpenAIModel` — GPT via the ``openai`` SDK
* :class:`FallbackModel` — ordered failover chain across models

The provider adapters import their SDK lazily inside ``__init__`` so
``from loomflow.model import AnthropicModel`` works even without
the corresponding extra installed; the ImportError is raised only when
the constructor needs to build a default client.
"""

from .anthropic import AnthropicModel
from .count_tokens import count_tokens
from .echo import EchoModel
from .fallback import FallbackModel, default_fall_on
from .litellm import LiteLLMModel
from .openai import OpenAIModel
from .scripted import ScriptedModel, ScriptedTurn

__all__ = [
    "AnthropicModel",
    "EchoModel",
    "FallbackModel",
    "LiteLLMModel",
    "OpenAIModel",
    "ScriptedModel",
    "ScriptedTurn",
    "count_tokens",
    "default_fall_on",
]
