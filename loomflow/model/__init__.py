"""Model adapters.

* :class:`EchoModel` — zero-key, echoes the prompt; default
* :class:`ScriptedModel` — replays canned turns for tests
* :class:`AnthropicModel` — Claude via the ``anthropic`` SDK
* :class:`OpenAIModel` — GPT via the ``openai`` SDK

The provider adapters import their SDK lazily inside ``__init__`` so
``from loomflow.model import AnthropicModel`` works even without
the corresponding extra installed; the ImportError is raised only when
the constructor needs to build a default client.
"""

from .anthropic import AnthropicModel
from .echo import EchoModel
from .litellm import LiteLLMModel
from .openai import OpenAIModel
from .scripted import ScriptedModel, ScriptedTurn

__all__ = [
    "AnthropicModel",
    "EchoModel",
    "LiteLLMModel",
    "OpenAIModel",
    "ScriptedModel",
    "ScriptedTurn",
]
