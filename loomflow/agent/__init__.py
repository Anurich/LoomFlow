"""The public :class:`Agent` class and its supporting machinery."""

from .api import Agent, Tuning
from .auto_compact import context_window_for, maybe_auto_compact
from .snip import snip_messages

__all__ = [
    "Agent",
    "Tuning",
    "context_window_for",
    "maybe_auto_compact",
    "snip_messages",
]
