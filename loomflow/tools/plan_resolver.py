"""Resolver for the ``living_plan=`` :class:`Agent` kwarg.

Recognised forms:

* ``None`` / ``False`` — disabled (default in v0.10.0; v0.11 will
  flip the ``None`` default to "auto" — on for tool-using agents).
* ``True`` — enabled. Workspace mirror auto-on when the Agent
  also has ``workspace=`` wired; otherwise in-memory only.
* ``"memory"`` / ``"inmemory"`` — enabled, in-memory only (never
  mirrors even if a workspace exists).
* ``"workspace"`` — enabled, REQUIRES the agent to also have
  ``workspace=`` set; raises :class:`ConfigError` otherwise.
* ``Mapping`` — declarative dict form. Recognised keys:

    * ``enabled: bool`` — explicit flag (default True).
    * ``mirror: "workspace" | "none"`` — where to persist.
      Default ``"workspace"`` when a workspace is set, else
      ``"none"``.
    * ``include_recall: bool`` — also wire ``recall_past_plans``
      (the cross-task plan-lineage tool). Default True when
      mirror is on (the "loomflow-native creative twist" —
      costs nothing if the model never calls it). Raises
      :class:`ConfigError` if True without a workspace mirror.
    * ``task_id: str`` — embedded in the workspace slug title
      for cross-task discoverability. Defaults to the active
      :class:`RunContext`'s ``run_id``.
    * ``author: str`` — author identity for mirror writes.
      Default ``"agent"`` (or the workspace member's name if a
      :class:`WorkspaceMembership` is set on the Agent).
    * Unknown keys → :class:`ConfigError`.

* :class:`LivingPlan` instance — passthrough; pre-seeds the run's
  plan with this state. Useful for resume / hand-off tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import ConfigError
from .plan import LivingPlan

__all__ = ["ResolvedLivingPlan", "resolve_living_plan"]


@dataclass(frozen=True, slots=True)
class ResolvedLivingPlan:
    """The post-resolve view of ``living_plan=``.

    :class:`Agent.__init__` builds this once, stashes it on
    ``self._living_plan_spec``, and reads from it at run time to
    decide which tools to wire and whether to mirror.
    """

    enabled: bool
    """``True`` if plan tools should be wired onto this agent."""

    mirror_to_workspace: bool
    """``True`` if every ``plan_write`` should also persist to
    the agent's workspace as a ``kind="plan"`` note."""

    include_recall: bool
    """``True`` if the ``recall_past_plans`` tool should also be
    exposed (only meaningful when ``mirror_to_workspace`` is
    True — otherwise nothing to recall)."""

    task_id: str | None
    """Optional explicit task identifier embedded in mirror note
    titles. ``None`` means "use ``RunContext.run_id`` at write
    time"."""

    author: str | None
    """Optional explicit author for mirror writes. ``None`` lets
    :class:`Agent.__init__` pick the workspace-membership name
    (if any) or fall back to ``"agent"``."""

    seed_plan: LivingPlan | None
    """Optional pre-seeded plan to populate the run's plan with
    before the model sees its first turn. Useful for resume."""

    auto_stop_hook: bool = True
    """When ``True`` (default), ``Agent.__init__`` auto-prepends
    a :class:`StopHook` that re-prompts the model when any plan
    step is still ``doing``/``todo`` after the architecture's
    final answer. This is the framework's Ralph-loop default —
    multi-step plans don't silently exit mid-stream. Set
    ``living_plan={"auto_stop_hook": False, ...}`` to disable;
    user-supplied ``stop_hooks=`` still run."""

    @classmethod
    def disabled(cls) -> ResolvedLivingPlan:
        return cls(
            enabled=False,
            mirror_to_workspace=False,
            include_recall=False,
            task_id=None,
            author=None,
            seed_plan=None,
            auto_stop_hook=False,
        )


def resolve_living_plan(
    spec: Any,
    *,
    workspace_present: bool,
) -> ResolvedLivingPlan:
    """Coerce ``living_plan=`` into a :class:`ResolvedLivingPlan`.

    ``workspace_present`` tells the resolver whether the agent has
    a workspace wired — needed to validate ``"workspace"`` /
    ``include_recall=True`` shapes and to pick the right default
    for the mirror.
    """
    if spec is None or spec is False:
        return ResolvedLivingPlan.disabled()
    if spec is True:
        return ResolvedLivingPlan(
            enabled=True,
            mirror_to_workspace=workspace_present,
            include_recall=workspace_present,
            task_id=None,
            author=None,
            seed_plan=None,
        )
    if isinstance(spec, LivingPlan):
        # Passthrough — pre-seed the run's plan with this state.
        return ResolvedLivingPlan(
            enabled=True,
            mirror_to_workspace=workspace_present,
            include_recall=workspace_present,
            task_id=None,
            author=None,
            seed_plan=spec,
        )
    if isinstance(spec, str):
        return _resolve_string(spec, workspace_present=workspace_present)
    if isinstance(spec, Mapping):
        return _resolve_dict(spec, workspace_present=workspace_present)
    raise ConfigError(
        "living_plan= must be bool, str, Mapping, LivingPlan, or None. "
        f"Got: {type(spec).__name__}"
    )


def _resolve_string(
    spec: str,
    *,
    workspace_present: bool,
) -> ResolvedLivingPlan:
    s = spec.strip().lower()
    if not s:
        raise ConfigError(
            "living_plan= empty string. Use 'memory', 'workspace', "
            "True, or False (or omit for the default)."
        )
    if s in ("memory", "inmemory", "ephemeral"):
        return ResolvedLivingPlan(
            enabled=True,
            mirror_to_workspace=False,
            include_recall=False,
            task_id=None,
            author=None,
            seed_plan=None,
        )
    if s in ("workspace", "disk", "persist", "mirror"):
        if not workspace_present:
            raise ConfigError(
                "living_plan='workspace' requires Agent(workspace=...) "
                "to also be set — there's nothing to mirror to."
            )
        return ResolvedLivingPlan(
            enabled=True,
            mirror_to_workspace=True,
            include_recall=True,
            task_id=None,
            author=None,
            seed_plan=None,
        )
    raise ConfigError(
        f"living_plan= string {spec!r} not recognised. Use 'memory', "
        "'workspace', True, or False."
    )


def _resolve_dict(
    spec: Mapping[str, Any],
    *,
    workspace_present: bool,
) -> ResolvedLivingPlan:
    accepted_keys = {
        "enabled",
        "mirror",
        "include_recall",
        "task_id",
        "author",
        "seed_plan",
        "auto_stop_hook",
    }
    extra = set(spec.keys()) - accepted_keys
    if extra:
        raise ConfigError(
            f"living_plan= dict has unknown keys: {sorted(extra)}. "
            f"Accepted: {sorted(accepted_keys)}."
        )

    enabled = bool(spec.get("enabled", True))
    if not enabled:
        return ResolvedLivingPlan.disabled()

    mirror_spec = spec.get("mirror")
    if mirror_spec is None:
        # Default: mirror when a workspace exists, otherwise not.
        mirror = workspace_present
    elif isinstance(mirror_spec, str):
        low = mirror_spec.strip().lower()
        if low in ("workspace", "disk", "persist"):
            if not workspace_present:
                raise ConfigError(
                    "living_plan= mirror='workspace' requires "
                    "Agent(workspace=...) to also be set."
                )
            mirror = True
        elif low in ("none", "off", "memory", "inmemory"):
            mirror = False
        else:
            raise ConfigError(
                f"living_plan= 'mirror' = {mirror_spec!r} not "
                "recognised. Use 'workspace' or 'none'."
            )
    else:
        raise ConfigError(
            "living_plan= 'mirror' must be a string. "
            f"Got: {type(mirror_spec).__name__}"
        )

    include_recall_spec = spec.get("include_recall")
    if include_recall_spec is None:
        # Default: include recall when mirror is on.
        include_recall = mirror
    else:
        include_recall = bool(include_recall_spec)
        if include_recall and not mirror:
            raise ConfigError(
                "living_plan= include_recall=True requires a "
                "workspace mirror (set mirror='workspace' AND "
                "Agent(workspace=...))."
            )

    task_id_val = spec.get("task_id")
    task_id = str(task_id_val) if task_id_val is not None else None

    author_val = spec.get("author")
    author = str(author_val) if author_val is not None else None

    seed_spec = spec.get("seed_plan")
    if seed_spec is None:
        seed_plan: LivingPlan | None = None
    elif isinstance(seed_spec, LivingPlan):
        seed_plan = seed_spec
    else:
        raise ConfigError(
            "living_plan= 'seed_plan' must be a LivingPlan instance "
            "(or omit it). Got: " + type(seed_spec).__name__
        )

    # Auto-stop-hook: defaults True; user opts out by passing False.
    auto_stop_hook = bool(spec.get("auto_stop_hook", True))

    return ResolvedLivingPlan(
        enabled=True,
        mirror_to_workspace=mirror,
        include_recall=include_recall,
        task_id=task_id,
        author=author,
        seed_plan=seed_plan,
        auto_stop_hook=auto_stop_hook,
    )
