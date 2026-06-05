# Changelog

All notable changes to Loom will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

For development-history detail (per-slice notes, file maps, gate
counts), see [`BUILD_LOG.md`](BUILD_LOG.md).

## [Unreleased]

### Added — Cache-hit-rate telemetry

`Usage.cache_hit_rate` and `RunResult.cache_hit_rate` expose the standard
prompt-cache efficacy ratio (`cached / (input + cache_write + cached)`),
and the ReAct loop now emits `loom.tokens.cached`, `loom.tokens.cache_write`,
`loom.cache.hit_rate` per turn plus a `loom.session.cache_hit_rate` rollup
at run end — all behind the existing `fast_telemetry` guard, so the no-op
path is untouched. Token economics is now measurable: a stable-prefix
workload with caching on should trend toward 0.7+ once warm. No behaviour
change; the underlying cache token counts were already parsed by both the
Anthropic and OpenAI adapters.

### Added — Lazy tool loading (`Tuning(lazy_tools=...)`)

Opt-in token reduction for large tool rosters. Instead of shipping every
tool's full JSON schema in the `tools` field each turn, the agent sends
only the eager tools plus one `expand_tool` meta-tool, and a compact tool
*catalog* (name + one-line description) in the **cached** system prompt.
The model calls `expand_tool(name)` to see a tool's arguments on demand;
the real tool still executes whether or not it was expanded first.

The key design property: the exposed tool list is **byte-stable across
turns**, so the prompt-cache tool breakpoint is never invalidated — you
get the schema-token savings *without* thrashing the cache (the failure
mode the caching literature warns about). Measured ~45-55% reduction in
per-turn tools-array tokens on a 12-16 tool coding roster.

`lazy_tools` accepts `True` (all lazy), `list[str]` (those names stay
eager), or `dict` (`{"eager": [...], "meta_tool_name": "..."}`). Default
`False` leaves the tool host untouched. v1 supports an `InProcessToolHost`
base only (skill / workspace / MCP-wrapped hosts raise `ConfigError`).
New: `loomflow.tools.LazyToolHost`.

### Added — `Tuning` config object for rarely-touched Agent knobs

`Agent.__init__` had grown to 37 keyword arguments, mixing the four
everyone uses (`instructions`, `model`, `tools`, `memory`) with a long
tail of knobs almost nobody sets. The signature is the most-read piece
of documentation a framework has, and that tail taxed every reader.

Ten of those knobs now live on a single optional, typed
`Tuning` dataclass (exported as `loomflow.Tuning`). Every field has a
production-safe default, so `Tuning()` alone is a complete config:

```python
Agent("be helpful", model="gpt-4o")                       # the 99%
Agent("...", model="...", tuning=Tuning(retry_policy=p))  # the 1%
```

**Moved into `Tuning`:** `retry_policy`, `tool_result_summary_threshold`,
`auto_compact_summariser`, `auto_compact_keep_recent_turns`,
`tool_transcript_max_bytes`, `max_stop_hook_iterations`, `stop_hooks`,
`secrets`, `auto_consolidate`, `response_tone`. These were chosen by
real usage (grepped across loom-code, the desktop sidecar, examples and
tests) — every knob still set by a real caller stayed top-level.

`Agent.__init__` drops from 37 → 27 top-level kwargs; the six
`Team.*` builders (`supervisor`/`swarm`/`router`/`debate`/
`actor_critic`/`blackboard`) now forward these knobs to the coordinator
as one `Tuning` instead of ten flat kwargs.

### Deprecated — flat form of the moved knobs

Passing any of the ten moved knobs **directly** to `Agent(...)` (the
pre-`Tuning` form) still works for now but emits a `DeprecationWarning`
naming the exact replacement (`pass them via tuning=Tuning(...)`). The
flat form will be removed in a future release. An explicit `tuning=`
wins over a legacy flat kwarg of the same name, and genuinely unknown
kwargs still raise `TypeError` (the shim absorbs only recognised
`Tuning` fields, so typos are not silently swallowed).

No behaviour changes for existing code on upgrade — only a warning.

## [0.10.20] — 2026-05-16

### Added — Comprehensive Agent-kwarg forwarding through Team.* builders

Closes the **complete** historical-gap audit between
``Agent.__init__`` and ``Team.supervisor`` / ``swarm`` / ``router``
/ ``debate`` / ``actor_critic`` / ``blackboard``. Prior releases
patched individual missing kwargs (``stop_hooks=`` in 0.10.10,
``prompt_caching=`` in 0.10.12, ``tool_result_summarizer=`` in
0.10.13). This release closes the rest in one sweep.

**Newly forwarded through all 6 Team.* builders:**

* ``snip_window=`` — 0.10.13's free list-slicing turn-trim
* ``auto_compact_at_tokens=`` — 0.10.13's mid-Ralph-loop compaction
* ``auto_compact_summariser=`` — companion summariser model
* ``auto_compact_keep_recent_turns=`` — verbatim-tail size
* ``retry_policy=`` — model retry policy (was: only Agent direct)
* ``auto_extract=`` — automatic fact extraction toggle
* ``approval_handler=`` — async approval callback for ask-mode
* ``secrets=`` — secrets manager for resolving API keys
* ``response_tone=`` — default response tone for the coordinator
* ``effort=`` — reasoning-effort dial
* ``strict_effort=`` — hard-error on effort capability mismatch

Eleven kwargs across six builders = 66 forwarding additions, all
mechanical. The full audit table now shows zero gaps; everything
``Agent`` accepts (except the per-call ``output_schema=`` and the
intentionally-overridden ``architecture=``) round-trips through
every ``Team.*`` builder.

**Loom-code impact:** the post-construction monkey-patches in
``loom_code/agent.py`` for ``_snip_window`` and
``_auto_compact_at_tokens`` (added in the previous loom-code
round as a bridge) can now be replaced with clean kwargs.

### Coverage

16 new tests in ``tests/test_team.py``: a comprehensive
``supervisor_with_all_kwargs`` fixture covering 11 kwargs at once
+ 6 spot-check tests on the other builders confirming the same
forwarding. Plus 3 "accepts kwarg without raise" tests for kwargs
(``auto_extract``, ``retry_policy``, ``approval_handler`` +
``secrets``) whose post-construction state isn't directly
introspectable. Full suite: 1598 passing.

## [0.10.19] — 2026-05-16

### Added — Auto-compact at framework level

The third tier of context-budget defence, completing the trio
that pairs with snip (0.10.16) and tool-result summarisation
(0.10.14). When ``session.messages`` accumulates past
``auto_compact_at_tokens`` mid-run, the older half is collapsed
into a single summary system message via an LLM call; the most
recent N user-anchored turn groups survive verbatim.

```python
agent = Agent(
    "you help",
    model="claude-opus-4-7",
    auto_compact_at_tokens=80_000,          # trigger threshold
    auto_compact_summariser="claude-haiku-4-5",  # defaults to main model
    auto_compact_keep_recent_turns=4,        # how many to keep verbatim
)
```

**Where it fires:** inside ``Agent._loop``'s Ralph loop between
architecture iterations. The first architecture pass runs untouched
(zero overhead on short single-turn runs). Before each subsequent
stop-hook-triggered iteration, we count tokens via the 0.10.17
``count_tokens`` helper; if the count exceeds the threshold, we
compact in place.

**What survives the compact:**

* Leading ``Role.SYSTEM`` head — always (identity / instructions).
* A NEW ``Role.SYSTEM`` message with the summary, prefixed
  ``[auto-compacted summary of N dropped messages]`` so the model
  knows what it's reading.
* The last N user-anchored turn groups — verbatim, so recent tool
  results / decisions stay concrete.

**Failure handling:** summariser exception, empty / whitespace-only
output, no anchor to slice at — all result in a graceful no-op
(conversation continues uncompacted). Framework-level token
optimisations must NEVER kill a turn.

**Why mid-Ralph, not at agent.run() boundary:** Each ``agent.run()``
creates a fresh ``AgentSession``; cross-run conversation continuity
comes from Memory rehydration, not from session.messages persisting.
Auto-compact's job is "we've been running for a while WITHIN one
agent.run(), prune before the next model call." The right moment is
between Ralph-loop iterations after enough turns have accumulated.

### Companion helpers

* ``loomflow.agent.auto_compact.context_window_for(model_name)`` —
  best-effort substring lookup for known model families
  (Claude / GPT / Gemini), 8k fallback for unknowns. Lifted from
  loom-code's local ``compact.py``.
* ``loomflow.agent.auto_compact.maybe_auto_compact()`` — the core
  helper, callable standalone from custom architectures.
* New `Event.architecture_event("auto_compacted", tokens_before=,
  messages_before=, messages_after=, messages_dropped=,
  summary_chars=)` for telemetry / ``/cost``-style UIs.

### Coverage

19 new tests in ``tests/test_auto_compact.py`` covering:
``context_window_for`` substring matching across model families
(Claude / GPT / unknown-fallback), ``_split_at_user_anchor``
slicing logic, ``maybe_auto_compact`` flow (under-threshold no-op,
over-threshold compact, no-anchor no-op, summariser-failure no-op,
empty-summary no-op, preamble-stripping), and Agent kwarg
validation (negative / zero threshold rejected, keep_recent
>= 1 enforced, default disabled, summariser defaults to main
model, explicit summariser wins). Full suite: 1582 passing.

### Comparison with the three tiers

| Tier | When | Cost | Granularity |
|---|---|---|---|
| `snip_window=N` (0.10.16) | Before each turn | Free (list slice) | Whole turn groups |
| `tool_result_summarizer=` (0.10.14) | Per tool result | 1 LLM call per oversized result | Single tool result |
| `auto_compact_at_tokens=N` (0.10.19) | Between Ralph iterations | 1 LLM call per compaction | Conversation prefix |

All three opt-in; default-off; compose freely.

## [0.10.18] — 2026-05-16

### Added — Subagent parent-attribution metadata

When ``SubagentInvocation`` spawns a child agent inside an
active parent run, the child's :class:`RunContext.metadata` now
carries two reserved keys recording who spawned it:

* ``_loomflow_parent_session_id`` — the parent's session_id
* ``_loomflow_parent_run_id`` — the parent's run_id

Useful for telemetry / audit attribution ("this child run was
spawned by parent X"), cache / memory partitioning by parent,
and for any custom tool that wants to know "am I running as a
subagent or directly?" — the keys are absent on direct
``Agent.run()`` calls.

### Design notes

The Claude Code "renderedSystemPrompt bytes" pattern doesn't
translate cleanly to loomflow's model — loomflow workers each
have their own ``instructions`` / ``tools`` / ``memory`` by
design (different agent, different identity), unlike Claude
Code's fork-as-copy. Persistent subagents (0.10.10) already
gives us cache-stable system prompts across delegations to the
same worker, so the cache-continuity goal is mostly already
covered.

What WAS missing was a way for the child to know it was
spawned by a specific parent run. Today's answer: reserved
metadata keys on the child's RunContext, set additively (never
clobbers user metadata, never overwrites a deeper ancestor's
attribution thanks to ``setdefault``).

### Coverage

4 new tests in
``tests/test_subagent_parent_attribution.py``: child gets parent
attribution inside an active parent run, child has NO
attribution when constructed outside any parent run (back-
compat), explicit ``context=`` overrides still get attribution
layered on (additive), and the ``setdefault`` rule that
preserves the OUTERmost ancestor identity through nested
subagent chains. Full suite: 1563 passing.

## [0.10.17] — 2026-05-16

### Added — Token counting helper with three-tier fallback

New `loomflow.model.count_tokens.count_tokens(model, messages,
tools=)` helper. The foundation 0.10.19's auto-compact will
build on; immediate use case is `/cost`-style UIs that want
"how close to the context window are we" without a round-trip.

Three-tier fallback chain:

1. **Provider-native** — if the model adapter exposes a
   `count_tokens(messages, tools=)` method, the helper delegates
   (exact byte-accurate counts).
2. **tiktoken** — `cl100k_base` encoding, the GPT-4 / Claude-3+
   family's tokenizer; accurate enough for budget decisions.
   Requires the `loader` extra (which already includes tiktoken).
3. **Char-based estimate** — `chars / 4` (configurable). Last
   resort, zero deps. Always succeeds — `count_tokens` never
   raises, since callers (compact triggers, budget bars) have no
   graceful fallback when counts go missing.

The `Model` protocol is **not** modified — `count_tokens` is
duck-typed via `hasattr` so custom Model impls inherit the
fallback automatically.

### Adapter implementations

* **Anthropic** — native via `client.messages.count_tokens(...)`
  (Anthropic's beta endpoint). Exact byte-accurate count
  matching what the actual completion would be billed for.
  Falls back to `client.beta.messages.count_tokens` on older
  SDKs. Located on `AnthropicModel.count_tokens`.
* **OpenAI / LiteLLM / Echo / Scripted / Retrying** — no
  native impl; inherit the tiktoken/char-based fallback for
  free.

### Coverage

10 new tests in `tests/test_count_tokens.py` covering: char-
based unit cases (empty / single-message / tools / custom
ratio), provider-native dispatch + exception fall-through,
tiktoken-path skipped-when-not-installed, no-native-method
fallback, tools-increase-total invariant, and the always-
positive-int contract. Full suite: 1559 passing.

## [0.10.16] — 2026-05-16

### Added — Snip: bounded conversation window before each turn

`Agent(snip_window=N)` keeps the last N user-anchored turn groups
in the rehydrated message list before each model call — pure
list slicing, no API call. The cheap always-on context-budget
defence that pairs with `tool_result_summarizer` (0.10.14) and
the future auto-compact (0.10.19) as the three tiers of context
management.

```python
agent = Agent(
    "you help",
    model="claude-opus-4-7",
    snip_window=10,   # keep last 10 user-anchored turns
)
```

**Where it fires:** inside `ReAct.run()` right after seed
messages are rehydrated from memory (architectures rehydrate
into `session.messages` at the top of each invocation; snip
happens immediately after that, before the first model call).
Pairs with `Dependencies.snip_window` + `Dependencies.fast_snip`
flag — default-disabled, zero-allocation hot path when not
opted in.

**Slicing rules** (in `loomflow.agent.snip.snip_messages`):

* Snips at `Role.USER` boundaries — never leaves an orphan
  `tool_result` before its preceding `tool_call`.
* Leading `Role.SYSTEM` messages survive every snip (they're the
  identity / instructions, not conversation history).
* `keep_last_n_turns=0` is a no-op (returns the same list object).
* No user messages in history → no-op (no anchor to slice at).

**Event:** emits `Event.architecture_event("messages_snipped",
dropped=N, kept=M, window_turns=W)` when a snip fires — telemetry
+ `/cost`-style UIs can show "trimmed N messages this turn."

### Coverage

12 new tests in `tests/test_snip.py`: 8 unit tests covering the
slicing helper edge cases (empty / zero-window / under-window /
exact-fit / over-window / system-head preservation / no-anchor /
tool-result pairing) + 3 Agent integration tests (negative
rejection, default behaviour, propagation) + 1 end-to-end via
`InMemoryMemory` rehydration confirming the snip event fires
across multiple `agent.run()` calls with a shared `session_id`.
Full suite: 1549 passing.

## [0.10.15] — 2026-05-16

### Added — `tool_result_summarizer=` forwarded through every `Team.*` builder

Closes the same papercut closed by 0.10.10 (``stop_hooks=``) and
0.10.12 (``prompt_caching=``): the 0.10.14 tool-result
summarisation feature now reaches ``Team.supervisor``, ``swarm``,
``router``, ``debate``, ``actor_critic``, and ``blackboard``
coordinators directly via the builder kwargs. Without this,
``Team.supervisor(tool_result_summarizer="haiku")`` raised
``TypeError`` and callers had to monkey-patch
``coord._tool_result_summarizer`` after construction.

```python
team = Team.supervisor(
    workers={"coder": coder_agent, "explorer": explorer_agent},
    model="claude-opus-4-7",
    tool_result_summarizer="claude-haiku-4-5",
    tool_result_summary_threshold=500,
)
```

### Coverage

7 new tests in ``tests/test_team.py`` — one per builder
confirming the kwarg + threshold propagate to the coordinator's
``_tool_result_summarizer`` / ``_tool_result_summary_threshold``,
plus a default-None test ensuring no kwarg means no summarisation
(back-compat). Full suite: 1537 passing.

## [0.10.14] — 2026-05-16

### Added — Tool-result summarization (Claude Code's `tool_use_summary`)

New opt-in `Agent(tool_result_summarizer=<model>)` kwarg. When
wired, the ReAct loop hands any tool result whose rendered
content exceeds `tool_result_summary_threshold` chars (default
500) to a small fast model (typically Haiku) BEFORE appending the
result to conversation history. The summary replaces the verbatim
output — every subsequent turn ships the short version, not the
original.

The single biggest per-turn token saver Claude Code uses. A 5KB
`read_file` result that used to occupy 5KB on every subsequent
turn now occupies ~500 bytes. Over a 10-turn session that
compounds — and unlike auto-compact (which triggers at 80%
context window), this kicks in immediately, before history starts
to bloat.

Failure handling: summariser exception, empty summary, or summary
larger than the original all fall back to shipping the original
verbatim. Principle: summarisation must NEVER kill a turn.

API shape:

```python
agent = Agent(
    "you help",
    model="claude-opus-4-7",
    tool_result_summarizer="claude-haiku-4-5",  # or a Model instance
    tool_result_summary_threshold=500,           # chars; default
)
```

Model-side caveat the developer should know: the agent is
DECEIVED — its next turn sees the summary as if it were the
original. If a multi-turn flow truly needs the verbatim text,
the agent must re-read the file / re-run the command. In
practice coding agents read → reason → write within one turn, so
the verbatim is available during the only turn that needs it.

### Implementation

* New module `loomflow/tools/result_summarizer.py` —
  `summarize_tool_result()` + summary prompt template +
  `DEFAULT_SUMMARY_THRESHOLD` constant.
* `Dependencies.tool_result_summarizer` +
  `Dependencies.tool_result_summary_threshold` +
  `Dependencies.fast_tool_summary` (auto-True when no summariser
  wired — hot path is fully short-circuited).
* `Agent.__init__` accepts `tool_result_summarizer: Model | str |
  None` and `tool_result_summary_threshold: int`. Summariser is
  resolved through the same `_resolve_model` path as the main
  `model=` kwarg, so `tool_result_summarizer="haiku"` shorthand
  works.
* ReAct's tool-dispatch loop (`react.py`) gains a single
  `if not deps.fast_tool_summary` branch right before the
  `Message(role=Role.TOOL, ...)` append. Emits a new
  `Event.architecture_event("tool_result_summarized", tool=, ...,
  original_chars=, summary_chars=)` so telemetry / `/cost`-style
  UIs can show "saved X chars on this turn."

### Coverage

11 new tests in `tests/test_tool_result_summarizer.py` covering:
below-threshold pass-through, above-threshold summarisation,
fall-back on summariser exception, fall-back on empty summary,
the `DEFAULT_SUMMARY_THRESHOLD` constant, Agent kwarg wiring,
negative-threshold rejection, end-to-end event emission for
oversized + undersized tool results, and confirmation that the
feature is fully off when not opted in. Full suite: 1530 passing.

## [0.10.13] — 2026-05-16

### Added — Multi-breakpoint Anthropic prompt caching (3rd/4th of 4)

Until 0.10.13 the Anthropic adapter used only 2 of the 4
``cache_control`` breakpoints the API allows: one on the last
system block, one on the last tool definition. Architectures emit
system content as MULTIPLE messages (instructions / memory blocks /
recall context), but the adapter joined them into a single string
before marking — so a memory-block change busted the entire cached
prefix, and the existence of a per-turn recall block (which changes
every turn) cache-busted EVERYTHING above it.

This release keeps the system parts separated through to the
cache-control helper and emits one ``cache_control``-marked content
block per part — up to 3 system markers + 1 tool marker = the full
4 breakpoints Anthropic supports:

* Single-part system (back-compat shape): 1 marker, identical to
  the pre-0.10.13 wire format.
* Two-part system (instructions + memory OR instructions + recall):
  both blocks marked — each cached independently.
* Three-part system (instructions + memory + recall): all three
  blocks marked. Even if the recall block is per-turn-volatile, the
  instructions + memory blocks still hit cache on every turn via
  their independent markers.
* Four+ part system (defensive cap): only the LAST 3 carry markers
  so we don't exceed the 4-breakpoint hard limit.

Impact on loom-code (the immediate consumer): the ``session_summary``
+ ``loom_index`` working blocks the REPL feeds through
``Memory.update_block`` finally get their own cache entries — they
hit cache turn-to-turn without being invalidated by recall churn.

### Cross-provider scope

Only the Anthropic adapter has work. OpenAI handles prompt caching
server-side without marker hints (`prompt_tokens_details.cached_tokens`
parsing was already in place); LiteLLM inherits from the upstream
provider; Echo/Scripted/Retrying accept the flag for signature
parity but have no backend. Gemini caching is not yet implemented
in loomflow (would require the separate ``CachedContent.create()``
endpoint — can't be expressed as a per-call marker).

### Coverage

5 new tests in ``tests/test_prompt_caching.py`` covering: single-part
back-compat, two-part memory-block marking, three-part full-coverage,
four-part defensive cap, and the cache-off no-op path. Updated
``tests/test_anthropic.py::test_system_messages_kept_as_list_for_cache_block_emission``
to reflect the new ``_to_anthropic_messages -> tuple[list[str], ...]``
signature. Full suite: 1519 passing.

## [0.10.12] — 2026-05-16

### Added — `prompt_caching=` forwarded through every `Team.*` builder

Closes the same papercut that 0.10.10 closed for `stop_hooks=` /
`max_stop_hook_iterations=`: `Team.supervisor`, `swarm`, `router`,
`debate`, `actor_critic`, and `blackboard` now accept
`prompt_caching: bool | Mapping[str, Any] | None = None` and
thread it into the coordinator `Agent(...)`. Previously the kwarg
silently didn't exist on these builders, so callers either had
to monkey-patch `coordinator._prompt_caching` post-construction
or live with an uncached coordinator (workers, which are built
as plain Agents, were already cacheable).

Accepted shapes are unchanged from the underlying Agent API:

```python
Team.supervisor(..., prompt_caching=True)                          # 5m TTL default
Team.supervisor(..., prompt_caching={"enabled": True, "ttl": "1h"})
Team.supervisor(..., prompt_caching={"enabled": True, "cache_key": "session_42"})
```

This unblocks loom-code's coordinator caching (it was already
caching workers, never the coordinator — the team-builder gap was
the reason).

### Coverage

7 new tests in `tests/test_team.py` — one per builder confirming
the kwarg propagates to the coordinator's `_prompt_caching` and
resolves to `enabled=True`, plus a dict-shape test confirming
`ttl="1h"` makes it through end-to-end. Full suite: 1515 passing.

## [0.10.11] — 2026-05-16

### Fixed — Windows compatibility (charmap codec + bash_tool shell)

Two production-blocking bugs surfaced by a remote loom-code user
running on Windows:

**1. ``UnicodeEncodeError: 'charmap' codec can't encode character
'≥'``** — ``LocalDiskWorkspace`` was writing notes via
``Path.write_text(content)`` without an explicit ``encoding=``,
so it picked up the system locale codec. On Windows that's
``cp1252`` ("charmap") which can't represent ``≥``, ``≤``, ``→``,
``✓``, emoji, and many other Unicode characters models routinely
emit. Notes containing npm version specifiers (``≥1.0.0``) crashed
the workspace.

Fix: all nine ``read_text`` / ``write_text`` sites in
``loomflow/workspace/disk.py`` now force UTF-8 via a pair of
helpers (``_read_utf8`` / ``_write_utf8``). Same fix applied
preventively to: ``loomflow/skills/skill.py`` (SKILL.md
loading), ``loomflow/tools/builtin.py`` (recursive read in the
search-and-replace path), ``loomflow/graph.py`` (Mermaid output),
and ``loomflow/vectorstore/inmemory.py`` (vector-store JSON
persistence — also gained ``ensure_ascii=False``).

**2. ``[WinError 2] The system cannot find the file specified``**
— ``bash_tool`` invoked the subprocess via ``["/bin/sh", "-c",
command]``. Windows doesn't have ``/bin/sh`` so the executable
lookup failed before any command could run.

Fix: detect ``sys.platform == "win32"`` and use
``["cmd.exe", "/c", command]`` there; keep ``/bin/sh -c`` on
POSIX. The tool's model-facing description now declares which
host shell it has (``Host shell: cmd.exe (use its native
syntax)``) so the model can adapt its commands to the platform.

### Coverage

5 new regression tests in ``tests/test_workspace_windows_encoding.py``
covering the bug-report scenario (notes containing ``≥``), the
WORKSPACE.md index regeneration with Unicode titles, and the
``.history/*.md`` snapshot path. Tests would FAIL on Windows
pre-fix; on POSIX hosts they pin the encoding via round-trips so
regressions can't sneak back. Full suite: 1508 passing on macOS;
the same suite is the contract Windows now has to meet too.

## [0.10.10] — 2026-05-16

### Added — Persistent subagents across every `Team.*` builder

`Team.supervisor`, `Team.swarm`, `Team.router`, `Team.debate`,
`Team.actor_critic`, and `Team.blackboard` all gain a
`persistent_subagents: bool = True` kwarg. When enabled (the
default), each worker is registered with a stable
`worker_<role>_<ULID>` ID and a stable `session_id` that is
reused on every spawn — so workers accumulate conversation memory
across handoffs, rounds, and ALSO across multiple `Agent.run()`
invocations on the same coordinator.

This is the Claude-Code parity feature: the supervisor's
"researcher" is the *same agent* across turn 1 and turn 5; the
swarm's "billing" peer remembers what it told the user last week;
the actor-critic pair iterates with full memory of the prior
critique rounds. Set `persistent_subagents=False` to restore the
legacy per-spawn stateless behaviour.

New `send_message(to=<worker_id>, content=<message>)` tool — the
companion to `delegate`. Where `delegate(target, instructions)`
spawns or re-engages a worker by ROLE, `send_message(to=...)`
addresses a specific worker by its persistent ID and continues
the worker's conversation thread. Auto-wired by
`Team.supervisor` when `persistent_subagents=True`.

Multi-tenant safety: worker handles pin `user_id` on first touch.
Cross-user `delegate` / `send_message` calls return a clear
tool-result error string (no raise) and never reach the
underlying agent. Per-handle `anyio.Lock` serialises concurrent
calls to the same worker; different workers stay parallel.

New module: `loomflow.agent.worker_registry` —
`_WorkerHandle`, `new_worker_id`, `build_worker_registry`,
`resolve_persistent_session` (used by every architecture's
spawn site). New tool factory:
`loomflow.tools.send_message.make_send_message_tool`.

### Coverage

All six team architectures (Supervisor, Swarm, Router, Debate,
ActorCritic, Blackboard) thread the registry through every
spawn site — judge, debaters per round, coordinator + decider,
actor + critic across rounds, blackboard contributors, swarm
handoff targets, router specialists. 21 new tests in
`tests/test_persistent_subagents.py` covering primitive
behaviour + opt-in/out across all six builders + error paths
on `send_message`.

### Added — `stop_hooks=` / `max_stop_hook_iterations=` on every `Team.*` builder

Closes the same papercut that `prompt_caching=` has on Team
builders today: callers previously had to mutate
`coordinator._stop_hooks` and `coordinator._max_stop_hook_iterations`
post-construction. Now passable directly through the builder
kwargs on `Team.supervisor` / `swarm` / `router` / `debate` /
`actor_critic` / `blackboard`.

### Added — `examples/20_persistent_subagents.py`

Zero-key example (uses `EchoModel`) demonstrating the persistent
vs legacy modes side-by-side + showing the registry contents on
`Team.debate` + `Team.actor_critic` for reference.

## [0.10.8] — 2026-05-16

### Added — `StopHook` protocol for framework-level Ralph loop

New `Agent(stop_hooks=[...], max_stop_hook_iterations=15)` lets
framework or user code force the agent to keep working when the
architecture would otherwise exit. Fixes the structural failure
mode where ReAct exits on text-without-tool-call mid-plan ("Now
let me scaffold the backend..." → ReAct treats as final answer →
plan sits with steps still in `doing`).

API:
- `loomflow.StopHook` — Protocol, async-callable. Returns a
  `StopHookResult` to force continuation, or `None` to vote stop.
  First non-None per iteration wins.
- `loomflow.StopHookResult` — frozen dataclass: `inject_message`
  becomes a user turn; `reason` lands on telemetry + audit.
- `Agent.max_stop_hook_iterations` — bounded loop; `0` disables.
- `Dependencies.fast_stop_hooks` — auto-set True when no hooks
  registered (no-op default keeps LangChain-class latency).

Auto-registration: `Agent(living_plan=True)` prepends a
LivingPlan stop hook that re-prompts when any step is `doing`/
`todo` and names the specific step. Opt out via
`living_plan={"auto_stop_hook": False}`.

LivingPlan tool prompt updated: enforces "exactly ONE `doing` at
a time", "mark `done` IMMEDIATELY", and "no final response while
any step is `doing`/`todo` — the framework will re-prompt you."

Telemetry: `Event.architecture_event` with `stop_hook.fired` /
`stop_hook.exhausted` names. No new EventKind value.

Industry precedent matches Claude Code's `handleStopHooks` +
TodoWriteTool discipline; AutoGen's `TerminationCondition`;
Cursor's judge agent; Anthropic `/goal`. Loomflow now has the
primitive at the framework layer instead of forcing each
consumer to re-implement the loop. 13 new tests; full 1480-test
suite green; `mypy --strict` clean.

## [0.10.6] — 2026-05-15

### Added — `web_tool` for agent web search (Serper + DuckDuckGo)

`from loomflow.tools import web_tool` — a single tool factory the
user wires into an Agent's `tools=` list. The model calls it as
`web_search(query=...)` and gets back a markdown-formatted top-N
result list (title, URL, snippet per result) — directly readable,
no JSON parsing in the prompt.

Two backends:

* `web_tool(backend="serper")` — Google via https://serper.dev.
  Best quality. Needs an API key (env `SERPER_API_KEY` or
  `api_key=` kwarg). Optional extra: `pip install loomflow[serper]`.
* `web_tool(backend="duckduckgo")` — Free, no key. Lower /
  variable quality, DDG rate-limits. Optional extra:
  `pip install loomflow[duckduckgo]`.

Convenience bundle: `pip install loomflow[web]` (both).

Both backends produce identical output shape so swapping doesn't
change what the model has learned to parse. Lazy SDK imports —
`import loomflow.tools` doesn't require either extra. Network
errors return a `"(web search failed: ...)"` string rather than
raise, so agents see the failure as a tool result and can decide
what to do next (retry, ask the user, try a different query).

14 new tests cover backend selection, missing-key ConfigError,
the result-formatting contract (both backends), the HTTP error
path, and identical tool shape across backends.

Future expansion: a `web_fetch_tool` for reading specific URLs
(HTML → markdown), additional backends (Brave, Tavily, Google
PSE). The `backend=` selector is the seam.

## [0.10.4] — 2026-05-15

### Fixed — sub-agent costs were silently lost in multi-agent architectures

Every architecture that spawned a sub-`Agent` via the
`SubagentInvocation` helper — `Supervisor`, `Swarm`, `Router`,
`ActorCritic`, `MultiAgentDebate`, `BlackboardArchitecture` — was
discarding the sub-agent's `RunResult` after extracting only its
`output`. The parent's `RunResult.cost_usd` and token counts
therefore reflected ONLY the parent's own model calls and silently
omitted every spawned worker. Anyone using `Team.supervisor` in
production saw a misleadingly small cost number.

Fix: `SubagentInvocation` gains a `rollup_into: AgentSession | None`
kwarg. When provided, the helper rolls the sub-agent's usage into
that session's `cumulative_usage` the moment the sub-agent's
`completed` event fires. Every architecture using the helper has
been updated to pass `rollup_into=session`. Supervisor's no-event-
sink fallback path (legacy `Agent.run()` call) does the rollup
inline.

Two regression tests in `tests/test_team.py` (Supervisor and Swarm
representatives) cover the contract — they assert the team's
`RunResult` reflects coordinator + worker costs combined.

No public API change; bug-fix release. Anyone using `Team.*` builders
will see their reported costs go up — that's the *real* number that
should always have been visible.

## [0.10.0] — unreleased

### Fixed — LivingPlan `plan_write` rejected weak-model step shapes

`_coerce_steps` filtered `steps` down to `isinstance(s, dict)` —
so when a weaker model (gpt-4.1-mini and friends) emitted `steps`
as a list of plain description *strings*, every element was
dropped and the plan came out empty (`0/0 done`). A bare step
dict the model forgot to wrap in a list errored outright.

Coercion is now genuinely lenient — it salvages: `list[str]`
(each string → a `todo` step), a list whose elements are
stringified-JSON dicts, a bare `dict` (single step, auto-wrapped),
and any of those inside a JSON string. It only errors when the
input is empty or has nothing salvageable. Per-element coercion
moves into `_coerce_one_step`; numbered-text into
`_coerce_numbered_text`.

### Fixed — `search_notes` multi-word queries (was returning nothing)

The workspace BM25 scorer tested the *entire query string* as one
substring (`q in body`), despite the docstring claiming
"BM25-ish". So any multi-word query — `search_notes("conda env
conflict")` — returned nothing unless that exact phrase appeared
verbatim. Multi-word queries are the norm for agent recall, so
this silently crippled the self-improvement loop.

Fix: the query is now tokenized; each term is scored independently
at the three tiers (title 1.0 > tag 0.7 > body 0.5) and the note's
score is the per-term tiers averaged over the query terms — OR
semantics ranked by coverage. A single-word query collapses to the
old substring-tier match, so existing callers are unaffected. The
scorer is now shared (`_common.score_bm25`) so the disk and
in-memory backends can't drift.

### Fixed — workspace rooted under a dot-directory listed zero notes

`_is_in_meta_dir` (workspace-v2) flagged *any* dot-prefixed path
part, intending to exclude `.history` revisions — but a workspace
rooted under a dot-directory (`.loom/notebook`, `.claude/workspace`)
has a dotted segment in every note's path, so every note got
filtered out of `list_notes` / `search_notes` / the index. Now
checks specifically for the `.history` segment.

### Fixed — `attribute_outcome` now works AFTER a run (was a silent no-op)

The self-improvement loop had a fatal gap: `attribute_outcome`
drained the per-run citation contextvar, but `Agent._loop` RESETS
that contextvar at run-end. So calling `attribute_outcome` after
`agent.run()` / `agent.stream()` returned always found an empty
set and did nothing — the citation counts never moved.

Fix:

* `RunResult` gains `cited_slugs: list[str]` — the workspace note
  slugs the agent read during the run, snapshotted from the
  contextvar just before it's reset.
* `Workspace.attribute_outcome` gains an explicit `slugs=` param.
  Pass `RunResult.cited_slugs` and it works reliably post-run.
  `slugs=None` keeps the old in-run contextvar-drain behaviour
  for tools / hooks that call it mid-run.

Canonical post-run pattern:

```python
result = await agent.run(prompt, user_id="u")
await workspace.attribute_outcome(
    success=<did it work?>,
    slugs=result.cited_slugs,
    user_id="u",
)
```

### Added — workspace retention: `Workspace.prune()`

`Workspace.prune()` is the workspace garbage-collector — citation-
aware retention that **hard-deletes** stale, low-value notes while
protecting valuable ones.

A note is pruned only when ALL hold:

* ``older_than`` is set AND the note's last activity
  (``max(updated_at, last_cited_at)``) is older than the window.
  When ``older_than`` is ``None``, age is not a filter (every note
  is age-eligible — the docstring strongly recommends passing it).
* ``cited_count`` is below ``min_cited_count`` (default 1, so a
  note cited even once survives).
* ``kind`` is not in ``keep_kinds`` (e.g. ``["decision"]`` to
  never GC decisions).

``keep_last_versions=N`` separately trims each surviving note's
``.history`` to the most recent N revisions — history grows
fastest, so it gets its own cap.

This is where the citation metadata (``cited_count`` /
``success_count`` / ``last_cited_at``) earns its keep: pruning is
SMART — "delete old uncited notes but keep anything that's been
referenced" — rather than dumb time-based deletion that loses
valuable old knowledge.

``prune`` is observation-class (no author-ownership check) — it's
an operator / maintenance op, not an agent action. Don't wire it
as an agent tool; call it from a cron job, an end-of-benchmark
hook, or manually. Mirrors ``Memory.forget``.

Returns a new ``PruneResult`` (Tier 1 export) with
``notes_deleted`` / ``versions_deleted`` / ``notes_kept`` counts.
Disk backend hard-deletes the ``.md`` file + its embedding
sidecar + its ``.history`` dir; in-memory backend drops the dict
entries. Multi-tenant: only touches the given ``user_id``'s
partition.

### Added — workspace self-improvement loop: citation tracking + outcome attribution + relevance-aware search

Three additions that turn the workspace from a passive notebook
into the substrate for an agent that gets smarter with every run:

1. **Citation tracking via contextvar.** A new
   ``_ambient_citations_var`` is installed by ``Agent._loop`` for
   the duration of every run. ``Workspace.read_note`` and
   ``read_version`` log the slug they returned into this set.
   Best-effort: outside a run (tests, direct tool calls) the set
   doesn't exist and the loggers no-op. No author check — citation
   is OBSERVATION, not authorship.

2. **Outcome attribution via ``Workspace.attribute_outcome(success,
   user_id)``.** Drains the per-run citation set and updates each
   cited note's metadata: ``cited_count`` += 1, ``success_count``
   += 1 when ``success=True``, ``last_cited_at`` = now. Returns
   the number of notes whose metadata was updated. User code
   calls this after ``agent.run()`` returns, passing a
   pass/fail signal — the workspace then knows WHICH past notes
   actually led to working outcomes.

3. **Relevance-aware search ranking.** ``search_notes`` gains an
   opt-in ``boost_relevance: bool = False`` kwarg. When True,
   each result's base score is multiplied by a citation-history
   boost: ``1 + log(1+cited_count) + 2*log(1+success_count)`` —
   log scaling so a single runaway-popular note doesn't drown
   out fresh ones, success-citations weighted 2x mere citations
   so "verified useful" outranks "merely read." Default off
   preserves back-compat. Available in all four modes (``auto``
   / ``bm25`` / ``semantic`` / ``hybrid``).

Public additions: three optional fields on ``Note`` and
``NoteSummary`` (``cited_count: int = 0``, ``success_count: int = 0``,
``last_cited_at: datetime | None = None``). Legacy notes (no
citation history in frontmatter) default to 0 and start
accumulating from first cite.

This is the **scaffold for self-improvement**, not the full
loop. The framework now KNOWS which past notes were useful;
follow-up work would add an offline "dreaming" pass that uses
that signal to consolidate / promote / archive notes. But just
the relevance signal alone — usable today — means search results
improve as the agent's history grows. Search results in run N+1
are biased toward notes that worked in runs 1..N.

### Added — workspace v2: namespacing, archive, versioning, questions, semantic search, shape-aware prompt

Six additions to `loomflow/workspace/`, all backward-compatible. Legacy `.md`
files load cleanly; existing Agent / Team calls keep working unchanged.

1. **Namespacing** — `write_note(..., namespace="...")` scopes writes to a
   sub-bucket. `list_notes` / `search_notes` ignore namespace by default
   (teammates' work in adjacent namespaces stays visible); pass
   `namespace=` to filter explicitly. `make_workspace_tools(workspace,
   namespace=...)` scopes a member's WRITES to one namespace.

2. **Archive** — `Workspace.archive_note(slug)` sets `archived_at`.
   Default behavior of `list_notes` / `search_notes` EXCLUDES archived
   notes; pass `include_archived=True` to see them. Archived notes
   remain readable by slug via `read_note`. New `archive_note` tool is
   on by default in `make_workspace_tools` (opt out via
   `include_archive=False`).

3. **Revision history** — every `update_note` snapshots the prior body
   into a `.history/<slug>/0001.md` sequence (4-digit monotonic
   counter). New methods `list_versions(slug)` / `read_version(slug, n)`
   on the Workspace protocol. Excluded from `list_notes` via a
   `_walk_note_files` filter so historical revisions never surface as
   live notes.

4. **Ask / answer pattern** — opt-in via `make_workspace_tools(...,
   questions=True)`. Adds `ask_question(title, content)` (writes
   `kind="question"` with `answered=False`), `answer_question(slug,
   content)` (writes a child finding + flips the question's
   `answered=True`), and `list_open_questions()` (lists unanswered).
   Cross-author safe via a new `mark_answered` carve-out on
   `update_note` — non-owners can flip the answered flag without
   touching the question body.

5. **Semantic search** — optional `embedder=` ctor param on
   `LocalDiskWorkspace` and `InMemoryWorkspace`. When wired,
   `write_note` also persists an embedding sidecar (with model name
   for stale-vector detection); `search_notes(query, mode=...)` accepts
   `"auto"` / `"bm25"` / `"semantic"` / `"hybrid"`. Hybrid mode uses
   reciprocal rank fusion (RRF) of BM25 + cosine. Default `auto`
   = hybrid when an embedder is wired, BM25 otherwise.

6. **Shape-aware prompt section** — `workspace_prompt_section` now
   produces TWO variants. When `teammates` is non-empty, the copy
   emphasises team coordination ("share findings with teammates"). When
   `teammates` is None / empty (single-agent cross-run mode), the copy
   emphasises persistent knowledge across runs ("this is YOUR
   persistent notebook"). Empirical motivation: Sonnet was skipping
   the notebook in single-agent mode because the team-language confused
   it about whether the notebook applied to its situation.

Public additions: `NoteVersion` (Tier 1). New optional fields on `Note` /
`NoteSummary`: `namespace`, `archived_at`, `answered`, `answered_by`,
`parent_slug`. `extra="ignore"` on `Note.model_config` for forward-compat
on future schema bumps.

Protocol additions (breaking for duck-typed `Workspace` implementations
— accept and document since workspace is young): `archive_note`,
`list_versions`, `read_version` are now part of `Workspace`. The
in-tree backends `LocalDiskWorkspace` and `InMemoryWorkspace` implement
all of them.

### Added — TodoWrite-style living plan via `living_plan=`

`Agent(living_plan=True)` wires two structured-plan tools onto the
agent: `plan_write(goal, steps)` (atomic full-list rewrite — every
call returns the rendered plan back as markdown, so the plan
becomes load-bearing in the conversation) and `plan_read()`. Each
step is `{description, status, finding}` where status is one of
`todo` | `doing` | `done` | `blocked` | `skipped`. Synonyms like
`in_progress` / `WIP` / `failed` are auto-normalised. The `steps`
argument accepts four serialisation shapes (native list, JSON
string of list, JSON string of `{"steps":[…]}` wrapper, free-form
numbered text) because providers serialise complex args differently.

When `workspace=` is also wired, the plan mirrors to a `kind="plan"`
note in the shared notebook — the first `plan_write` creates the
note, subsequent calls `update_note` the same slug. A
`recall_past_plans(query)` tool is auto-added in this case, so
future runs can search past plans by free-text query and bootstrap
from ones that match. Cross-task plan lineage is multi-tenant by
`user_id`, partitioned via the standard `RunContext`.

Per-run plan state lives in a new `_ambient_living_plan_var`
contextvar (mirroring `_ambient_workspace_var`), so concurrent
`agent.run()` invocations on the same `Agent` instance have
isolated plans. Custom architectures / hooks read the active plan
via `loomflow.tools.plan.get_active_plan()`.

Public exports: `LivingPlan` and `LivingPlanStep` at the top level;
`make_plan_tools`, `make_recall_past_plans_tool`,
`living_plan_prompt_section`, `resolve_living_plan`,
`ResolvedLivingPlan` under `loomflow.tools`.

Default is **opt-in** for v0.10.0 (`living_plan=None` → disabled).
v0.11 will flip the default to "auto" — on when `tools=` is
non-empty, off otherwise — after the primitive has been dogfooded.

Empirical motivation: Terminal-Bench 2.0 tasks where the agent
previously failed 3× in a row (across 3 different architectures)
were resolved on the first attempt after adding a structured plan
tool. The pattern is mainstream in 2026 (Claude Code's TodoWrite,
Devin's plan-mode, OpenHands' task graph) — empirical SWE-bench
data shows scaffolding moved scores from 1.96% to 78.4% holding
the model constant.



This release turns Loom from a working agent harness into a
**production framework**. Five themed milestones (M1–M5) close the
gaps that separate a demo loop from something you put in front of
paying users — multi-tenant safety, conversation continuity, typed
outputs, retry/error taxonomy, and a fast path that skips no-op
integrations. Plus an audit pass that aligns the public surface,
a Sphinx docs site, migration guides for LangGraph and the raw
OpenAI SDK, and a live integration suite that hits paid endpoints.

Every change is **additive**; existing code keeps working unchanged.
Multi-tenancy and structured outputs are opt-in by passing
`user_id=` / `output_schema=`. Retry-on-transient is auto-applied
to in-tree network adapters (OpenAI, Anthropic, LiteLLM); custom
models opt in.

### Added — verbatim audit capture via `audit_log={...}` dict

The default audit log is compliance-friendly: prompts are truncated
to 500 chars, the model's final output isn't recorded, and tool
results carry only `ok` / `denied` / `error` / `reason`. That's
right for regimes that prohibit logging customer PII verbatim.

For **debugging**, **post-incident replay**, or **internal
investigations**, opt into verbatim capture by passing the existing
`audit_log=` parameter a config dict — no extra class to import:

```python
agent = Agent(
    "...",
    audit_log={
        "name": "./audit.jsonl",   # path; omit for in-memory
        "scope_full": True,         # capture prompts + outputs + tool bodies
        "secret": "hmac-key",       # optional signing key
    },
)
```

The same dict works on `Workflow(...)`. With `scope_full: True` the
audit payload includes the full prompt, the model's final output
(and `parsed` for structured runs), the full tool result body, and
tool call duration. Signatures still verify cleanly.

`Agent(audit_log=...)` and `Workflow(audit_log=...)` now share one
resolver — accepts `None`, an `AuditLog` instance, a `str` / `Path`
sugar for `FileAuditLog`, or the new dict form. The class
`FullTranscriptAuditLog` is still available for power users who
want to wrap a hand-built backend; `isinstance(log,
FullTranscriptAuditLog)` is the audit reviewer's signal that PII
may be in the log.

### Added — dict-form `model=` config

The `model=` parameter now accepts a config dict, mirroring the
`audit_log={...}` shape — one parameter carries the model spec and
related agent-level defaults (no need for separate `effort` /
`strict_effort` kwargs at the call site):

```python
agent = Agent(
    "Plan the migration in detail.",
    model={
        "name": "claude-opus-4-7",
        "effort": "high",
        "strict_effort": True,
    },
)
```

Recognised dict keys: `name` (the model spec — required; `model`
is an alias), `effort`, `strict_effort`. Top-level kwargs win when
both are specified, so `Agent(model={"effort": "low"},
effort="high")` uses "high". Unknown keys raise `ConfigError`.

Equivalent to the explicit form `Agent(model="claude-opus-4-7",
effort="high", strict_effort=True)` — pick whichever reads better
at the call site.

### Added — `effort` dial for reasoning-capable models

One enum, one kwarg, every provider's shape — agents can now
request "more thinking" without learning each lab's API.

```python
agent = Agent(
    "Plan the migration in detail.",
    model="claude-opus-4-7",
    effort="xhigh",       # Loom translates per provider
)
# Per-call wins over the agent default:
await agent.run("...", effort="low")
```

`effort` accepts ``"minimal" | "low" | "medium" | "high" | "xhigh"
| "max"`` (exported as ``loomflow.Effort``). Each adapter
translates into the provider's native shape:

* **OpenAI** o-series / GPT-5 — `reasoning_effort` (`xhigh` / `max`
  clamp to `high` — OpenAI's enum tops out there).
* **Anthropic Opus 4.7 / Mythos** — adaptive `thinking` +
  `output_config.effort`, the only regime where `xhigh` and `max`
  pass through unclamped.
* **Anthropic Opus 4.6 / Sonnet 4.6** — adaptive + effort enum
  (xhigh/max clamp to high).
* **Anthropic Sonnet 3.7 / 4 / 4.5** — legacy
  `thinking.budget_tokens` integer (1024 → 32768 across the
  range).
* **LiteLLM** — `reasoning_effort` forwarded; LiteLLM handles
  per-provider routing.

Adapters whose model doesn't support reasoning effort emit a
one-time warning per `(model, effort)` pair and drop the kwarg —
opt into hard-fail with `Agent(..., strict_effort=True)` to catch
typos or capability mismatches loudly during development.

### Added — `FileTelemetry` — JSONL telemetry on disk

A no-collector-required sink for users who want spans + metrics
persisted to disk in a parseable format. Each completed span
and metric emit becomes a single JSON line, with a ``"kind"``
discriminator so downstream pipelines can split them.

```python
from loomflow.observability import FileTelemetry

agent = Agent(..., telemetry=FileTelemetry("./traces.jsonl"))
await agent.run("...")
```

Output:

```jsonl
{"kind":"span","name":"loom.turn","trace_id":"...","parent_span_id":"...","duration_ms":380,"attributes":{"turn":1},"exception":null}
{"kind":"span","name":"loom.run","parent_span_id":null,"duration_ms":420,...}
{"kind":"metric","name":"loom.tokens.input","value":42,"instrument_kind":"counter","attributes":{},"emitted_at":"..."}
```

Query offline with `jq`:

```shell
jq -c 'select(.kind=="span" and .duration_ms > 1000)' traces.jsonl
jq -c 'select(.attributes.session_id=="sess_xyz")'     traces.jsonl
```

Mirrors `FileAuditLog`'s pattern: parent dir auto-created,
writes through `anyio.to_thread.run_sync` so the event loop
never blocks on disk I/O, internal lock serialises concurrent
emits from parallel tool dispatches, file is append-only so
restart-recovery is free.

**Complements** `FileAuditLog`; doesn't replace it. Audit log =
business events for compliance; telemetry = performance /
diagnostic spans. Run both together in production for full
offline forensics. No rotation built in — use `logrotate` /
`journald`.

5 new tests: span JSONL format, metric kind tag, parent dir
auto-creation, exception recording, append-across-restart.

### Added — `InMemoryTelemetry` + `ConsoleTelemetry` + `MultiTelemetry`

Three new telemetry sinks that need no OTel collector deploy.
Previously the only "see what spans my agent emits" path was
`OTelTelemetry` plus ~10 lines of OTel SDK boilerplate
(``TracerProvider`` + ``SpanProcessor`` + ``InMemorySpanExporter``).
That's the right primitive for production, but pure friction
for tests, exploration, and dev "tail my agent" workflows.

```python
from loomflow.observability import (
    InMemoryTelemetry, ConsoleTelemetry, MultiTelemetry,
)

# Assert on spans + metrics in unit tests
tel = InMemoryTelemetry()
agent = Agent(..., telemetry=tel)
await agent.run(...)
assert any(s.name == "loom.tool" for s in tel.spans())

# Watch a flow live in stderr while developing
agent = Agent(..., telemetry=ConsoleTelemetry())

# Fan out — see live AND inspect after
in_mem = InMemoryTelemetry()
agent = Agent(..., telemetry=MultiTelemetry([
    ConsoleTelemetry(), in_mem,
]))
```

`InMemoryTelemetry` records each span as a
:class:`~loomflow.observability.CapturedSpan` (name, trace_id,
span_id, parent_span_id, started_at, ended_at, duration_ms,
attributes, optional exception repr). Metrics become
:class:`~loomflow.observability.CapturedMetric` with the
auto-detected instrument kind (counter vs histogram) — same
suffix-based dispatch rule as `OTelTelemetry`.

`ConsoleTelemetry` prints one line per span completion + one
line per metric emit. Indented by parent depth so the trace
tree is visible. Default stream is `sys.stderr`; `show_metrics=
False` available for span-only output.

`MultiTelemetry` enters every sink's ``trace()`` contextmanager
via `AsyncExitStack`, so cleanup runs in reverse order even
when one sink raises mid-emit. ``MultiTelemetry([])`` is rejected
with a clear "use NoTelemetry() for a no-op" error.

Production path unchanged — `OTelTelemetry()` with no args still
picks up the globally-configured OTel `TracerProvider` /
`MeterProvider`.

10 new tests across span hierarchy, metric kind dispatch, clear-
state reset, exception recording, console stream output, multi
fan-out, empty-sink rejection.

Updated `examples/13_telemetry.py` — was 80 lines of OTel SDK
plumbing, now 30 lines of Loom-native sink usage.

### Added — `response_tone=` on Agent + Workflow

A new optional kwarg that steers *how* the agent phrases its
output — not *what* it answers. Pass a preset name or any
free-form string; the framework appends a one-line style
directive to the system prompt. Default is ``None`` (no
directive, no behaviour change).

```python
# Preset
agent = Agent("...", model="gpt-4.1-mini", response_tone="legal")

# Free-form passthrough — preset map is convenience, not gatekeeper
agent = Agent("...", response_tone="warm but precise, like a doctor")

# Per-call override beats agent default
result = await agent.run("...", response_tone="casual")

# Workflow ambient flows to nested agents that didn't set their own
wf = Workflow.chain([agent_a, agent_b], response_tone="executive")
```

**Shipped presets** (one sentence each, intentionally tight):
``casual``, ``professional``, ``technical``, ``legal``,
``finance``, ``executive``, ``academic``.

**Resolution order**: per-call > agent default > workflow ambient
> ``None``. Same propagation pattern as ``Workflow(memory=...)``
— via a contextvar in :mod:`loomflow.core.context` that the
workflow installs in ``stream()`` and resets in ``finally``.

**Tone vs persona vs instructions** — three orthogonal levers:

* **Instructions** (``Agent(instructions=...)``) — *what* the
  agent does.
* **Persona** (text inside instructions, e.g. "You are a tax
  lawyer...") — *who* the agent is.
* **Tone** (``response_tone=``) — *how* the agent phrases its
  output.

Free-form strings let users pin a custom org voice without
registering it as a preset. The framework treats the value as
opaque text appended to the system prompt — the model handles
the actual styling.

**Interaction with structured output:** when both
``response_tone`` and ``output_schema`` are set on a non-native
model adapter (so the schema directive does get injected), the
tone is appended AFTER the schema so it's the last thing the
model reads. For native-structured-output adapters (OpenAI,
Anthropic), only the tone is appended; the schema is the
API-level constraint as before.

### Fixed — Don't double-inject schema for native structured output

When the model adapter declares
``supports_native_structured_output = True`` (OpenAI, Anthropic,
LiteLLM-passthrough), the agent loop no longer ALSO appends the
JSON Schema as text into the system prompt. The native API-level
constraint (``response_format=json_schema, strict=True`` /
forced ``__output__`` tool call) is sufficient on its own;
duplicating the schema in the prompt was just dead tokens —
~2k extra input tokens per structured-output call.

Concrete impact, measured on ``gpt-4.1-mini`` against the
benchmark scenario in ``test/bench/`` (RAG + Pydantic
``PdfSummary`` schema):

* Input tokens: **~3,091 → ~1,100** (≈64% reduction).
* Cost per call: **$1.535m → ~$0.55m** (Loom now cheaper than
  LangGraph's $0.729m on the same workload).
* Reliability: unchanged — validate-with-retry still injects
  the schema into the retry message if the model ever produces
  invalid JSON.

Custom user-supplied model adapters that don't set the flag keep
the prompt-augmentation safety net (default off). Two tests guard
against regression: one for the native-skip path, one for the
non-native still-injects path.

### Added — `Memory.recall_scored()` — hybrid BM25+vector retrieval with score breakdown

A protocol-evolution change so adding rerankers / MMR / hybrid
weighting later isn't a breaking surface change. The new
``recall_scored`` method returns a list of :class:`EpisodeMatch`
— each carrying the raw episode plus a fused score and the
component scores (BM25, vector cosine, optional reranker) used
to rank it. Callers that want to apply a downstream reranker, an
MMR diversifier, or a score-threshold filter can now do so
without re-running recall.

```python
matches = await agent.memory.recall_scored(
    "postgres replication",
    user_id="alice",
    alpha=0.5,        # 0=BM25 only, 1=vector only, 0.5=balanced (RRF)
)
for m in matches:
    print(m.episode.input, m.score, m.bm25_score, m.vector_score)
```

**Native hybrid implementations:**

* :class:`InMemoryMemory` now ships a real BM25 ranker (replaces
  the prior substring-match-then-recency behaviour). Episodes
  whose ``input`` / ``output`` lexically match the query rank
  ahead of unrelated recent episodes — a real recall-quality
  upgrade for the default backend.
* :class:`VectorMemory` does the full BM25 + cosine + Reciprocal
  Rank Fusion (RRF) hybrid that the framework's vectorstore
  module already used for RAG, now extended to agent memory.

**Other backends (Chroma, Postgres, Redis, Sqlite, AutoExtract,
Lazy)** ship a thin shim via the new
:func:`loomflow.memory.default_recall_scored` helper that wraps
their existing recall results with neutral scores. The protocol
stays coherent; native hybrid implementations for those backends
can land later without breaking callers.

**Why this matters competitively:** before this change, Loom's
recall was cosine + token-overlap fallback only — weaker than
Zep (BM25 + vector + graph BFS + reranker) and CrewAI (composite
+ deep mode). After this change, Loom matches the hybrid-recall
baseline of the field for in-process memory, with the protocol
shape ready for a reranker / MMR / cross-encoder layer when
someone needs one. The full bi-temporal + auto-extract +
multi-tenant + hybrid-recall combination doesn't exist anywhere
else open-source.

### Changed — Skills `tools.py` now imports lazily + `build_tools(ctx)` factory

Two related improvements to how Loom skills load their Python
tools — both fixing real friction and adding a capability that
beats deepagents' "no tools.py at all" punt.

**Lazy `tools.py` import (breaking, but fixes a footgun).**
Previously, ``Skill("path/")`` imported the skill's ``tools.py``
*at construction time*. If that file did module-level event-loop
work — e.g., ``asyncio.run(setup_vectorstore())`` — and the
caller was already inside a Jupyter event loop, the import
crashed with ``RuntimeError: asyncio.run() cannot be called from
a running event loop``.

Now the import is deferred to first-use:

* ``Skill(...)`` only *detects* ``tools.py``; doesn't import it.
* ``Skill.materialize_tools(ctx)`` does the import on first call
  and caches the result. Subsequent calls reuse the cache.
* The framework's built-in ``load_skill`` tool calls
  ``materialize_tools`` from inside the running agent loop, where
  doing event-loop work is fine.

**Behaviour change:** import errors in ``tools.py`` now surface
when the model first calls ``load_skill(name)``, not at
``Skill(...)`` construction. Most skill code paths are unaffected;
the Mode C subprocess-tool path stays eager. Migration: if your
tests asserted "construction raises on bad ``tools.py``", call
``skill.materialize_tools()`` explicitly to trigger the import.

**`build_tools(ctx)` factory protocol (new feature).** Skills
can now ship tools that close over caller-supplied state without
globals or module-level setup:

```python
# skills/pdf-retrieval/tools.py
from loomflow import tool

def build_tools(ctx):
    vectorstore = ctx.metadata["vectorstore"]
    @tool
    async def retreiver(query: str) -> list:
        return await vectorstore.search_hybrid(query=query)
    return [retreiver]
```

```python
# In your script:
agent = Agent(..., skills=[Skill("skills/")])
result = await agent.run(
    "...",
    metadata={"vectorstore": vectorstore},
)
# load_skill('pdf-retrieval') passes the live RunContext to
# build_tools(ctx); the resulting `retreiver` is registered with
# the agent's tool host and visible on the next turn.
```

When ``tools.py`` *doesn't* export ``build_tools``, the framework
falls back to discovering module-level ``@tool``-decorated
globals (the prior behaviour). Back-compatible.

**Bonus: clearer error on the asyncio gotcha.** When a skill's
``tools.py`` does ``asyncio.run(...)`` at module level and the
import does fire inside a running loop, the resulting
``SkillError`` now includes a hint pointing at the
``build_tools(ctx)`` pattern instead of just the raw asyncio
traceback.

### Fixed — Workflow router classifier can now be `async def`

The classifier passed to ``add_router`` (both ``add_router(node,
fn=...)`` and ``add_router(START, fn=...)``) is now awaited when
it's an async function. Previously the framework called it
synchronously, so an ``async def`` classifier returned a coroutine
object that ``str()`` rendered as ``<coroutine object …>`` —
which never matched any route key, causing:

> ``RuntimeError: entry router on '...' produced key
> <coroutine object ... at 0x...> with no matching route and no
> default``

Common case this hits: a classifier that calls a model to decide
the branch. New ``_eval_classifier`` helper detects coroutine
functions via ``inspect.iscoroutinefunction`` and awaits them;
also handles sync wrappers around async inner calls
(``lambda v: some_async_fn(v)``) by awaiting the returned
coroutine.

### Added — `add_router(START, ...)` — branch at the entry of the graph

You can now classify the workflow's input and route directly to
one of N first nodes, without an artificial passthrough "entry"
step. Mirrors LangGraph's ``add_conditional_edges(START, ...)``:

```python
from loomflow import Workflow, START, END

wf = Workflow()
wf.add_node("step_1", step_1)
wf.add_node("step_3", step_3)
wf.add_router(
    START,
    fn=lambda q: "step_1" if "work" in q else "step_3",
    routes={"step_1": "step_1", "step_3": "step_3"},
    default=END,                         # optional fallback
)
wf.add_edge("step_1", END)
wf.add_edge("step_3", END)
```

Validation at build time: route targets must be registered nodes
or ``END``, so typos raise a clear error before the run starts.
``set_start`` and ``add_router(START, ...)`` are mutually
exclusive — calling one resets the other. The mermaid / DOT
diagrams show ``START -->|key| node`` directly so the visual
matches what you wrote.

Ships alongside fix to ``WorkflowResult`` reconstruction so the
zero-step ``default=END`` path returns the original input as
``result.output`` (it was returning ``None`` before).

### Added — `Workflow(memory=...)` shared agent memory across the graph

Workflows can now own a single :class:`~loomflow.Memory` and
propagate it to every nested :class:`~loomflow.Agent` step that
didn't specify its own — so episodes / facts written by one
agent are recall-able by the next without per-agent wiring:

```python
from loomflow import Workflow, Agent, InMemoryMemory

mem = InMemoryMemory()
agent_a = Agent(model="gpt-4.1-mini", instructions="...")
agent_b = Agent(model="gpt-4.1-mini", instructions="...")

wf = Workflow.chain([agent_a, agent_b], memory=mem)
await wf.run("hi", user_id="alice", session_id="conv-1")
# Both agents wrote to / read from `mem`.
```

Resolution order is **explicit always wins**:

1. ``Agent(memory=my_mem)`` keeps using ``my_mem`` even inside
   a workflow with ``memory=``.
2. Otherwise ``Workflow(memory=mem)`` is used as the fallback.
3. Otherwise the agent's per-instance default (in-memory).

Implemented via a contextvar in :mod:`loomflow.core.context`
that the workflow installs at the start of ``run`` / ``stream``
and resets in ``finally``, so memory does not leak across
workflow runs. Available on the explicit ``Workflow(memory=...)``
constructor and on all sugar constructors (``chain``, ``route``,
``parallel``).

### Added — `add_edge(START, "node")` as alias for `set_start("node")`

The ``START`` sentinel was previously inert (just a name). It now
works as an edge source so graphs read symmetrically with the
``END`` sentinel:

```python
wf.add_edge(START, "first")     # alias for set_start("first")
wf.add_edge("first", "second")
wf.add_edge("second", END)
```

Matches the pattern users coming from LangGraph expect, and lets
the entry / exit show up in mermaid diagrams via the same edge
syntax. ``set_start`` continues to work — no breaking change.
``add_edge(START, END)`` and ``add_edge(END, ...)`` are rejected
with messages that point at the right method.

### Added — Workflow visualisation: `to_mermaid()` / `to_dot()` / Jupyter

* **`Workflow.to_mermaid() -> str`** — returns a Mermaid
  ``flowchart TD`` diagram of the graph. Pastes directly into
  GitHub Markdown (renders inline) or https://mermaid.live for
  PNG / SVG export. Solid arrows are unconditional edges,
  labelled solid arrows are router branches, dotted arrows are
  router *defaults*. ``START`` and ``END`` are stadium-shaped.
* **`Workflow.to_dot() -> str`** — same picture in Graphviz DOT
  for users who prefer the Graphviz toolchain (`dot -Tpng -o
  graph.png`). Optional — Mermaid is the recommended path since
  it needs no install.
* **`Workflow._repr_markdown_`** — Jupyter / VS Code / JupyterLab
  auto-render the diagram inline when you type ``wf`` into a cell.
  No imports, no extra calls.

Tests cover linear chains, routers (labelled + default branches),
empty workflows, DOT shape declarations, and the markdown wrapper.

### Changed — Boundary-input errors now name the fix, not just the failure

A pass over the most-hit user entry points: when the framework
rejects a wrong-shape argument, the error message now (1) names
the offending value, (2) lists the accepted forms, and (3) shows
a working example. The goal is "fail fast at the point of
mis-use, with the next step in the message" — replacing several
errors that previously surfaced deep inside the runtime as
generic Python exceptions (`'str' can't be used in 'await'
expression`, `list.append() takes no keyword arguments`, etc.).

* **`Workflow(audit_log=...)` accepts `str` / `Path`** — auto-
  wraps as :class:`~loomflow.security.FileAuditLog` so users can
  write `Workflow.chain(..., audit_log="run.log")` without
  importing the backend. Anything else (e.g. a bare `list`) is
  rejected at construction time with a `TypeError` listing the
  four valid forms (`InMemoryAuditLog`, `FileAuditLog`, path
  string/`Path`, `None`).
* **`Agent(tools=...)` rejection** — non-list / non-Tool / non-
  callable / non-`ToolHost` values now fail with a message that
  names every accepted form (`tools=None`, list, single tool,
  single callable, `ToolHost` instance).
* **`tools=[entry]` rejection** — non-callable list entries now
  fail with a `@tool`-decorated function example in the error
  text, so the user sees the fix inline.

### Changed — Workflow `@step` decorator: clearer error on sync functions

* **`@step` now raises at decoration time** when applied to a
  synchronous `def`. Previously, wrapping a sync function silently
  succeeded and the workflow only failed deep inside the runner
  with `'str' can't be used in 'await' expression` — a cryptic
  message that gave no hint that the user's function was the
  cause. The new `TypeError` names the offending function and
  spells out both fixes:
    1. Add `async` to the `def` (gets telemetry / audit / journaling).
    2. Drop `@step` and pass the plain function directly —
       `Workflow.chain` and `.route` already accept sync callables
       and dispatch them to a worker thread.
  Failure now surfaces at module import, not after a workflow
  has started running.

### Added — Multi-tenancy by default (M1)

* **`RunContext`** (frozen `dataclass(slots=True)`) — typed,
  immutable per-run scope with `user_id`, `session_id`, `run_id`,
  and `metadata`. First-class framework primitive, not strings in
  a `configurable` dict. `with_overrides(...)` for sub-agent
  inheritance.
* **`get_run_context()`** — read the live context inside any tool /
  hook / sub-agent. Backed by a `ContextVar` set in `Agent._loop`;
  `anyio` task groups propagate it across parallel tool dispatch
  and spawned sub-agents automatically. Returns the empty default
  outside an active run — never raises, so direct `@tool` calls in
  tests keep working.
* **`set_run_context(ctx)`** — async context manager for installing
  a context outside an active run (background workers that share
  tool implementations with the agent).
* **`Agent.run(user_id=, session_id=, metadata=, context=)`** —
  flat kwargs; LangGraph-style `config={"configurable": {...}}`
  nesting deliberately avoided. Same kwargs added to
  `Agent.stream` and `Agent.resume`.
* **`Episode.user_id` + `Fact.user_id`** — Pydantic fields, optional.
  Backends partition on these as a hard namespace boundary:
  episodes / facts stored under one `user_id` are never visible to
  a recall scoped to a different one. `None` is its own
  ("anonymous / single-tenant") bucket.
* **`Memory.recall(user_id=)`, `Memory.recall_facts(user_id=)`,
  `FactStore.query(user_id=)`, `FactStore.recall_text(user_id=)`** —
  partition filter wired through every implementation:
  `InMemoryMemory`, `VectorMemory`, `ChromaMemory`,
  `PostgresMemory`, `RedisMemory`, plus all four fact stores.
  Postgres / SQLite use `IS NOT DISTINCT FROM` for safe
  NULL-bucket comparisons; Chroma uses native `where` filters;
  Redis stores `user_id` as a Hash field and post-filters.
* **Schema migrations** — `episodes` and `facts` Postgres tables
  gain a `user_id TEXT` column with idempotent
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for in-place
  upgrades; SQLite gets the same with a duplicate-column-tolerant
  `try`/`except`. New `(namespace, user_id, occurred_at DESC)`
  index on Postgres episodes; `(user_id, subject, predicate)` on
  the facts tables.
* **Namespace-scoped supersession** — fact-store supersession (the
  bi-temporal `valid_until = new.valid_from` write when a new
  fact replaces an old one) is now scoped by `user_id`: alice's
  new claim never invalidates bob's currently-valid claim on the
  same `(subject, predicate)`. Across all four fact-store
  backends.

### Added — Conversation continuity (M2)

* **`Memory.session_messages(session_id, *, user_id=, limit=)`** —
  new protocol method returning prior user/assistant turns from
  the named conversation, scoped to the `user_id` partition,
  oldest-first. Implemented across all five memory backends.
* **`_build_seed_messages` rehydration** — when `session_id` was
  reused, the agent loop loads the prior turns as real `Message`
  history (proper USER / ASSISTANT roles) before the current
  prompt. Cross-session episodic recall filters out the current
  session's episodes to avoid duplication. Reusing the same
  `session_id` across `agent.run()` calls now genuinely continues
  the conversation — no reducer protocol, no `add_messages` magic.

### Added — Footgun protection + multi-agent inheritance (M3)

* **`IsolationWarning`** (subclass of `UserWarning`) — fires when
  `Memory.recall(user_id=None)` runs against a store whose
  episodes / facts include a non-None `user_id`. The partition is
  still safe, but the developer probably forgot to pass
  `user_id=` somewhere; loud failure beats silent confusion. Goes
  through Python's `warnings` machinery — apps promote to error
  with `warnings.simplefilter("error", IsolationWarning)`.
* **Multi-agent context inheritance** — `SubagentInvocation` (used
  by `Supervisor`, `Debate`, `Swarm`, `Router`, `ActorCritic`,
  `Blackboard`) now reads `get_run_context()` automatically when
  no explicit `context=` is passed. The parent's `user_id` and
  `metadata` propagate to every sub-agent without each
  architecture having to plumb them by hand. Sub-agents get a
  fresh `session_id` so each worker has its own thread while
  inheriting the parent's namespace.

### Added — Structured outputs (M4)

* **`Agent.run(output_schema=, output_validation_retries=)`** —
  pass any Pydantic `BaseModel` and the framework augments the
  per-run system prompt with a `STRUCTURED OUTPUT REQUIRED`
  directive embedding the schema's JSON Schema, parses the final
  assistant text, and returns a validated typed instance on
  `RunResult.parsed`. Static `agent._instructions` is not
  mutated; the augmentation is per-run.
* **Retry-with-feedback** — on parse failure, the framework gives
  the model up to `output_validation_retries` (default 1) extra
  single-shot turns to fix the output, feeding the validation
  error back as a USER message. After the retry budget is
  exhausted, raises `OutputValidationError`.
* **`OutputValidationError`** — carries the raw text (`raw`), the
  schema being targeted (`schema`), and the underlying Pydantic
  `ValidationError` (`cause`, also linked via `__cause__`).
* **`RunResult.parsed: Any | None`** — typed, validated instance
  when `output_schema=` was supplied; `None` otherwise.
  `RunResult.output` keeps the raw (cleaned) JSON text for
  logging / audit.
* **Markdown-fence tolerance** — strips `` ```json `` / `` ``` ``
  fences before parsing, since real models occasionally wrap
  output despite being told not to.
* **`RunResult.value`** — smart accessor that returns `parsed`
  when a schema validated, else the raw `output` string. Removes
  the `result.output` vs `result.parsed` "did the schema even
  fire?" footgun: `result.value` is always "the answer" in the
  shape the caller expects. The original `parsed` / `output`
  fields stay untouched for code that reads them directly.
* **`Agent(output_schema=...)`** — agent-bound default schema.
  Pass it once on construction and every `agent.run()` /
  `agent.stream()` applies it; a per-call `output_schema=` still
  overrides for one-off shapes. Mirrors Pydantic AI's
  `output_type=` ergonomics.
* **Tagged-union output schemas** — `output_schema=A | B` (or
  `Union[A, B]`) lets an agent return one of multiple shapes per
  call. Validation tries each member in declaration order and
  accepts the first that fits, so callers can model
  "valid result vs structured error" without a discriminator
  field.

### Added — Production hardening (M10)

The "best-in-class" pass: closes seven holes a Google reviewer
called out (unbounded per-user state, Postgres empty-string-for-
NULL hack, silent auto-extract, silent ``ask`` bypass, missing
deprecation infrastructure, hard-coded API-key resolution, no
multi-tenant load proof). Every change is additive; default
behaviour is preserved unless a caller opts in.

* **Bounded per-user state (M10.1).** ``StandardBudget._by_user``
  and ``InMemoryMemory._blocks`` now use a new
  :class:`loomflow.core._eviction.BoundedDict` with LRU +
  idle-TTL eviction. Defaults: ``max_users=100_000``,
  ``user_idle_ttl_seconds=86_400`` (24h). Pass ``None`` to
  either constructor kwarg to disable bounding for single-tenant
  deployments. Eviction *drops* a user's bucket — callers needing
  durable spill-to-disk should pick :class:`SqliteMemory` /
  :class:`PostgresMemory` instead of relying on the in-process
  bound.
* **Postgres anonymous-bucket sentinel (M10.2).** The empty-
  string-for-NULL hack on ``memory_blocks.user_id`` is replaced
  with a reserved sentinel ``__jeeves_anon_user__``. Schema DDL
  includes an idempotent migration that rewrites legacy ``''``
  rows. Callers that try to use the sentinel as a real
  ``user_id`` get a ``ValueError`` — defense against impersonating
  the anonymous bucket.
* **Auto-extract observability (M10.3).** :class:`AutoExtractMemory`
  now emits ``jeeves.auto_extract.duration_ms`` (histogram) and
  ``jeeves.auto_extract.invocations`` (counter) per extraction,
  tagged with ``user_id`` and ``status``. A one-time-per-process
  ``INFO`` log notice fires when the wrapper is enabled by the
  default-on heuristic, so ops teams learn about it before the
  LLM bill arrives.
* **Permission "ask" approval handler (M10.4).**
  ``Agent(approval_handler=callable)`` resolves
  ``Decision.ask_(...)`` outcomes from the permissions layer.
  Without one, ``ask`` falls back to deny — closes a security
  hole where the fast-hooks default-allow was silently bypassing
  the approval gate.
* **Deprecation infrastructure (M10.5).** New
  :class:`LoomDeprecationWarning` subclass + ``warn_legacy_
  protocol(...)`` helper; every protocol-evolution
  ``except TypeError`` shim now warns once-per-process pointing
  at the v1.0 removal target so callers can migrate.
* **Secrets protocol wired into model resolution (M10.6).** New
  concrete :class:`EnvSecrets` (default) and
  :class:`DictSecrets` impls plus a ``lookup_sync(ref)`` method
  on the Secrets protocol. ``Agent(secrets=...)`` flows through
  to model adapters; ``OpenAIModel`` / ``AnthropicModel`` /
  ``LiteLLMModel`` resolve API keys via ``api_key=`` →
  ``secrets.lookup_sync`` → ``os.environ`` precedence.
  ``redact()`` masks common API-key shapes (OpenAI / Anthropic /
  AWS / GitHub) so audit logs don't leak credentials.
* **Multi-tenant load benchmark (M10.7).** New
  ``bench/multi_tenant.py`` simulates N concurrent users × M turns
  on one shared Agent and reports p50 / p99 latency, RSS growth,
  isolation violations, budget mismatches. Smoke-test variant in
  ``tests/test_multi_tenant_load.py`` runs as part of the regular
  suite.
* **Tests + docs.** 980 tests pass (up from 933 at the start of
  M10); mypy ``--strict`` clean across 112 source files; ruff
  clean. CHANGELOG entry, capability matrix updated.

### Added — Multi-tenant by default *everywhere* (M9)

Closes the remaining gaps so every stateful primitive partitions by
``user_id``. Memory was already done (M1–M8); M9 covers working
blocks, budget, audit log, permissions, hooks.

* **Working memory blocks** — ``Memory.working(user_id=)`` /
  ``update_block(name, content, user_id=)`` /
  ``append_block(name, content, user_id=)``. All six backends
  partition: in-memory dicts re-keyed to ``(user_id, name)`` tuples;
  SQLite + Postgres got migrations adding a ``user_id`` PK column
  (idempotent, with table-rebuild fallback for SQLite which can't
  ``ALTER`` a PK). Pinned-order is per-user — adding bob's first
  block doesn't bump alice's slots. The agent loop reads
  ``deps.memory.working(user_id=deps.context.user_id)``; legacy
  custom Memory impls without the kwarg fall back gracefully.
* **Per-user budget accounting** — ``StandardBudget`` now tracks
  tokens / cost per ``user_id``. New ``BudgetConfig`` fields
  ``per_user_max_tokens``, ``per_user_max_input_tokens``,
  ``per_user_max_output_tokens``, ``per_user_max_cost_usd``,
  ``per_user_max_wall_clock`` enforce per-user caps alongside (or
  instead of) the global ones. ``Budget.allows_step(user_id=)`` /
  ``consume(user_id=)`` are the new protocol shape.
  ``StandardBudget.usage_for(user_id)`` snapshots a single user's
  totals for ops dashboards.
* **Audit log: top-level ``user_id``** — ``AuditEntry.user_id`` is
  now a first-class field (was buried in payload). ``AuditLog.append``
  + ``AuditLog.query`` gain the kwarg. The HMAC signature covers
  ``user_id`` so a tampered entry that swaps user identity won't
  verify. Both ``InMemoryAuditLog`` and ``FileAuditLog`` updated.
* **Permissions: ``user_id`` kwarg** — ``Permissions.check(call,
  context=, user_id=)``. ``StandardPermissions`` accepts and ignores
  it; the new ``PerUserPermissions`` routes to per-user policies::

      perms = PerUserPermissions(
          policies={
              "admin_alice": StandardPermissions(mode=Mode.BYPASS),
              "paid_user_42": StandardPermissions(mode=Mode.ACCEPT_EDITS),
          },
          default=StandardPermissions(
              mode=Mode.DEFAULT, denied_tools=["bash"]
          ),
      )
      Agent(..., permissions=perms)

* **Hooks: ``user_id`` kwarg** — ``HookHost.pre_tool`` /
  ``post_tool`` accept ``user_id`` so custom hook hosts can dispatch
  per-user. The bundled ``HookRegistry`` accepts and ignores the
  kwarg; individual hook callables continue to receive only
  ``(call,)`` / ``(call, result)`` for API stability.
* **Backwards-compatible everywhere** — every protocol change is a
  keyword-only add. The agent loop wraps every kwarg-bearing call in
  a ``try / except TypeError`` fallback to the legacy signature, so
  custom implementations users wrote pre-M9 keep working unchanged.
* **Public exports** — ``PerUserPermissions`` is exported from both
  ``loomflow.security`` and the top-level ``loomflow`` package.
* **Tests** — 14 new in ``tests/test_user_id_isolation_full.py``
  covering every primitive: working-block partition (in-memory +
  SQLite-persistent), pinned-order per-user, budget per-user
  totals + per-user caps + global+per-user combination, audit
  top-level user_id + filter + combined filter, permissions
  routing + default fallback, full end-to-end with one Agent
  serving alice and bob through Memory + Budget + AuditLog all at
  once.

### Added — Auto fact extraction (M8)

* **`AutoExtractMemory`** — a :class:`Memory` wrapper that runs the
  bundled :class:`Consolidator` on every persisted episode,
  extracting structured ``(subject, predicate, object)`` claims
  into the inner backend's fact store. Implements the full
  :class:`Memory` protocol; forwards every method through to the
  inner backend; only ``remember`` adds the extraction pass.
* **`Agent(auto_extract=...)`** — new kwarg, default-picked by
  model class. ON for in-tree network adapters (``OpenAIModel`` /
  ``AnthropicModel`` / ``LiteLLMModel``); OFF for in-process fakes
  (``ScriptedModel`` / ``EchoModel``) and unrecognised custom
  Models. Pass ``auto_extract=True``/``False`` to override.
* **Internal split: `_memory` vs `_wrapped_memory`** —
  ``agent.memory`` (the public accessor) keeps returning the
  user-supplied / resolver-built backend so introspection and
  ``agent.memory.profile(...)`` style code work transparently.
  ``_wrapped_memory`` is the loop-facing view that runs through
  the auto-extract layer. Same dual-attribute pattern as
  ``_model`` / ``_wrapped_model`` for the retry layer.
* **Best-effort by design** — extraction failures (model errors,
  malformed JSON, rate limits) never break the run. The episode
  write succeeds first; extraction runs after and either appends
  facts or logs and moves on. The agent's primary contract
  (return a result, persist the episode) is preserved unchanged.
* **End-to-end UX** — a single ``agent.run("I prefer dark mode",
  user_id="alice")`` against a real model now leaves a
  ``Fact(user_id="alice", subject="alice", predicate="prefers",
  object="dark_mode")`` in the store, partition-respecting, ready
  for future ``recall_facts`` queries to surface.

### Added — Memory inspection + GDPR helpers (M7)

* **`Memory.profile(user_id=)`** — returns a `MemoryProfile`
  carrying episode count, fact count, last-seen timestamp, the 10
  most-recent sessions touched, and a sample of the most-recently-
  recorded facts. Per-user, partition-respecting; suitable for
  rendering "what does the bot know about me?" views to end users
  or ops dashboards.
* **`Memory.forget(*, user_id=, session_id=, before=)`** —
  right-to-erasure. With ``user_id`` only, erases all episodes +
  facts for that user. With ``session_id``, narrows to that
  conversation. With ``before``, narrows to a retention window.
  Filters AND together. Returns the count of records deleted.
  ``user_id=None`` erases the anonymous bucket only — same hard
  partition rule as `recall`. Erasing every user is deliberately
  per-user-explicit so it can't happen by accident.
* **`Memory.export(user_id=)`** — full data dump for portability /
  DSAR responses. Returns a `MemoryExport` with every episode and
  fact for the user; serialise with `.model_dump_json()` for
  download.
* **`MemoryProfile`, `MemoryExport`** — new Pydantic types in
  `core/types.py`, exported from both `loomflow.core` and the
  top-level `loomflow` package.
* **Cross-backend implementations** — all six backends
  (`InMemoryMemory`, `SqliteMemory`, `VectorMemory`, `ChromaMemory`,
  `PostgresMemory`, `RedisMemory`) honour the new methods.
  Postgres uses native `IS NOT DISTINCT FROM`-aware DELETEs;
  Chroma uses native `where` filters; SQLite uses `DELETE` against
  the same `.db` file the FactStore lives in; in-memory backends
  filter dicts directly. `LazyMemory` forwards through to the
  inner backend on first use, same as the other protocol methods.
* **Bug fix in `Consolidator._build_fact`** — extracted facts now
  inherit the source episode's ``user_id``. Prior to this, every
  consolidator-extracted fact landed in the anonymous bucket
  regardless of which user the episode belonged to, breaking
  multi-tenant fact recall.
* **Example** — `examples/05_memory_showcase.py` walks every
  backend (Postgres + Redis skip gracefully without DSNs),
  exercises the resolver in all three tiers (string / dict /
  instance), demonstrates profile / forget / export across
  backends, and runs the `Consolidator` to extract structured
  facts from raw episodes. Single runnable file.

### Added — Memory string resolver + SqliteMemory (M6)

* **`memory=` URL/dict/instance resolver** — `Agent(...)` accepts:
  * `None` → default `InMemoryMemory()`
  * `"inmemory"` / `"sqlite:./bot.db"` / `"sqlite"` /
    `"chroma:./vec"` / `"chroma"` / `"postgres://..."` /
    `"redis://..."` (URL scheme picks the backend)
  * `{"backend": ..., "path": ..., "namespace": ...,
    "embedder": ..., "with_facts": ...}` (config dict)
  * any already-constructed `Memory` instance (today's API, unchanged)
  Mirrors the design of the existing `model=` resolver.
* **`resolve_memory(spec)`** — public helper for the same resolution
  logic. Used internally by `Agent.__init__`; exposed so external
  config systems (TOML, YAML, env-driven configs) can drive memory
  picks.
* **`SqliteMemory`** — new backend at
  `loomflow.memory.sqlite.SqliteMemory`. Episodes, working
  blocks, session messages, and the bi-temporal fact store all in
  one sqlite file. Single-file persistence, no server, idempotent
  schema migrations (`CREATE TABLE IF NOT EXISTS`,
  `ALTER TABLE ADD COLUMN`-with-duplicate-tolerant exception).
  Honours the M1 `user_id` partition contract; emits
  `IsolationWarning` on mixed-bucket recall, same as
  `InMemoryMemory`. Use `SqliteMemory(":memory:")` for an
  ephemeral in-process database.
* **`LazyMemory`** — wraps async-construct backends (Postgres /
  Redis) so `Agent(...)` stays synchronous. Connection opens on
  first protocol method call; concurrent first-uses serialise
  through an `anyio.Lock`; backend exceptions get normalised to
  `MemoryStoreError` with the original on `__cause__`. The proxy
  forwards every Memory protocol method (`working`,
  `update_block`, `append_block`, `remember`, `recall`,
  `recall_facts`, `session_messages`, `consolidate`) and exposes
  `.facts` once resolved.
* **Auto-picked embedder** — string and dict specs pick
  `OpenAIEmbedder("text-embedding-3-small")` when
  `OPENAI_API_KEY` is set, `HashEmbedder()` otherwise. Override
  via `embedder=` in the dict form, taking either an `Embedder`
  instance or one of `"hash"`, `"openai"`, `"openai-large"`,
  `"voyage"`, `"cohere"`.
* **Auto-attached fact store on resolver path** —
  `with_facts=True` is the default for string and dict specs;
  semantic-recall layer is on out of the box. Pass
  `with_facts=False` in the dict form to skip it. Explicit
  `Memory(...)` instances keep their existing per-backend
  defaults so today's call sites are unchanged.
* **Public exports** — `SqliteMemory`, `LazyMemory`,
  `resolve_memory` exported from both `loomflow.memory` and
  the top-level `loomflow` package.

### Added — Resilient model calls (M5)

* **Error taxonomy** — `ModelError` base + `TransientModelError`
  (retryable; carries `retry_after`) + `RateLimitError` (subclass
  of transient) + `PermanentModelError` + `AuthenticationError` +
  `InvalidRequestError` + `ContentFilterError`. All inherit from
  `LoomError`; existing `except LoomError` catches
  keep working. Each carries a `cause` slot and chains through
  `__cause__` so debug code can still inspect the raw SDK error.
* **`RetryPolicy`** (frozen dataclass) — `max_attempts`,
  `initial_delay_s`, `max_delay_s`, `multiplier`, `jitter`. Plus
  `RetryPolicy.disabled()` and `RetryPolicy.aggressive()` factories.
  Sensible default: 3 attempts, 1 → 2 → 4 s with ±10% jitter,
  capped at 30 s.
* **`compute_backoff(policy, attempt, retry_after=)`** — exponential
  growth, capped at `max_delay_s`, jittered. Provider-supplied
  `Retry-After` is a **floor**: can exceed the cap because the
  provider is more authoritative than our heuristic.
* **`classify_model_error(exc)`** — maps OpenAI / Anthropic /
  httpx exceptions to the taxonomy via lazy imports (no hard
  dependency on any SDK). Returns `None` for unrecognised
  exceptions — the framework refuses to silently retry errors it
  doesn't understand.
* **`RetryingModel`** — wraps any `Model`; auto-applied to in-tree
  network adapters (`OpenAIModel`, `AnthropicModel`,
  `LiteLLMModel`) by default. Custom Models are not auto-wrapped
  (we can't reason about their error types); pass
  `retry_policy=RetryPolicy()` to `Agent(...)` to opt in.
  Streaming retries fire only **before** the first chunk — once
  tokens are flowing, errors propagate.
* **`Agent(retry_policy=RetryPolicy.disabled())`** — opt out of
  retries when handling errors at a higher layer.

### Added — Documentation + migration

* **Sphinx docs site** at <https://loomflow.readthedocs.io>
  (`docs/conf.py`, Furo theme, `sphinx-autoapi` for the full API
  reference, `myst-parser` so existing `.md` content mounts
  cleanly). Build locally with `pip install -e ".[docs]"` and
  `sphinx-build -b html docs docs/_build/html`. ReadTheDocs
  integration via `.readthedocs.yaml`.
* **`docs/migrations/from-langgraph.md`** — concrete side-by-side
  translations: hello world, tools, multi-tenant memory (the
  `user_id`-as-convention vs `user_id`-as-primitive contrast),
  session continuity, structured output, streaming, multi-agent.
  Plus a "things Loom does NOT have" section.
* **`docs/migrations/from-openai-sdk.md`** — translation guide for
  hand-rolled-loop users: tool definitions, multi-turn state,
  structured output, retries, streaming, parallel tool calls.
* **`pyproject.toml` `docs` extras** — `sphinx`, `furo`,
  `sphinx-autoapi`, `myst-parser`, `linkify-it-py`.
* **Live integration tests** (`tests/test_live_openai.py`) — 9
  tests, ~16 s wall clock, marked `live` (deselected by default).
  Run with `pytest -m live` once `OPENAI_API_KEY` is set. Covers
  the differentiating contracts end-to-end against `gpt-4.1-mini`:
  basic round-trip, tool dispatch, `user_id` partition, `session_id`
  continuity, tool reads `user_id` via `get_run_context()`,
  structured output, structured output retry-on-failure, real
  auth-error → `AuthenticationError` classification, streaming.
* **CHANGELOG.md** — version-by-version release notes (this file).

### Added — Examples

* `examples/01_rag_pdf.py` — single-agent RAG over a folder of
  PDFs. Loader → `RecursiveChunker` → `ChromaVectorStore` →
  `@tool` retriever → `Agent`.
* `examples/02_specialist_debate.py` — five domain specialists
  (IT / physics / medicine / finance / law), each with their own
  Chroma collection, composed via `Team.debate(...)` with a
  synthesising judge.
* `examples/03_multi_user_sessions.py` — live demo of M1+M2 on one
  shared `Agent` + `InMemoryMemory`. Two users, distinct sessions,
  no cross-contamination, tool reads `user_id` via
  `get_run_context()`.
* `examples/04_structured_outputs.py` — extracts a `MeetingSummary`
  (with nested `ActionItem` lists, ISO dates, sentiment enum)
  from a raw transcript.

### Changed — Public API surface

* **`Agent.resume` signature aligned with `Agent.run`** — gained
  `context=`, `output_schema=`, `output_validation_retries=` kwargs.
  The three call methods (`run`, `stream`, `resume`) now have the
  same kwarg surface in the same order.
* **`OutputValidationError.cause` typed as `BaseException | None`**
  (was `Exception | None`) to match `ModelError.cause`.
* **README intro rewritten** — quickstart now demonstrates `user_id`
  partitioning, `session_id` continuity, and structured outputs in
  a single ~25-line example. The "Why pick this over LangGraph"
  framing is preserved but the differentiating bullets reference
  the actual M1–M5 work.
* **README "API stability" section** — four-tier table (Stable /
  Stable backends / Experimental / Internal) so adopters know what
  they can pin against in production.
* **`Dependencies.context: RunContext`** — new field on the
  per-run dependency bundle architectures receive. Existing
  architectures read `deps.context.user_id` to scope memory recall.

### Performance

* **Fast path by default** — every layer (audit, telemetry,
  permissions, hooks, runtime, budget) is detected as no-op or
  production-wired at construction time. The hot path skips the
  integration when the layer is no-op, so a barebones `Agent`
  runs at LangChain-class latency (parity ±2 % on tool-using
  scenarios; see `bench/jeeves_vs_langchain.py`). The moment any
  of those layers is wired up, the integration becomes active —
  no flag, no constructor split.
* **Non-streaming `Model.complete()`** — every model adapter now
  has a single-shot `complete(...)` method alongside `stream(...)`.
  `agent.run` (no consumer reading from `stream()`) prefers
  `complete()` and skips per-chunk yield + Event construction.
  About 100–200 ms saved per turn on token-heavy responses.

### Quality

* **866 offline tests pass** in ~6 s (5 env-gated integrations
  skip without `JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL`).
* **9 live tests pass** against real OpenAI in ~16 s.
* **mypy `--strict`** clean across **105 production source files**.
* **ruff** clean across `loomflow`, `tests`, `examples`,
  including `flake8-async` lints.

### Compatibility

* All new fields on `Episode`, `Fact`, `RunResult`, and
  `Dependencies` are optional with safe defaults. Existing
  pickled / JSON-serialised records load unchanged.
* All new kwargs on `Agent.run` / `stream` / `resume` are
  keyword-only with `None` / sensible defaults. Existing call
  sites keep working unchanged.
* All new memory-protocol methods (`session_messages`) ship with
  default implementations on every shipped backend; custom
  `Memory` implementations gain a graceful `[]` fallback in
  `_build_seed_messages` so old backends keep working too.

---

## [0.3.0] — unreleased

### Added — Architectures

* **`MultiAgentDebate`** — N debater Agents argue across rounds with
  optional judge synthesis. Round 0 is independent (parallel via
  anyio task group); rounds 1..K each debater sees the full prior
  transcript and defends or updates its position. Naive convergence
  check (whitespace-normalized exact match) terminates early when
  all debaters agree; pass `convergence_check=False` to disable.
  When `judge=None`, falls back to majority vote on the final
  round (modal answer wins, original casing preserved). Each
  debater + judge invocation uses deterministic session ids
  (`{parent}__debater_<i>_round_<r>` / `{parent}__judge`) for
  replay correctness. Min 2 debaters; min 1 round. Useful for
  high-stakes contested questions; reserve for cases where a wrong
  answer is expensive (3-5× cost over single-agent).
* **`TreeOfThoughts`** + **`ThoughtNode`** — branching exploration
  with per-node evaluation (Yao et al. 2023). BFS beam search:
  each level generates `branch_factor` candidates per frontier
  node, evaluator scores each, top `beam_width` survive to the
  next level. Best leaf wins. Early-exit on
  `score >= solved_threshold` (default 1.0). Per-call uses
  `text_only_model_call` so every propose / evaluate is journaled.
  Architecture events surface tree state at each step (`tot.proposed`,
  `tot.evaluated`, `tot.pruned`, `tot.solved`, `tot.completed`).
  Full search tree is stashed on `session.metadata["tot_nodes"]`
  for post-hoc rendering. Resolver string `"tree-of-thoughts"`.
* **`ActorCritic`** — generator + adversarial critic. Both the
  actor and critic are required, separate `Agent` instances —
  same-model self-iteration is what `SelfRefine` is for; ActorCritic
  earns its complexity only when there's actual asymmetry (different
  models, different prompts, different blind spots). Round 0 is
  actor; each round is critic → approve check → actor refine. Critic
  output parsed as JSON (with markdown-fence stripping) into
  `CriticOutput(issues, score, summary)`; regex-only fallback when
  JSON parsing fails. Each actor / critic invocation uses a
  deterministic session id (`{parent}__actor_<round>` /
  `{parent}__critic_<round>`) for replay correctness. Constructor:
  `ActorCritic(actor=..., critic=..., max_rounds=3,
  approval_threshold=0.9)`. Composes inside Supervisor (per-worker
  quality control) and inside Reflexion (cross-session learning of
  effective critique patterns).
* **`Supervisor`** — second multi-agent architecture; the
  hierarchical pattern. Workers (dict of `Agent` instances) +
  a base architecture (default `ReAct`). The supervisor's
  ToolHost is wrapped to inject one extra tool —
  `delegate(worker, instructions)` — that routes calls to the
  named worker `Agent` and returns its output as the tool result.
  Multiple `delegate` calls in a single supervisor turn run in
  parallel for free (ReAct's tool dispatch is already a
  task-group). Custom `delegate_tool_name=` to avoid clashes.
  Worker session ids are uniquely generated per delegation, so
  the same worker can be invoked multiple times in one turn
  without journal collisions. Composition: workers can be any
  architecture themselves (DeepAgent for research, Reflexion for
  cross-session learning). The agent's own `instructions` are
  preserved and the supervisor template is appended.
* **`Agent.instructions`** public property — symmetric with
  `model`/`memory`/`runtime`/`architecture`/etc. Surfaced so
  multi-agent architectures (Supervisor, future Actor-Critic) can
  read each worker's role description when composing supervising
  prompts.
* **`Router`** + **`RouterRoute`** — first multi-agent architecture.
  Classify input → dispatch to ONE specialist `Agent`. Each route is
  a fully-constructed `Agent` (its own model / memory / tools /
  architecture). Specialist runs with a deterministic session_id
  (`{parent}__route_{route_name}`) so replay flows through both the
  parent's classifier journal and the specialist's own journal.
  Optional `fallback_route` + `require_confidence_above` for graceful
  handling of ambiguous inputs and unknown routes from the
  classifier. `declared_workers()` exposes routes by name for
  introspection. NOT registered as a resolver string — Router needs
  config; pass an instance:
  `architecture=Router(routes=[RouterRoute(name="billing",
  agent=billing_agent), ...], fallback_route="general")`.
* **`Reflexion`** — verbal reinforcement learning via memory
  (Shinn et al. 2023). Wraps any base architecture (default
  `ReAct`); each attempt, an evaluator scores the output (0-1) and
  if below `threshold` (default 0.8) a reflector produces a
  one-sentence lesson. Lessons are appended via
  `memory.append_block(lessons_block_name, ...)` so the base
  architecture's own `memory.working()` recall picks them up on
  the next attempt — zero plumbing on the base side. With a
  persistent memory backend (Sqlite / Postgres / Redis), lessons
  carry across process restarts (cross-session learning).
  Constructor: `architecture=Reflexion(base=ReAct(),
  threshold=0.8, max_attempts=3, lessons_block_name="...")` or
  `architecture="reflexion"`.
* **`SelfRefine`** — iterative refinement via critique
  (Madaan et al. 2023). Wraps any `base` architecture (default
  `ReAct`); each round, the same model plays critic and refiner.
  Stops on `stop_phrase` (default `"no issues"`) or after
  `max_rounds`. Composable: `architecture=SelfRefine(base=ReAct(...),
  max_rounds=3)` or `architecture="self-refine"`.
* **`EventKind.ARCHITECTURE_EVENT`** + `Event.architecture_event(
  session_id, name, **data)` factory — generic progress event for
  architecture-specific milestones. Each architecture uses a
  namespaced name (`"self_refine.critique"`,
  `"self_refine.refined"`, `"self_refine.converged"`,
  `"self_refine.max_rounds_reached"`) so consumers can pattern-match
  without expanding `EventKind` per architecture.
* **`loomflow.architecture.helpers`** — shared utilities
  architectures reuse: `text_only_model_call(deps, step_name,
  messages) -> (text, usage)` (one-shot text-only model call,
  journaled for replay) and `add_usage(a, b)` (sum two `Usage`
  records).
* **12 new SelfRefine tests** covering protocol satisfaction,
  stop-phrase early exit, full critique → refine cycles,
  `max_rounds` enforcement, budget gating, progress events, and
  `architecture="self-refine"` resolver string.
* **19 new Reflexion tests** covering protocol, constructor
  validation, score parsing (`"score: X"` patterns +
  fallbacks + clamping), threshold-met early exit, full
  evaluate → reflect → retry cycles, `max_attempts` enforcement,
  lesson persistence into the memory block, lesson visibility on
  the next attempt's seed_context, end-to-end via `ReAct` base,
  and resolver-string construction.
* **21 new Router tests** covering protocol, constructor validation
  (empty / duplicate / invalid fallback / confidence range),
  classification regex (`route:` + `confidence:` + `=`-separator +
  defaults + clamping), successful dispatch, fallback paths
  (low confidence, unknown route), specialist interruption
  propagation, deterministic specialist session_id, architecture
  events surfacing through `Agent.stream`.
* **13 new Supervisor tests** covering protocol, constructor
  validation, single delegation, parallel delegations (two
  delegate calls in one turn), unknown-worker error handling,
  instructions composition (user prompt + template + worker
  descriptions), unique worker session ids per delegation, custom
  delegate tool name, and the helper `_make_delegate_tool`
  building a valid `Tool`.
* **19 new ActorCritic tests** covering protocol, constructor
  validation, critique-parsing (pure JSON / markdown-fenced JSON /
  regex fallback / empty-string default), single-round approval
  (no refine call when critic approves on round 1), full
  refine-then-approve cycles, max_rounds enforcement, full event
  sequence emission, deterministic actor/critic session ids per
  round, and interruption propagation from sub-agents.
* **16 new TreeOfThoughts tests** covering protocol, constructor
  validation (branch_factor / max_depth / beam_width /
  solved_threshold ranges), single-level beam pruning, multi-level
  expansion with deterministic top-by-score selection, early
  termination on solved_threshold, max_depth enforcement, helper
  `_chain_to_root`, full event sequence emission, and beam pruning
  to top-N per level.
* **20 new MultiAgentDebate tests** covering protocol, constructor
  validation (≥2 debaters, ≥1 rounds), helpers (`_normalize`,
  `_converged`, `_majority_vote` including casing preservation),
  parallel round 0, multi-round with history visibility, convergence
  early-exit, judge synthesis path, majority-vote fallback,
  deterministic debater + judge session ids, and full event
  sequence emission.
* **7 new examples** — `examples/09_self_refine.py`,
  `10_reflexion.py`, `11_router.py`, `12_supervisor.py`,
  `13_actor_critic.py`, `14_tree_of_thoughts.py`, `15_debate.py`.
  Each runs deterministically with `ScriptedModel` (no API key) and
  prints a streaming event view plus the final answer.
* **`parse_score(text) -> float`** promoted to
  `loomflow.architecture.helpers` (was private in `reflexion.py`).
  Used by Reflexion, Tree of Thoughts, and any future architecture
  with an evaluator step.
* **32 new ReWOO tests** covering protocol, resolver string,
  constructor validation, placeholder helpers (`_extract_placeholders` /
  `_substitute_placeholders` recursion + dedupe + non-string
  passthrough + unresolved-placeholder leave-as-is), topological
  helpers (`_topological_levels` linear chain / collapsed
  independent-steps level / cycle detection / unknown-dep
  treatment), plan parser (clean JSON / markdown fences /
  malformed-step skipping / auto-id assignment), full
  end-to-end loop with real tools, parallel level execution,
  step-error path that doesn't crash the architecture,
  `max_steps` cap, cyclic plan handling, full event sequence,
  and Pydantic round-trip.
* Total tests: **560** (was 341 in v0.2.0; +219 across the v0.3
  architecture work — 15 foundation + 12 SelfRefine + 19 Reflexion +
  21 Router + 13 Supervisor + 19 ActorCritic + 16 TreeOfThoughts +
  20 MultiAgentDebate + 17 Swarm + 18 BlackboardArchitecture +
  17 PlanAndExecute + 32 ReWOO).

### Added — Architecture protocol foundation

* **`loomflow.architecture`** package — pluggable agent-loop
  strategies. The `Architecture` protocol lets users swap iteration
  patterns (ReAct, Plan-and-Execute, Reflexion, Router, Supervisor,
  ...) without touching memory / runtime / tools / governance. See
  `Subagent.md` in the repo root for the design rationale and
  catalogue of architectures.
* **`Architecture` protocol** (`runtime_checkable`) — every
  architecture implements `name: str`, `async def run(session, deps,
  prompt) -> AsyncIterator[Event]`, and `declared_workers() ->
  dict[str, Agent]`. Architectures are async generators that yield
  `Event` values for milestones; setup / teardown stays in `Agent`.
* **`AgentSession`** — mutable per-run state (id, instructions,
  messages, turns, output, cumulative_usage, interrupted,
  interruption_reason, metadata). Architectures mutate this as they
  iterate; `Agent` reads the final state to build a `RunResult`.
* **`Dependencies`** — bundles every protocol implementation an
  architecture might need (model, memory, runtime, tools, budget,
  permissions, hooks, telemetry, audit_log, max_turns) into one
  struct so `run()` signatures stay short.
* **`ReAct`** — the canonical observe/think/act loop, lifted out of
  `Agent._loop` verbatim. Now the framework's default architecture.
  Constructor takes optional `max_turns` override (useful when
  composing inside other architectures, e.g. `Reflexion(base=ReAct(max_turns=10))`).
* **`Agent(architecture=...)`** kwarg — accepts an `Architecture`
  instance, a known string (`"react"`), or `None` (defaults to
  `ReAct()`). Public `agent.architecture` property exposes it.
* **`resolve_architecture(spec)`** — string / instance / None →
  concrete `Architecture`. Unknown strings raise `ConfigError` with
  the list of known names.
* **15 new tests** (`tests/test_architecture.py`) covering the
  Protocol surface, the resolver, Agent integration, and a custom
  architecture driving an end-to-end run.
* **`Subagent.md`** committed as the architecture reference manual
  for upcoming v0.4+ work (Reflexion, Self-Refine, Plan-and-Execute,
  Router, Supervisor, ...).

### Changed (non-breaking)

* `Agent._loop` is now ~100 lines (was ~330): setup wraps the
  runtime/telemetry context and audits the run boundary, then
  delegates iteration to `self._architecture.run(session, deps,
  prompt)`, then teardown persists the episode and builds the
  `RunResult`. Helpers (`_seed_context`, `_take_one_turn`,
  `_dispatch_tools`, `_run_single_tool`) moved into
  `loomflow/architecture/react.py`.
* The `jeeves.run` telemetry span carries an `architecture` attribute
  alongside `model` / `max_turns` / `session_id`. The audit
  `run_started` payload also includes the architecture name.
* All 341 v0.2.0 tests still pass without modification — the refactor
  is behaviour-preserving.

---

## [0.2.0] — 2026-05-06

### Changed (breaking)

* **`Agent(...)` requires `model=`**. The previous behaviour silently
  defaulted to `EchoModel`, which produced `"Echo: ..."` output that
  users misread as a real LLM response. Forgetting the kwarg now
  raises `ConfigError` with a suggestion list. Migration: pass
  `model="echo"` for tests / zero-key dev, or one of the real
  strings (`"claude-opus-4-7"`, `"gpt-4o"`, `"mistral-large"`, ...).
* **Resolver errors harmonised to `ConfigError`**. `_resolve_model`
  used to raise `ValueError` for unknown specs; now it's
  `ConfigError` with a message that lists every supported prefix and
  the explicit `litellm/` opt-in.

### Added

* **`LiteLLMModel`** — single adapter for ~100 providers via the
  `litellm` SDK (Cohere, Mistral, Bedrock, Vertex, Together, Ollama,
  Gemini, Groq, Replicate, Azure, …). Inherits from `OpenAIModel`
  since LiteLLM normalises every provider's chunks to OpenAI's
  shape — zero new chunk-aggregation code, just a different
  underlying client.
* **String resolver dispatches more prefixes** to `LiteLLMModel`:
  `mistral-`, `command-`, `bedrock/`, `vertex_ai/`,
  `together_ai/`, `ollama/`, `gemini/`, `groq/`, `replicate/`,
  `azure/`. Plus `litellm/<spec>` as an explicit opt-in that strips
  the prefix and forces the LiteLLM path even for specs the direct
  adapters would otherwise grab.
* **`VoyageEmbedder`** — embeddings via Voyage AI's `voyageai` SDK.
  Models: `voyage-3` / `voyage-3-large` / `voyage-code-3` (1024
  dim), `voyage-3-lite` (512 dim). Configurable `input_type`
  (``"document"`` / ``"query"``).
* **`CohereEmbedder`** — embeddings via Cohere's `cohere` SDK.
  Models: `embed-english-v3.0` / `embed-multilingual-v3.0` (1024),
  `embed-english-light-v3.0` / `embed-multilingual-light-v3.0`
  (384). Required `input_type` (``"search_document"`` /
  ``"search_query"``) plus `embedding_types=["float"]` baked in.
* **`Agent.__repr__()`** — concise dev-time inspection:
  ``Agent(model='claude-opus-4-7', memory=InMemoryMemory,
  runtime=InProcRuntime, tools=InProcessToolHost, max_turns=50)``.
* **`RunResult.total_tokens`** (`tokens_in + tokens_out`) and
  **`RunResult.duration`** (`finished_at - started_at`) convenience
  properties.
* **`Agent.consolidate()` returns the count of new facts** extracted.
  ``0`` when no consolidator is configured or the memory backend
  doesn't expose a `.facts` store. Useful for batch consolidation
  loops that want to know whether anything changed.
* **`tools=` accepts a single callable or `Tool`**. Previously you
  had to wrap one tool in a list (`tools=[my_fn]`); now
  `tools=my_fn` and `tools=Tool(...)` both work. List form is
  unchanged.
* **30 new tests** for embedders + polish + tool ergonomics,
  bringing total tests to 279 from 244.
* New `voyage` and `cohere` extras in `pyproject.toml`
  (`pip install 'loomflow[voyage,cohere]'`).
* **`ConsolidationWorker`** — long-running anyio task that calls
  `memory.consolidate()` every N seconds. For long-lived agents
  where per-run `auto_consolidate=True` is wasteful. Surfaces new
  fact counts via `on_consolidated(count)` and consolidator failures
  via `on_error(exc)` so a transient LLM hiccup doesn't kill the
  worker. Doubles as an async-context-manager
  (`async with worker: ...` runs in the background, exiting cancels
  cleanly).
* **`agent.add_tool(fn)`** — register tools after construction.
  Friendly for plugin patterns. Works with the default
  `InProcessToolHost`; raises `ConfigError` with a clear message for
  custom hosts that don't support post-hoc registration.
* **`examples/07_litellm.py`** — runnable LiteLLM dispatch demo,
  picks a model based on which provider key is in the environment.
* **CI install line** — adds `litellm`, `voyage`, `cohere` extras so
  mypy resolves the SDK type hints in their lazy-import paths.
* **Plugin API**: `agent.remove_tool(name)` and
  `agent.tools_list()` round out post-construction tool management.
* **Public introspection** properties on `Agent`: `model`, `memory`,
  `runtime`, `tool_host`, `budget`, `permissions`. Use these instead
  of the `_model` / `_memory` / etc. private attributes.
* **`agent.recall(query, kind=, limit=)`** — convenience wrapper
  around `agent.memory.recall(...)`.
* **Migration guide** at `docs/migration_0.1_to_0.2.md` covering
  both breaking changes and the stale-install-shadowing pitfall.
* **`SubprocessSandbox`** — runs each tool call in a fresh child
  Python process via `multiprocessing` (spawn). Process isolation,
  hard timeout, memory boundary. Wraps any `InProcessToolHost`;
  rejects other host types with a clear error. Picklable
  module-level functions are required.
* **`Memory.recall_facts(query, *, limit, valid_at)` protocol
  method** — formalises the previously-duck-typed `.facts` access
  path. Every memory backend implements it: backends with a fact
  store forward to `self.facts.recall_text`; backends without one
  return `[]`. The agent loop calls `memory.recall_facts(...)`
  directly now.
* **Batch embedding in `InMemoryFactStore.append_many`** — coalesces
  the per-fact `embed()` calls into a single `embed_batch()`
  round-trip when an embedder is configured. The `Consolidator`
  uses `append_many` internally so multi-fact extraction from one
  episode hits the embedder API once instead of N times. Falls back
  to per-fact `append` for stores without `append_many`.
* **`Agent.from_config(toml_path)`** — load an `Agent` from a TOML
  file. Supports declarative `instructions`, `model`, `max_turns`,
  `auto_consolidate`, and a `[budget]` block. Concrete instances
  for `memory` / `runtime` / `tools` / model overrides can be passed
  as kwargs.
* **`Agent.from_dict(cfg)`** — same shape as `from_config` but skips
  the file read. Useful when config comes from env vars, Pydantic
  settings, YAML, an HTTP API, etc. `from_config` now delegates to
  `from_dict` for the parsing logic.
* **`@agent.with_tool` decorator** — register a tool inline:
  ```python
  @agent.with_tool
  async def search(q: str) -> str: ...
  ```
  Returns the function unchanged so it stays directly callable;
  registers it on the underlying `InProcessToolHost`.
* **`PostgresJournalStore` + `PostgresRuntime`** — Phase 5
  production durable runtime. Same `JournaledRuntime` architecture
  as `SqliteRuntime`, but the journal lives in two Postgres tables
  (`journal_steps`, `journal_streams`). Lazy `asyncpg` import.
  Idempotent `init_schema()`. **Note**: this isn't a DBOS-specific
  adapter — DBOS Python's workflow model requires
  `@DBOS.workflow()` / `@DBOS.communicator()` decoration at
  module-load time, which doesn't compose with our generic
  `runtime.step(name, fn, *args)` API. `PostgresJournalStore` gives
  the same durability guarantee with no decorator intrusion; users
  who want the full DBOS workflow surface can layer DBOS on top of
  their own tool functions.
* **`examples/08_from_config.py`** + companion `examples/agent.toml`
  — runnable demo of `from_config` / `from_dict` / `@agent.with_tool`.

---

## [0.1.0] — 2026-05-06

First public release.

### Added

* **Phase 1 — Protocols + types.** 18 Pydantic value objects, 14
  module-boundary `Protocol` definitions, exception hierarchy, ULID
  helpers.
* **Phase 2 — Agent loop.** `Agent` class with `run()` / `stream()` /
  `resume()`. Parallel tool dispatch via `anyio.create_task_group`.
  Streaming events through bounded `anyio.create_memory_object_stream`
  with backpressure + clean cancellation.
* **Phase 3 — MCP spine.** `MCPClient` (lazy `mcp` SDK), `MCPRegistry`
  (auto name-disambiguation across servers), `JeevesGateway`
  one-line wrapper.
* **Phase 4 — Memory + bi-temporal facts.** Five backends:
  `InMemoryMemory`, `VectorMemory`, `ChromaMemory`, `PostgresMemory`,
  `RedisMemory`. Two embedders: `HashEmbedder` (zero-deps,
  deterministic), `OpenAIEmbedder`. **Bi-temporal `FactStore` in
  every backend** with supersession, `valid_at` historical queries,
  embedder-driven cosine recall. LLM-driven `Consolidator` for fact
  extraction with `auto_consolidate=True` opt-in.
* **Phase 5 — Durable runtime.** `JournaledRuntime` + `SqliteRuntime`
  for crash-recovery replay across process restarts. Session-id
  tracking via `contextvars`.
* **Phase 6 — Security + governance + observability.**
  `StandardPermissions` (mode + allow/deny), `HookRegistry`
  (timeout-shielded), `StandardBudget` (token/cost/wall-clock with
  soft warnings), `OTelTelemetry` (spans + metrics for every
  milestone), `FilesystemSandbox` (path-arg validation, symlink
  resolution), `InMemoryAuditLog` + `FileAuditLog` (HMAC-signed),
  `FreshnessPolicy` + `LineagePolicy` validators.
* **Provider adapters.** `AnthropicModel` (`claude-opus-4-7`, etc.),
  `OpenAIModel` (`gpt-4o`, etc.). String-based resolver:
  `Agent(model="claude-opus-4-7")` dispatches by prefix.
* **Resume API.** `agent.resume(session_id, prompt)` + optional
  `session_id` kwarg on `run()` / `stream()`. Pairs with
  `SqliteRuntime` for cross-process replay.

### Documentation

* `README.md` — value prop, capability matrix, install matrix,
  30-second quickstart.
* `docs/quickstart.md` — 14 copy-pasteable examples.
* `docs/recipes.md` — 8 production patterns + 24-item production
  checklist.
* `docs/architecture.md` — module map, lifecycle walkthrough,
  extension-points table.
* `examples/` — 7 runnable scripts mirroring the recipes.

### Engineering

* **Async-only** — anyio everywhere; zero raw `asyncio.create_task`
  / `gather` calls.
* **mypy `--strict` clean** across 53 production source files.
* **243 tests passing** + 4 env-gated live-integration skips.
* **CI** on Python 3.11 + 3.12 (ruff + mypy + pytest + examples
  smoke).
* **Release** workflow via PyPI trusted publishing on `v*` tags.
