# Migrating from 0.1 → 0.2

JeevesAgent 0.2.0 is mostly additive but ships **two breaking
changes**. This guide shows what to change and why.

## TL;DR

```diff
- agent = Agent("You are helpful.")
+ agent = Agent("You are helpful.", model="claude-opus-4-7")
```

```diff
- with pytest.raises(ValueError):
-     Agent("hi", model="bad-spec")
+ from jeevesagent.core.errors import ConfigError
+ with pytest.raises(ConfigError):
+     Agent("hi", model="bad-spec")
```

That's it for breaking changes. Everything else in 0.2.0 is new
features and polish.

---

## Breaking change 1: `model=` is required

### What changed

`Agent(...)` used to silently default to `EchoModel` when `model=`
was omitted. Users got `"Echo: ..."` output and assumed they were
talking to an LLM. v0.2.0 fails fast with a helpful error:

```
ConfigError: Agent() requires a `model` argument. Pass one of:
  model='claude-opus-4-7'   (Anthropic, needs ANTHROPIC_API_KEY)
  model='gpt-4o'            (OpenAI, needs OPENAI_API_KEY)
  model='echo'              (zero-key fake — text echoes the prompt; for dev/tests)
  model='mistral-large'     (LiteLLM, also: command-, bedrock/, vertex_ai/, ollama/, ...)
  model=AnthropicModel(...) or any Model-protocol instance for full control.
```

### Migration

Pick one of:

| If you want… | Pass |
|---|---|
| To talk to Claude (most readers' real use case) | `model="claude-opus-4-7"` |
| To talk to GPT | `model="gpt-4o"` |
| To talk to Mistral / Cohere / Bedrock / etc. | `model="mistral-large"` (etc.) |
| The zero-key fake for tests / dev | `model="echo"` |
| Full control over construction | `model=AnthropicModel("claude-opus-4-7", ...)` |

### Why we did it

Default behaviour that produces *plausible-looking but wrong*
output is the worst kind of footgun. If users want the echo behaviour
they have to opt in explicitly — easier to debug, easier to read, no
silent fallback.

---

## Breaking change 2: Resolver errors → `ConfigError`

### What changed

`_resolve_model("totally-unknown-spec")` used to raise `ValueError`.
Now it raises `ConfigError` (from `jeevesagent.core.errors`),
matching the rest of the configuration-error vocabulary.

### Migration

```diff
- with pytest.raises(ValueError, match="unknown model spec"):
-     Agent("hi", model="bad-spec")
+ from jeevesagent.core.errors import ConfigError
+ with pytest.raises(ConfigError, match="unknown model spec"):
+     Agent("hi", model="bad-spec")
```

If you were catching the resolver error in production code, change
the catch clause:

```diff
  try:
      agent = Agent(prompt, model=user_input_spec)
- except ValueError:
+ except ConfigError:
      ...
```

### Why we did it

`ConfigError` already existed for "user supplied invalid
configuration." Resolver errors fit that exactly; harmonising means
catching `ConfigError` covers all configuration-level failures.

---

## What's new in 0.2.0 (non-breaking)

### Provider coverage

* **`LiteLLMModel`** — single adapter for ~100 providers via the
  LiteLLM SDK. The string resolver dispatches the common prefixes:

  ```python
  Agent("...", model="mistral-large")        # → LiteLLMModel
  Agent("...", model="command-r-plus")       # → LiteLLMModel
  Agent("...", model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
  Agent("...", model="vertex_ai/gemini-pro")
  Agent("...", model="ollama/llama3")        # local Ollama
  Agent("...", model="groq/llama-3.1-70b")
  Agent("...", model="litellm/claude-3-haiku")  # explicit opt-in
  ```

* New embedders: `VoyageEmbedder`, `CohereEmbedder`. Same shape as
  `OpenAIEmbedder`.

### Polish

* `Agent.__repr__()` for dev-time inspection.
* `RunResult.total_tokens` and `RunResult.duration` properties.
* `agent.consolidate() -> int` returns the count of new facts
  extracted (was `None`).
* `tools=my_fn` — pass a single callable or `Tool` directly without
  list-wrapping. Lists still work.

### New plugin API

* `agent.add_tool(fn)` — register a tool after construction.
* `agent.remove_tool(name)` — unregister by name.
* `agent.tools_list()` — list registered tool names.
* Public introspection properties: `agent.model`, `agent.memory`,
  `agent.runtime`, `agent.tool_host`, `agent.budget`,
  `agent.permissions`. Replaces `_model` / `_memory` / etc. as the
  supported access path.
* `agent.recall(query, kind=, limit=)` — convenience wrapper around
  `agent.memory.recall(...)`.

### Background work

* `ConsolidationWorker(memory, interval_seconds=60)` — long-running
  anyio task that periodically calls `memory.consolidate()`.
  Surfaces new fact counts via `on_consolidated(count)` callback;
  consolidator failures via `on_error(exc)` so a transient hiccup
  doesn't kill the worker. Doubles as an async context manager:

  ```python
  async with ConsolidationWorker(memory, interval_seconds=60):
      await main()  # worker runs in background
  # Worker is cancelled when the block exits.
  ```

### Examples

* **`examples/07_litellm.py`** — runnable LiteLLM dispatch demo.

---

## Common pitfall: stale install shadowing the editable

If you have a previous `pip install jeevesagent` in your env, an
editable install (`pip install -e .`) of 0.2.0 may not take
precedence because pip's regular install sometimes wins on the
import path:

```
$ python -c "import jeevesagent; print(jeevesagent.__version__)"
0.1.0   # ← stale; expected 0.2.0
```

Fix:

```bash
pip uninstall jeevesagent -y
pip install -e '.[dev,...]'
```

Verify with:

```bash
python -c "import jeevesagent; print(jeevesagent.__version__, jeevesagent.__file__)"
# Should print: 0.2.0 /your/checkout/path/jeevesagent/__init__.py
```

This is a Python packaging quirk, not a JeevesAgent bug — but worth
flagging since it produces confusing `AttributeError: ... has no
attribute 'total_tokens'` failures when 0.1's `RunResult` is
imported instead of 0.2's.
