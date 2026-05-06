# Examples

Each script is self-contained — `python examples/<name>.py` runs it.
Most fall back to `EchoModel` / `ScriptedModel` when no API key is
set, so the entire folder is exercise-able in a fresh checkout with
just `pip install -e '.[dev]'`.

| File | What it shows | Needs |
|---|---|---|
| [`00_hello.py`](00_hello.py) | Smallest possible agent (zero-key, zero-infra) | nothing |
| [`01_real_model.py`](01_real_model.py) | String-based model resolver, real LLM call with graceful fallback | optional `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| [`02_tools_parallel.py`](02_tools_parallel.py) | `@tool` decorator + parallel dispatch in one turn | nothing |
| [`03_streaming.py`](03_streaming.py) | `agent.stream()` event flow | nothing |
| [`04_facts.py`](04_facts.py) | Bi-temporal facts: supersession + `valid_at` queries + auto-consolidation | nothing |
| [`05_durable.py`](05_durable.py) | `SqliteRuntime` cross-instance replay | nothing |
| [`06_production.py`](06_production.py) | Full production shape: every cross-cutting concern wired up | optional API key |

Read in order; each builds on the last conceptually.

## Running

```bash
pip install -e '.[dev]'

# To make 01 and 06 hit a real model:
export ANTHROPIC_API_KEY=sk-ant-...     # or
export OPENAI_API_KEY=sk-...

python examples/00_hello.py
python examples/01_real_model.py
python examples/02_tools_parallel.py
python examples/03_streaming.py
python examples/04_facts.py
python examples/05_durable.py
python examples/06_production.py
```

## See also

* [`docs/quickstart.md`](../docs/quickstart.md) — narrative walkthrough
  of every public API surface.
* [`docs/recipes.md`](../docs/recipes.md) — production patterns these
  examples are condensed from.
* [`docs/architecture.md`](../docs/architecture.md) — module map and
  lifecycle deep dive.
