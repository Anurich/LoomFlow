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
| [`07_litellm.py`](07_litellm.py) | LiteLLM dispatch — ~100 providers (Mistral / Cohere / Bedrock / Vertex / Ollama / Groq / Gemini / ...) through one adapter | optional `MISTRAL_API_KEY` / `COHERE_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` |
| [`08_from_config.py`](08_from_config.py) | Declarative `Agent.from_config(toml_path)` + `Agent.from_dict(cfg)` + `@agent.with_tool` decorator | nothing |
| [`09_self_refine.py`](09_self_refine.py) | `SelfRefine` architecture — generator → critic → refiner cycles with stop-phrase convergence | nothing |
| [`10_reflexion.py`](10_reflexion.py) | `Reflexion` architecture — verbal RL: lessons from failed attempts persist via `memory.working()` and shape the next attempt | nothing |
| [`11_router.py`](11_router.py) | `Router` architecture — classify input, dispatch to one specialist `Agent` (with confidence threshold + fallback) | nothing |
| [`12_supervisor.py`](12_supervisor.py) | `Supervisor` architecture — workers + a `delegate(...)` tool, parallel delegations in one turn | nothing |
| [`13_actor_critic.py`](13_actor_critic.py) | `ActorCritic` architecture — actor + adversarial critic with structured JSON critique parsing | nothing |
| [`14_tree_of_thoughts.py`](14_tree_of_thoughts.py) | `TreeOfThoughts` architecture — BFS beam search with per-node evaluation; observable search tree | nothing |
| [`15_debate.py`](15_debate.py) | `MultiAgentDebate` architecture — N debaters argue across rounds, judge synthesizes; parallel debater dispatch per round | nothing |
| [`16_swarm.py`](16_swarm.py) | `Swarm` architecture — peer agents pass control via a `handoff` tool; cycle detection + max_handoffs | nothing |
| [`17_blackboard.py`](17_blackboard.py) | `BlackboardArchitecture` — coordinator + agents share a state board; LLM-driven agent selection per round | nothing |
| [`18_plan_and_execute.py`](18_plan_and_execute.py) | `PlanAndExecute` architecture — planner → step executor → synthesizer; cheaper than ReAct on predictable multi-step tasks | nothing |
| [`19_rewoo.py`](19_rewoo.py) | `ReWOO` architecture — plan-then-tool-execute with `{{En}}` placeholder substitution; parallel independent steps; 2 LLM calls + N tool calls | nothing |
| [`15_debate.py`](15_debate.py) | `MultiAgentDebate` architecture — N debaters argue in parallel rounds, judge synthesizes | nothing |

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
