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
| [`09_self_refine.py`](09_self_refine.py) | `SelfRefine` — iterative copywriting (tweet polish via critique + refine) | `OPENAI_API_KEY` |
| [`10_reflexion.py`](10_reflexion.py) | `Reflexion` — math word problem solver with verbal RL (evaluator + reflector) | `OPENAI_API_KEY` |
| [`11_router.py`](11_router.py) | `Router` — customer-support intent classification with billing/tech/general specialists, each with their own tools | `OPENAI_API_KEY` |
| [`12_supervisor.py`](12_supervisor.py) | `Supervisor` — software-dev team (researcher + coder + reviewer) building a Python function | `OPENAI_API_KEY` |
| [`13_actor_critic.py`](13_actor_critic.py) | `ActorCritic` — code generation with adversarial review (different prompts on the same model) | `OPENAI_API_KEY` |
| [`14_tree_of_thoughts.py`](14_tree_of_thoughts.py) | `TreeOfThoughts` — BFS beam search to solve the Game of 24 puzzle | `OPENAI_API_KEY` |
| [`15_debate.py`](15_debate.py) | `MultiAgentDebate` — investment-decision debate (optimist + skeptic + analyst → judge) | `OPENAI_API_KEY` |
| [`16_swarm.py`](16_swarm.py) | `Swarm` — customer-support peers (triage → billing/tech) with `handoff` and cycle detection | `OPENAI_API_KEY` |
| [`17_blackboard.py`](17_blackboard.py) | `BlackboardArchitecture` — root-cause analysis (hypothesis + evidence + critic agents, coordinator-led) | `OPENAI_API_KEY` |
| [`18_plan_and_execute.py`](18_plan_and_execute.py) | `PlanAndExecute` — Tokyo trip-itinerary planner (planner + step executor + synthesizer) | `OPENAI_API_KEY` |
| [`19_rewoo.py`](19_rewoo.py) | `ReWOO` — country fact-sheet builder (parallel tool execution; 2 LLM calls total) | `OPENAI_API_KEY` |
| [`20_rag_supervisor.py`](20_rag_supervisor.py) | `Supervisor` + RAG — three-worker pipeline (Researcher → Curator → Synthesizer) over a small fake corpus, with the Curator catching DRAFT-vs-FINAL hallucinations | `OPENAI_API_KEY` |
| [`21_research_pipeline.py`](21_research_pipeline.py) | **Showcase**: Plan + parallel research + real `.md` file I/O + review + update cycle. Three-worker pipeline over a semantic-indexed corpus. Writes a real markdown report to disk, reads it back, has it reviewed, applies fixes via `update_section`. Demonstrates the framework's full depth. | `OPENAI_API_KEY` |
| [`22_rag_with_loader.py`](22_rag_with_loader.py) | **Full RAG showcase**: Loader (`load(path)`) → MarkdownChunker (preserves header trail) → HashEmbedder cosine index → Supervisor with researcher / writer / reviewer using the framework's built-in `read_tool` / `write_tool` / `edit_tool`. End-to-end production-shape RAG pipeline using every layer of the framework. | `OPENAI_API_KEY` |
| [`15_debate.py`](15_debate.py) | `MultiAgentDebate` architecture — N debaters argue in parallel rounds, judge synthesizes | nothing |

Read in order; each builds on the last conceptually.

## Running

```bash
pip install -e '.[dev,openai]'

# Examples 00-08 run zero-key (Echo / Scripted models).
# Examples 09-20 require OPENAI_API_KEY for real LLM calls.
# Recommended: put the key in a .env file at the repo root,
# all the architecture examples auto-load it via python-dotenv:
echo "OPENAI_API_KEY=sk-..." > .env

python examples/00_hello.py
python examples/09_self_refine.py
python examples/19_rewoo.py
python examples/20_rag_supervisor.py
```

### How the architecture examples handle keys

Each of `09`-`20` does this at the top:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY required. Add to .env at repo root.")
```

So as long as `.env` lives at the repo root (next to `pyproject.toml`),
they pick it up automatically.

## See also

* [`docs/quickstart.md`](../docs/quickstart.md) — narrative walkthrough
  of every public API surface.
* [`docs/recipes.md`](../docs/recipes.md) — production patterns these
  examples are condensed from.
* [`docs/architecture.md`](../docs/architecture.md) — module map and
  lifecycle deep dive.
