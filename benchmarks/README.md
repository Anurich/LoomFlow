# Benchmarks

Apples-to-apples comparisons of loomflow against other agent frameworks
on identical tool-using tasks, same model, same tools.

## Running

```bash
pip install -e ".[litellm]" langgraph langchain-openai pydantic-ai python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env      # in the repo root
python benchmarks/framework_shootout.py   # simple: 2 typed tools
python benchmarks/shootout_complex.py     # complex: 6 chained typed tools
```

Every framework gets the **same model** (`gpt-4.1-mini`), the **same
tools** (with identical type annotations), and the **same task**. We
measure total tokens (input+output), USD cost, model round-trips, and
correctness (the answer must match the known ground truth).

## What's measured, and why it's fair

- **Typed tools on every framework.** Each tool declares the same
  parameter types (`qty: int`, `rate_pct: float`). A framework that
  doesn't coerce a model-supplied `"8"` to `8` crashes the tool and
  burns turns retrying — so this is a real test of the tool-call path,
  not just the prompt.
- **The tools do the arithmetic.** The model only *orchestrates* the
  tool calls; it never computes. This isolates framework efficiency
  from the model's raw math ability (a small model can't reliably do
  multi-step arithmetic in its head, which would otherwise swamp the
  signal).
- **Correctness gates cost.** A cheap wrong answer loses. Only runs
  that produce the exact ground-truth number count.

## Results (gpt-4.1-mini)

Representative single runs; absolute numbers vary slightly with model
non-determinism. Re-run to reproduce.

### Simple task — 2 typed tools (`get_price` → `add_tax`)

| framework   | correct | tokens | calls | cost (USD) |
|-------------|:------:|-------:|------:|-----------:|
| **loomflow**    | ✅ | **472** | 3 | **0.000236** |
| pydantic-ai | ✅ | 486 | 3 | 0.000244 |
| raw-openai  | ✅ | 570 | 3 | 0.000314 |
| langgraph   | ✅ | 585 | 3 | 0.000324 |

### Complex task — 6 chained typed tools (line totals → add → discount → tax)

| framework   | correct | tokens | calls | cost (USD) | secs |
|-------------|:------:|-------:|------:|-----------:|-----:|
| raw-openai  | ✅ | 2217 | 6 | 0.001049 | 6.5 |
| **loomflow**    | ✅ | **2409** | 6 | **0.001126** | **5.1** |
| langgraph   | ✅ | 2409 | 6 | 0.001133 | 5.9 |
| pydantic-ai | ✅ | 2465 | 6 | 0.001153 | 6.7 |

## Reading the results

- On the **simple** task loomflow is the **most token-efficient** of the
  field, and on the **complex** task it's tied with LangGraph and ahead
  of Pydantic-AI on tokens, while being the **fastest** wall-clock.
- **raw-openai** (a hand-rolled loop with no framework) is marginally
  cheaper on the complex task — expected, since it carries zero
  framework overhead (no memory recall scaffolding, audit, permissions,
  or multi-tenant plumbing). loomflow's ~9% gap there is the price of
  those production features, and it stays *below* the other full
  frameworks.

These are micro-benchmarks on small tasks — directional, not a
leaderboard. They exist to show loomflow's per-task overhead is
competitive, and to guard against regressions in the tool-call path.
