"""Apples-to-apples framework shootout — same task, same tools, same model.

Task: "What is the total price of 3 widgets and 2 gadgets, with 8% tax
added? Reply with just the final number." (widget=$10, gadget=$25 →
3*10 + 2*25 = 80; +8% = 86.40)

Tools given to every framework:
  - get_price(item) -> float
  - add_tax(amount, rate_pct) -> float

Measures per framework: total tokens (in+out), USD cost, #model-calls,
and whether the final answer contains 86.4 (correctness).

Frameworks: raw OpenAI SDK (baseline), loomflow, loom_code, LangGraph,
Pydantic-AI. Same gpt-4.1-mini for all.
"""
from dotenv import load_dotenv

load_dotenv()  # reads .env from cwd; set OPENAI_API_KEY

import asyncio  # noqa: E402
import time  # noqa: E402

MODEL = "gpt-4.1-mini"
QUESTION = (
    "Look up the price of a widget with get_price, then add 8% tax to "
    "that price using add_tax (rate_pct=8). Reply with ONLY the final "
    "number that add_tax returns."
)
EXPECT = "54"

# OpenAI gpt-4.1-mini pricing (per 1M tokens), for a uniform cost calc.
PRICE_IN = 0.40 / 1_000_000
PRICE_OUT = 1.60 / 1_000_000

PRICES = {"widget": 50.0, "gadget": 25.0}


def get_price(item: str) -> float:
    return PRICES.get(item.lower().rstrip("s"), 0.0)


def add_tax(amount: float, rate_pct: float) -> float:
    return round(amount * (1 + rate_pct / 100), 2)


def correct(ans: str) -> bool:
    return EXPECT in (ans or "").replace(",", "")


def cost(tin: int, tout: int) -> float:
    return tin * PRICE_IN + tout * PRICE_OUT


RESULTS = []


def record(name, tin, tout, calls, ans, secs):
    RESULTS.append(dict(
        name=name, tin=tin, tout=tout, total=tin + tout,
        cost=cost(tin, tout), calls=calls, ok=correct(ans),
        secs=secs, ans=(ans or "").strip()[:30],
    ))


# --------------------------------------------------------------------------
# 1. RAW OPENAI SDK (hand-rolled tool loop) — the baseline
# --------------------------------------------------------------------------
async def run_openai():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    tools = [
        {"type": "function", "function": {"name": "get_price",
            "description": "Price of an item in USD.",
            "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
        {"type": "function", "function": {"name": "add_tax",
            "description": "Add tax percent to an amount.",
            "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "rate_pct": {"type": "number"}}, "required": ["amount", "rate_pct"]}}},
    ]
    msgs = [{"role": "user", "content": QUESTION}]
    tin = tout = calls = 0
    t0 = time.time()
    for _ in range(8):
        r = await client.chat.completions.create(model=MODEL, messages=msgs, tools=tools)
        calls += 1
        tin += r.usage.prompt_tokens
        tout += r.usage.completion_tokens
        m = r.choices[0].message
        if not m.tool_calls:
            record("raw-openai", tin, tout, calls, m.content, time.time() - t0)
            return
        msgs.append(m.model_dump(exclude_none=True))
        import json
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            fn = {"get_price": get_price, "add_tax": add_tax}[tc.function.name]
            out = fn(**args)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)})
    record("raw-openai", tin, tout, calls, "(no final)", time.time() - t0)


# --------------------------------------------------------------------------
# 2. LOOMFLOW
# --------------------------------------------------------------------------
async def run_loomflow():
    from loomflow import Agent
    from loomflow.tools import tool

    # Typed tools (annotations present), same as the langgraph /
    # pydantic-ai tools — fair apples-to-apples.
    @tool(name="get_price", description="Price of an item in USD.")
    def gp(item: str) -> float:
        return get_price(item)

    @tool(name="add_tax", description="Add tax percent to an amount.")
    def at(amount: float, rate_pct: float) -> float:
        return add_tax(amount, rate_pct)

    agent = Agent("Use the tools. Be terse.", model=MODEL, tools=[gp, at])
    t0 = time.time()
    r = await agent.run(QUESTION)
    record("loomflow", r.tokens_in + r.cached_tokens_in, r.tokens_out, r.turns, r.output, time.time() - t0)


# --------------------------------------------------------------------------
# 3. LOOM-CODE (build_agent) — note: it's a coding agent, heavier prompt
# --------------------------------------------------------------------------
async def run_loomcode():
    import tempfile

    from loom_code.agent import build_agent
    from loom_code.project import detect_project
    agent, _ = build_agent(detect_project(tempfile.mkdtemp()), model=MODEL)
    t0 = time.time()
    r = await agent.run(QUESTION)
    record("loom-code", r.tokens_in + r.cached_tokens_in, r.tokens_out, r.turns, r.output, time.time() - t0)


# --------------------------------------------------------------------------
# 4. LANGGRAPH (prebuilt react agent)
# --------------------------------------------------------------------------
async def run_langgraph():
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    @lc_tool
    def get_price_t(item: str) -> float:
        "Price of an item in USD."
        return get_price(item)

    @lc_tool
    def add_tax_t(amount: float, rate_pct: float) -> float:
        "Add tax percent to an amount."
        return add_tax(amount, rate_pct)

    llm = ChatOpenAI(model=MODEL, temperature=0)
    agent = create_react_agent(llm, [get_price_t, add_tax_t])
    cb = UsageMetadataCallbackHandler()
    t0 = time.time()
    out = await agent.ainvoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config={"callbacks": [cb]},
    )
    final = out["messages"][-1].content
    # sum usage across model calls
    tin = tout = calls = 0
    for u in cb.usage_metadata.values():
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
    # count AI messages as calls
    calls = sum(1 for m in out["messages"] if m.__class__.__name__ == "AIMessage")
    record("langgraph", tin, tout, calls, final, time.time() - t0)


# --------------------------------------------------------------------------
# 5. PYDANTIC-AI
# --------------------------------------------------------------------------
async def run_pydantic_ai():
    from pydantic_ai import Agent as PAgent

    agent = PAgent(f"openai:{MODEL}", system_prompt="Use the tools. Be terse.")

    @agent.tool_plain
    def get_price_p(item: str) -> float:
        "Price of an item in USD."
        return get_price(item)

    @agent.tool_plain
    def add_tax_p(amount: float, rate_pct: float) -> float:
        "Add tax percent to an amount."
        return add_tax(amount, rate_pct)

    t0 = time.time()
    r = await agent.run(QUESTION)
    u = r.usage()
    tin = getattr(u, "input_tokens", 0) or getattr(u, "request_tokens", 0) or 0
    tout = getattr(u, "output_tokens", 0) or getattr(u, "response_tokens", 0) or 0
    calls = getattr(u, "requests", 0) or 0
    record("pydantic-ai", tin, tout, calls, str(r.output), time.time() - t0)


async def main():
    runners = [
        ("raw-openai", run_openai),
        ("loomflow", run_loomflow),
        ("loom-code", run_loomcode),
        ("langgraph", run_langgraph),
        ("pydantic-ai", run_pydantic_ai),
    ]
    for name, fn in runners:
        try:
            await fn()
        except Exception as e:
            print(f"  ! {name} failed: {type(e).__name__}: {str(e)[:120]}")

    print(f"\n=== FRAMEWORK SHOOTOUT — {MODEL} — task: widgets+gadgets+tax (expect {EXPECT}) ===\n")
    print(f"{'framework':<14}{'ok':>4}{'tokens':>9}{'calls':>7}{'cost($)':>11}{'secs':>7}  answer")
    print("-" * 72)
    for r in sorted(RESULTS, key=lambda x: x["cost"]):
        print(f"{r['name']:<14}{('Y' if r['ok'] else 'N'):>4}{r['total']:>9}{r['calls']:>7}"
              f"{r['cost']:>11.6f}{r['secs']:>7.1f}  {r['ans']}")


if __name__ == "__main__":
    asyncio.run(main())
