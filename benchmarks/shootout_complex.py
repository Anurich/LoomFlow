"""Complex multi-step shootout — chained typed-tool calls + reasoning.

Task: an order-pricing problem that forces MULTIPLE typed tool calls in
sequence and arithmetic across them:

  "A customer orders 3 'widget' and 4 'gadget'. Look up each unit price
   with get_price. Apply a quantity discount with apply_discount based on
   total item count (use get_discount_rate to find the rate for the total
   quantity). Then add 8% tax with add_tax. Reply with ONLY the final
   total, rounded to 2 decimals."

Ground truth:
  widget=12.5, gadget=7.25
  subtotal = 3*12.5 + 4*7.25 = 37.5 + 29 = 66.5
  total qty = 7 → get_discount_rate(7) = 10 (>=5 → 10%)
  after discount = apply_discount(66.5, 10) = 59.85
  +8% tax = add_tax(59.85, 8) = 64.64 (round(59.85*1.08,2)=64.64)

Tools (all typed — the fair setup):
  get_price(item: str) -> float
  get_discount_rate(quantity: int) -> float
  apply_discount(amount: float, rate_pct: float) -> float
  add_tax(amount: float, rate_pct: float) -> float

Measures tokens / calls / cost / correctness across frameworks.
"""
from dotenv import load_dotenv

load_dotenv()  # reads .env from cwd; set OPENAI_API_KEY

import asyncio  # noqa: E402
import time  # noqa: E402

MODEL = "gpt-4.1-mini"
QUESTION = (
    "Compute an order total by calling tools in sequence (let the tools "
    "do ALL arithmetic — never calculate yourself):\n"
    "1. line_total(item='widget', qty=3)\n"
    "2. line_total(item='gadget', qty=4)\n"
    "3. add(a=<result1>, b=<result2>) to get the subtotal\n"
    "4. get_discount_rate(quantity=7)\n"
    "5. apply_discount(amount=<subtotal>, rate_pct=<rate>)\n"
    "6. add_tax(amount=<discounted>, rate_pct=8)\n"
    "Reply with ONLY the number from step 6."
)
EXPECT = "64.64"

PRICE_IN = 0.40 / 1_000_000
PRICE_OUT = 1.60 / 1_000_000

PRICES = {"widget": 12.5, "gadget": 7.25}


def get_price(item: str) -> float:
    return PRICES.get(item.lower().rstrip("s"), 0.0)


def line_total(item: str, qty: int) -> float:
    return round(get_price(item) * qty, 2)


def add(a: float, b: float) -> float:
    return round(a + b, 2)


def get_discount_rate(quantity: int) -> float:
    # >=10 → 15%, >=5 → 10%, else 0
    return 15.0 if quantity >= 10 else (10.0 if quantity >= 5 else 0.0)


def apply_discount(amount: float, rate_pct: float) -> float:
    return round(amount * (1 - rate_pct / 100), 2)


def add_tax(amount: float, rate_pct: float) -> float:
    return round(amount * (1 + rate_pct / 100), 2)


FUNCS = {
    "line_total": line_total,
    "add": add,
    "get_discount_rate": get_discount_rate,
    "apply_discount": apply_discount,
    "add_tax": add_tax,
}


def correct(ans: str) -> bool:
    return EXPECT in (ans or "").replace(",", "")


def cost(tin, tout):
    return tin * PRICE_IN + tout * PRICE_OUT


RESULTS = []


def record(name, tin, tout, calls, ans, secs):
    RESULTS.append(dict(name=name, total=tin + tout, cost=cost(tin, tout),
                        calls=calls, ok=correct(ans), secs=secs,
                        ans=(ans or "").strip()[:24]))


async def run_openai():
    import json

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    defs = [
        ("line_total", {"item": "string", "qty": "integer"}, ["item", "qty"]),
        ("add", {"a": "number", "b": "number"}, ["a", "b"]),
        ("get_discount_rate", {"quantity": "integer"}, ["quantity"]),
        ("apply_discount", {"amount": "number", "rate_pct": "number"}, ["amount", "rate_pct"]),
        ("add_tax", {"amount": "number", "rate_pct": "number"}, ["amount", "rate_pct"]),
    ]
    tools = [{"type": "function", "function": {"name": n,
              "description": n.replace("_", " "),
              "parameters": {"type": "object",
                  "properties": {k: {"type": v} for k, v in p.items()},
                  "required": r}}} for n, p, r in defs]
    msgs = [{"role": "user", "content": QUESTION}]
    tin = tout = calls = 0
    t0 = time.time()
    for _ in range(12):
        rr = await client.chat.completions.create(model=MODEL, messages=msgs, tools=tools)
        calls += 1
        tin += rr.usage.prompt_tokens
        tout += rr.usage.completion_tokens
        m = rr.choices[0].message
        if not m.tool_calls:
            record("raw-openai", tin, tout, calls, m.content, time.time() - t0)
            return
        msgs.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            a = json.loads(tc.function.arguments)
            out = FUNCS[tc.function.name](**a)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)})
    record("raw-openai", tin, tout, calls, "(no final)", time.time() - t0)


async def run_loomflow():
    from loomflow import Agent
    from loomflow.tools import tool

    @tool(name="line_total", description="qty * unit price of an item.")
    def lt(item: str, qty: int) -> float:
        return line_total(item, qty)

    @tool(name="add", description="Add two numbers.")
    def ad2(a: float, b: float) -> float:
        return add(a, b)

    @tool(name="get_discount_rate", description="Discount % for a total quantity.")
    def gd(quantity: int) -> float:
        return get_discount_rate(quantity)

    @tool(name="apply_discount", description="Subtract a discount % from an amount.")
    def ad(amount: float, rate_pct: float) -> float:
        return apply_discount(amount, rate_pct)

    @tool(name="add_tax", description="Add a tax % to an amount.")
    def at(amount: float, rate_pct: float) -> float:
        return add_tax(amount, rate_pct)

    agent = Agent("Solve step by step using the tools. Be terse.",
                  model=MODEL, tools=[lt, ad2, gd, ad, at])
    t0 = time.time()
    r = await agent.run(QUESTION)
    record("loomflow", r.tokens_in + r.cached_tokens_in, r.tokens_out, r.turns,
           r.output, time.time() - t0)


async def run_langgraph():
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    @lc_tool
    def line_total_t(item: str, qty: int) -> float:
        "qty * unit price of an item."
        return line_total(item, qty)

    @lc_tool
    def add_t(a: float, b: float) -> float:
        "Add two numbers."
        return add(a, b)

    @lc_tool
    def get_discount_rate_t(quantity: int) -> float:
        "Discount % for a total quantity."
        return get_discount_rate(quantity)

    @lc_tool
    def apply_discount_t(amount: float, rate_pct: float) -> float:
        "Subtract a discount % from an amount."
        return apply_discount(amount, rate_pct)

    @lc_tool
    def add_tax_t(amount: float, rate_pct: float) -> float:
        "Add a tax % to an amount."
        return add_tax(amount, rate_pct)

    llm = ChatOpenAI(model=MODEL, temperature=0)
    agent = create_react_agent(llm, [line_total_t, add_t, get_discount_rate_t, apply_discount_t, add_tax_t])
    cb = UsageMetadataCallbackHandler()
    t0 = time.time()
    out = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]},
                              config={"callbacks": [cb]})
    tin = tout = 0
    for u in cb.usage_metadata.values():
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
    calls = sum(1 for m in out["messages"] if m.__class__.__name__ == "AIMessage")
    record("langgraph", tin, tout, calls, out["messages"][-1].content, time.time() - t0)


async def run_pydantic_ai():
    from pydantic_ai import Agent as PAgent
    agent = PAgent(f"openai:{MODEL}", system_prompt="Solve step by step using the tools. Be terse.")

    @agent.tool_plain
    def line_total_p(item: str, qty: int) -> float:
        "qty * unit price of an item."
        return line_total(item, qty)

    @agent.tool_plain
    def add_p(a: float, b: float) -> float:
        "Add two numbers."
        return add(a, b)

    @agent.tool_plain
    def get_discount_rate_p(quantity: int) -> float:
        "Discount % for a total quantity."
        return get_discount_rate(quantity)

    @agent.tool_plain
    def apply_discount_p(amount: float, rate_pct: float) -> float:
        "Subtract a discount % from an amount."
        return apply_discount(amount, rate_pct)

    @agent.tool_plain
    def add_tax_p(amount: float, rate_pct: float) -> float:
        "Add a tax % to an amount."
        return add_tax(amount, rate_pct)

    t0 = time.time()
    r = await agent.run(QUESTION)
    u = r.usage()
    tin = getattr(u, "input_tokens", 0) or getattr(u, "request_tokens", 0) or 0
    tout = getattr(u, "output_tokens", 0) or getattr(u, "response_tokens", 0) or 0
    calls = getattr(u, "requests", 0) or 0
    record("pydantic-ai", tin, tout, calls, str(r.output), time.time() - t0)


async def main():
    print(f"=== COMPLEX SHOOTOUT — {MODEL} — chained typed tools (expect {EXPECT}) ===")
    for name, fn in [("raw-openai", run_openai), ("loomflow", run_loomflow),
                     ("langgraph", run_langgraph), ("pydantic-ai", run_pydantic_ai)]:
        try:
            await fn()
        except Exception as e:
            print(f"  ! {name} failed: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n{'framework':<13}{'ok':>4}{'tokens':>9}{'calls':>7}{'cost($)':>11}{'secs':>7}  answer")
    print("-" * 60)
    for r in sorted(RESULTS, key=lambda x: (not x["ok"], x["cost"])):
        print(f"{r['name']:<13}{('Y' if r['ok'] else 'N'):>4}{r['total']:>9}{r['calls']:>7}"
              f"{r['cost']:>11.6f}{r['secs']:>7.1f}  {r['ans']}")


if __name__ == "__main__":
    asyncio.run(main())
