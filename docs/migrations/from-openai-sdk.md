# Migrating from the raw OpenAI SDK

If you've been driving the agent loop yourself with
`openai.AsyncOpenAI`, this is the path of least resistance to a
production framework. The framework owns the loop + tool dispatch +
retries + memory; you keep your tools and your prompts.

## The agent loop you're probably running

```python
# Hand-rolled
from openai import AsyncOpenAI

client = AsyncOpenAI()

tools = [
    {"type": "function", "function": {"name": "get_weather",
     "parameters": {...}, "description": "..."}},
]

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Weather in Tokyo?"},
]

while True:
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, tools=tools
    )
    msg = resp.choices[0].message
    messages.append(msg.model_dump())

    if not msg.tool_calls:
        break

    for tc in msg.tool_calls:
        result = await dispatch_tool(tc.function.name, tc.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

print(messages[-1]["content"])
```

What this skeleton is missing for production:

* No retries on transient 5xx / 429s.
* No structured output validation.
* No multi-tenant isolation.
* No conversation continuity (each request rebuilds state).
* No telemetry / audit / budget.
* No parallel tool dispatch.
* No backpressure-aware streaming.

## The same thing on Loom

```python
import asyncio
from loomflow import Agent, tool

@tool
async def get_weather(city: str) -> str:
    """Look up weather for a city."""
    return await my_weather_api(city)

async def main():
    agent = Agent(
        "You are helpful.",
        model="gpt-4.1-mini",
        tools=[get_weather],
    )
    result = await agent.run("Weather in Tokyo?")
    print(result.output)

asyncio.run(main())
```

Eight lines. Retries, parallel tool dispatch, cancellation,
streaming, and the integration points (audit, telemetry, budget,
permissions, hooks) are all there waiting for you to wire them up.

## Translating common patterns

### Tool definitions

```python
# Raw SDK — JSON schema by hand
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
```

```python
# Loom — schema derived from type hints + docstring
from loomflow import tool

@tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return await my_weather_api(city)
```

### Multi-turn conversation

```python
# Raw SDK — you maintain the message history yourself
state = {"alice": [], "bob": []}

async def respond(user_id: str, prompt: str) -> str:
    state[user_id].append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini", messages=state[user_id]
    )
    state[user_id].append(resp.choices[0].message.model_dump())
    return resp.choices[0].message.content
```

```python
# Loom — session_id rehydrates, user_id partitions
agent = Agent("...", model="gpt-4.1-mini")

async def respond(user_id: str, prompt: str) -> str:
    r = await agent.run(
        prompt, user_id=user_id, session_id=f"chat-{user_id}"
    )
    return r.output
```

### Structured output

```python
# Raw SDK — response_format + manual parsing
from pydantic import BaseModel

class Invoice(BaseModel):
    amount_cents: int
    vendor: str

resp = await client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {
        "name": "Invoice",
        "schema": Invoice.model_json_schema(),
        "strict": True,
    }},
)
invoice = Invoice.model_validate_json(resp.choices[0].message.content)
```

```python
# Loom — output_schema kwarg, automatic validate-and-retry
agent = Agent("...", model="gpt-4.1-mini")
r = await agent.run("...", output_schema=Invoice)
invoice: Invoice = r.parsed
```

### Retries on transient failures

```python
# Raw SDK — handle exceptions yourself
import openai
import asyncio

async def with_retry(fn, attempts=3, base_delay=1.0):
    for i in range(attempts):
        try:
            return await fn()
        except openai.RateLimitError as e:
            if i == attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2**i))
```

```python
# Loom — happens automatically, configurable
from loomflow import Agent
from loomflow.governance import RetryPolicy

# Default policy is sensible (3 attempts, exp backoff, jitter, honours
# Retry-After). Override only if you need something different:
agent = Agent(
    "...",
    model="gpt-4.1-mini",
    retry_policy=RetryPolicy.aggressive(),  # or .disabled()
)
```

### Streaming

```python
# Raw SDK
async with client.chat.completions.stream(
    model="gpt-4.1-mini", messages=[...]
) as stream:
    async for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
```

```python
# Loom — Event stream, type-safe filtering
async for event in agent.stream("..."):
    if event.kind.value == "model_chunk":
        chunk = event.payload["chunk"]
        if chunk["text"]:
            print(chunk["text"], end="", flush=True)
```

### Parallel tool calls

The raw SDK returns multiple `tool_calls` and you decide whether to
run them in parallel. Most hand-rolled loops do them sequentially —
which is wrong, just hard to notice.

```python
# Loom — automatic parallel dispatch via anyio task groups
@tool
async def search_db(query: str) -> str: ...

@tool
async def search_docs(query: str) -> str: ...

agent = Agent("...", model="gpt-4.1-mini",
              tools=[search_db, search_docs])
# Model emits both tool_calls; the framework runs them in parallel
# under one task group, propagates cancellation correctly, and
# preserves arrival order in the message log.
```

## What you keep

* Your prompts, your tool implementations, your data, your
  monitoring stack. All of it.
* The OpenAI SDK is still the underlying transport — Loom's
  `OpenAIModel` wraps it.
* You can pass your own `AsyncOpenAI` instance if you have custom
  client config:

```python
from openai import AsyncOpenAI
from loomflow.model.openai import OpenAIModel
from loomflow import Agent

client = AsyncOpenAI(timeout=30.0, max_retries=0)
agent = Agent(
    "...",
    model=OpenAIModel("gpt-4.1-mini", client=client),
)
```

## What you can drop

The agent loop. The tool dispatch. The schema-validate-and-retry
glue. The "remember to namespace per user" boilerplate. The retry
wrapper. Roughly 200-400 LOC of hand-rolled framework code,
depending on how complete your previous loop was.
