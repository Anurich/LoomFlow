# Migrating from LangGraph

Side-by-side translations of the patterns LangGraph users hit most
often. The table at the top is the executive summary; the sections
below show concrete code.

| LangGraph concept | JeevesAgent equivalent |
|---|---|
| `StateGraph` + nodes + edges + reducers | `Agent` + `Architecture` (twelve shipped) |
| `add_messages` reducer | Automatic — message history rehydrates from `session_id` |
| `config={"configurable": {"thread_id": "..."}}` | `agent.run(prompt, session_id="...")` |
| `config={"configurable": {"user_id": "..."}}` (convention) | `agent.run(prompt, user_id="...")` (first-class) |
| Checkpointer (`MemorySaver`, `SqliteSaver`) | `SqliteRuntime` + `Memory` together — split into journaling and recall, joined by `session_id` |
| Store API (`store.put(namespace, key, value)`) | `Memory.recall_facts` + `Fact` (typed, bi-temporal) |
| `tool_node` + tool routing | `tools=[...]` on `Agent` — framework dispatches |
| `RunnableConfig` propagation | `RunContext` via `get_run_context()` |
| `stream_mode="values" / "updates" / "messages" / "debug"` | One `Event` stream with backpressure |
| `interrupt` / human-in-the-loop | Hooks (`@agent.before_tool` returns a denial) + permission policies |
| Subgraphs | Multi-agent architectures (`Supervisor`, `Debate`, `Swarm`, …) compose `Agent` instances directly |

## Hello world

```python
# LangGraph
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "Hi."}]})
print(result["messages"][-1].content)
```

```python
# JeevesAgent
import asyncio
from jeevesagent import Agent

async def main():
    agent = Agent("Be helpful.", model="gpt-4.1-mini")
    result = await agent.run("Hi.")
    print(result.output)

asyncio.run(main())
```

## Tool calling

```python
# LangGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"

agent = create_react_agent(model="gpt-4.1-mini", tools=[get_weather])
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in Tokyo?"}]}
)
```

```python
# JeevesAgent
from jeevesagent import Agent, tool

@tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"

agent = Agent(
    "Use the weather tool when asked about weather.",
    model="gpt-4.1-mini",
    tools=[get_weather],
)
result = await agent.run("Weather in Tokyo?")
```

## Multi-tenant memory

The biggest correctness gap in LangGraph: `user_id` is a string in
`config["configurable"]`. Typo it once and you silently leak data
across tenants. JeevesAgent makes `user_id` a typed primitive that
the framework honours — every memory backend partitions by it
automatically.

```python
# LangGraph — user_id is a CONVENTION you have to honour by hand
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

def my_node(state, config):
    user_id = config["configurable"]["user_id"]   # ← typo here = leak
    namespace = (user_id, "memories")
    memories = store.search(namespace, query=state["messages"][-1].content)
    # ...
```

```python
# JeevesAgent — user_id is a typed primitive; partition is automatic
from jeevesagent import Agent, get_run_context

# Inside any tool — never plumb user_id through signatures.
@tool
async def fetch_orders() -> str:
    ctx = get_run_context()
    return await db.query("orders", user_id=ctx.user_id)

agent = Agent("...", model="gpt-4.1-mini", tools=[fetch_orders])

# user_id is the call kwarg; framework partitions memory recall.
await agent.run("show my orders", user_id="alice")
```

## Conversation continuity

```python
# LangGraph — the checkpointer wires up state replay
from langgraph.checkpoint.memory import MemorySaver

graph = ... .compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "conv-42"}}

graph.invoke({"messages": ["Hi, I'm Alice."]}, config)
graph.invoke({"messages": ["What's my name?"]}, config)
# → "Alice", because the checkpointer rehydrated the thread state.
```

```python
# JeevesAgent — same session_id reused = conversation continues
agent = Agent("...", model="gpt-4.1-mini")
await agent.run("Hi, I'm Alice.", session_id="conv-42", user_id="alice")
await agent.run("What's my name?", session_id="conv-42", user_id="alice")
# → "Alice", because session_messages rehydrated the prior turns.
```

## Structured output

LangGraph offloads this to whatever model adapter you wire up
(`ChatOpenAI(model="...").with_structured_output(MySchema)`).
JeevesAgent ships it as a first-class kwarg with retry-on-validation-failure.

```python
# LangGraph
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(MyOutput)
result = llm.invoke([{"role": "user", "content": "..."}])
# result is a MyOutput instance — but errors aren't fed back as a
# retry; you handle ValidationError yourself.
```

```python
# JeevesAgent
from jeevesagent import Agent

agent = Agent("...", model="gpt-4.1-mini")
result = await agent.run("...", output_schema=MyOutput)
output: MyOutput = result.parsed   # ← validated; retry-on-fail built in
```

## Streaming

```python
# LangGraph — pick a stream_mode (4 incompatible flavours)
async for chunk in graph.astream(input_, config, stream_mode="messages"):
    ...
```

```python
# JeevesAgent — one Event stream, backpressure-aware
async for event in agent.stream(prompt, session_id="conv-42"):
    if event.kind.value == "model_chunk":
        print(event.payload["chunk"]["text"], end="", flush=True)
```

## Multi-agent

LangGraph's subgraphs are compiled-graph-inside-compiled-graph,
which has known issues around config propagation and
checkpointing. JeevesAgent multi-agent architectures compose
`Agent` instances directly — sub-agents inherit the parent's
`RunContext` automatically.

```python
# JeevesAgent — Team facade for the common shapes
from jeevesagent import Agent, Team

researcher = Agent("Research the topic.", model="gpt-4.1-mini")
writer = Agent("Draft the answer.", model="gpt-4.1-mini")

team = Team.supervisor(
    workers={"researcher": researcher, "writer": writer},
    model="gpt-4.1-mini",
)
result = await team.run("Write a brief about Acme Corp.", user_id="alice")
```

## Things JeevesAgent does NOT have

* No graph editor / state-graph DSL. The agent loop is a strategy;
  twelve are shipped. If you need something custom, implement the
  `Architecture` protocol (one async generator method).
* No `RunnableConfig` / `configurable` dict. Use kwargs (`user_id`,
  `session_id`, `metadata`) and the typed `RunContext`.
* No 4-mode streaming. One `Event` stream covers all cases; filter
  by `event.kind` if you only want a subset.
