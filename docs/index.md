# JeevesAgent

**Production-ready async agent harness. Multi-tenant by default,
typed outputs, retries on transient errors, model-agnostic,
MCP-native.**

```python
import asyncio
from pydantic import BaseModel
from jeevesagent import Agent

class WeatherReport(BaseModel):
    city: str
    temp_c: float
    conditions: str

async def main():
    agent = Agent("Be precise.", model="gpt-4.1-mini")

    r = await agent.run("Hi, my name is Alice.", user_id="alice")
    print(r.output)

    r = await agent.run(
        "Weather in Tokyo: sunny, 22°C, light wind. Extract.",
        user_id="alice",
        session_id="conv_42",
        output_schema=WeatherReport,
    )
    report: WeatherReport = r.parsed   # typed, validated
    print(f"{report.city}: {report.temp_c}°C — {report.conditions}")

asyncio.run(main())
```

## Why pick this

* `user_id` is a first-class typed primitive. One `Agent` + one
  `Memory` partitions automatically across N tenants. **No more
  "forgot to namespace" data leaks.**
* `output_schema=` accepts any Pydantic model. Framework augments
  the system prompt, parses, validates, retries with feedback on
  failure. **Typed outputs by default.**
* Network model adapters auto-wrapped with a typed error taxonomy
  + retry policy. Rate limits, 5xx, network blips don't blow up
  your run. **Resilient by default.**
* `session_id` is a real conversation handle. Reuse it and prior
  turns rehydrate as real chat history. **No reducer protocol.**
* Twelve agent-loop architectures shipped behind one `Agent`
  constructor. **One kwarg flips the iteration pattern.**

```{toctree}
:maxdepth: 2
:caption: Get started

quickstart
recipes
architecture
workflow_vs_agent
```

```{toctree}
:maxdepth: 2
:caption: Production

production_hardening
```

```{toctree}
:maxdepth: 2
:caption: Migrating from

migrations/from-langgraph
migrations/from-openai-sdk
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/jeevesagent/index
migration_0.1_to_0.2
```

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
