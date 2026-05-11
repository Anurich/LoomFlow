---
name: Feature request
about: Propose a new capability or API
title: "feat: "
labels: ["enhancement", "needs-triage"]
assignees: []
---

<!--
Before filing: bigger proposals are worth discussing in GitHub
Discussions first. Issues are for "I want this to ship" — once
the design is roughly agreed.
-->

## The problem you're trying to solve

<!--
Concrete, please. "I'd like a feature for X" without the problem
context is hard to evaluate. What are you actually trying to do
that's currently painful or impossible?
-->

## Proposed solution

<!--
What would the API look like? Even a sketch is fine:

```python
agent = Agent(..., new_kwarg="...")
result = await agent.some_new_method(...)
```

How does it interact with existing primitives (Agent, Workflow,
Memory, etc.)?
-->

```python
# sketch of what you'd like to write
```

## Alternatives considered

<!--
What did you think about and rule out? Workarounds in current
Loom? Doing it outside the framework?
-->

## Why this belongs in Loom

<!--
Honest question. Could this just live in your app, in a helper
library, or in OTel / mem0 / LangGraph instead? What makes it
specifically *framework-level*?

Loom tries to stay focused. If a feature is easy to write as a
3-line user function, it might not belong in the framework.
-->

## Are you willing to send a PR?

- [ ] Yes, I can implement this myself
- [ ] Yes, with guidance from a maintainer
- [ ] No — flagging it for someone else to pick up
