---
name: Bug report
about: Something doesn't work the way you expect
title: "bug: "
labels: ["bug", "needs-triage"]
assignees: []
---

## Describe the bug

<!-- 1-3 sentences. What's broken? What did you expect? -->

## Minimal reproducer

<!--
The smaller the better. Paste a runnable Python snippet. If it
needs an API key, mark it explicitly. If it depends on `EchoModel`
or `ScriptedModel`, no key needed.

Goal: someone reading this issue can copy-paste and reproduce in
under 60 seconds.
-->

```python
from loomflow import Agent, ScriptedModel, ScriptedTurn

agent = Agent(
    "instructions",
    model=ScriptedModel([ScriptedTurn(text="hi")]),
)
# ... the part that fails
```

## Expected behaviour

<!-- What should happen? -->

## Actual behaviour

<!-- What does happen? Include the full traceback if there is one. -->

```
paste traceback here
```

## Environment

- **Loom version**: <!-- `python -c "import loomflow; print(loomflow.__version__)"` -->
- **Python version**: <!-- `python --version` -->
- **Operating system**: <!-- e.g. macOS 15.1, Ubuntu 24.04 -->
- **Installed extras**: <!-- e.g. `[openai,postgres]`. Default `pip install loomflow` = none. -->

## Anything else?

<!--
Workarounds you've tried. Related issues. Whether the bug
reproduces on a clean venv. Add any context that might help.
-->
