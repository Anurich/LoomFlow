"""28_skills — Packaged playbooks loaded on demand (3 modes).

Skills extend Anthropic's Agent Skills (Oct 2025) format and the
LangChain DeepAgents layered-source pattern: each skill is a folder
with ``SKILL.md`` (frontmatter + markdown body) plus optional
supporting files.

This example shows all three modes side by side
-----------------------------------------------

* **Mode A** — pure markdown. ``invoice-processing`` is just
  instructions; the model uses the agent's built-in ``read`` /
  ``bash`` tools to do the work itself.
* **Mode C** — frontmatter manifest declares a script as a tool.
  ``calc`` ships a Python script, but the SKILL.md ``tools:``
  block tells the framework to wrap it in a subprocess-backed
  Tool with typed args. The model calls ``calc__add(a, b)`` like
  any normal tool — script runs in a subprocess, stdout returns.
  Works for any language (Python, bash, Node, …).
* **Mode B** — ``tools.py`` ships @tool functions. ``greeter``
  drops a ``tools.py`` next to its SKILL.md; the framework imports
  it on construction and registers any ``@tool``-decorated
  callables when the skill is loaded. In-process Python.

Plus the layered source pattern: ``system`` skills + ``project``
skills that override by name. Plus an inline skill via
:meth:`Skill.from_text` for one-off definitions in code.

Each scenario in this run shows a different interaction:

* Invoice question → loads ``invoice-processing`` (Mode A)
* Math question → loads ``calc``, calls ``calc__add(a, b)`` (Mode C)
* Greeting → loads ``greeter``, calls ``greeter__say_hi(name)`` (Mode B)
* Standup → loads ``standup-format`` (inline)
* Generic → no skill loaded, model just answers

Run::

    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/28_skills.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import (  # noqa: E402
    Agent,
    FileAuditLog,
    Mode,
    Skill,
    StandardPermissions,
    bash_tool,
    edit_tool,
    read_tool,
    write_tool,
)
from jeevesagent.governance.budget import (  # noqa: E402
    BudgetConfig,
    StandardBudget,
)

# ---------------------------------------------------------------------------
# Skill content (would normally live in version-controlled folders)
# ---------------------------------------------------------------------------

INVOICE_SKILL_MD = """\
---
name: invoice-processing
description: Extract amount, vendor, and date from invoice files. Use when the user mentions invoices, receipts, bills, or asks for invoice totals.
license: MIT
allowed_tools: [read, bash]
metadata:
  author: jeeves-team
  version: "1.0"
---

# Invoice Processing

## Steps

1. Use `read(path)` to load the invoice file (PDF text, .txt, .md).
2. Look for these patterns to extract structured data:
   - **Total**: regex `(?:Total|TOTAL|Grand Total)[:\\s]+\\$?([\\d,]+\\.\\d{2})`
   - **Vendor**: usually the first non-blank line
   - **Date**: ISO 8601 if present (`YYYY-MM-DD`), otherwise `MM/DD/YYYY`
3. For complex multi-line-item invoices, run the bundled helper:
   `bash("python skills/invoice-processing/scripts/extract_total.py FILE")`
4. Return a one-paragraph summary: vendor, date, total, line-item count.
"""

INVOICE_HELPER_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Tiny invoice line-item summer. Demo helper for the invoice
processing skill — sums every '$X.YY' figure in a file.\"\"\"
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: extract_total.py FILE", file=sys.stderr)
    sys.exit(2)

text = Path(sys.argv[1]).read_text()
amounts = [float(m.replace(",", "")) for m in re.findall(r"\\$?([\\d,]+\\.\\d{2})", text)]
print(f"line_items={len(amounts)}")
print(f"sum=${sum(amounts):.2f}")
print(f"max=${max(amounts) if amounts else 0:.2f}")
"""

EMAIL_SKILL_MD = """\
---
name: customer-emails
description: Draft replies to customer-support emails. Use when the user wants to write a customer reply, respond to a complaint, or thank a customer.
allowed_tools: [write]
---

# Customer Emails

## Tone
Warm, concise, action-oriented. Three short paragraphs maximum.

## Structure
1. Acknowledge the issue specifically (don't paraphrase generically).
2. State exactly what you'll do, with a date if applicable.
3. Sign off with the agent's name and a contact channel.

## Save the draft
Always save the draft to `draft.md` via the `write` tool so the user
can review before sending.
"""

# Skill that overrides the on-disk customer-emails one — for the demo.
PROJECT_EMAIL_SKILL_MD = """\
---
name: customer-emails
description: Draft replies to customer-support emails using OUR company's specific tone (project-level override).
allowed_tools: [write]
---

# Customer Emails (Project Override)

This is the PROJECT-LEVEL version. Same shape as the system one but
with company-specific tone:

## Tone
Empathetic but action-oriented. Always lead with "Thanks for reaching
out about <specific issue>."
"""


# ---------------------------------------------------------------------------
# Mode C — frontmatter declares a Python script as a typed Tool.
# The script is plain — no @tool decorator. Framework wraps it.
# ---------------------------------------------------------------------------

CALC_SKILL_MD = """\
---
name: calc
description: Arithmetic helpers. Use when the user asks to add, sum, or compute a numeric result from two integers.
tools:
  add:
    description: Sum two integers and return the result.
    script: scripts/add.py
    args:
      a:
        type: string
        description: First integer (as a string; the script parses it).
      b:
        type: string
        description: Second integer.
---

# Calc

Use the `calc__add` tool to sum two integers. Pass them as strings;
the bundled script parses and adds them.
"""

CALC_ADD_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Plain Python script — no decorators. Reads argv, prints stdout.
The framework wraps THIS file as `calc__add` because the SKILL.md
frontmatter declares it.\"\"\"
import sys
print(int(sys.argv[1]) + int(sys.argv[2]))
"""


# ---------------------------------------------------------------------------
# Mode B — tools.py ships an @tool function. Framework imports it.
# ---------------------------------------------------------------------------

GREETER_SKILL_MD = """\
---
name: greeter
description: Greet a person warmly. Use when the user says hello or asks for a friendly greeting.
---

# Greeter

Use the `greeter__say_hi` tool to generate a warm greeting for a
named person.
"""

GREETER_TOOLS_PY = """\
\"\"\"Skill-shipped Python tools. Imported on construction;
registered with the agent's tool host on load_skill('greeter').\"\"\"
from jeevesagent import tool

@tool
async def say_hi(name: str) -> str:
    \"\"\"Generate a warm greeting for NAME.\"\"\"
    return f\"Hi {name}! Hope your day is going well.\"
"""


def _materialize_skills(workdir: Path) -> tuple[Path, Path]:
    """Drop the skill folders to disk.

    System layer has invoice-processing + customer-emails + calc + greeter.
    Project layer overrides customer-emails with a company-specific tone.

    Returns ``(system_dir, project_dir)``."""
    system = workdir / "skills" / "system"
    project = workdir / "skills" / "project"

    # Mode A — pure markdown skill (with optional bash-callable script)
    (system / "invoice-processing" / "scripts").mkdir(parents=True)
    (system / "invoice-processing" / "SKILL.md").write_text(INVOICE_SKILL_MD)
    (system / "invoice-processing" / "scripts" / "extract_total.py").write_text(
        INVOICE_HELPER_SCRIPT
    )

    # Mode A — pure markdown skill, no scripts
    (system / "customer-emails").mkdir(parents=True)
    (system / "customer-emails" / "SKILL.md").write_text(EMAIL_SKILL_MD)

    # Project layer override
    (project / "customer-emails").mkdir(parents=True)
    (project / "customer-emails" / "SKILL.md").write_text(PROJECT_EMAIL_SKILL_MD)

    # Mode C — frontmatter manifest wraps a script as a typed Tool
    (system / "calc" / "scripts").mkdir(parents=True)
    (system / "calc" / "SKILL.md").write_text(CALC_SKILL_MD)
    (system / "calc" / "scripts" / "add.py").write_text(CALC_ADD_SCRIPT)

    # Mode B — tools.py with @tool functions, auto-discovered
    (system / "greeter").mkdir(parents=True)
    (system / "greeter" / "SKILL.md").write_text(GREETER_SKILL_MD)
    (system / "greeter" / "tools.py").write_text(GREETER_TOOLS_PY)

    return system, project


# ---------------------------------------------------------------------------
# Sample invoice text (would normally be uploaded by the user)
# ---------------------------------------------------------------------------

SAMPLE_INVOICE = """\
TechCorp Inc.
123 Innovation Way

Invoice #TC-9842
Date: 2026-04-12

Items:
  API credits (10K)     $250.00
  Premium support       $99.00

Subtotal: $349.00
Tax: $28.42
Total: $377.42
"""


# ---------------------------------------------------------------------------
# Inline skill — an in-code playbook with no folder needed
# ---------------------------------------------------------------------------

STANDUP_SKILL = Skill.from_text(
    """---
name: standup-format
description: Format a daily-standup update from rough notes. Use when the user gives you bullet points and asks for a daily standup or status update.
---

# Standup format

Always exactly three sections, in this order:

## Yesterday
[What got done]

## Today
[What's planned]

## Blockers
[What needs help; "None." if nothing]

Keep each section to 3 bullets max.
"""
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_skills_")
    ).resolve()
    print("=" * 72)
    print("Skills demo — packaged playbooks with progressive disclosure")
    print("=" * 72)
    print(f"Workdir: {workdir}\n")

    # Stage 1: lay down skill folders.
    system_skills, project_skills = _materialize_skills(workdir)
    invoice_path = workdir / "invoice.txt"
    invoice_path.write_text(SAMPLE_INVOICE)

    print("Materialised skill sources:")
    print(f"  system:  {system_skills}")
    print(f"  project: {project_skills} (overrides customer-emails)")
    print(f"  inline:  {STANDUP_SKILL.name}\n")

    # Stage 2: build the agent with layered skill sources.
    audit_path = workdir / "audit.jsonl"
    agent = Agent(
        instructions=(
            "You are a versatile assistant. Use the skills you have "
            "when relevant; otherwise just answer normally."
        ),
        model="gpt-4.1-mini",
        tools=[
            read_tool(workdir=workdir),
            write_tool(workdir=workdir),
            edit_tool(workdir=workdir),
            bash_tool(workdir=workdir),
        ],
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        audit_log=FileAuditLog(audit_path, secret="skills-demo"),
        budget=StandardBudget(
            BudgetConfig(max_tokens=80_000, max_cost_usd=1.0)
        ),
        skills=[
            (system_skills, "System"),       # base layer
            (project_skills, "Project"),     # project override
            STANDUP_SKILL,                    # inline skill
        ],
    )

    print(f"Agent's skill catalog ({len(agent.skills)} skills):")
    assert agent.skills is not None
    for skill in agent.skills:
        print(
            f"  - {skill.name} "
            f"[{skill.metadata.source_label or 'inline'}]: "
            f"{skill.description[:60]}..."
        )
    print()

    # Stage 3: five requests covering all three modes.
    scenarios = [
        (
            "Mode A — invoice (markdown skill, model uses bash itself)",
            f"What's the total amount in {invoice_path}?",
        ),
        (
            "Mode C — math (frontmatter declares script as tool)",
            "What's 47 plus 95? Use the calc skill.",
        ),
        (
            "Mode B — greeting (tools.py @tool function)",
            "Use the greeter skill to say hi to Anupam.",
        ),
        (
            "Mode A — customer reply (project override wins)",
            "A customer named Sam complained their order ORD-1002 was "
            "duplicated. Draft a reply.",
        ),
        (
            "Generic question — no skill loaded",
            "What's the capital of Spain?",
        ),
    ]

    for i, (title, prompt) in enumerate(scenarios, 1):
        print(f"\n{'═' * 72}")
        print(f"SCENARIO {i}: {title}")
        print(f"{'═' * 72}")
        print(f"User: {prompt}\n")

        loaded_skills: list[str] = []
        async for ev in agent.stream(prompt):
            kind = ev.kind.value
            if kind == "model_chunk":
                chunk = ev.payload.get("chunk", {})
                if chunk.get("kind") == "text" and chunk.get("text"):
                    print(chunk["text"], end="", flush=True)
            elif kind == "tool_call":
                call = ev.payload.get("call", {})
                tool_name = call.get("tool", "")
                args = call.get("args", {})
                if tool_name == "load_skill":
                    skill_name = args.get("name", "?")
                    loaded_skills.append(skill_name)
                    print(f"\n  ◇ load_skill({skill_name!r})")
                else:
                    arg_str = ", ".join(
                        f"{k}={str(v)[:40]!r}"
                        for k, v in args.items()
                    )
                    print(f"\n  ◇ {tool_name}({arg_str})")
            elif kind == "completed":
                result = ev.payload.get("result") or {}
                print(
                    f"\n\n  [skills loaded: {loaded_skills or 'none'} | "
                    f"turns={result.get('turns')} "
                    f"tokens in={result.get('tokens_in')} "
                    f"out={result.get('tokens_out')} "
                    f"cost=${float(result.get('cost_usd', 0) or 0):.4f}]"
                )

    # Final audit roll-up.
    print(f"\n{'═' * 72}")
    print("AUDIT TRAIL (selected entries)")
    print(f"{'═' * 72}")
    log = FileAuditLog(audit_path, secret="skills-demo")
    entries = await log.query()
    skill_loads = [
        e for e in entries if "load_skill" in str(e.action).lower()
    ]
    print(f"  Total entries:   {len(entries)}")
    print(f"  Skill loads:     {len(skill_loads)}")
    print(f"  Audit file:      {audit_path}")
    print(f"\n(Workdir kept at {workdir} for inspection.)")


if __name__ == "__main__":
    asyncio.run(main())
