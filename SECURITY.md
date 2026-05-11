# Security policy

## Reporting a vulnerability

**Please do not file a public issue for security problems.**
Public issues alert attackers before a fix is available.

Report privately via one of these channels (in preferred order):

1. **GitHub Security Advisory** (recommended)
   <https://github.com/Anurich/LoomFlow/security/advisories/new>
   Lets the maintainer triage and ship a fix without the report
   ever being public until you're ready.

2. **Email**: `nautiyalanupam98@gmail.com` with subject prefix
   `[loomflow-security]`.

What to include:

- A description of the issue and its impact.
- Steps to reproduce (minimal repro preferred).
- The Loom version and Python version you observed it on.
- Whether you'd like to be credited in the fix's CHANGELOG entry.

## What to expect

- **Acknowledgement within 72 hours.** If you don't hear back by
  then, the email may have been missed — please ping again or
  switch to the GitHub Security Advisory channel.
- **Initial assessment within 7 days.** Confirmed vs not, severity,
  rough timeline.
- **Coordinated disclosure.** We'll work with you on a disclosure
  date once a fix is ready. Default is to ship the fix in a patch
  release and publish the advisory + credit on the same day.
- **No bug bounty program** today. The project is volunteer-run;
  we offer credit and our sincere thanks, not money.

## Supported versions

| Version line | Supported with security fixes? |
|---|---|
| `0.9.x` (current) | ✅ Yes |
| `0.8.x` and older | ❌ Please upgrade |

The `0.x` series doesn't promise long-term backports; patches go
to the latest minor only. Once `1.0` ships, we'll publish an
explicit support window here.

## Out of scope

The following are **not** vulnerabilities for the purpose of this
policy:

- Issues that require an attacker to already have code-execution
  rights on the user's machine (Python is not a sandbox).
- Issues in optional dependencies (`opentelemetry`, `chromadb`,
  `psycopg`, etc.) that don't have a Loom-specific component —
  please report those upstream.
- "Default settings could be more restrictive" suggestions that
  aren't a real attack — file these as enhancement issues, not
  security reports.
- Prompt injection / model-output-trust issues that are a property
  of LLMs generally and not specific to Loom's framework code.

If you're unsure whether something qualifies — err on the side of
reporting privately. We'd rather get a few false positives than
have a real issue filed publicly.

Thank you for helping keep Loom and its users safe.
