"""``loom`` — the loomflow command line (G12).

Stdlib ``argparse`` only; no click/typer dependency. Wired up via
``[project.scripts]`` in ``pyproject.toml``::

    loom run "summarise this" --model openai:gpt-4o
    loom serve my_service:agent --host 0.0.0.0 --port 8000
    loom eval cases.jsonl --agent my_service:agent --threshold exact_match=0.9
    loom version

``loom serve`` needs an ASGI server; install the extra::

    pip install 'loomflow[serve]'
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import anyio

from . import Agent, __version__

__all__ = ["main"]

_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


def _load_target(spec: str) -> Any:
    """Import ``module:attribute`` and return the attribute."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"module {module_name!r} has no attribute {attr!r}") from exc


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"loomflow {__version__}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        agent = Agent(args.instructions, model=args.model)
    except Exception as exc:  # noqa: BLE001 — e.g. an unknown model spec
        return _fail(f"{type(exc).__name__}: {exc}")

    if args.stream:

        async def _stream() -> None:
            async for event in agent.stream(
                args.prompt,
                user_id=args.user_id,
                session_id=args.session_id,
                response_tone=args.tone,
            ):
                print(f"{event.kind.value}\t{event.model_dump_json()}")

        try:
            anyio.run(_stream)
        except Exception as exc:  # noqa: BLE001 — CLI boundary: message, not traceback
            return _fail(f"{type(exc).__name__}: {exc}")
        return 0

    async def _run() -> Any:
        return await agent.run(
            args.prompt,
            user_id=args.user_id,
            session_id=args.session_id,
            response_tone=args.tone,
        )

    try:
        result = anyio.run(_run)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: message, not traceback
        return _fail(f"{type(exc).__name__}: {exc}")
    print(result.output)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    # Lazy import: uvicorn is the optional ``serve`` extra, checked
    # FIRST so the error message doesn't depend on the target module
    # importing cleanly.
    try:
        import uvicorn  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        return _fail(
            "uvicorn is not installed; the serve command needs an ASGI "
            "server.\n  pip install 'loomflow[serve]'"
        )

    from .serve import create_app  # deferred: keep `loom version` import-light

    try:
        agent = _load_target(args.target)
    except (ImportError, ValueError) as exc:
        return _fail(str(exc))
    app = create_app(agent)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval import Contains, Dataset, EvalHarness, ExactMatch  # tier-2 import

    thresholds: dict[str, float] = {}
    for spec in args.threshold or []:
        name, sep, value = spec.partition("=")
        if not sep or not name:
            return _fail(f"--threshold expects name=value, got {spec!r}")
        try:
            thresholds[name] = float(value)
        except ValueError:
            return _fail(f"--threshold value must be a number, got {spec!r}")

    try:
        agent = _load_target(args.agent)
    except (ImportError, ValueError) as exc:
        return _fail(str(exc))
    try:
        dataset = Dataset.from_jsonl(args.dataset)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    metric = ExactMatch() if args.metric == "exact" else Contains()
    harness = EvalHarness(agent, metrics=[metric])
    try:
        report = anyio.run(harness.run, dataset)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: message, not traceback
        return _fail(f"{type(exc).__name__}: {exc}")

    print(json.dumps({"passed": report.passed, "metrics": report.summary()}, indent=2))
    if thresholds:
        try:
            report.assert_thresholds(thresholds)
        except AssertionError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0 if report.passed else 1


# ---------------------------------------------------------------------------
# Parser + entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom", description="loomflow — run, serve, and eval agents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a single prompt through an Agent")
    p_run.add_argument("prompt", help="the user prompt")
    p_run.add_argument(
        "--model", default="echo", help="model spec (e.g. openai:gpt-4o; default: echo)"
    )
    p_run.add_argument(
        "--instructions",
        default=_DEFAULT_INSTRUCTIONS,
        help=f"system instructions (default: {_DEFAULT_INSTRUCTIONS!r})",
    )
    p_run.add_argument("--user-id", dest="user_id", default=None, help="tenancy partition")
    p_run.add_argument("--session-id", dest="session_id", default=None)
    p_run.add_argument("--tone", default=None, help="response tone directive")
    p_run.add_argument(
        "--stream", action="store_true", help="print one line per agent event instead"
    )
    p_run.set_defaults(func=_cmd_run)

    p_serve = sub.add_parser("serve", help="serve an Agent over HTTP (ASGI + uvicorn)")
    p_serve.add_argument("target", help="import path of an Agent, as module:attribute")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_eval = sub.add_parser("eval", help="run a JSONL dataset through loomflow.eval")
    p_eval.add_argument("dataset", help="path to a JSONL dataset (one Case per line)")
    p_eval.add_argument(
        "--agent", required=True, help="import path of an Agent, as module:attribute"
    )
    p_eval.add_argument(
        "--metric",
        choices=("exact", "contains"),
        default="exact",
        help="scoring metric (default: exact)",
    )
    p_eval.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=MIN",
        help="CI gate, e.g. exact_match=0.9 (repeatable); exit 1 when unmet",
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_version = sub.add_parser("version", help="print the loomflow version")
    p_version.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``loom`` console script. Returns the exit code."""
    args = _build_parser().parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
