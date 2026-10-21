"""Studio's storage: one SQLite file.

The recording process writes it from a single background thread
(:class:`~loomflow.studio._recorder.Recorder`); the ``loom studio``
web app opens it read-only. WAL mode lets both run at once, in one
process or several.

Times are stored as UTC epoch seconds (``REAL``); JSON columns hold
compact ``json.dumps`` text.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    agent_key TEXT,
    agent_version INTEGER,
    run_id TEXT,
    user_id TEXT,
    session_id TEXT,
    input TEXT,
    output TEXT,
    status TEXT NOT NULL,
    error TEXT,
    interruption_reason TEXT,
    model TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    duration_ms REAL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    tags TEXT,
    metadata TEXT,
    experiment_id TEXT,
    case_key TEXT
);
CREATE INDEX IF NOT EXISTS traces_by_time ON traces (project, started_at DESC);
CREATE INDEX IF NOT EXISTS traces_by_session ON traces (session_id);
CREATE INDEX IF NOT EXISTS traces_by_agent ON traces (project, agent_key, agent_version);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    duration_ms REAL,
    status TEXT NOT NULL,
    error TEXT,
    attributes TEXT,
    content TEXT
);
CREATE INDEX IF NOT EXISTS spans_by_trace ON spans (trace_id, started_at);
CREATE INDEX IF NOT EXISTS spans_by_run ON spans (run_id);
CREATE TABLE IF NOT EXISTS agents (
    project TEXT NOT NULL,
    agent_key TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    PRIMARY KEY (project, agent_key)
);
CREATE TABLE IF NOT EXISTS agent_versions (
    project TEXT NOT NULL,
    agent_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    definition TEXT NOT NULL,
    changes TEXT NOT NULL,
    first_seen REAL NOT NULL,
    PRIMARY KEY (project, agent_key, version),
    UNIQUE (project, agent_key, fingerprint)
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT,
    span_id TEXT,
    run_id TEXT,
    key TEXT NOT NULL,
    score REAL,
    value TEXT,
    comment TEXT,
    source TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS scores_by_trace ON scores (trace_id);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    agent_key TEXT,
    agent_version INTEGER,
    agent_definition TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS experiment_cases (
    experiment_id TEXT NOT NULL,
    case_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    input TEXT,
    expected TEXT,
    output TEXT,
    trace_id TEXT,
    run_id TEXT,
    scores TEXT,
    error TEXT,
    PRIMARY KEY (experiment_id, case_key)
);
"""


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str, separators=(",", ":"))


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _use_wal(conn: sqlite3.Connection) -> None:
    """Put the file in WAL mode. Switching can report "locked" without
    waiting while another process creates the same file, so retry."""
    delay = 0.02
    deadline = time.monotonic() + 10.0
    while True:
        try:
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def _team(definition: Mapping[str, Any]) -> tuple[Any, ...]:
    """The agents in a structure and how they're connected, tools left
    out (the tools line covers those)."""
    graph = definition.get("structure") or {}
    nodes = {n["id"]: n for n in graph.get("nodes", []) if n.get("kind") != "tool"}
    labels = sorted((n["kind"], n["label"], n.get("detail", "")) for n in nodes.values())
    edges = sorted(
        (nodes[e["source"]]["label"], nodes[e["target"]]["label"], e.get("label", ""))
        for e in graph.get("edges", [])
        if e["source"] in nodes and e["target"] in nodes
    )
    return tuple(labels), tuple(edges)


def describe_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    """One readable line saying how ``new`` differs from ``old``
    (two agent definitions), e.g. ``"Model a to b. Tool added: x."``."""
    parts: list[str] = []
    for key in ("model", "architecture", "classifier"):
        if old.get(key) != new.get(key) and (old.get(key) or new.get(key)):
            parts.append(f"{key.capitalize()} {old.get(key) or 'none'} to {new.get(key) or 'none'}.")
    old_tools = {t["name"] for t in old.get("tools") or []}
    new_tools = {t["name"] for t in new.get("tools") or []}
    if added := sorted(new_tools - old_tools):
        parts.append(f"Tool{'s' if len(added) > 1 else ''} added: {', '.join(added)}.")
    if removed := sorted(old_tools - new_tools):
        parts.append(f"Tool{'s' if len(removed) > 1 else ''} removed: {', '.join(removed)}.")
    if old.get("instructions") != new.get("instructions"):
        parts.append("Instructions edited.")
    if old.get("nodes") != new.get("nodes") or old.get("edges") != new.get("edges"):
        parts.append("Graph changed.")
    elif _team(old) != _team(new):
        parts.append("Team or routing changed.")
    elif old.get("structure") != new.get("structure") and not (added or removed):
        parts.append("A team member's tools changed.")
    shown = {"model", "architecture", "classifier", "tools", "instructions", "nodes", "edges", "structure"}
    others = sorted(k for k in set(old) | set(new) if k not in shown and old.get(k) != new.get(k))
    if others:
        parts.append(f"Changed: {', '.join(others)}.")
    return " ".join(parts) or "No visible change."


class StudioStore:
    """The Studio database at ``path``.

    A store opened for writing creates the file and schema; a
    read-only store (``readonly=True``, what the web app uses) never
    creates or changes anything. Every method opens its own short
    connection, except :meth:`writer`, which the recorder's thread
    holds for its lifetime.
    """

    def __init__(self, path: str | Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = readonly
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    # ---- connections ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # ``timeout`` waits out another process's lock (several
        # processes may record to one file), including while the
        # first of them switches the file to WAL.
        if self.readonly:
            conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, timeout=10.0, check_same_thread=False
            )
        else:
            conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
            _use_wal(conn)
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def writer(self) -> sqlite3.Connection:
        """A connection for the recorder's writer thread."""
        if self.readonly:
            raise RuntimeError("a read-only StudioStore can't write")
        return self._connect()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # ---- writes (called on the writer thread's connection) -------------

    @staticmethod
    def upsert_agent(
        conn: sqlite3.Connection,
        project: str,
        agent: Mapping[str, Any],
        at: float,
    ) -> int:
        """Record ``agent`` (``key``, ``name``, ``kind``, ``fingerprint``,
        ``definition``) in the catalog; return its version number. A
        fingerprint not seen before for this agent becomes the next
        version, with a note on what changed since the previous one."""
        key, fingerprint = agent["key"], agent["fingerprint"]
        conn.execute(
            "INSERT INTO agents (project, agent_key, name, kind, first_seen, last_seen)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (project, agent_key) DO UPDATE SET"
            " name = excluded.name, kind = excluded.kind,"
            " last_seen = MAX(agents.last_seen, excluded.last_seen)",
            (project, key, agent["name"], agent["kind"], at, at),
        )
        row = conn.execute(
            "SELECT version FROM agent_versions WHERE project = ? AND agent_key = ? AND fingerprint = ?",
            (project, key, fingerprint),
        ).fetchone()
        if row is not None:
            return int(row["version"])
        prev = conn.execute(
            "SELECT version, definition FROM agent_versions WHERE project = ? AND agent_key = ?"
            " ORDER BY version DESC LIMIT 1",
            (project, key),
        ).fetchone()
        version = int(prev["version"]) + 1 if prev is not None else 1
        changes = (
            describe_changes(json.loads(prev["definition"]), agent["definition"])
            if prev is not None
            else "First seen."
        )
        conn.execute(
            "INSERT INTO agent_versions"
            " (project, agent_key, version, fingerprint, definition, changes, first_seen)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project, key, version, fingerprint, _dumps(agent["definition"]), changes, at),
        )
        return version

    @staticmethod
    def write_trace(
        conn: sqlite3.Connection,
        trace: Mapping[str, Any],
        spans: Sequence[Mapping[str, Any]],
        agents: Sequence[Mapping[str, Any]],
    ) -> None:
        """Insert or replace a trace with its spans, cataloguing every
        agent / workflow that ran in it."""
        project = trace["project"]
        versions: dict[str, int] = {}
        for agent in agents:
            versions[agent["key"]] = StudioStore.upsert_agent(
                conn, project, agent, trace["started_at"]
            )
        row = dict(trace)
        if row.get("agent_key") in versions:
            row["agent_version"] = versions[row["agent_key"]]
        row["tags"] = _dumps(row.get("tags") or [])
        row["metadata"] = _dumps(row.get("metadata") or {})
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        conn.execute(f"INSERT OR REPLACE INTO traces ({cols}) VALUES ({marks})", tuple(row.values()))
        conn.executemany(
            "INSERT OR REPLACE INTO spans (span_id, trace_id, parent_id, name, kind, run_id,"
            " started_at, ended_at, duration_ms, status, error, attributes, content)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    s["span_id"], trace["trace_id"], s.get("parent_id"), s["name"], s["kind"],
                    s.get("run_id"), s["started_at"], s.get("ended_at"), s.get("duration_ms"),
                    s.get("status", "ok"), s.get("error"),
                    _dumps(s.get("attributes") or {}), _dumps(s.get("content") or {}),
                )
                for s in spans
            ],
        )

    @staticmethod
    def add_score(conn: sqlite3.Connection, score: Mapping[str, Any]) -> None:
        """Insert a score, resolving its trace and span from ``run_id``
        when only that is known."""
        row = dict(score)
        if row.get("trace_id") is None and row.get("run_id"):
            found = conn.execute(
                "SELECT trace_id, span_id FROM spans WHERE run_id = ? LIMIT 1", (row["run_id"],)
            ).fetchone()
            if found is not None:
                row["trace_id"], row["span_id"] = found["trace_id"], found["span_id"]
        conn.execute(
            "INSERT INTO scores (trace_id, span_id, run_id, key, score, value, comment, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.get("trace_id"), row.get("span_id"), row.get("run_id"), row["key"],
                row.get("score"), row.get("value"), row.get("comment"), row["source"],
                row.get("created_at", time.time()),
            ),
        )

    @staticmethod
    def start_experiment(conn: sqlite3.Connection, exp: Mapping[str, Any]) -> None:
        version: int | None = None
        agent = exp.get("agent")
        if agent is not None:
            version = StudioStore.upsert_agent(conn, exp["project"], agent, exp["started_at"])
        conn.execute(
            "INSERT OR REPLACE INTO experiments (id, project, name, dataset, agent_key,"
            " agent_version, agent_definition, started_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exp["id"], exp["project"], exp["name"], exp["dataset"],
                agent["key"] if agent else None, version,
                _dumps(agent["definition"]) if agent else None,
                exp["started_at"], _dumps(exp.get("metadata") or {}),
            ),
        )

    @staticmethod
    def add_experiment_case(conn: sqlite3.Connection, case: Mapping[str, Any]) -> None:
        trace_id = case.get("trace_id")
        if trace_id is None and case.get("run_id"):
            found = conn.execute(
                "SELECT trace_id FROM spans WHERE run_id = ? LIMIT 1", (case["run_id"],)
            ).fetchone()
            trace_id = found["trace_id"] if found is not None else None
        if trace_id is None:
            # A run that raised has no run id to go by.
            found = conn.execute(
                "SELECT trace_id FROM traces WHERE experiment_id = ? AND case_key = ?"
                " ORDER BY started_at DESC LIMIT 1",
                (case["experiment_id"], case["case_key"]),
            ).fetchone()
            trace_id = found["trace_id"] if found is not None else None
        conn.execute(
            "INSERT OR REPLACE INTO experiment_cases (experiment_id, case_key, position, input,"
            " expected, output, trace_id, run_id, scores, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case["experiment_id"], case["case_key"], case["position"], case.get("input"),
                case.get("expected"), case.get("output"), trace_id, case.get("run_id"),
                _dumps(case.get("scores") or {}), case.get("error"),
            ),
        )

    @staticmethod
    def finish_experiment(conn: sqlite3.Connection, exp_id: str, ended_at: float, summary: Any) -> None:
        conn.execute(
            "UPDATE experiments SET ended_at = ?, summary = ? WHERE id = ?",
            (ended_at, _dumps(summary), exp_id),
        )

    # ---- reads (the web app) -------------------------------------------

    def projects(self) -> list[str]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT project FROM traces UNION SELECT project FROM experiments ORDER BY 1"
            ).fetchall()
        return [r[0] for r in rows]

    def list_traces(
        self,
        *,
        project: str | None = None,
        search: str | None = None,
        agent: str | None = None,
        user: str | None = None,
        model: str | None = None,
        session: str | None = None,
        status: str | None = None,
        version: int | None = None,
        since: float | None = None,
        problems: bool = False,
        include_experiments: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Root runs, newest first, plus totals over every match."""
        where: list[str] = ["1 = 1"]
        args: list[Any] = []
        if project:
            where.append("project = ?")
            args.append(project)
        if not include_experiments:
            where.append("experiment_id IS NULL")
        for column, value in (("agent_key", agent), ("user_id", user), ("model", model),
                              ("session_id", session), ("status", status)):
            if value:
                where.append(f"{column} = ?")
                args.append(value)
        if version is not None:
            where.append("agent_version = ?")
            args.append(version)
        if since is not None:
            where.append("started_at >= ?")
            args.append(since)
        if problems:
            where.append("status IN ('error', 'interrupted')")
        if search:
            # Root input / output, or anything any span in the trace
            # recorded (messages, tool arguments and results).
            where.append(
                "(input LIKE ? OR output LIKE ? OR trace_id IN"
                " (SELECT trace_id FROM spans WHERE content LIKE ? OR name LIKE ?))"
            )
            needle = f"%{search}%"
            args.extend([needle] * 4)
        clause = " AND ".join(where)
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT * FROM traces WHERE {clause} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            stats = conn.execute(
                f"SELECT COUNT(*) AS n, SUM(status = 'error') AS errors,"
                f" SUM(status = 'interrupted') AS interrupted, SUM(cost_usd) AS cost"
                f" FROM traces WHERE {clause}",
                args,
            ).fetchone()
            durations = [
                r[0]
                for r in conn.execute(
                    f"SELECT duration_ms FROM traces WHERE {clause} AND duration_ms IS NOT NULL"
                    " ORDER BY started_at DESC LIMIT 1000",
                    args,
                ).fetchall()
            ]
            ids = [r["trace_id"] for r in rows]
            scores = self._scores_for(conn, ids)
        traces = [self._trace_row(r) for r in rows]
        for t in traces:
            t["scores"] = scores.get(t["trace_id"], [])
        return {
            "traces": traces,
            "total": int(stats["n"] or 0),
            "errors": int(stats["errors"] or 0),
            "interrupted": int(stats["interrupted"] or 0),
            "cost_usd": float(stats["cost"] or 0.0),
            "median_ms": statistics.median(durations) if durations else None,
        }

    def filter_options(self, project: str | None) -> dict[str, list[str]]:
        """Distinct agents, users and models seen, for the filter menus."""
        clause, args = ("WHERE project = ?", [project]) if project else ("", [])
        with self._read() as conn:
            def distinct(column: str) -> list[str]:
                extra = "AND" if clause else "WHERE"
                return [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT {column} FROM traces {clause} {extra} {column} IS NOT NULL"
                        f" ORDER BY {column} LIMIT 200",
                        args,
                    ).fetchall()
                ]
            return {"agents": distinct("agent_key"), "users": distinct("user_id"), "models": distinct("model")}

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                return None
            spans = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at, rowid", (trace_id,)
            ).fetchall()
            scores = self._scores_for(conn, [trace_id]).get(trace_id, [])
            turns = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE session_id = ? AND project = ?",
                (row["session_id"], row["project"]),
            ).fetchone()[0] if row["session_id"] else 0
        trace = self._trace_row(row)
        trace["spans"] = [
            {
                **{k: r[k] for k in r.keys() if k not in ("attributes", "content")},
                "attributes": _loads(r["attributes"]) or {},
                "content": _loads(r["content"]) or {},
            }
            for r in spans
        ]
        trace["scores"] = scores
        trace["thread_runs"] = int(turns)
        return trace

    def list_agents(self, project: str | None) -> list[dict[str, Any]]:
        clause, args = ("WHERE a.project = ?", [project]) if project else ("", [])
        week_ago = time.time() - 7 * 86400
        with self._read() as conn:
            rows = conn.execute(
                "SELECT a.*, (SELECT MAX(version) FROM agent_versions v"
                " WHERE v.project = a.project AND v.agent_key = a.agent_key) AS current,"
                " (SELECT COUNT(*) FROM traces t WHERE t.project = a.project"
                " AND t.agent_key = a.agent_key AND t.experiment_id IS NULL"
                " AND t.started_at >= ?) AS runs_week"
                f" FROM agents a {clause} ORDER BY a.last_seen DESC",
                (week_ago, *args),
            ).fetchall()
        return [dict(r) for r in rows]

    def agent_versions(self, project: str, agent_key: str) -> list[dict[str, Any]]:
        """Each version with its run count and error rate (eval runs
        aside, as in the traces list) and its mean scores (eval runs
        included: they're often the best measure of a version)."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_versions WHERE project = ? AND agent_key = ? ORDER BY version DESC",
                (project, agent_key),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                stats = conn.execute(
                    "SELECT COUNT(*) AS n, SUM(status = 'error') AS errors FROM traces"
                    " WHERE project = ? AND agent_key = ? AND agent_version = ?"
                    " AND experiment_id IS NULL",
                    (project, agent_key, r["version"]),
                ).fetchone()
                score_rows = conn.execute(
                    "SELECT s.key, AVG(s.score) AS mean, COUNT(s.score) AS n FROM scores s"
                    " JOIN traces t ON t.trace_id = s.trace_id"
                    " WHERE t.project = ? AND t.agent_key = ? AND t.agent_version = ?"
                    " AND s.score IS NOT NULL GROUP BY s.key ORDER BY n DESC",
                    (project, agent_key, r["version"]),
                ).fetchall()
                out.append({
                    **{k: r[k] for k in r.keys() if k != "definition"},
                    "definition": _loads(r["definition"]),
                    "runs": int(stats["n"] or 0),
                    "errors": int(stats["errors"] or 0),
                    "scores": {s["key"]: float(s["mean"]) for s in score_rows},
                })
        return out

    def version_definition(self, project: str, agent_key: str, version: int) -> dict[str, Any] | None:
        """One version's recorded definition (with its ``structure``)."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT definition FROM agent_versions WHERE project = ? AND agent_key = ? AND version = ?",
                (project, agent_key, version),
            ).fetchone()
        result: dict[str, Any] | None = _loads(row["definition"]) if row is not None else None
        return result

    def list_experiments(self, project: str | None) -> list[dict[str, Any]]:
        clause, args = ("WHERE project = ?", [project]) if project else ("", [])
        with self._read() as conn:
            rows = conn.execute(
                "SELECT e.*, (SELECT COUNT(*) FROM experiment_cases c WHERE c.experiment_id = e.id) AS cases"
                f" FROM experiments e {clause} ORDER BY started_at DESC",
                args,
            ).fetchall()
        return [{**dict(r), "summary": _loads(r["summary"]), "metadata": _loads(r["metadata"])} for r in rows]

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                "SELECT * FROM experiment_cases WHERE experiment_id = ? ORDER BY position", (exp_id,)
            ).fetchall()
            durations = {
                r["trace_id"]: r["duration_ms"]
                for r in conn.execute(
                    "SELECT trace_id, duration_ms, cost_usd FROM traces WHERE experiment_id = ?", (exp_id,)
                ).fetchall()
            }
            costs = {
                r["trace_id"]: r["cost_usd"]
                for r in conn.execute(
                    "SELECT trace_id, cost_usd FROM traces WHERE experiment_id = ?", (exp_id,)
                ).fetchall()
            }
        exp = {
            **dict(row),
            "summary": _loads(row["summary"]) or {},
            "metadata": _loads(row["metadata"]) or {},
            "agent_definition": _loads(row["agent_definition"]),
        }
        exp["cases"] = [
            {
                **dict(c),
                "scores": _loads(c["scores"]) or {},
                "duration_ms": durations.get(c["trace_id"]),
                "cost_usd": costs.get(c["trace_id"]),
            }
            for c in cases
        ]
        return exp

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _scores_for(conn: sqlite3.Connection, trace_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        if not trace_ids:
            return {}
        marks = ", ".join("?" for _ in trace_ids)
        rows = conn.execute(
            f"SELECT * FROM scores WHERE trace_id IN ({marks}) ORDER BY created_at", tuple(trace_ids)
        ).fetchall()
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(r["trace_id"], []).append(dict(r))
        return out

    @staticmethod
    def _trace_row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["tags"] = _loads(row["tags"]) or []
        out["metadata"] = _loads(row["metadata"]) or {}
        return out
