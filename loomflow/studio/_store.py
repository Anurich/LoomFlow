"""Studio's storage: one SQLite file.

The recording process writes it from a single background thread
(:class:`~loomflow.studio._recorder.Recorder`); the ``loom studio``
web app reads it, and writes only what people add from its pages
(scores, dataset cases). WAL mode lets them run at once, in one
process or several.

Search goes through an FTS5 full-text index (the ``search`` table,
with ``search_docs`` mapping each finished trace to its row there)
that the writer keeps up to date, when the SQLite build has FTS5;
otherwise it scans the recorded text.

Times are stored as UTC epoch seconds (``REAL``); JSON columns hold
compact ``json.dumps`` text.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

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
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (project, name)
);
CREATE TABLE IF NOT EXISTS dataset_cases (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    input TEXT NOT NULL,
    expected TEXT,
    trace_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS dataset_cases_by_dataset ON dataset_cases (dataset_id, created_at);
"""

_SEARCH_TABLE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(body, tokenize = 'unicode61 remove_diacritics 2')"
)
# A trace's row in ``search``, so it can be replaced or deleted without
# scanning the index.
_SEARCH_DOCS = "CREATE TABLE IF NOT EXISTS search_docs (trace_id TEXT PRIMARY KEY, doc INTEGER NOT NULL)"
# Text indexed per trace, at most.
_SEARCH_MAX = 100_000


class _Connection(sqlite3.Connection):
    # Whether this (writer) connection keeps the search index up to date.
    search_ready = False


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


def _strings(value: Any, out: list[str]) -> None:
    """Every string and number in ``value``, and its mappings' keys."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            out.append(str(key))
            _strings(item, out)
    elif isinstance(value, list | tuple):
        for item in value:
            _strings(item, out)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        out.append(str(value))


def search_body(trace: Mapping[str, Any], spans: Sequence[Mapping[str, Any]]) -> str:
    """The text a trace is found by: its ids, input, output and error,
    and each step's name and content. Model messages are left out:
    each call repeats the conversation so far, which the other steps
    already hold."""
    out = [
        str(trace[key])
        for key in ("trace_id", "run_id", "name", "user_id", "session_id", "input", "output", "error")
        if trace.get(key)
    ]
    for span in spans:
        out.extend(str(span[key]) for key in ("name", "run_id", "error") if span.get(key))
        content = span.get("content") or {}
        if isinstance(content, str):
            content = _loads(content) or {}
        for key, value in content.items():
            if key != "messages":
                _strings(value, out)
    return "\n".join(out)[:_SEARCH_MAX]


def _fts_query(text: str) -> str | None:
    """``text`` as an FTS5 query: every word has to appear, matching
    the start of a word (``refund`` finds "refunded")."""
    terms = [t for t in text.split() if any(c.isalnum() for c in t)]
    if not terms:
        return None
    return " ".join('"' + t.replace('"', '""') + '"*' for t in terms)


def _index(conn: sqlite3.Connection, trace_id: str, body: str) -> None:
    row = conn.execute("SELECT doc FROM search_docs WHERE trace_id = ?", (trace_id,)).fetchone()
    if row is not None:
        conn.execute("UPDATE search SET body = ? WHERE rowid = ?", (body, row[0]))
    else:
        doc = conn.execute("INSERT INTO search (body) VALUES (?)", (body,)).lastrowid
        conn.execute("INSERT INTO search_docs (trace_id, doc) VALUES (?, ?)", (trace_id, doc))


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    # A file an older loomflow wrote lacks the newer tables until a
    # writer opens it; the read-only web app mustn't fail on that.
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


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
    read-only store (``readonly=True``, what the web app reads with)
    never creates or changes anything. Every method opens its own short
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
                conn.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version' AND CAST(value AS INTEGER) < ?",
                    (str(SCHEMA_VERSION), SCHEMA_VERSION),
                )

    # ---- connections ---------------------------------------------------

    def _connect(self) -> _Connection:
        # ``timeout`` waits out another process's lock (several
        # processes may record to one file), including while the
        # first of them switches the file to WAL.
        conn: _Connection
        if self.readonly:
            conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, timeout=10.0, check_same_thread=False,
                factory=_Connection,
            )
        else:
            conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False, factory=_Connection)
            if conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0:
                # A new file: let pruning hand space back. SQLite takes
                # this only before the first table and the switch to WAL.
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            _use_wal(conn)
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def writer(self) -> _Connection:
        """A connection for the recorder's writer thread."""
        if self.readonly:
            raise RuntimeError("a read-only StudioStore can't write")
        return self._connect()

    @contextmanager
    def write(self) -> Iterator[_Connection]:
        """One transaction on a new connection (the web app's writes)."""
        conn = self.writer()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- search and pruning (the writer) -------------------------------

    @staticmethod
    def prepare_search(conn: _Connection) -> bool:
        """Get the search index ready for ``conn`` to keep up to date:
        create it, and index the traces already recorded, unless it is
        complete. Returns ``False`` when this SQLite has no FTS5; search
        then scans the recorded text, for every reader of the file."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            state = _meta(conn, "search")
            try:
                conn.execute(_SEARCH_TABLE)
            except sqlite3.OperationalError:
                if state != "off":
                    _set_meta(conn, "search", "off")
                conn.commit()
                conn.search_ready = False
                return False
            conn.execute(_SEARCH_DOCS)
            if state != "fts5":
                conn.execute("DELETE FROM search")
                conn.execute("DELETE FROM search_docs")
                traces = conn.execute(
                    "SELECT trace_id, run_id, name, user_id, session_id, input, output, error FROM traces"
                    " WHERE status != 'running'"
                ).fetchall()
                for trace in traces:
                    spans = conn.execute(
                        "SELECT name, run_id, error, content FROM spans WHERE trace_id = ?",
                        (trace["trace_id"],),
                    ).fetchall()
                    _index(conn, trace["trace_id"], search_body(dict(trace), [dict(s) for s in spans]))
                _set_meta(conn, "search", "fts5")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        conn.search_ready = True
        return True

    @staticmethod
    def prune(conn: _Connection, before: float, *, batch: int = 500) -> int:
        """Delete the runs started before ``before`` (epoch seconds) with
        their steps, scores and search entries; eval runs are kept.
        Commits after each ``batch`` traces, so other writers aren't
        held up. Returns how many traces were deleted."""
        deleted = 0
        while True:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT trace_id FROM traces WHERE started_at < ? AND experiment_id IS NULL LIMIT ?",
                    (before, batch),
                ).fetchall()
            ]
            if not ids:
                break
            marks = ", ".join("?" for _ in ids)
            with conn:
                conn.execute(f"DELETE FROM spans WHERE trace_id IN ({marks})", ids)
                conn.execute(f"DELETE FROM scores WHERE trace_id IN ({marks})", ids)
                if conn.search_ready:
                    conn.execute(
                        f"DELETE FROM search WHERE rowid IN (SELECT doc FROM search_docs WHERE trace_id IN ({marks}))",
                        ids,
                    )
                    conn.execute(f"DELETE FROM search_docs WHERE trace_id IN ({marks})", ids)
                conn.execute(f"DELETE FROM traces WHERE trace_id IN ({marks})", ids)
            deleted += len(ids)
        if deleted:
            conn.execute("PRAGMA incremental_vacuum").fetchall()
        return deleted

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
        # A finished trace is indexed; its ``running`` row isn't.
        if getattr(conn, "search_ready", False) and row["status"] != "running":
            _index(conn, trace["trace_id"], search_body(trace, spans))

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

    @staticmethod
    def delete_score(conn: sqlite3.Connection, score_id: int) -> None:
        conn.execute("DELETE FROM scores WHERE id = ?", (score_id,))

    @staticmethod
    def add_dataset_case(
        conn: sqlite3.Connection,
        *,
        project: str,
        dataset: str,
        input: str,
        expected: str | None,
        trace_id: str | None,
    ) -> str:
        """Add a case to the dataset named ``dataset`` in ``project``
        (created if new). Returns the dataset's id."""
        row = conn.execute(
            "SELECT id FROM datasets WHERE project = ? AND name = ?", (project, dataset)
        ).fetchone()
        now = time.time()
        if row is None:
            dataset_id = "ds_" + secrets.token_hex(8)
            conn.execute(
                "INSERT INTO datasets (id, project, name, created_at) VALUES (?, ?, ?, ?)",
                (dataset_id, project, dataset, now),
            )
        else:
            dataset_id = str(row["id"])
        conn.execute(
            "INSERT INTO dataset_cases (id, dataset_id, input, expected, trace_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("case_" + secrets.token_hex(8), dataset_id, input, expected, trace_id, now),
        )
        return dataset_id

    @staticmethod
    def delete_dataset_case(conn: sqlite3.Connection, dataset_id: str, case_id: str) -> None:
        conn.execute("DELETE FROM dataset_cases WHERE id = ? AND dataset_id = ?", (case_id, dataset_id))

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
        unscored: bool = False,
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
        if unscored:
            where.append("NOT EXISTS (SELECT 1 FROM scores s WHERE s.trace_id = traces.trace_id)")
        with self._read() as conn:
            if search:
                sql, extra = self._search_filter(conn, search)
                where.append(sql)
                args.extend(extra)
            clause = " AND ".join(where)
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

    def average_cost(self, project: str, agent_key: str) -> float | None:
        """What one run of ``agent_key`` cost on average, over its last
        100 runs that cost anything (``None`` if none did)."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT AVG(cost_usd) FROM (SELECT cost_usd FROM traces WHERE project = ? AND agent_key = ?"
                " AND cost_usd > 0 ORDER BY started_at DESC LIMIT 100)",
                (project, agent_key),
            ).fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def previous_experiment(self, exp: Mapping[str, Any]) -> dict[str, Any] | None:
        """The run of the same dataset, in the same project, that
        started last before ``exp``."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, name, started_at FROM experiments WHERE project = ? AND dataset = ? AND id != ?"
                " AND started_at < ? ORDER BY started_at DESC LIMIT 1",
                (exp["project"], exp["dataset"], exp["id"], exp["started_at"]),
            ).fetchone()
        return None if row is None else dict(row)

    def score_keys(self, project: str | None) -> list[str]:
        """Score keys in use, most used first (suggestions for the form)."""
        clause, args = ("WHERE t.project = ?", [project]) if project else ("", [])
        with self._read() as conn:
            rows = conn.execute(
                "SELECT s.key FROM scores s JOIN traces t ON t.trace_id = s.trace_id"
                f" {clause} GROUP BY s.key ORDER BY COUNT(*) DESC LIMIT 50",
                args,
            ).fetchall()
        return [r[0] for r in rows]

    def score_trace(self, score_id: int) -> str | None:
        with self._read() as conn:
            row = conn.execute("SELECT trace_id FROM scores WHERE id = ?", (score_id,)).fetchone()
        return None if row is None else row[0]

    def list_datasets(self, project: str | None) -> list[dict[str, Any]]:
        clause, args = ("WHERE d.project = ?", [project]) if project else ("", [])
        with self._read() as conn:
            if not _has_table(conn, "datasets"):
                return []
            rows = conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM dataset_cases c WHERE c.dataset_id = d.id) AS cases,"
                " (SELECT MAX(created_at) FROM dataset_cases c WHERE c.dataset_id = d.id) AS updated_at"
                f" FROM datasets d {clause} ORDER BY d.name",
                args,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """A dataset with its cases, oldest first; ``trace_exists`` says
        whether the run a case came from is still recorded."""
        with self._read() as conn:
            if not _has_table(conn, "datasets"):
                return None
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                "SELECT c.*, t.trace_id IS NOT NULL AS trace_exists FROM dataset_cases c"
                " LEFT JOIN traces t ON t.trace_id = c.trace_id"
                " WHERE c.dataset_id = ? ORDER BY c.created_at, c.rowid",
                (dataset_id,),
            ).fetchall()
        return {**dict(row), "cases": [dict(c) for c in cases]}

    def find_datasets(self, name: str, project: str | None = None) -> list[dict[str, Any]]:
        clause, args = ("AND project = ?", [project]) if project else ("", [])
        with self._read() as conn:
            if not _has_table(conn, "datasets"):
                return []
            rows = conn.execute(f"SELECT * FROM datasets WHERE name = ? {clause}", (name, *args)).fetchall()
        return [dict(r) for r in rows]

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _search_filter(conn: sqlite3.Connection, text: str) -> tuple[str, list[Any]]:
        """The WHERE clause for a search: the full-text index when the
        file has a complete one this SQLite can read, else a scan of
        the root input / output and everything any step recorded."""
        query = _fts_query(text)
        if query is not None and _meta(conn, "search") == "fts5":
            try:
                conn.execute("SELECT 1 FROM search LIMIT 0")
            except sqlite3.OperationalError:
                pass  # this SQLite has no FTS5
            else:
                return (
                    "trace_id IN (SELECT trace_id FROM search_docs WHERE doc IN"
                    " (SELECT rowid FROM search WHERE search MATCH ?))",
                    [query],
                )
        needle = f"%{text}%"
        return (
            "(input LIKE ? OR output LIKE ? OR trace_id IN"
            " (SELECT trace_id FROM spans WHERE content LIKE ? OR name LIKE ?))",
            [needle] * 4,
        )

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
