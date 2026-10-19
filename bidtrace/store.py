# SPDX-License-Identifier: MPL-2.0
"""Small local store. Source snapshots and review history are append-only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def digest(value) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def document(text, title, role, version, provenance, source_url=None):
    pages = [{"page": i + 1, "text": part.strip()} for i, part in enumerate(text.split("\f"))]
    return {"id": uid("doc"), "title": title, "role": role, "version": version,
            "provenance": provenance, "source_url": source_url, "sha256": digest(text),
            "text": text, "pages": pages, "created_at": now()}


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "bidtrace.sqlite"
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, project_id TEXT, payload TEXT NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def all(self):
        with self.connect() as con:
            return [json.loads(r[0]) for r in con.execute("SELECT payload FROM projects ORDER BY rowid")]

    def get(self, project_id):
        with self.connect() as con:
            row = con.execute("SELECT payload FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return json.loads(row[0])

    def save(self, detail, review=None):
        project_id = detail["project"]["id"]
        with self.connect() as con:
            for doc in detail["documents"]:
                con.execute("INSERT OR IGNORE INTO documents VALUES(?,?)", (doc["id"], json.dumps(doc)))
            con.execute("INSERT INTO projects VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                        (project_id, json.dumps(detail)))
            if review:
                con.execute("INSERT INTO reviews VALUES(?,?,?)", (review["id"], project_id, json.dumps(review)))

    def historical_document(self, document_id):
        with self.connect() as con:
            row = con.execute("SELECT payload FROM documents WHERE id=?", (document_id,)).fetchone()
        return json.loads(row[0]) if row else None


def event(detail, kind, message):
    detail["events"].insert(0, {"id": uid("event"), "type": kind, "message": message, "created_at": now()})


def counts(requirements):
    return {"total": len(requirements),
            "needs_review": sum(r["status"] == "needs_review" for r in requirements),
            "gaps": sum(r["assessment"] == "gap" for r in requirements),
            "supported": sum(r["assessment"] == "supported" for r in requirements),
            "unknown": sum(r["assessment"] == "unknown" for r in requirements),
            "changed": sum(r["change_type"] != "unchanged" for r in requirements)}


def review_fingerprint(req, docs):
    # Preserve review on unchanged clause text, but invalidate after any supplier-evidence revision.
    return digest({"key": req["key"], "after": req["after"], "assessment": req["assessment"],
                   "evidence": sorted(d["sha256"] for d in docs if d["role"] == "evidence")})
