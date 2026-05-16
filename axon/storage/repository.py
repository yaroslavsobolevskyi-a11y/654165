from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Notes ──────────────────────────────────────────────────────────────────────

def get_note_hash(conn: sqlite3.Connection, note_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    return row["content_hash"] if row else None


def upsert_note(conn: sqlite3.Connection, *, id: str, path: str, title: str,
                content: str, content_hash: str,
                created_at: str = "", modified_at: str = "") -> None:
    conn.execute(
        """
        INSERT INTO notes (id, path, title, content, content_hash,
                           created_at, modified_at, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            path         = excluded.path,
            title        = excluded.title,
            content      = excluded.content,
            content_hash = excluded.content_hash,
            created_at   = excluded.created_at,
            modified_at  = excluded.modified_at,
            indexed_at   = excluded.indexed_at
        """,
        (id, path, title, content, content_hash, created_at, modified_at, _now()),
    )


def all_note_ids(conn: sqlite3.Connection) -> list[str]:
    return [r["id"] for r in conn.execute("SELECT id FROM notes").fetchall()]


def get_note(conn: sqlite3.Connection, note_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()


def get_note_by_title(conn: sqlite3.Connection, title: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM notes WHERE lower(title) = lower(?)", (title,)
    ).fetchone()


# ── Tags ───────────────────────────────────────────────────────────────────────

def replace_tags(conn: sqlite3.Connection, note_id: str, tags: list[str]) -> None:
    conn.execute("DELETE FROM tags WHERE note_id = ?", (note_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO tags (note_id, tag) VALUES (?, ?)",
        [(note_id, t) for t in tags],
    )


# ── Wiki links ─────────────────────────────────────────────────────────────────

def replace_wiki_links(conn: sqlite3.Connection, note_id: str,
                        links: list[str]) -> None:
    conn.execute("DELETE FROM wiki_links WHERE source_id = ?", (note_id,))
    for title in links:
        target = get_note_by_title(conn, title)
        target_id = target["id"] if target else None
        conn.execute(
            """INSERT OR IGNORE INTO wiki_links (source_id, target_title, target_id)
               VALUES (?, ?, ?)""",
            (note_id, title, target_id),
        )


def resolve_dangling_links(conn: sqlite3.Connection) -> None:
    """Fill in target_id for wiki_links that were created before the target existed."""
    conn.execute(
        """
        UPDATE wiki_links
        SET target_id = (
            SELECT id FROM notes WHERE lower(title) = lower(wiki_links.target_title)
        )
        WHERE target_id IS NULL
        """
    )


# ── Embeddings ─────────────────────────────────────────────────────────────────

def upsert_embedding(conn: sqlite3.Connection, note_id: str,
                      model: str, vector_blob: bytes) -> None:
    conn.execute(
        """INSERT INTO embeddings (note_id, model, vector) VALUES (?, ?, ?)
           ON CONFLICT(note_id) DO UPDATE SET model=excluded.model, vector=excluded.vector""",
        (note_id, model, vector_blob),
    )


def all_embeddings(conn: sqlite3.Connection) -> list[tuple[str, bytes]]:
    """Return [(note_id, vector_blob), ...]."""
    rows = conn.execute("SELECT note_id, vector FROM embeddings").fetchall()
    return [(r["note_id"], r["vector"]) for r in rows]


# ── Semantic links ─────────────────────────────────────────────────────────────

def replace_semantic_links(conn: sqlite3.Connection,
                            links: list[tuple[str, str, float]]) -> None:
    """links = [(source_id, target_id, score), ...]"""
    conn.execute("DELETE FROM semantic_links")
    conn.executemany(
        "INSERT OR IGNORE INTO semantic_links (source_id, target_id, score) VALUES (?,?,?)",
        links,
    )


def get_semantic_neighbors(conn: sqlite3.Connection, note_id: str,
                            limit: int = 5) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT n.id, n.title, n.path, sl.score
        FROM semantic_links sl
        JOIN notes n ON n.id = sl.target_id
        WHERE sl.source_id = ?
        ORDER BY sl.score DESC
        LIMIT ?
        """,
        (note_id, limit),
    ).fetchall()


# ── Stats ──────────────────────────────────────────────────────────────────────

def stats(conn: sqlite3.Connection) -> dict:
    return {
        "notes": conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
        "embeddings": conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        "wiki_links": conn.execute("SELECT COUNT(*) FROM wiki_links").fetchone()[0],
        "semantic_links": conn.execute("SELECT COUNT(*) FROM semantic_links").fetchone()[0],
        "tags": conn.execute("SELECT COUNT(DISTINCT tag) FROM tags").fetchone()[0],
    }
