from __future__ import annotations

import sqlite3

import numpy as np

from axon.embeddings.encoder import Encoder
from axon.storage import repository as repo


def semantic_search(conn: sqlite3.Connection, encoder: Encoder,
                    query: str, top_k: int = 10) -> list[dict]:
    """Embed the query and rank all indexed notes by cosine similarity."""
    query_vec = encoder.encode_one(query)  # already normalised

    rows = repo.all_embeddings(conn)
    if not rows:
        return []

    ids = [r[0] for r in rows]
    matrix = np.stack([Encoder.from_blob(r[1]) for r in rows])
    scores = (matrix @ query_vec).tolist()

    ranked = sorted(zip(scores, ids), reverse=True)[:top_k]
    results = []
    for score, note_id in ranked:
        note = repo.get_note(conn, note_id)
        if note:
            results.append({
                "id": note["id"],
                "title": note["title"],
                "path": note["path"],
                "score": round(score, 4),
            })
    return results


def keyword_search(conn: sqlite3.Connection, query: str,
                   top_k: int = 10) -> list[dict]:
    """FTS5 full-text search across title and content."""
    rows = conn.execute(
        """
        SELECT n.id, n.title, n.path,
               rank AS bm25_score
        FROM notes_fts
        JOIN notes n ON n.rowid = notes_fts.rowid
        WHERE notes_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, top_k),
    ).fetchall()

    return [
        {"id": r["id"], "title": r["title"], "path": r["path"],
         "score": round(float(r["bm25_score"]), 4)}
        for r in rows
    ]
