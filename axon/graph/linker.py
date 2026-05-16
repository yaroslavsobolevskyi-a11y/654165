from __future__ import annotations

import numpy as np

from axon.embeddings.encoder import Encoder
from axon.storage import repository as repo


def build_semantic_links(conn, encoder: Encoder, top_k: int = 5,
                          min_similarity: float = 0.35) -> int:
    """
    Compute cosine similarity between all note embeddings and store
    the top-K most similar pairs per note.

    Returns number of links stored.
    """
    rows = repo.all_embeddings(conn)
    if len(rows) < 2:
        return 0

    ids = [r[0] for r in rows]
    matrix = np.stack([Encoder.from_blob(r[1]) for r in rows])  # shape (N, D)

    # Vectors are already L2-normalised by encoder → dot product == cosine similarity
    sim = matrix @ matrix.T  # shape (N, N)

    links: list[tuple[str, str, float]] = []
    for i, source_id in enumerate(ids):
        row = sim[i].copy()
        row[i] = -1.0  # exclude self
        top_indices = np.argpartition(row, -min(top_k, len(ids) - 1))[-top_k:]
        for j in top_indices:
            score = float(row[j])
            if score >= min_similarity:
                links.append((source_id, ids[j], score))

    repo.replace_semantic_links(conn, links)
    conn.commit()
    return len(links)
