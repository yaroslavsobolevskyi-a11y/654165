import sqlite3
import numpy as np
import pytest

from axon.storage.db import connect
from axon.storage import repository as repo
from axon.embeddings.encoder import Encoder


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    yield c
    c.close()


def _add_note(conn, id="n1", path="/tmp/n1.md", title="Note 1", content="Hello"):
    repo.upsert_note(conn, id=id, path=path, title=title,
                     content=content, content_hash="abc")
    conn.commit()


def test_upsert_and_get_note(conn):
    _add_note(conn)
    note = repo.get_note(conn, "n1")
    assert note["title"] == "Note 1"
    assert note["path"] == "/tmp/n1.md"


def test_upsert_is_idempotent(conn):
    _add_note(conn)
    _add_note(conn, title="Updated")
    note = repo.get_note(conn, "n1")
    assert note["title"] == "Updated"


def test_tags_replace(conn):
    _add_note(conn)
    repo.replace_tags(conn, "n1", ["ai", "memory"])
    conn.commit()
    tags = [r["tag"] for r in conn.execute("SELECT tag FROM tags WHERE note_id='n1'")]
    assert set(tags) == {"ai", "memory"}

    repo.replace_tags(conn, "n1", ["search"])
    conn.commit()
    tags = [r["tag"] for r in conn.execute("SELECT tag FROM tags WHERE note_id='n1'")]
    assert tags == ["search"]


def test_embedding_roundtrip(conn):
    _add_note(conn)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    repo.upsert_embedding(conn, "n1", "test-model", Encoder.to_blob(vec))
    conn.commit()

    rows = repo.all_embeddings(conn)
    assert len(rows) == 1
    recovered = Encoder.from_blob(rows[0][1])
    np.testing.assert_allclose(recovered, vec)


def test_stats(conn):
    _add_note(conn, id="n1", path="/tmp/n1.md", title="A")
    _add_note(conn, id="n2", path="/tmp/n2.md", title="B")
    s = repo.stats(conn)
    assert s["notes"] == 2


def test_semantic_links(conn):
    _add_note(conn, id="n1", path="/tmp/n1.md", title="A")
    _add_note(conn, id="n2", path="/tmp/n2.md", title="B")
    repo.replace_semantic_links(conn, [("n1", "n2", 0.9)])
    conn.commit()
    neighbors = repo.get_semantic_neighbors(conn, "n1", limit=5)
    assert len(neighbors) == 1
    assert neighbors[0]["score"] == pytest.approx(0.9)
