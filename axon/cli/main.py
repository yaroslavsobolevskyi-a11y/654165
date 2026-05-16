from __future__ import annotations

from pathlib import Path

import click
from tqdm import tqdm

from axon.config import Config
from axon.embeddings.encoder import Encoder
from axon.graph.linker import build_semantic_links
from axon.ingestion.scanner import scan_vault
from axon.parser.extractor import extract
from axon.parser.markdown import parse_note
from axon.search.searcher import keyword_search, semantic_search
from axon.storage import db, repository as repo


def _get_conn(cfg: Config):
    return db.connect(cfg.db_path)


@click.group()
@click.option("--db", "db_path", default=None,
              help="Path to SQLite database (default: ~/.axon/axon.db)")
@click.pass_context
def cli(ctx, db_path):
    """Axon AI — local-first semantic memory for Markdown notes."""
    cfg = Config.from_env()
    if db_path:
        cfg.db_path = Path(db_path)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


# ── index ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("vault", type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, help="Re-embed all notes, ignoring hash cache.")
@click.pass_context
def index(ctx, vault: str, force: bool):
    """Scan VAULT folder, parse notes, generate embeddings, build semantic links."""
    cfg: Config = ctx.obj["cfg"]
    cfg.vault_path = Path(vault).resolve()
    conn = _get_conn(cfg)
    encoder = Encoder(cfg.embedding_model)

    files = list(scan_vault(cfg.vault_path))
    click.echo(f"Found {len(files)} Markdown file(s) in {cfg.vault_path}")

    changed = []
    for f in files:
        stored_hash = repo.get_note_hash(conn, f["id"])
        if force or stored_hash != f["content_hash"]:
            changed.append(f)

    click.echo(f"{len(changed)} file(s) need indexing.")

    if not changed:
        click.echo("Nothing to do. Use --force to re-index everything.")
        _finish(conn, cfg)
        return

    # Parse + store metadata
    texts: list[str] = []
    parsed_records: list[dict] = []
    for f in tqdm(changed, desc="Parsing"):
        parsed = parse_note(f["path"])
        extracted = extract(parsed["raw_body"], parsed["frontmatter"])

        repo.upsert_note(
            conn,
            id=f["id"],
            path=f["path"],
            title=parsed["title"],
            content=parsed["content"],
            content_hash=f["content_hash"],
            created_at=parsed["created_at"],
            modified_at=parsed["modified_at"],
        )
        repo.replace_tags(conn, f["id"], extracted.tags)
        repo.replace_wiki_links(conn, f["id"], extracted.wiki_links)

        texts.append(f"{parsed['title']}. {parsed['content']}")
        parsed_records.append({"id": f["id"]})

    conn.commit()
    repo.resolve_dangling_links(conn)
    conn.commit()

    # Embed in batch
    click.echo(f"Embedding {len(texts)} note(s) with '{cfg.embedding_model}'…")
    vectors = encoder.encode(texts, batch_size=cfg.batch_size)
    for record, vec in zip(parsed_records, vectors):
        repo.upsert_embedding(conn, record["id"], cfg.embedding_model,
                               Encoder.to_blob(vec))
    conn.commit()

    _finish(conn, cfg)


def _finish(conn, cfg: Config):
    encoder = Encoder(cfg.embedding_model)
    click.echo("Building semantic links…")
    n = build_semantic_links(conn, encoder, cfg.top_k_links, cfg.min_similarity)
    click.echo(f"Stored {n} semantic link(s).")

    s = repo.stats(conn)
    click.echo(
        f"\nDone — {s['notes']} notes | {s['wiki_links']} wiki-links | "
        f"{s['semantic_links']} semantic links | {s['tags']} unique tags"
    )


# ── search ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--top", default=5, show_default=True, help="Number of results.")
@click.option("--keyword", is_flag=True, help="Use FTS5 keyword search instead of semantic.")
@click.pass_context
def search(ctx, query: str, top: int, keyword: bool):
    """Search indexed notes by meaning (default) or keyword."""
    cfg: Config = ctx.obj["cfg"]
    conn = _get_conn(cfg)

    if keyword:
        results = keyword_search(conn, query, top_k=top)
        label = "keyword"
    else:
        encoder = Encoder(cfg.embedding_model)
        results = semantic_search(conn, encoder, query, top_k=top)
        label = "semantic"

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop {len(results)} {label} results for: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        click.echo(f"  {i}. [{r['score']:+.4f}]  {r['title']}")
        click.echo(f"       {r['path']}")


# ── links ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("note_path", type=click.Path(exists=True))
@click.option("--top", default=5, show_default=True)
@click.pass_context
def links(ctx, note_path: str, top: int):
    """Show semantic neighbors for a single note."""
    cfg: Config = ctx.obj["cfg"]
    conn = _get_conn(cfg)

    abs_path = str(Path(note_path).resolve())
    row = conn.execute("SELECT id, title FROM notes WHERE path = ?", (abs_path,)).fetchone()
    if not row:
        click.echo("Note not indexed. Run `axon index <vault>` first.")
        return

    neighbors = repo.get_semantic_neighbors(conn, row["id"], limit=top)
    if not neighbors:
        click.echo("No semantic neighbors found.")
        return

    click.echo(f"\nSemantic neighbors of: {row['title']}\n")
    for n in neighbors:
        click.echo(f"  [{n['score']:+.4f}]  {n['title']}")
        click.echo(f"           {n['path']}")


# ── status ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def status(ctx):
    """Show database statistics."""
    cfg: Config = ctx.obj["cfg"]
    conn = _get_conn(cfg)
    s = repo.stats(conn)
    click.echo(f"Database : {cfg.db_path}")
    click.echo(f"Notes    : {s['notes']}")
    click.echo(f"Embedded : {s['embeddings']}")
    click.echo(f"Wiki links    : {s['wiki_links']}")
    click.echo(f"Semantic links: {s['semantic_links']}")
    click.echo(f"Unique tags   : {s['tags']}")


if __name__ == "__main__":
    cli()
