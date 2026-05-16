from __future__ import annotations

from pathlib import Path

import frontmatter


def parse_note(path: str | Path) -> dict:
    """
    Parse a Markdown file and return a dict with:
      title, content (plain text body), raw_body, frontmatter metadata,
      created_at, modified_at.
    """
    path = Path(path)
    post = frontmatter.load(str(path))

    raw_body: str = post.content
    metadata: dict = dict(post.metadata)

    title = metadata.get("title") or _extract_title(raw_body) or path.stem
    created_at = str(metadata.get("created", metadata.get("date", "")))
    modified_at = str(metadata.get("updated", metadata.get("modified", "")))

    return {
        "title": title,
        "raw_body": raw_body,
        "content": _strip_markdown(raw_body),
        "frontmatter": metadata,
        "created_at": created_at,
        "modified_at": modified_at,
    }


def _extract_title(body: str) -> str | None:
    """Return text of first H1 line, if present."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _strip_markdown(text: str) -> str:
    """
    Lightweight Markdown → plain text for embedding.
    Removes headings markers, bold/italic, links, code fences.
    Not a full renderer — good enough for semantic embedding.
    """
    import re

    text = re.sub(r"```[\s\S]*?```", " ", text)   # fenced code blocks
    text = re.sub(r"`[^`]+`", " ", text)            # inline code
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)    # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links → label
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)  # wiki-links → title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)  # bold/italic
    text = re.sub(r"#\w+", " ", text)               # hashtags
    text = re.sub(r"\s+", " ", text).strip()
    return text
