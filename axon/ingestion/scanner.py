from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _note_id(path: Path, vault_root: Path) -> str:
    """Stable ID: SHA256 of the path relative to vault root."""
    rel = str(path.relative_to(vault_root))
    return hashlib.sha256(rel.encode()).hexdigest()


def scan_vault(vault_path: Path) -> Iterator[dict]:
    """Yield one record per .md file with id, path, and content_hash."""
    for md_file in sorted(vault_path.rglob("*.md")):
        if any(part.startswith(".") for part in md_file.parts):
            continue  # skip hidden dirs like .obsidian
        yield {
            "id": _note_id(md_file, vault_path),
            "path": str(md_file),
            "content_hash": _content_hash(md_file),
        }
