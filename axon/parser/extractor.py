from __future__ import annotations

import re
from typing import NamedTuple


class NoteExtract(NamedTuple):
    wiki_links: list[str]   # raw titles from [[...]]
    tags: list[str]          # without leading #


_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]")
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_/-]*)")


def extract(raw_body: str, frontmatter: dict) -> NoteExtract:
    wiki_links = _unique(_WIKI_LINK_RE.findall(raw_body))
    inline_tags = _TAG_RE.findall(raw_body)
    fm_tags = _normalise_fm_tags(frontmatter.get("tags", []))
    tags = _unique(inline_tags + fm_tags)
    return NoteExtract(wiki_links=wiki_links, tags=tags)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _normalise_fm_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(t).lstrip("#") for t in value]
    if isinstance(value, str):
        return [t.lstrip("#") for t in value.split()]
    return []
