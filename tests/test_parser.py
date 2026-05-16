import textwrap
import tempfile
from pathlib import Path

from axon.parser.markdown import parse_note, _extract_title, _strip_markdown
from axon.parser.extractor import extract


def _write_note(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def test_extract_title_from_h1():
    assert _extract_title("# My Note\nSome text") == "My Note"


def test_extract_title_none():
    assert _extract_title("No heading here") is None


def test_strip_markdown_removes_wiki_links():
    result = _strip_markdown("See [[Other Note]] for details")
    assert "[[" not in result
    assert "Other Note" in result


def test_strip_markdown_removes_code_blocks():
    result = _strip_markdown("```python\nprint('hi')\n```\nafter")
    assert "print" not in result
    assert "after" in result


def test_parse_note_frontmatter():
    content = textwrap.dedent("""\
        ---
        title: Test Note
        tags: [ai, memory]
        ---
        # Test Note
        Body text here.
    """)
    p = _write_note(content)
    note = parse_note(p)
    assert note["title"] == "Test Note"
    assert "Body text here" in note["content"]


def test_extract_wiki_links():
    body = "See [[Alpha]] and [[Beta|display]] for more."
    result = extract(body, {})
    assert "Alpha" in result.wiki_links
    assert "Beta" in result.wiki_links


def test_extract_tags_inline():
    body = "This is #important and #ai/ml related."
    result = extract(body, {})
    assert "important" in result.tags
    assert "ai/ml" in result.tags


def test_extract_tags_frontmatter():
    result = extract("No inline tags.", {"tags": ["python", "memory"]})
    assert "python" in result.tags
    assert "memory" in result.tags


def test_extract_deduplicates():
    body = "#ai some text #ai"
    result = extract(body, {"tags": ["ai"]})
    assert result.tags.count("ai") == 1
