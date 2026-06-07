"""Load guideline markdown files with YAML frontmatter."""

from pathlib import Path

import yaml


def load_guideline(path: Path) -> tuple[dict, str]:
    """Load a single guideline file.

    Returns:
        (frontmatter_dict, body_markdown_text)

    Raises:
        ValueError: if the file has no frontmatter delimiter.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name} has no YAML frontmatter")

    # Split into: leading empty string, frontmatter YAML, body
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path.name} frontmatter not properly delimited")

    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].lstrip()
    return frontmatter, body


def load_all_guidelines(guidelines_dir: Path) -> list[dict]:
    """Load every .md file in the guidelines directory.

    Returns a list of dicts: {"file", "frontmatter", "body"}.
    """
    documents = []
    for md_path in sorted(guidelines_dir.glob("*.md")):
        fm, body = load_guideline(md_path)
        documents.append({
            "file": md_path.name,
            "frontmatter": fm,
            "body": body,
        })
    return documents
