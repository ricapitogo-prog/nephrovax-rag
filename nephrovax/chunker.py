"""Structural chunking of guideline markdown.

Splits each guideline by heading hierarchy and emits one chunk per leaf
section (deepest heading with recommendation content). Each chunk gets
a natural-language context header derived from the breadcrumb, which
the embedder uses for stronger semantic matching.
"""

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def expand_age_band(age_str: str) -> str:
    """Convert structured age band strings into natural language.

    Embeddings handle natural-language age descriptions more reliably than
    symbol-heavy headings like 'Age ≥19 years'.
    """
    s = age_str.lower()
    if "≤5" in age_str or "<5" in s:
        return "young children aged 5 years or younger"
    if "<2" in s or "2–23 months" in s or "2-23 months" in s:
        return "infants younger than 2 years (2 to 23 months)"
    if "2–9" in age_str or "2-9" in s:
        return "children aged 2 to 9 years"
    if "6–18" in age_str or "6-18" in s:
        return "children and adolescents aged 6 to 18 years"
    if "6–19" in age_str or "6-19" in s:
        return "children and adolescents aged 6 to 19 years"
    if "≥10" in age_str or ">=10" in s:
        return "adolescents and adults aged 10 years or older"
    if "≥19" in age_str or ">=19" in s:
        return "adults aged 19 years or older"
    if "≥5" in age_str or ">=5" in s:
        return "patients aged 5 years or older"
    if "all ages" in s:
        return "patients of all ages"
    return age_str  # fallback for unrecognized patterns


def build_context_header(chunk: dict) -> str:
    """Build a natural-language sentence describing who/what the chunk applies to.

    This is prepended to the chunk text before embedding to give the embedder
    explicit semantic signal about pathway, prior vaccine, age, and population.
    """
    bc = chunk["breadcrumb"]
    vaccine = chunk["vaccine_class"]

    pathway_text = bc[0] if bc else ""
    if "Pathway A" in pathway_text:
        history = f"no or unknown prior {vaccine} vaccination"
    elif "Pathway B" in pathway_text:
        history = f"known prior {vaccine} vaccination"
    else:
        history = ""

    # Prior vaccine appears as an intermediate breadcrumb element in Pathway B
    prior_vaccine = ""
    for crumb in bc[1:-1]:
        if crumb.startswith("Prior vaccine:"):
            prior_vaccine = crumb.replace("Prior vaccine:", "").strip()
            break

    age_band = bc[-1] if bc else ""
    age_natural = expand_age_band(age_band)

    parts = [
        f"This recommendation applies to {age_natural}",
        "on complement inhibitor therapy",
    ]
    if history:
        parts.append(f"with {history}")
    if prior_vaccine:
        parts.append(f"(specifically prior {prior_vaccine})")

    return ", ".join(parts) + "."


def chunk_by_leaf(body: str, frontmatter: dict) -> list[dict]:
    """Walk the markdown by headings, emit one chunk per leaf section.

    A leaf is a heading at level 4 or 5 (the deepest in our schema) that has
    recommendation content beneath it. Each chunk carries:
      - frontmatter metadata (doc_id, vaccine_class, authority_tier, version)
      - breadcrumb path (list of heading texts from H3 downward)
      - text (the full content under the leaf)
    """
    lines = body.splitlines()
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, heading_text)
    current_text: list[str] = []
    leaf_open = False

    def flush():
        nonlocal leaf_open
        if not leaf_open:
            return
        breadcrumb = [t for (lvl, t) in stack if lvl >= 3]
        full_text = "\n".join(current_text).strip()
        if full_text and breadcrumb:
            chunks.append({
                "doc_id": frontmatter["doc_id"],
                "vaccine_class": frontmatter["vaccine_class"],
                "authority_tier": frontmatter["authority_tier"],
                "version": frontmatter["version"],
                "breadcrumb": breadcrumb,
                "breadcrumb_str": " > ".join(breadcrumb),
                "text": full_text,
            })
        leaf_open = False

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            current_text.clear()

            level = len(m.group(1))
            heading_text = m.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))

            if level >= 4:
                leaf_open = True
            continue

        if leaf_open:
            current_text.append(line)

    flush()
    return chunks


def chunk_to_embed_text(chunk: dict) -> str:
    """Build the string that gets embedded.

    Prepends a natural-language context header and the breadcrumb so the
    embedding represents both semantic and structural context.
    """
    context = build_context_header(chunk)
    return f"{context}\n\n{chunk['breadcrumb_str']}\n\n{chunk['text']}"


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Run chunking across all loaded documents.

    Returns a flat list of chunks across every guideline.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_by_leaf(doc["body"], doc["frontmatter"])
        all_chunks.extend(chunks)
    return all_chunks
