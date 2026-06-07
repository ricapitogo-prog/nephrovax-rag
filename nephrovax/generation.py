"""Answer generation with Claude over retrieved guideline chunks.

Uses footnote-style citations: Claude references retrieved chunks by number
([1], [2], etc.) in the answer. The mapping from number to chunk is rendered
by the app, not embedded in the answer text.
"""

import anthropic

from . import config


SYSTEM_PROMPT = """You are a clinical decision support assistant for the NephroVax \
vaccination guideline. The guideline covers pneumococcal, meningococcal, and Hib \
vaccination for patients on complement inhibitor therapy.

You MUST:
- Answer ONLY using the retrieved guideline excerpts provided below.
- Use footnote-style citations to reference excerpts: write [1] to reference \
Excerpt 1, [2] for Excerpt 2, etc. Multiple references can be combined: [1, 3].
- Place each citation immediately after the claim it supports.
- If the retrieved context does not contain a clear answer for the question, say \
so explicitly. Do not invent recommendations or fall back on general medical \
knowledge.
- If the question requires information from multiple excerpts (e.g., a new patient \
needing pneumococcal AND meningococcal AND Hib vaccinations), use all relevant \
excerpts and cite each separately.
- This is for licensed healthcare providers. Use clinical terminology.

FORMATTING RULES (strict):
- DO NOT use markdown headings (#, ##, ###, ####) anywhere in your response.
- DO NOT use horizontal rules (---) to separate sections.
- For section labels, use **bold text** on its own line, e.g., \
"**Meningococcal vaccination.**" then continue with the content on the next line.
- Use markdown tables for structured comparisons (vaccine schedules, products, etc.).
- Use bulleted lists (- ) for itemized recommendations.
- Keep paragraphs concise. Prefer short, declarative sentences.

You MUST NOT:
- Use the format "[doc_id > section]" — use only the numeric format [1], [2], etc.
- Recommend specific dosing schedules that are not present in the retrieved excerpts.
- Generalize from one vaccine to another (e.g., do not assume Hib guidance based \
on pneumococcal guidance).
- Provide reassurance or estimates of risk; stay strictly within what the guidelines say.

End every response with this exact sentence as the final line:
"This is a research prototype for decision support. Clinical judgment and \
verification against current guidelines remain the responsibility of the \
treating clinician."
"""


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as numbered excerpts for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"[Excerpt {i}]\n"
            f"Section: {hit['breadcrumb_str']}\n\n"
            f"{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


class Generator:
    """Wraps the Claude client. Stateless beyond the API key."""

    def __init__(self, anthropic_api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)

    def answer(self, question: str, hits: list[dict]) -> str:
        """Generate an answer using the retrieved hits as context."""
        context = format_context(hits)
        user_msg = (
            f"Retrieved guideline excerpts:\n\n{context}\n\n"
            f"---\n\n"
            f"Clinical question: {question}"
        )

        response = self._client.messages.create(
            model=config.GENERATION_MODEL,
            max_tokens=config.MAX_GENERATION_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
