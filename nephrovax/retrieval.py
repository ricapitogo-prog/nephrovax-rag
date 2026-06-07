"""Embedding and FAISS-backed vector retrieval.

Supports two retrieval modes:
  - single: top-K chunks across all vaccine classes (default)
  - multi:  top-K_per_class chunks from each named vaccine class, merged

The mode is decided by a lightweight LLM classifier that runs before
retrieval. Designed to fail safe — any classifier error falls back to
single mode.
"""

import json

import faiss
import numpy as np
import voyageai
import anthropic

from . import config
from .chunker import chunk_to_embed_text


# Valid vaccine classes the classifier may return.
# Anything outside this set is treated as a classifier error.
VALID_VACCINE_CLASSES = {"pneumococcal", "meningococcal", "hib"}


CLASSIFIER_SYSTEM_PROMPT = """You classify clinical questions about vaccination \
for patients on complement inhibitor therapy.

The guideline covers three vaccine classes: pneumococcal, meningococcal, hib.

For each question, decide:
1. Whether the question is about a SINGLE vaccine class or MULTIPLE classes
2. Which classes are relevant

A question is multi-vaccine when it:
- Asks about a "complete vaccination plan", "all vaccines", or "what to give" \
for a new patient with no prior immunizations
- Explicitly mentions multiple vaccine classes
- Asks for prophylaxis or full immunization for a patient starting complement \
inhibitor therapy (which typically requires all three classes)

A question is single-vaccine when it mentions or implies just one class:
- Names a specific vaccine product (e.g., PCV13, Menveo, ActHIB) → that class only
- Mentions one disease (pneumococcal, meningococcal, Hib) → that class only

Output ONLY valid JSON in this exact shape, with no additional text:
{"mode": "single", "classes": ["pneumococcal"]}
or
{"mode": "multi", "classes": ["pneumococcal", "meningococcal", "hib"]}

The "classes" list must contain only values from: ["pneumococcal", \
"meningococcal", "hib"]. If unsure, prefer "multi" with all three classes \
(better to over-retrieve than miss).
"""


class Retriever:
    """Embeds chunks once on construction, then serves retrieval queries.

    Supports both single-mode (top-K across all chunks) and multi-mode
    (top-K per vaccine class). The mode is chosen by a Claude classifier.
    """

    def __init__(
        self,
        chunks: list[dict],
        voyage_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.chunks = chunks
        self._voyage = voyageai.Client(api_key=voyage_api_key)
        self._claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self._build_index()
        self._build_class_indices()

    def _build_index(self):
        """Embed every chunk and store the FAISS index over the full corpus."""
        embed_texts = [chunk_to_embed_text(c) for c in self.chunks]
        result = self._voyage.embed(
            embed_texts,
            model=config.EMBEDDING_MODEL,
            input_type="document",
        )
        embeddings = np.array(result.embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self._embeddings = embeddings  # kept for per-class search
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _build_class_indices(self):
        """For each vaccine class, precompute the list of chunk indices.

        This lets us filter cheaply in multi-mode without rebuilding a
        FAISS index per class.
        """
        self._class_to_chunk_indices = {}
        for vc in VALID_VACCINE_CLASSES:
            self._class_to_chunk_indices[vc] = [
                i for i, c in enumerate(self.chunks)
                if c["vaccine_class"] == vc
            ]

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = config.DEFAULT_K) -> list[dict]:
        """Single-mode retrieval: top-K chunks across the full corpus.

        This is the unchanged behavior — kept for direct use and as the
        fallback for classification errors.
        """
        q_emb = self._embed_query(query)
        scores, idxs = self.index.search(q_emb, k)
        return self._format_hits(scores[0], idxs[0])

    def retrieve_per_class(
        self,
        query: str,
        classes: list[str],
        k_per_class: int = 3,
    ) -> list[dict]:
        """Multi-mode retrieval: top-K_per_class chunks from each named class.

        Results are merged across classes and re-sorted by score, so the
        returned list still reflects similarity ranking.
        """
        # Validate class names — silently drop anything unknown
        valid_classes = [c for c in classes if c in VALID_VACCINE_CLASSES]
        if not valid_classes:
            # Bad input — fall back to single mode
            return self.retrieve(query, k=k_per_class * 3)

        q_emb = self._embed_query(query)
        all_hits = []

        for vc in valid_classes:
            class_idx = self._class_to_chunk_indices[vc]
            if not class_idx:
                continue
            # Pull the embedding sub-matrix for this class
            class_emb = self._embeddings[class_idx]  # shape (N_class, dim)
            # Compute inner product against the query
            sims = (class_emb @ q_emb.T).flatten()  # shape (N_class,)
            # Top k_per_class
            top_within_class = np.argsort(-sims)[:k_per_class]
            for local_pos in top_within_class:
                global_idx = class_idx[local_pos]
                all_hits.append((float(sims[local_pos]), global_idx))

        # Sort merged results by score (descending) so highest-confidence first
        all_hits.sort(key=lambda x: -x[0])
        scores = np.array([h[0] for h in all_hits])
        idxs = np.array([h[1] for h in all_hits])
        return self._format_hits(scores, idxs)

    def retrieve_smart(self, query: str, k_per_class: int = 3, k_single: int = config.DEFAULT_K) -> dict:
        """Classify the query, then route to single or multi retrieval.

        Returns a dict with:
          - hits: the retrieved chunks (list)
          - mode: "single" or "multi"
          - classes: vaccine classes used (only meaningful in multi mode)
          - classifier_error: optional string if classification failed

        The 'mode' and 'classes' fields are returned so the UI can show
        users when multi-vaccine retrieval was triggered.
        """
        classification = self._classify_query(query)

        if classification.get("error"):
            # Classifier failed — fall back to single mode, surface the error
            hits = self.retrieve(query, k=k_single)
            return {
                "hits": hits,
                "mode": "single",
                "classes": [],
                "classifier_error": classification["error"],
            }

        mode = classification["mode"]
        classes = classification["classes"]

        if mode == "multi":
            hits = self.retrieve_per_class(query, classes, k_per_class=k_per_class)
        else:
            hits = self.retrieve(query, k=k_single)

        return {
            "hits": hits,
            "mode": mode,
            "classes": classes,
            "classifier_error": None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string, return normalized (1, dim) array."""
        q_result = self._voyage.embed(
            [query],
            model=config.EMBEDDING_MODEL,
            input_type="query",
        )
        q_emb = np.array(q_result.embeddings, dtype=np.float32)
        faiss.normalize_L2(q_emb)
        return q_emb

    def _format_hits(self, scores, idxs) -> list[dict]:
        """Convert FAISS-style scores+idxs into the standard hit dict format."""
        results = []
        for score, idx in zip(scores, idxs):
            chunk = self.chunks[int(idx)]
            results.append({
                "doc_id": chunk["doc_id"],
                "vaccine_class": chunk["vaccine_class"],
                "breadcrumb": chunk["breadcrumb"],
                "breadcrumb_str": chunk["breadcrumb_str"],
                "text": chunk["text"],
                "score": float(score),
            })
        return results

    def _classify_query(self, query: str) -> dict:
        """Ask Claude whether the query is single- or multi-vaccine.

        Returns a dict shaped like:
          {"mode": "single"|"multi", "classes": [...]}
        Or on any error:
          {"error": "<reason>"}

        Designed to fail safe — caller treats errors as "fall back to single".
        """
        try:
            response = self._claude.messages.create(
                model="claude-haiku-4-5",  # cheap+fast for classification
                max_tokens=200,
                system=CLASSIFIER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            raw = response.content[0].text.strip()

            # Tolerate code-fence wrapping if Claude adds one despite instructions
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()

            parsed = json.loads(raw)

            mode = parsed.get("mode")
            classes = parsed.get("classes", [])

            if mode not in ("single", "multi"):
                return {"error": f"invalid mode: {mode!r}"}
            if not isinstance(classes, list):
                return {"error": "classes is not a list"}
            # Filter to valid classes only — silently drop hallucinations
            classes = [c for c in classes if c in VALID_VACCINE_CLASSES]
            if not classes:
                return {"error": "no valid classes in classifier output"}

            return {"mode": mode, "classes": classes}

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failure: {e}"}
        except Exception as e:
            # Catch-all so retrieval never breaks because of classifier issues
            return {"error": f"{type(e).__name__}: {e}"}
