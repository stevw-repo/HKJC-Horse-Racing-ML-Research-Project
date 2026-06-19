"""Turn comment-on-running text into numeric NLP signals (PLAN.md §2 M4).

Two layers, per PLAN's stack:

* **rules + lexicon** -- a spaCy ``PhraseMatcher`` over :data:`lexicon.LEXICON` yields an
  interpretable count per category (trouble / slow_start / ran_on / easing / weakened / wide /
  health).
* **embeddings** -- MiniLM sentence embeddings, reduced to a few interpretable *anchor
  similarities* (semantic closeness to "troubled run" / "won easing" / "no excuse"), so the
  384-dim vector becomes a handful of ablatable features rather than an opaque block.

The encoder is the per-*run* signal; lagging (prior runs only) happens in the feature builder.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hkjc.features.nlp.lexicon import ANCHORS, LEXICON
from hkjc.models.base import FloatArray
from hkjc.models.device import gpu_available

EMBED_MODEL = "all-MiniLM-L6-v2"


class NlpEncoder:
    """Encode comments into lexicon-flag counts + MiniLM anchor similarities."""

    def __init__(self, use_embeddings: bool = True, use_gpu: bool | None = None) -> None:
        import spacy
        from spacy.matcher import PhraseMatcher

        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for category, phrases in LEXICON.items():
            self.matcher.add(category, [self.nlp.make_doc(p) for p in phrases])
        self.categories = list(LEXICON)
        self.anchor_names = list(ANCHORS)
        self.use_embeddings = use_embeddings
        self._use_gpu = gpu_available() if use_gpu is None else use_gpu
        self._model: object | None = None
        self._anchor_vecs: FloatArray | None = None

    @property
    def flag_names(self) -> list[str]:
        return [f"nlp_{c}" for c in self.categories]

    def feature_names(self) -> list[str]:
        return self.flag_names + (self.anchor_names if self.use_embeddings else [])

    def _flags(self, comment: str) -> list[int]:
        doc = self.nlp(comment.lower())
        counts: Counter[str] = Counter(
            self.nlp.vocab.strings[mid] for mid, _s, _e in self.matcher(doc)
        )
        return [counts.get(c, 0) for c in self.categories]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        device = "cuda" if self._use_gpu else "cpu"
        self._model = SentenceTransformer(EMBED_MODEL, device=device)
        self._anchor_vecs = self._embed([ANCHORS[name] for name in self.anchor_names])

    def _embed(self, texts: list[str]) -> FloatArray:
        assert self._model is not None
        vecs = self._model.encode(  # type: ignore[attr-defined]
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float64)

    def encode(self, comments: list[str]) -> dict[str, list[float]]:
        """Return one column per feature (flag counts + anchor similarities) for ``comments``."""
        flags = np.array([self._flags(c) for c in comments], dtype=np.float64).reshape(
            len(comments), -1
        )
        cols: dict[str, list[float]] = {
            name: flags[:, i].tolist() for i, name in enumerate(self.flag_names)
        }
        if self.use_embeddings and comments:
            self._ensure_model()
            assert self._anchor_vecs is not None
            sims = self._embed(comments) @ self._anchor_vecs.T  # cosine (normalised) -> (n, k)
            for i, name in enumerate(self.anchor_names):
                cols[name] = sims[:, i].tolist()
        elif self.use_embeddings:
            for name in self.anchor_names:
                cols[name] = []
        return cols
