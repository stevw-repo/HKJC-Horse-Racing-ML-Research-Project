"""English NLP track (M4): comment-on-running -> lagged structured signals.

Rules + lexicon (interpretable flags) + MiniLM anchor-similarity embeddings, enforced
**lagged** (`text_event_time < race_off_time`, PLAN.md §1C) by the feature builder.
"""

from __future__ import annotations

from hkjc.features.nlp.build import build_comment_features, comment_features_path
from hkjc.features.nlp.encode import NlpEncoder

__all__ = ["NlpEncoder", "build_comment_features", "comment_features_path"]
