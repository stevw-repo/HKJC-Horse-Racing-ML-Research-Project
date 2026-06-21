"""Curated lexicon of HKJC comment-on-running phrases, grouped into interpretable signals.

These power the spaCy ``PhraseMatcher`` (case-insensitive). Each category becomes a lagged
count feature; the idea is the classic regression-to-mean signals: a horse that was *unlucky*
(blocked/checked) last start is underrated, one that *won easing* may be ahead of its mark,
one that *weakened* may be over its peak. Phrases are lower-cased substrings of real comments.
"""

from __future__ import annotations

LEXICON: dict[str, list[str]] = {
    "trouble": [
        "interfered",
        "interference",
        "crowded",
        "checked",
        "blocked",
        "no clear run",
        "denied a run",
        "denied the run",
        "steadied",
        "tightened",
        "hampered",
        "carried wide",
        "had to be checked",
        "lost ground",
        "awkwardly placed",
        "held up for a run",
        "no room",
        "shut out",
        "bumped",
        "made contact",
    ],
    "slow_start": [
        "slow to begin",
        "slow into stride",
        "began awkwardly",
        "missed the break",
        "slowly away",
        "fractious",
        "reared",
        "dwelt",
        "jumped awkwardly",
        "slow to stride",
    ],
    "ran_on": [
        "ran on",
        "kept on",
        "finished well",
        "finished strongly",
        "strong to the line",
        "stuck on",
        "kept on well",
        "ran on strongly",
        "rallied",
        "responded",
    ],
    "easing": [
        "easing",
        "eased down",
        "comfortably",
        "in hand",
        "untroubled",
        "won easily",
        "with something in hand",
        "going away",
        "cleverly",
        "as he pleased",
    ],
    "weakened": [
        "weakened",
        "faded",
        "tired",
        "gave way",
        "no extra",
        "found nothing",
        "dropped out",
        "could not quicken",
        "one paced",
        "failed to respond",
    ],
    "wide": [
        "wide",
        "three deep",
        "four deep",
        "caught wide",
        "raced wide",
        "wide throughout",
        "no cover",
        "exposed",
    ],
    "health": [
        "bled",
        "lame",
        "lost action",
        "pulled up",
        "distressed",
        "lost its action",
        "unbalanced",
        "stumbled",
        "struck into",
        "cardiac",
        "irregular",
    ],
}

# Anchor sentences for the sentence-transformer similarity features (semantic, beyond keywords).
ANCHORS: dict[str, str] = {
    "nlp_sim_trouble": (
        "the horse had a troubled or unlucky run, interfered with or denied a clear passage"
    ),
    "nlp_sim_easywin": (
        "the horse won or finished strongly with plenty in reserve and was eased down"
    ),
    "nlp_sim_noexcuse": (
        "the horse weakened and simply had no excuses, beaten on merit and not good enough"
    ),
}
