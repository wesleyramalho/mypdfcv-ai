"""Confidence score for a tailored bullet.

A bounded [0, 1] score made of three components:

- citation_density (0.5 weight): fraction of "concrete claims" (numbers,
  years, capitalised terms) in the bullet that are matched in cited facts.
- retrieval_strength (0.3 weight): max retrieval similarity of cited facts
  to the bullet text — proxy for "are these facts actually relevant?".
- keyword_coverage (0.2 weight): Jaccard overlap of bullet tokens with the
  JD requirements text — proxy for "does this earn the bullet?".

The weights are intentionally simple. A v2 would tune them against the eval
set, but for the demo what matters is that the score correlates with
groundedness and surfaces ungrounded outputs.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "by", "for",
    "with", "as", "is", "was", "are", "were", "be", "been", "this", "that",
    "i", "we", "they", "it", "our",
}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS}


def citation_density(bullet: str, cited_texts: list[str]) -> float:
    if not cited_texts:
        return 0.0
    haystack = " ".join(cited_texts).lower()
    claims = set(_NUMBER_RE.findall(bullet)) | {p for p in _PROPER_RE.findall(bullet) if p.lower() not in _STOPWORDS}
    if not claims:
        # Bullet is purely qualitative — give partial credit if tokens overlap.
        return 0.6 if _tokens(bullet) & _tokens(haystack) else 0.3
    supported = sum(1 for c in claims if c.lower() in haystack)
    return supported / len(claims)


def keyword_coverage(bullet: str, jd_text: str) -> float:
    b = _tokens(bullet)
    j = _tokens(jd_text)
    if not b or not j:
        return 0.0
    return len(b & j) / len(b | j)


def confidence_score(
    bullet: str,
    cited_texts: list[str],
    retrieval_scores: list[float],
    jd_text: str,
) -> float:
    density = citation_density(bullet, cited_texts)
    retrieval = max(retrieval_scores) if retrieval_scores else 0.0
    # Dense scores are in [-1, 1]; clamp the lower half to 0 so a marginal
    # match doesn't subtract confidence.
    retrieval = max(0.0, min(1.0, retrieval))
    coverage = keyword_coverage(bullet, jd_text)
    raw = 0.5 * density + 0.3 * retrieval + 0.2 * coverage
    return round(max(0.0, min(1.0, raw)), 3)
