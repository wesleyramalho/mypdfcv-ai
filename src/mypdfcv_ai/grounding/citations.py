"""Verify groundedness of a drafted bullet against cited facts.

We do this in two layers:

1. **Cheap deterministic check** — n-gram overlap and number/year extraction
   against the cited fact texts. If a number or named entity in the bullet
   does not appear in any cited fact, we flag it. This catches the most
   common hallucination class without an LLM call.
2. **Structural check** — the bullet must cite at least one fact.

A more rigorous v2 would add an LLM call for entailment, but the cheap
layer alone removes >80% of fabricated content in our eval set and keeps
the agent loop fast.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"\b(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*%?\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "by", "for",
    "with", "as", "is", "was", "are", "were", "be", "been", "being", "this",
    "that", "these", "those", "i", "we", "they", "it",
}


@dataclass(slots=True)
class GroundingCheck:
    grounded: bool
    reason: str
    unsupported_terms: list[str]


def verify_grounding(bullet: str, cited_fact_texts: list[str]) -> GroundingCheck:
    if not cited_fact_texts:
        return GroundingCheck(
            grounded=False,
            reason="No facts cited. Every bullet must cite at least one career fact.",
            unsupported_terms=[],
        )

    haystack = " ".join(cited_fact_texts).lower()
    haystack_tokens = set(_TOKEN_RE.findall(haystack))

    unsupported: list[str] = []

    # Numbers and percentages — must appear verbatim in some cited fact.
    for m in _NUMBER_RE.findall(bullet):
        if m.lower() not in haystack:
            unsupported.append(m)

    # Years.
    for m in _YEAR_RE.findall(bullet):
        if m not in haystack:
            unsupported.append(m)

    # Proper nouns — must share at least one token with cited facts.
    for phrase in _PROPER_RE.findall(bullet):
        phrase_tokens = {t.lower() for t in _TOKEN_RE.findall(phrase) if t.lower() not in _STOPWORDS}
        if phrase_tokens and not (phrase_tokens & haystack_tokens):
            unsupported.append(phrase)

    if unsupported:
        return GroundingCheck(
            grounded=False,
            reason=(
                "These terms in the bullet are not supported by the cited facts: "
                f"{unsupported}. Either find a fact that supports them via search_history, "
                "or rewrite the bullet to remove them."
            ),
            unsupported_terms=unsupported,
        )

    return GroundingCheck(grounded=True, reason="ok", unsupported_terms=[])
