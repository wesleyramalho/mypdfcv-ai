from mypdfcv_ai.grounding.citations import verify_grounding
from mypdfcv_ai.grounding.confidence import (
    citation_density,
    confidence_score,
    keyword_coverage,
)


def test_grounding_rejects_invented_percent():
    facts = ["Migrated monolith to microservices, reducing p99 latency from 850ms to 410ms."]
    bullet = "Cut latency by 52% via microservices migration."
    check = verify_grounding(bullet, facts)
    assert not check.grounded
    assert "52" in " ".join(check.unsupported_terms)


def test_grounding_accepts_supported_percent():
    facts = ["Lifted checkout CTR by 9.4% in A/B test."]
    bullet = "Lifted checkout CTR by 9.4%."
    check = verify_grounding(bullet, facts)
    assert check.grounded


def test_grounding_rejects_empty_bullet_even_with_citations():
    facts = ["Built ETL pipelines at Acme."]
    check = verify_grounding("", facts)
    assert not check.grounded


def test_grounding_rejects_trivially_short_bullet():
    facts = ["Built ETL pipelines at Acme."]
    check = verify_grounding("ok.", facts)
    assert not check.grounded


def test_grounding_rejects_invented_employer():
    facts = ["At Acme Logistics, built ETL pipelines."]
    bullet = "At Globex Corporation, built ETL pipelines."
    check = verify_grounding(bullet, facts)
    assert not check.grounded
    assert any("Globex" in t for t in check.unsupported_terms)


def test_grounding_no_citations_is_ungrounded():
    check = verify_grounding("Some text.", [])
    assert not check.grounded


def test_confidence_high_when_grounded_and_relevant():
    bullet = "At Acme Logistics, cut p99 latency by 52% migrating to FastAPI."
    cited = ["At Acme Logistics, migrated to FastAPI and cut p99 latency by 52%."]
    jd = "We need FastAPI experience and latency optimisation."
    score = confidence_score(bullet, cited, retrieval_scores=[0.82], jd_text=jd)
    assert score > 0.5


def test_confidence_low_when_no_citations():
    score = confidence_score("anything", cited_texts=[], retrieval_scores=[], jd_text="anything")
    assert score < 0.4


def test_keyword_coverage_basic():
    cov = keyword_coverage("Python and FastAPI", "Python developer needed with FastAPI")
    assert 0.0 < cov <= 1.0


def test_citation_density_with_numbers():
    bullet = "Cut latency by 52%."
    cited = ["Cut latency by 52% during the rewrite."]
    assert citation_density(bullet, cited) == 1.0
