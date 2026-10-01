import pytest
from backend.services.speaking_scoring_service import SpeakingScoringService


@pytest.fixture
def scoring_service():
    return SpeakingScoringService()


def test_speaking_scoring_valid_positive_similarity(scoring_service):
    res = scoring_service.calculate_score(
        transcript="Online learning provides flexibility.",
        semantic_similarity=0.8520,
    )

    assert res["final_score"] == 85.2
    assert res["scoring_method"] == "deterministic_semantic_v1"
    assert res["version"] == "1.0"
    assert res["components"]["semantic"]["raw_similarity"] == 0.8520
    assert res["components"]["semantic"]["score"] == 85.2
    assert res["components"]["semantic"]["weight"] == 1.0


def test_speaking_scoring_medium_similarity(scoring_service):
    res = scoring_service.calculate_score(
        transcript="Some partially relevant response.",
        semantic_similarity=0.5000,
    )

    assert res["final_score"] == 50.0
    assert res["components"]["semantic"]["raw_similarity"] == 0.5
    assert res["components"]["semantic"]["score"] == 50.0


def test_speaking_scoring_negative_and_zero_similarity(scoring_service):
    # Negative cosine similarity (unrelated text)
    res_neg = scoring_service.calculate_score(
        transcript="Unrelated audio transcript.",
        semantic_similarity=-0.0628,
    )
    assert res_neg["final_score"] == 0.0
    assert res_neg["components"]["semantic"]["raw_similarity"] == -0.0628
    assert res_neg["components"]["semantic"]["score"] == 0.0

    # Zero similarity
    res_zero = scoring_service.calculate_score(
        transcript="Another text.",
        semantic_similarity=0.0,
    )
    assert res_zero["final_score"] == 0.0
    assert res_zero["components"]["semantic"]["score"] == 0.0


def test_speaking_scoring_upper_bound_clamp(scoring_service):
    # Exact 1.0
    res_one = scoring_service.calculate_score(
        transcript="Identical text.",
        semantic_similarity=1.0,
    )
    assert res_one["final_score"] == 100.0
    assert res_one["components"]["semantic"]["score"] == 100.0

    # Greater than 1.0 (clamping guard)
    res_over = scoring_service.calculate_score(
        transcript="Identical text.",
        semantic_similarity=1.05,
    )
    assert res_over["final_score"] == 100.0
    assert res_over["components"]["semantic"]["score"] == 100.0


def test_speaking_scoring_deterministic_output(scoring_service):
    res1 = scoring_service.calculate_score(
        transcript="Consistency test.",
        semantic_similarity=0.7412,
    )
    res2 = scoring_service.calculate_score(
        transcript="Consistency test.",
        semantic_similarity=0.7412,
    )
    assert res1 == res2
    assert res1["final_score"] == 74.12


def test_speaking_scoring_none_and_empty_inputs(scoring_service):
    res = scoring_service.calculate_score(
        transcript="",
        semantic_similarity=None,
        asr_result={},
    )
    assert res["final_score"] == 0.0
    assert 0.0 <= res["final_score"] <= 100.0
    assert res["components"]["semantic"]["raw_similarity"] == 0.0

