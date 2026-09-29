from backend.ai.semantic_agent import (
    MODEL_NAME,
    calculate_semantic_similarity,
    evaluate_semantic_similarity,
)


def test_semantic_similarity_related_higher_than_unrelated():
    golden = "Online education provides flexibility for students."
    candidate_related = (
        "Students can study online from different locations "
        "with flexible schedules."
    )
    candidate_unrelated = "The weather is pleasant today."

    sim_related = calculate_semantic_similarity(golden, candidate_related)
    sim_unrelated = calculate_semantic_similarity(golden, candidate_unrelated)

    assert sim_related > sim_unrelated
    assert sim_related > 0.5
    assert sim_unrelated < 0.2


def test_evaluate_semantic_similarity_result_structure():
    golden = "Online education provides flexibility for students."
    candidate = "Students can study online from different locations with flexible schedules."

    result = evaluate_semantic_similarity(golden, candidate)

    assert result["model"] == MODEL_NAME
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
