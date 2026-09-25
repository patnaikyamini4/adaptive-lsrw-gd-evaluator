from backend.ai.gd.gd_agent_result import GDAgentResult
from backend.ai.gd.gd_scorer import GDScorer


def build_results(
    scores=None,
    agents=None,
):
    scores = scores or [80, 90, 70, 60]
    agents = agents or [
        "relevance",
        "coherence",
        "fluency",
        "participation",
    ]

    return [
        GDAgentResult(
            participant_id="P001",
            agent=agent,
            score=score,
        )
        for agent, score in zip(agents, scores)
    ]


def test_valid_four_agent_result():
    result = GDScorer().calculate(build_results())

    assert result["final_score"] == 75.0
    assert result["scores"] == {
        "relevance": 80.0,
        "coherence": 90.0,
        "fluency": 70.0,
        "participation": 60.0,
    }


def test_rejects_duplicate_agent():
    results = build_results(
        agents=[
            "relevance",
            "relevance",
            "fluency",
            "participation",
        ]
    )

    try:
        GDScorer().calculate(results)
    except ValueError as exc:
        assert str(exc) == "Duplicate agent results: ['relevance']"
    else:
        raise AssertionError("Expected duplicate agent rejection")


def test_rejects_missing_agent():
    results = build_results()[:-1]

    try:
        GDScorer().calculate(results)
    except ValueError as exc:
        assert str(exc) == "Missing agent results: ['participation']"
    else:
        raise AssertionError("Expected missing agent rejection")


def test_rejects_unexpected_agent():
    results = build_results(
        agents=["relevance", "coherence", "fluency", "unsupported"]
    )

    try:
        GDScorer().calculate(results)
    except ValueError as exc:
        assert str(exc) == "Unexpected agent results: ['unsupported']"
    else:
        raise AssertionError("Expected unexpected agent rejection")


def test_rejects_invalid_scores():
    cases = [
        (-1, "Score for relevance must be between 0 and 100"),
        (101, "Score for relevance must be between 0 and 100"),
        (float("nan"), "Score for relevance must be finite"),
        (float("inf"), "Score for relevance must be finite"),
        (float("-inf"), "Score for relevance must be finite"),
        ("80", "Score for relevance must be numeric"),
    ]

    for invalid_score, message in cases:
        results = build_results(scores=[invalid_score, 90, 70, 60])

        try:
            GDScorer().calculate(results)
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError(
                f"Expected score rejection for {invalid_score!r}"
            )


def main():

    results = build_results()

    scorer = GDScorer()

    result = scorer.calculate(results)

    print("=" * 60)
    print("GD SCORER TEST")
    print("=" * 60)

    print("Scores:")
    print(result["scores"])

    print("\nWeights:")
    print(result["weights"])

    print("\nFinal score:")
    print(result["final_score"])

    assert result["final_score"] == 75.0

    print("\nGD SCORER TEST: PASSED")


if __name__ == "__main__":
    main()
