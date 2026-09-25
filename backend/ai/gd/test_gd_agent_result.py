"""
Test GD Agent Result Contract
"""

from backend.ai.gd.gd_agent_result import (
    GDAgentResult,
)


def main():

    print("=" * 70)
    print("GD AGENT RESULT CONTRACT TEST")
    print("=" * 70)

    result = GDAgentResult(
        participant_id="P001",
        agent="relevance",
        score=82.0,

        reasoning=(
            "The participant addressed the GD topic "
            "and provided relevant points."
        ),

        strengths=[
            "Stayed on topic",
            "Provided a relevant example",
        ],

        weaknesses=[
            "Could provide more supporting evidence",
        ],

        evidence=[
            "technology can improve education",
        ],

        metadata={
            "model": "test",
        },
    )

    print()
    print("PARTICIPANT:")
    print(result.participant_id)

    print()
    print("AGENT:")
    print(result.agent)

    print()
    print("SCORE:")
    print(result.score)

    print()
    print("REASONING:")
    print(result.reasoning)

    print()
    print("STRENGTHS:")
    print(result.strengths)

    print()
    print("WEAKNESSES:")
    print(result.weaknesses)

    print()
    print("EVIDENCE:")
    print(result.evidence)

    result_dict = result.to_dict()

    print()
    print("=" * 70)
    print("DICTIONARY OUTPUT")
    print("=" * 70)

    print(result_dict)

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert result_dict["participant_id"] == "P001"
    assert result_dict["agent"] == "relevance"
    assert result_dict["score"] == 82.0

    assert len(result_dict["strengths"]) == 2

    assert len(result_dict["weaknesses"]) == 1

    assert len(result_dict["evidence"]) == 1

    assert result_dict["metadata"]["model"] == "test"

    print()
    print("=" * 70)
    print("ALL GD AGENT RESULT ASSERTIONS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()