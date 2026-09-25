import pytest
from unittest.mock import patch

from backend.ai.gd.agents import (
    CoherenceAgent,
    FluencyAgent,
    ParticipationAgent,
    RelevanceAgent,
)
from backend.ai.gd.gd_agent_input import GDParticipantInput


MOCK_LLM_RESPONSE = """
{
    "score": 82,
    "reasoning": "The participant provides a relevant and understandable contribution.",
    "strengths": [
        "Relevant ideas",
        "Clear contribution"
    ],
    "weaknesses": [
        "Some ideas could be developed further"
    ],
    "evidence": [
        "The participant connects their point to the GD topic."
    ]
}
"""


def create_test_input():
    return GDParticipantInput(
        session_id="GD001",
        participant_id="P001",
        topic="Importance of traditional food",
        participant_transcript=(
            "Traditional food represents culture and "
            "helps preserve regional identity."
        ),
        group_transcript=(
            "[1.0-4.0] P001: Traditional food represents culture "
            "and helps preserve regional identity."
        ),
        features={
            "speaking_time": 3.0,
            "speaking_ratio": 0.05,
            "word_count": 11,
            "words_per_minute": 220.0,
            "turn_count": 1,
            "average_turn_duration": 3.0,
            "longest_turn": 3.0,
            "shortest_turn": 3.0,
        },
        interaction_features={
            "first_speaking_time": 1.0,
            "last_speaking_time": 4.0,
            "responses": 0,
            "overlap_events": 0,
            "other_speakers_before": 0,
            "other_speakers_after": 1,
            "average_gap_between_turns": 0.0,
        },
        group_segments=[],
        other_participants=["P002", "P003"],
    )


def run_agent_test(agent):
    participant_input = create_test_input()

    with patch.object(
        agent,
        "call_llm",
        return_value=MOCK_LLM_RESPONSE
    ):
        result = agent.evaluate(participant_input)

    assert result.participant_id == "P001"
    assert result.agent == agent.agent_name
    assert result.score == 82.0
    assert isinstance(result.reasoning, str)
    assert isinstance(result.strengths, list)
    assert isinstance(result.weaknesses, list)
    assert isinstance(result.evidence, list)

    print(f"{agent.agent_name}: PASSED")
    print(f"  Score: {result.score}")
    print(f"  Reasoning: {result.reasoning}")


@pytest.mark.parametrize(
    "agent_class",
    [
        RelevanceAgent,
        CoherenceAgent,
        FluencyAgent,
        ParticipationAgent,
    ],
)
def test_agent(agent_class):
    agent = agent_class()
    run_agent_test(agent)


def main():

    print("=" * 60)
    print("GD AGENT ARCHITECTURE TEST")
    print("=" * 60)

    agents = [
        RelevanceAgent(),
        CoherenceAgent(),
        FluencyAgent(),
        ParticipationAgent(),
    ]

    for agent in agents:
        run_agent_test(agent)

    print("=" * 60)
    print("ALL GD AGENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
    