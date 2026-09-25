from unittest.mock import patch

from backend.ai.gd.gd_agent_input import GDParticipantInput
from backend.ai.gd.gd_orchestrator import GDOrchestrator


MOCK_LLM_RESPONSE = """
{
    "score": 80,
    "reasoning": "The participant makes a clear contribution.",
    "strengths": [
        "Clear contribution"
    ],
    "weaknesses": [
        "Could provide more detail"
    ],
    "evidence": [
        "The participant provides a relevant point."
    ]
}
"""


def create_input():

    return GDParticipantInput(
        session_id="GD001",
        participant_id="P001",
        topic="Importance of traditional food",
        participant_transcript=(
            "Traditional food represents culture "
            "and regional identity."
        ),
        group_transcript=(
            "[1.0-4.0] P001: Traditional food represents "
            "culture and regional identity."
        ),
        features={
            "speaking_time": 3.0,
            "speaking_ratio": 0.05,
            "word_count": 8,
            "words_per_minute": 160.0,
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
        other_participants=["P002"],
    )


def main():

    print("=" * 60)
    print("GD ORCHESTRATOR TEST")
    print("=" * 60)

    orchestrator = GDOrchestrator()

    participant_input = create_input()

    with patch(
        "backend.ai.gd.agents.base_agent.GDBaseAgent.call_llm",
        return_value=MOCK_LLM_RESPONSE
    ):

        report = (
            orchestrator.evaluate_participant(
                participant_input
            )
        )

    print("\nParticipant:")
    print(report["participant_id"])

    print("\nAgent scores:")

    for result in report["agent_results"]:
        print(
            f"  {result['agent']}: "
            f"{result['score']}"
        )

    print("\nFinal score:")
    print(
        report["scorecard"]["final_score"]
    )

    assert len(report["agent_results"]) == 4

    assert (
        report["scorecard"]["final_score"]
        == 80.0
    )

    print("\nGD ORCHESTRATOR TEST: PASSED")


if __name__ == "__main__":
    main()