from backend.ai.gd.agents import RelevanceAgent
from backend.ai.gd.gd_agent_input import GDParticipantInput


def main():

    print("=" * 60)
    print("REAL GD QWEN AGENT TEST")
    print("=" * 60)

    participant_input = GDParticipantInput(
        session_id="GD_TEST_001",
        participant_id="P001",
        topic="Importance of communication skills",
        participant_transcript=(
            "Communication skills are important because "
            "they help students express their ideas clearly "
            "and work effectively with others."
        ),
        group_transcript=(
            "[1.0-5.0] P001: Communication skills are important "
            "because they help students express their ideas "
            "clearly and work effectively with others."
        ),
        features={
            "speaking_time": 4.0,
            "speaking_ratio": 0.10,
            "word_count": 23,
            "words_per_minute": 345.0,
            "turn_count": 1,
            "average_turn_duration": 4.0,
            "longest_turn": 4.0,
            "shortest_turn": 4.0,
        },
        interaction_features={
            "first_speaking_time": 1.0,
            "last_speaking_time": 5.0,
            "responses": 0,
            "overlap_events": 0,
            "other_speakers_before": 0,
            "other_speakers_after": 1,
            "average_gap_between_turns": 0.0,
        },
        group_segments=[],
        other_participants=["P002"],
    )

    agent = RelevanceAgent()

    result = agent.evaluate(
        participant_input
    )

    print("\n===== REAL QWEN RESULT =====")

    print("Agent:")
    print(result.agent)

    print("Participant:")
    print(result.participant_id)

    print("Score:")
    print(result.score)

    print("Reasoning:")
    print(result.reasoning)

    print("Strengths:")
    print(result.strengths)

    print("Weaknesses:")
    print(result.weaknesses)

    print("Evidence:")
    print(result.evidence)

    print("\nREAL GD QWEN AGENT TEST: PASSED")


if __name__ == "__main__":
    main()