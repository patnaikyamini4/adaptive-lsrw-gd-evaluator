"""
Test GD Agent Input Contract
"""

from backend.ai.gd.gd_agent_input import (
    GDParticipantInput,
)


def main():

    print("=" * 70)
    print("GD AGENT INPUT CONTRACT TEST")
    print("=" * 70)

    agent_input = GDParticipantInput(
        session_id="GD001",
        participant_id="P001",

        topic="Technology in Education",

        participant_transcript=(
            "I think technology can improve education."
        ),

        group_transcript=(
            "[P001] I think technology can improve education.\n"
            "[P002] I agree with that point.\n"
            "[P003] Technology also creates new opportunities."
        ),

        features={
            "speaking_time": 15.4,
            "speaking_ratio": 0.2567,
            "word_count": 32,
            "turn_count": 4,
            "words_per_minute": 124.68,
            "average_turn_duration": 3.9,
            "longest_turn": 6.8,
            "shortest_turn": 2.0,
        },

        interaction_features={
            "turn_count": 2,
            "responses": 1,
            "overlap_events": 0,
            "other_speakers_before": 1,
            "other_speakers_after": 2,
        },

        group_segments=[
            {
                "participant_id": "P001",
                "start": 2.0,
                "end": 5.0,
                "text": (
                    "I think technology can improve education."
                ),
            },
            {
                "participant_id": "P002",
                "start": 6.0,
                "end": 8.0,
                "text": "I agree with that point.",
            },
        ],

        other_participants=[
            "P002",
            "P003",
        ],

        metadata={
            "session_duration": 60.0,
        },
    )

    print()
    print("SESSION:")
    print(agent_input.session_id)

    print()
    print("PARTICIPANT:")
    print(agent_input.participant_id)

    print()
    print("TOPIC:")
    print(agent_input.topic)

    print()
    print("PARTICIPANT TRANSCRIPT:")
    print(agent_input.participant_transcript)

    print()
    print("FEATURES:")
    print(agent_input.features)

    print()
    print("INTERACTION FEATURES:")
    print(agent_input.interaction_features)

    print()
    print("OTHER PARTICIPANTS:")
    print(agent_input.other_participants)

    print()
    print("=" * 70)
    print("DICTIONARY OUTPUT")
    print("=" * 70)

    result = agent_input.to_dict()

    print()
    print(result)

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert result["session_id"] == "GD001"
    assert result["participant_id"] == "P001"
    assert result["topic"] == "Technology in Education"

    assert (
        result["participant_transcript"]
        == "I think technology can improve education."
    )

    assert result["features"]["word_count"] == 32

    assert (
        result["interaction_features"]["responses"]
        == 1
    )

    assert (
        result["interaction_features"]["overlap_events"]
        == 0
    )

    assert len(result["group_segments"]) == 2

    assert result["other_participants"] == [
        "P002",
        "P003",
    ]

    print()
    print("=" * 70)
    print("ALL GD AGENT INPUT ASSERTIONS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()