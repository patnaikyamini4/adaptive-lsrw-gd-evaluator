from backend.ai.gd.gd_agent_input import GDParticipantInput
from backend.ai.gd.gd_orchestrator import GDOrchestrator


def create_participant(
    participant_id,
    participant_transcript,
    speaking_time,
    speaking_ratio,
    word_count,
    turn_count,
    first_speaking_time,
    last_speaking_time,
    responses,
    overlap_events,
    other_speakers_before,
    other_speakers_after,
):
    """
    Create a GDParticipantInput for testing.
    """

    return GDParticipantInput(
        session_id="GD_TEST_001",
        participant_id=participant_id,

        topic=(
            "Should artificial intelligence be used "
            "in education?"
        ),

        participant_transcript=participant_transcript,

        group_transcript=(
            "[2.0-8.0] P001: Artificial intelligence can "
            "provide personalized learning and help students.\n"

            "[9.0-15.0] P002: AI can also reduce repetitive "
            "work for teachers.\n"

            "[16.0-22.0] P003: However, students should not "
            "depend completely on AI.\n"

            "[23.0-30.0] P004: AI should support teachers "
            "rather than replace them."
        ),

        features={
            "speaking_time": speaking_time,
            "speaking_ratio": speaking_ratio,
            "word_count": word_count,
            "words_per_minute": (
                (word_count / speaking_time) * 60
                if speaking_time > 0
                else 0
            ),
            "turn_count": turn_count,
            "average_turn_duration": (
                speaking_time / turn_count
                if turn_count > 0
                else 0
            ),
            "longest_turn": speaking_time,
            "shortest_turn": speaking_time,
        },

        interaction_features={
            "first_speaking_time": first_speaking_time,
            "last_speaking_time": last_speaking_time,
            "responses": responses,
            "overlap_events": overlap_events,
            "other_speakers_before": other_speakers_before,
            "other_speakers_after": other_speakers_after,
            "average_gap_between_turns": 2.0,
        },

        group_segments=[],

        other_participants=[
            "P001",
            "P002",
            "P003",
            "P004",
        ],
    )


def main():

    print("=" * 70)
    print("FULL GD EVALUATION TEST")
    print("=" * 70)

    participants = [

        create_participant(
            participant_id="P001",
            participant_transcript=(
                "Artificial intelligence can provide "
                "personalized learning and help students "
                "understand difficult topics."
            ),
            speaking_time=6.0,
            speaking_ratio=0.20,
            word_count=17,
            turn_count=1,
            first_speaking_time=2.0,
            last_speaking_time=8.0,
            responses=0,
            overlap_events=0,
            other_speakers_before=0,
            other_speakers_after=3,
        ),

        create_participant(
            participant_id="P002",
            participant_transcript=(
                "AI can reduce repetitive work for teachers "
                "and give them more time to focus on students."
            ),
            speaking_time=6.0,
            speaking_ratio=0.20,
            word_count=19,
            turn_count=1,
            first_speaking_time=9.0,
            last_speaking_time=15.0,
            responses=1,
            overlap_events=0,
            other_speakers_before=1,
            other_speakers_after=2,
        ),

        create_participant(
            participant_id="P003",
            participant_transcript=(
                "Students should not depend completely on AI "
                "because they still need to develop their "
                "own thinking and problem solving skills."
            ),
            speaking_time=6.0,
            speaking_ratio=0.20,
            word_count=23,
            turn_count=1,
            first_speaking_time=16.0,
            last_speaking_time=22.0,
            responses=1,
            overlap_events=0,
            other_speakers_before=2,
            other_speakers_after=1,
        ),

        create_participant(
            participant_id="P004",
            participant_transcript=(
                "AI should support teachers rather than "
                "replace them because education also needs "
                "human interaction and guidance."
            ),
            speaking_time=7.0,
            speaking_ratio=0.23,
            word_count=22,
            turn_count=1,
            first_speaking_time=23.0,
            last_speaking_time=30.0,
            responses=1,
            overlap_events=0,
            other_speakers_before=3,
            other_speakers_after=0,
        ),
    ]

    orchestrator = GDOrchestrator()

    print("\nEvaluating participants...\n")

    reports = orchestrator.evaluate_session(
        participants
    )

    print("=" * 70)
    print("GD EVALUATION RESULTS")
    print("=" * 70)

    for report in reports:

        print("\nParticipant:", report["participant_id"])

        print("\nAgent Scores:")

        for result in report["agent_results"]:

            print(
                f"  {result['agent']}: "
                f"{result['score']}"
            )

        print(
            "\nFinal Score:",
            report["scorecard"]["final_score"]
        )

        print("\nFeedback evidence:")

        for result in report["agent_results"]:

            print(
                f"\n{result['agent'].upper()}:"
            )

            for evidence in result["evidence"]:
                print(
                    "  -",
                    evidence
                )

    print("\n" + "=" * 70)
    print("FULL GD EVALUATION TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()