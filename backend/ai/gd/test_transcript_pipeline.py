"""
GD Transcript Pipeline Test

Tests chronological combination of participant
transcripts.

No speaker diarization is used.
"""


from backend.ai.gd.audio_session import (
    ParticipantAudio,
)

from backend.ai.gd.transcript_service import (
    GDTranscriptService,
)


def main() -> None:

    print("=" * 70)
    print("GD TRANSCRIPT PIPELINE TEST")
    print("=" * 70)

    session_id = "GD001"

    p1 = ParticipantAudio(
        session_id=session_id,
        participant_id="P001",
    )

    p2 = ParticipantAudio(
        session_id=session_id,
        participant_id="P002",
    )

    p3 = ParticipantAudio(
        session_id=session_id,
        participant_id="P003",
    )

    p4 = ParticipantAudio(
        session_id=session_id,
        participant_id="P004",
    )

    # Simulated ASR results.
    # Real ASR will be connected later.

    p1.add_transcript_segment(
        1.0,
        5.0,
        "I think artificial intelligence can improve education.",
    )

    p2.add_transcript_segment(
        5.5,
        9.0,
        "I agree, but there are also some challenges.",
    )

    p3.add_transcript_segment(
        9.5,
        14.0,
        "One important challenge is the responsible use of AI.",
    )

    p4.add_transcript_segment(
        14.5,
        18.0,
        "I believe proper guidelines can solve many of these issues.",
    )

    participants = [
        p1,
        p2,
        p3,
        p4,
    ]

    service = GDTranscriptService()

    print()
    print("PARTICIPANT TRANSCRIPTS")
    print("-" * 70)

    for participant in participants:

        result = (
            service.build_participant_transcript(
                participant
            )
        )

        print()
        print(
            f"Participant: "
            f"{result['participant_id']}"
        )

        print(
            f"Transcript: "
            f"{result['transcript']}"
        )

        print(
            f"Words: "
            f"{result['word_count']}"
        )

    print()
    print("=" * 70)
    print("CHRONOLOGICAL GROUP TRANSCRIPT")
    print("=" * 70)

    print()

    print(
        service.build_group_text(
            participants
        )
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()