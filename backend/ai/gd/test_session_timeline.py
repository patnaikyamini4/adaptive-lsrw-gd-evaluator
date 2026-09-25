"""
GD Session Timeline Test

This test verifies that participant-local
timestamps can be converted into a shared
GD session timeline.

No diarization is used.
"""


from backend.ai.gd.audio_session import (
    ParticipantAudio,
)

from backend.ai.gd.transcript_service import (
    GDTranscriptService,
)


SESSION_ID = "GD001"


def create_participant(
    participant_id: str,
    session_offset: float,
) -> ParticipantAudio:

    participant = ParticipantAudio(
        session_id=SESSION_ID,
        participant_id=participant_id,
        session_offset=session_offset,
    )

    return participant


def main():

    print("=" * 70)
    print("GD SESSION TIMELINE TEST")
    print("=" * 70)

    print()

    # --------------------------------------------------
    # CREATE PARTICIPANTS
    # --------------------------------------------------

    p001 = create_participant(
        "P001",
        0.0,
    )

    p002 = create_participant(
        "P002",
        5.0,
    )

    p003 = create_participant(
        "P003",
        2.0,
    )

    p004 = create_participant(
        "P004",
        8.0,
    )

    # --------------------------------------------------
    # ADD LOCAL TRANSCRIPT SEGMENTS
    # --------------------------------------------------

    p001.add_transcript_segment(
        start=1.0,
        end=4.0,
        text="I think traditional food is important.",
    )

    p002.add_transcript_segment(
        start=1.0,
        end=3.0,
        text="I agree with that point.",
    )

    p003.add_transcript_segment(
        start=1.0,
        end=3.5,
        text="It also reflects regional culture.",
    )

    p004.add_transcript_segment(
        start=1.0,
        end=3.0,
        text="Different regions have different dishes.",
    )

    # --------------------------------------------------
    # BUILD SHARED TIMELINE
    # --------------------------------------------------

    participants = [
        p001,
        p002,
        p003,
        p004,
    ]

    service = GDTranscriptService()

    segments = (
        service.build_group_transcript(
            participants
        )
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("SHARED GD SESSION TIMELINE")
    print("-" * 70)

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        print(
            f"{index:03d}. "
            f"{segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s "
            f"[{segment['participant_id']}] "
            f"{segment['text']}"
        )

    print()
    print("=" * 70)
    print("EXPECTED ORDER")
    print("=" * 70)

    print(
        "P001 -> P003 -> P002 -> P004"
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()