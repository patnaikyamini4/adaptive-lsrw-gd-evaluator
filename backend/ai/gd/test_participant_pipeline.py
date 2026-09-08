"""
Complete GD Participant Pipeline Test.

Tests:

    Participant Audio
          ↓
        VAD
          ↓
        ASR
          ↓
    ParticipantAudio

Tests P001-P004 separately.

No speaker diarization is used.
"""


from backend.ai.gd.participant_pipeline import (
    GDParticipantPipeline,
)


SESSION_ID = "GD001"


PARTICIPANTS = [
    (
        "P001",
        "data/gd/audio/p001_test.wav",
    ),
    (
        "P002",
        "data/gd/audio/p002_test.wav",
    ),
    (
        "P003",
        "data/gd/audio/p003_test.wav",
    ),
    (
        "P004",
        "data/gd/audio/p004_test.wav",
    ),
]


def main() -> None:

    print("=" * 70)
    print("GD COMPLETE PARTICIPANT PIPELINE TEST")
    print("=" * 70)

    print()
    print(
        f"Session: {SESSION_ID}"
    )

    # One shared VAD + one shared Whisper model.
    pipeline = GDParticipantPipeline(
        asr_model="base"
    )

    participants = []

    for participant_id, audio_path in PARTICIPANTS:

        print()
        print("#" * 70)
        print(
            f"PROCESSING {participant_id}"
        )
        print("#" * 70)

        participant = pipeline.process(
            session_id=SESSION_ID,
            participant_id=participant_id,
            audio_path=audio_path,
        )

        participants.append(
            participant
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PARTICIPANT SUMMARY")
    print("=" * 70)

    for participant in participants:

        print()
        print(
            participant.participant_id
        )

        print(
            f"  Speaking time: "
            f"{participant.total_speaking_time:.3f}s"
        )

        print(
            f"  VAD segments: "
            f"{len(participant.vad_segments)}"
        )

        print(
            f"  Transcript segments: "
            f"{len(participant.transcript_segments)}"
        )

        print(
            f"  Word count: "
            f"{participant.word_count}"
        )

        print(
            f"  Transcript: "
            f"{participant.transcript}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()