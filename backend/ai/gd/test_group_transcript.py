"""
GD Group Transcript Integration Test

Tests the REAL pipeline:

    Participant Audio
            ↓
          VAD
            ↓
          ASR
            ↓
    ParticipantAudio
            ↓
    GDTranscriptService
            ↓
    Chronological Group Transcript

No speaker diarization is used.
"""

from pathlib import Path

from backend.ai.gd.participant_pipeline import (
    GDParticipantPipeline,
)

from backend.ai.gd.transcript_service import (
    GDTranscriptService,
)


SESSION_ID = "GD001"

AUDIO_FILES = {
    "P001": Path("data/gd/audio/p001_test.wav"),
    "P002": Path("data/gd/audio/p002_test.wav"),
    "P003": Path("data/gd/audio/p003_test.wav"),
    "P004": Path("data/gd/audio/p004_test.wav"),
}


def main():

    print("=" * 70)
    print("GD GROUP TRANSCRIPT INTEGRATION TEST")
    print("=" * 70)

    print()
    print(f"Session: {SESSION_ID}")

    # --------------------------------------------------
    # CREATE PARTICIPANT PIPELINE
    # --------------------------------------------------

    pipeline = GDParticipantPipeline(
        asr_model="base"
    )

    participants = []

    # --------------------------------------------------
    # PROCESS EVERY PARTICIPANT
    # --------------------------------------------------

    for participant_id, audio_path in AUDIO_FILES.items():

        print()
        print("#" * 70)
        print(f"PROCESSING {participant_id}")
        print("#" * 70)

        if not audio_path.exists():

            print(
                f"WARNING: Audio not found: "
                f"{audio_path}"
            )

            continue

        participant = pipeline.process(
            session_id=SESSION_ID,
            participant_id=participant_id,
            audio_path=str(audio_path),
        )

        participants.append(
            participant
        )

    # --------------------------------------------------
    # CHECK PARTICIPANTS
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("PARTICIPANTS READY")
    print("=" * 70)

    print(
        f"Participants processed: "
        f"{len(participants)}"
    )

    for participant in participants:

        print(
            f"{participant.participant_id}: "
            f"{participant.total_speaking_time:.3f}s "
            f"speaking time, "
            f"{len(participant.transcript_segments)} "
            f"transcript segments"
        )

    # --------------------------------------------------
    # BUILD GROUP TRANSCRIPT
    # --------------------------------------------------

    transcript_service = GDTranscriptService()

    group_segments = (
        transcript_service.build_group_transcript(
            participants
        )
    )

    group_text = (
        transcript_service.build_group_text(
            participants
        )
    )

    # --------------------------------------------------
    # DISPLAY CHRONOLOGICAL TRANSCRIPT
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CHRONOLOGICAL GROUP TRANSCRIPT")
    print("=" * 70)

    if not group_segments:

        print()
        print("No transcript segments found.")

    else:

        for index, segment in enumerate(
            group_segments,
            start=1,
        ):

            print(
                f"{index:03d}. "
                f"[{segment['start']:.3f}s -> "
                f"{segment['end']:.3f}s] "
                f"[{segment['participant_id']}] "
                f"{segment['text']}"
            )

    # --------------------------------------------------
    # GROUP TEXT
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("GROUP TEXT")
    print("=" * 70)

    print()

    print(group_text)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()