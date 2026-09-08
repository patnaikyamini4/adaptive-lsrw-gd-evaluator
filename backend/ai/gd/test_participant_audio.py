"""
GD Participant Audio Test

Tests the correct GD architecture:

    P001 audio -> P001 VAD
    P002 audio -> P002 VAD
    P003 audio -> P003 VAD
    P004 audio -> P004 VAD

No speaker diarization is used.
"""


from pathlib import Path

from backend.ai.gd.audio_session import (
    ParticipantAudio,
)

from backend.ai.gd.vad_service import (
    VADService,
)


SESSION_ID = "GD001"

AUDIO_DIR = (
    Path("data")
    / "gd"
    / "audio"
)


PARTICIPANTS = [
    ("P001", "p001_test.wav"),
    ("P002", "p002_test.wav"),
    ("P003", "p003_test.wav"),
    ("P004", "p004_test.wav"),
]


def main() -> None:

    print("=" * 70)
    print("GD PARTICIPANT AUDIO TEST")
    print("=" * 70)

    print()
    print(f"Session: {SESSION_ID}")
    print()

    vad_service = VADService()

    participants = []

    for participant_id, filename in PARTICIPANTS:

        audio_path = AUDIO_DIR / filename

        print()
        print("-" * 70)
        print(
            f"Processing {participant_id}"
        )
        print(
            f"Audio: {audio_path}"
        )
        print("-" * 70)

        if not audio_path.exists():

            print(
                f"ERROR: Audio file not found: "
                f"{audio_path}"
            )

            continue

        participant = ParticipantAudio(
            session_id=SESSION_ID,
            participant_id=participant_id,
            audio_path=str(audio_path),
        )

        result = vad_service.detect(
            audio_path=str(audio_path),
            session_id=SESSION_ID,
            participant_id=participant_id,
        )

        for segment in result["segments"]:

            participant.add_vad_segment(
                start=segment["start"],
                end=segment["end"],
            )

        participants.append(
            participant
        )

        print()
        print(
            f"Participant: {participant.participant_id}"
        )

        print(
            f"Audio duration: "
            f"{result['audio_duration']:.3f}s"
        )

        print(
            f"Speech duration: "
            f"{result['speech_duration']:.3f}s"
        )

        print(
            f"Speech ratio: "
            f"{result['speech_ratio']:.2%}"
        )

        print(
            f"Speech segments: "
            f"{len(result['segments'])}"
        )

    print()
    print("=" * 70)
    print("PARTICIPANT SUMMARY")
    print("=" * 70)

    for participant in participants:

        print()
        print(
            f"{participant.participant_id}"
        )

        print(
            f"  Audio: "
            f"{participant.audio_path}"
        )

        print(
            f"  Speaking time: "
            f"{participant.total_speaking_time:.3f}s"
        )

        print(
            f"  VAD segments: "
            f"{len(participant.vad_segments)}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()