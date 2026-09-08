"""
GD Shared ASR Test

Tests the shared ASR service on P001 audio.

No speaker diarization is used.
"""


from backend.ai.asr_service import ASRService


def main() -> None:

    print("=" * 70)
    print("GD SHARED ASR TEST")
    print("=" * 70)

    audio_path = (
        "data/gd/audio/p001_test.wav"
    )

    service = ASRService(
        model_name="base"
    )

    result = service.transcribe(
        audio_path=audio_path,
        language="en",
    )

    print()
    print("=" * 70)
    print("ASR RESULT")
    print("=" * 70)

    print()
    print(
        "Detected language:",
        result["language"],
    )

    print(
        "Device:",
        result["device"],
    )

    print(
        "Model:",
        result["model"],
    )

    print()
    print("FULL TRANSCRIPT:")
    print(
        result["text"]
    )

    print()
    print("SEGMENTS:")
    print("-" * 70)

    for index, segment in enumerate(
        result["segments"],
        start=1,
    ):

        print(
            f"{index:03d}. "
            f"{segment['start']:.3f}s -> "
            f"{segment['end']:.3f}s "
            f"({segment['duration']:.3f}s)"
        )

        print(
            f"      {segment['text']}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()