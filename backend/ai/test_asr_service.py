from backend.ai.asr_service import transcribe_audio


def main():

    audio_path = "test_audio.ogg"

    print("===== ASR SERVICE TEST =====")

    result = transcribe_audio(audio_path)

    print("\nTranscript:")
    print(result["transcript"])

    print("\nDetected Language:")
    print(result["language"])

    print("\nLanguage Probability:")
    print(result["language_probability"])

    print("\nDuration:")
    print(f"{result['duration']} seconds")

    print("\nSegments:")

    for index, segment in enumerate(
        result["segments"],
        start=1
    ):
        print(
            f"{index}. "
            f"[{segment['start']}s -> {segment['end']}s] "
            f"{segment['text']}"
        )

    print("\nValidation: SUCCESS")
    print("============================")


if __name__ == "__main__":
    main()