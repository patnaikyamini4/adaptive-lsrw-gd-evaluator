from faster_whisper import WhisperModel


def main():
    print("Loading Whisper model...")

    model = WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8"
    )

    print("Model loaded.")
    print("Starting transcription...\n")

    segments, info = model.transcribe(
        "test_audio.ogg",
        language="en"
    )

    print("===== ASR RESULT =====")
    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.2f}")
    print()

    full_transcript = []

    for segment in segments:
        text = segment.text.strip()

        print(
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
            f"{text}"
        )

        full_transcript.append(text)

    transcript = " ".join(full_transcript)

    print()
    print("===== FULL TRANSCRIPT =====")
    print(transcript)
    print("============================")


if __name__ == "__main__":
    main()