from faster_whisper import WhisperModel


def main():
    print("Loading Whisper model...")

    model = WhisperModel(
        "small.en",
        device="cpu",
        compute_type="int8"
    )

    print("Whisper model loaded successfully!")
    print("Model: small.en")
    print("Device: CPU")
    print("Compute type: int8")


if __name__ == "__main__":
    main()