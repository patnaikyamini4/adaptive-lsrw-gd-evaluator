"""
GD Feature Extraction Test
"""

from backend.ai.gd.participant_pipeline import (
    GDParticipantPipeline,
)

from backend.ai.gd.feature_extraction import (
    GDFeatureExtractor,
)


SESSION_ID = "GD001"


AUDIO_FILES = {
    "P001": "data/gd/audio/p001_test.wav",
    "P002": "data/gd/audio/p002_test.wav",
    "P003": "data/gd/audio/p003_test.wav",
    "P004": "data/gd/audio/p004_test.wav",
}


def main():

    print("=" * 70)
    print("GD FEATURE EXTRACTION TEST")
    print("=" * 70)

    print()
    print("Session:", SESSION_ID)

    # --------------------------------------------------
    # PARTICIPANT PIPELINE
    # --------------------------------------------------

    pipeline = GDParticipantPipeline(
        asr_model="base"
    )

    participants = []

    for participant_id, audio_path in (
        AUDIO_FILES.items()
    ):

        print()
        print(
            f"Processing {participant_id}..."
        )

        participant = pipeline.process(
            session_id=SESSION_ID,
            participant_id=participant_id,
            audio_path=audio_path,
        )

        participants.append(
            participant
        )

    # --------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------

    extractor = GDFeatureExtractor(
        session_duration=60.0
    )

    features = extractor.extract_all(
        participants
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTED GD FEATURES")
    print("=" * 70)

    for feature in features:

        print()
        print(
            f"Participant: "
            f"{feature['participant_id']}"
        )

        print(
            f"Speaking time: "
            f"{feature['speaking_time']:.3f}s"
        )

        print(
            f"Speaking ratio: "
            f"{feature['speaking_ratio']:.2%}"
        )

        print(
            f"Word count: "
            f"{feature['word_count']}"
        )

        print(
            f"Turn count: "
            f"{feature['turn_count']}"
        )

        print(
            f"Words/minute: "
            f"{feature['words_per_minute']:.2f}"
        )

        print(
            f"Average turn: "
            f"{feature['average_turn_duration']:.3f}s"
        )

        print(
            f"Longest turn: "
            f"{feature['longest_turn']:.3f}s"
        )

        print(
            f"Shortest turn: "
            f"{feature['shortest_turn']:.3f}s"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()