"""
Test GD Interaction Feature Extraction
"""

from backend.ai.gd.interaction_features import (
    GDInteractionFeatureExtractor,
)


def main():

    print("=" * 70)
    print("GD INTERACTION FEATURE TEST")
    print("=" * 70)

    # Synthetic shared GD timeline.
    #
    # P001 speaks first.
    # P002 responds after P001.
    # P003 speaks after P002.
    # P001 responds after P003.
    # P002 overlaps with P001.

    group_segments = [
    {
        "participant_id": "P001",
        "start": 2.0,
        "end": 5.0,
        "text": "I think technology can improve education.",
    },
    {
        "participant_id": "P002",
        "start": 6.0,
        "end": 8.0,
        "text": "I agree with that point.",
    },
    {
        "participant_id": "P003",
        "start": 12.0,
        "end": 15.0,
        "text": "Technology also creates new opportunities.",
    },
    {
        "participant_id": "P001",
        "start": 16.0,
        "end": 19.0,
        "text": "Yes, especially through online learning.",
    },
    {
        "participant_id": "P002",
        "start": 18.0,
        "end": 21.0,
        "text": "I would also mention accessibility.",
    },
]
    print()
    print("SHARED GD TIMELINE")
    print("-" * 70)

    for index, segment in enumerate(group_segments, start=1):

        print(
            f"{index:03d}. "
            f"{segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s "
            f"[{segment['participant_id']}] "
            f"{segment['text']}"
        )

    extractor = GDInteractionFeatureExtractor()

    participant_ids = [
        "P001",
        "P002",
        "P003",
    ]

    print()
    print("=" * 70)
    print("INTERACTION FEATURES")
    print("=" * 70)

    results = extractor.extract_all(
        group_segments,
        participant_ids,
    )

    for result in results:

        print()
        print(
            f"Participant: "
            f"{result['participant_id']}"
        )

        print(
            f"Turn count: "
            f"{result['turn_count']}"
        )

        print(
            f"First speaking time: "
            f"{result['first_speaking_time']}"
        )

        print(
            f"Last speaking time: "
            f"{result['last_speaking_time']}"
        )

        print(
            f"Average gap between turns: "
            f"{result['average_gap_between_turns']:.3f}s"
        )

        print(
            f"Responses: "
            f"{result['responses']}"
        )

        print(
            f"Overlap events: "
            f"{result['overlap_events']}"
        )

        print(
            f"Other speakers before: "
            f"{result['other_speakers_before']}"
        )

        print(
            f"Other speakers after: "
            f"{result['other_speakers_after']}"
        )

    print()
    print("=" * 70)
    print("EXPECTED BEHAVIOUR")
    print("=" * 70)

    print()
    print("P001:")
    print("  - 2 turns")
    print("  - 1 response")
    print("  - 1 other speaker before second turn")
    print("  - 0 overlap events")

    print()
    print("P002:")
    print("  - 2 turns")
    print("  - 1 response")
    print("  - 1 overlap event")

    print()
    print("P003:")
    print("  - 1 turn")
    print("  - 0 responses")
    print("  - 0 overlap events")

    # -------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------

    p001 = results[0]
    p002 = results[1]
    p003 = results[2]

    assert p001["turn_count"] == 2
    assert p001["responses"] == 1
    assert p001["overlap_events"] == 0

    assert p002["turn_count"] == 2
    assert p002["responses"] == 1
    assert p002["overlap_events"] == 1

    assert p003["turn_count"] == 1
    assert p003["responses"] == 0
    assert p003["overlap_events"] == 0

    print()
    print("=" * 70)
    print("ALL INTERACTION ASSERTIONS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()