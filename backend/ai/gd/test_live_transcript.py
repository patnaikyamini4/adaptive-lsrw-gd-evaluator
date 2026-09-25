from backend.ai.gd.live_transcript import LiveTranscriptManager


def main():

    manager = LiveTranscriptManager()

    manager.add_segment(
        session_id="GD_LIVE_TEST_001",
        participant_id="P001",
        start=1.0,
        end=4.0,
        text="North Indian cuisine uses wheat and dairy.",
    )

    manager.add_segment(
        session_id="GD_LIVE_TEST_001",
        participant_id="P002",
        start=4.5,
        end=8.0,
        text="South Indian cuisine relies heavily on rice.",
    )

    manager.add_segment(
        session_id="GD_LIVE_TEST_001",
        participant_id="P001",
        start=8.5,
        end=11.0,
        text="I agree, and both regions have distinctive dishes.",
    )

    print("=" * 70)
    print("LIVE TRANSCRIPT TEST")
    print("=" * 70)

    print()
    print("GROUP TRANSCRIPT:")
    print(manager.get_group_text())

    print()
    print("P001 TRANSCRIPT:")
    print(manager.get_participant_text("P001"))

    print()
    print("P002 TRANSCRIPT:")
    print(manager.get_participant_text("P002"))

    print()
    print("LIVE TRANSCRIPT TEST: PASSED")


if __name__ == "__main__":
    main()