from backend.ai.gd.live_session import LiveGDSession


def main():

    session = LiveGDSession(
        session_id="GD_LIVE_TEST_001",
        topic="Regional Food Culture in India",
        duration_seconds=600,
    )

    # Register participants
    session.add_participant("P001")
    session.add_participant("P002")
    session.add_participant("P003")
    session.add_participant("P004")

    # Start session
    session.start()

    # Participants join
    session.participant_join("P001")
    session.participant_join("P002")
    session.participant_join("P003")
    session.participant_join("P004")

    # Simulate P001 speaking
    session.update_audio_activity(
        "P001",
        speaking=True,
    )

    session.add_speaking_time(
        "P001",
        4.5,
    )

    session.add_turn("P001")
    session.add_words("P001", 18)

    # P001 stops speaking
    session.update_audio_activity(
        "P001",
        speaking=False,
    )

    # Simulate P002 speaking
    session.update_audio_activity(
        "P002",
        speaking=True,
    )

    session.add_speaking_time(
        "P002",
        5.2,
    )

    session.add_turn("P002")
    session.add_words("P002", 22)

    session.update_audio_activity(
        "P002",
        speaking=False,
    )

    print("=" * 70)
    print("LIVE GD SESSION TEST")
    print("=" * 70)

    print(session.get_status())

    # End session
    session.end()

    print()
    print("=" * 70)
    print("FINAL SESSION STATE")
    print("=" * 70)

    print(session.get_status())

    print()
    print("LIVE SESSION TEST: PASSED")


if __name__ == "__main__":
    main()