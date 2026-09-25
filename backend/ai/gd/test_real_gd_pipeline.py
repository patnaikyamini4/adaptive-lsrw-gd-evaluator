"""
Real GD Audio Integration Test

Flow:

    Participant WAV
        |
        v
    VAD
        |
        v
    Shared ASR
        |
        v
    ParticipantAudio
        |
        v
    Session Timeline
        |
        v
    Group Transcript
        |
        v
    Feature Extraction
        |
        v
    Interaction Features
        |
        v
    GD Agents
        |
        v
    GD Scorer
"""


from pathlib import Path

from backend.ai.gd.participant_pipeline import (
    GDParticipantPipeline,
)

from backend.ai.gd.transcript_service import (
    GDTranscriptService,
)

from backend.ai.gd.feature_extraction import (
    GDFeatureExtractor,
)

from backend.ai.gd.interaction_features import (
    GDInteractionFeatureExtractor,
)

from backend.ai.gd.gd_agent_input import (
    GDParticipantInput,
)

from backend.ai.gd.gd_orchestrator import (
    GDOrchestrator,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SESSION_ID = "GD_REAL_001"

TOPIC = "Regional Food Culture in India"

BASE_AUDIO_DIR = Path("data/gd/audio")


PARTICIPANTS = {
    "P001": {
        "audio": BASE_AUDIO_DIR / "p001_test.wav",
        "session_offset": 0.0,
    },

    "P002": {
        "audio": BASE_AUDIO_DIR / "p002_test.wav",
        "session_offset": 16.994,
    },

    "P003": {
        "audio": BASE_AUDIO_DIR / "p003_test.wav",
        "session_offset": 39.028,
    },

    "P004": {
        "audio": BASE_AUDIO_DIR / "p004_test.wav",
        "session_offset": 68.502,
    },
}


# ==========================================================
# VALIDATE AUDIO FILES
# ==========================================================

def validate_audio_files():

    print()
    print("=" * 80)
    print("VALIDATING GD AUDIO FILES")
    print("=" * 80)

    for participant_id, config in PARTICIPANTS.items():

        audio_path = config["audio"]

        print()
        print(
            f"{participant_id}: "
            f"{audio_path}"
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        print(
            f"Found: {audio_path}"
        )


# ==========================================================
# PROCESS PARTICIPANT AUDIO
# ==========================================================

def process_participants():

    print()
    print("=" * 80)
    print("STEP 1: PROCESS PARTICIPANT AUDIO")
    print("=" * 80)

    pipeline = GDParticipantPipeline(
        asr_model="base"
    )

    participants = []

    for participant_id, config in PARTICIPANTS.items():

        audio_path = config["audio"]

        session_offset = config[
            "session_offset"
        ]

        print()
        print("#" * 80)
        print(
            f"PROCESSING {participant_id}"
        )
        print("#" * 80)

        participant = pipeline.process(
            session_id=SESSION_ID,
            participant_id=participant_id,
            audio_path=str(audio_path),
        )

        # --------------------------------------------------
        # Assign offline test session offset
        # --------------------------------------------------

        participant.session_offset = (
            session_offset
        )

        participants.append(
            participant
        )

        print()
        print(
            f"{participant_id} session offset: "
            f"{participant.session_offset:.3f}s"
        )

    return participants


# ==========================================================
# BUILD GROUP TRANSCRIPT
# ==========================================================

def build_group_data(participants):

    print()
    print("=" * 80)
    print("STEP 2: BUILD GROUP TRANSCRIPT")
    print("=" * 80)

    transcript_service = (
        GDTranscriptService()
    )

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

    print()
    print("GROUP TRANSCRIPT")
    print("-" * 80)

    print(group_text)

    print()
    print(
        f"Total group segments: "
        f"{len(group_segments)}"
    )

    return (
        transcript_service,
        group_segments,
        group_text,
    )


# ==========================================================
# EXTRACT OBJECTIVE FEATURES
# ==========================================================

def extract_features(participants):

    print()
    print("=" * 80)
    print("STEP 3: EXTRACT PARTICIPANT FEATURES")
    print("=" * 80)

    # ------------------------------------------------------
    # Determine test session duration
    # ------------------------------------------------------

    session_end = 0.0

    for participant in participants:

        for segment in (
            participant.transcript_segments
        ):

            session_end = max(
                session_end,
                participant.local_to_session_time(
                    segment["end"]
                ),
            )

    print()
    print(
        f"Calculated test session duration: "
        f"{session_end:.3f}s"
    )

    feature_extractor = (
        GDFeatureExtractor(
            session_duration=session_end
        )
    )

    features_by_participant = {}

    for participant in participants:

        features = (
            feature_extractor.extract(
                participant
            )
        )

        features_by_participant[
            participant.participant_id
        ] = features

        print()
        print(
            f"{participant.participant_id}"
        )

        print(
            f"  Speaking time: "
            f"{features['speaking_time']:.3f}s"
        )

        print(
            f"  Speaking ratio: "
            f"{features['speaking_ratio']:.2%}"
        )

        print(
            f"  Word count: "
            f"{features['word_count']}"
        )

        print(
            f"  Turn count: "
            f"{features['turn_count']}"
        )

        print(
            f"  WPM: "
            f"{features['words_per_minute']:.2f}"
        )

        print(
            f"  Average turn: "
            f"{features['average_turn_duration']:.3f}s"
        )

    return features_by_participant


# ==========================================================
# EXTRACT INTERACTION FEATURES
# ==========================================================

def extract_interaction_features(
    participants,
    group_segments,
):

    print()
    print("=" * 80)
    print("STEP 4: EXTRACT INTERACTION FEATURES")
    print("=" * 80)

    interaction_extractor = (
        GDInteractionFeatureExtractor()
    )

    interaction_by_participant = {}

    participant_ids = [
        participant.participant_id
        for participant in participants
    ]

    for participant_id in participant_ids:

        interaction = (
            interaction_extractor.extract(
                group_segments=group_segments,
                participant_id=participant_id,
            )
        )

        interaction_by_participant[
            participant_id
        ] = interaction

        print()
        print(
            f"{participant_id}"
        )

        print(
            f"  Turns: "
            f"{interaction['turn_count']}"
        )

        print(
            f"  Responses: "
            f"{interaction['responses']}"
        )

        print(
            f"  Overlap events: "
            f"{interaction['overlap_events']}"
        )

        print(
            f"  Other speakers before: "
            f"{interaction['other_speakers_before']}"
        )

        print(
            f"  Other speakers after: "
            f"{interaction['other_speakers_after']}"
        )

    return interaction_by_participant


# ==========================================================
# BUILD GD AGENT INPUTS
# ==========================================================

def build_agent_inputs(
    participants,
    group_segments,
    group_text,
    features_by_participant,
    interaction_by_participant,
):

    print()
    print("=" * 80)
    print("STEP 5: BUILD GD AGENT INPUTS")
    print("=" * 80)

    participant_ids = [
        participant.participant_id
        for participant in participants
    ]

    inputs = []

    for participant in participants:

        participant_id = (
            participant.participant_id
        )

        # --------------------------------------------------
        # Participant-specific transcript
        # --------------------------------------------------

        participant_transcript = (
            participant.transcript
        )

        # --------------------------------------------------
        # Other participants
        # --------------------------------------------------

        other_participants = [
            pid
            for pid in participant_ids
            if pid != participant_id
        ]

        # --------------------------------------------------
        # Build agent input
        # --------------------------------------------------

        agent_input = GDParticipantInput(

            session_id=SESSION_ID,

            participant_id=participant_id,

            topic=TOPIC,

            participant_transcript=(
                participant_transcript
            ),

            group_transcript=group_text,

            features=(
                features_by_participant[
                    participant_id
                ]
            ),

            interaction_features=(
                interaction_by_participant[
                    participant_id
                ]
            ),

            group_segments=group_segments,

            other_participants=(
                other_participants
            ),

            metadata={
                "source": "real_audio_test",

                "audio_file": (
                    str(
                        PARTICIPANTS[
                            participant_id
                        ]["audio"]
                    )
                ),

                "session_offset": (
                    participant.session_offset
                ),
            },
        )

        inputs.append(
            agent_input
        )

        print()
        print(
            f"Built agent input for "
            f"{participant_id}"
        )

    return inputs


# ==========================================================
# RUN GD EVALUATION
# ==========================================================

def evaluate_session(agent_inputs):

    print()
    print("=" * 80)
    print("STEP 6: RUN GD EVALUATION")
    print("=" * 80)

    orchestrator = GDOrchestrator()

    results = (
        orchestrator.evaluate_session(
            agent_inputs
        )
    )

    return results


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

def print_results(results):

    print()
    print("=" * 80)
    print("FINAL GD EVALUATION")
    print("=" * 80)

    for result in results:

        participant_id = (
            result["participant_id"]
        )

        scorecard = (
            result["scorecard"]
        )

        print()
        print(
            f"{participant_id}"
        )

        print(
            "-" * 60
        )

        print(
            f"Relevance: "
            f"{scorecard['scores']['relevance']:.2f}"
        )

        print(
            f"Coherence: "
            f"{scorecard['scores']['coherence']:.2f}"
        )

        print(
            f"Fluency: "
            f"{scorecard['scores']['fluency']:.2f}"
        )

        print(
            f"Participation: "
            f"{scorecard['scores']['participation']:.2f}"
        )

        print(
            f"FINAL SCORE: "
            f"{scorecard['final_score']:.2f}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 80)
    print("REAL GD AUDIO EVALUATION TEST")
    print("=" * 80)

    print(
        f"Session: {SESSION_ID}"
    )

    print(
        f"Topic: {TOPIC}"
    )

    # ------------------------------------------------------
    # 1. Validate audio
    # ------------------------------------------------------

    validate_audio_files()

    # ------------------------------------------------------
    # 2. VAD + ASR
    # ------------------------------------------------------

    participants = (
        process_participants()
    )

    # ------------------------------------------------------
    # 3. Group transcript
    # ------------------------------------------------------

    (
        transcript_service,
        group_segments,
        group_text,
    ) = build_group_data(
        participants
    )

    # ------------------------------------------------------
    # 4. Objective features
    # ------------------------------------------------------

    features_by_participant = (
        extract_features(
            participants
        )
    )

    # ------------------------------------------------------
    # 5. Interaction features
    # ------------------------------------------------------

    interaction_by_participant = (
        extract_interaction_features(
            participants,
            group_segments,
        )
    )

    # ------------------------------------------------------
    # 6. Agent inputs
    # ------------------------------------------------------

    agent_inputs = (
        build_agent_inputs(
            participants,
            group_segments,
            group_text,
            features_by_participant,
            interaction_by_participant,
        )
    )

    # ------------------------------------------------------
    # 7. GD evaluation
    # ------------------------------------------------------

    results = (
        evaluate_session(
            agent_inputs
        )
    )

    # ------------------------------------------------------
    # 8. Print results
    # ------------------------------------------------------

    print_results(
        results
    )

    print()
    print("=" * 80)
    print("REAL GD AUDIO TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()