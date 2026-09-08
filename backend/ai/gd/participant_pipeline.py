"""
GD Participant Audio Pipeline.

Correct GD architecture:

    session_id
        +
    participant_id
        +
    participant audio
             |
             v
            VAD
             |
             v
            ASR
             |
             v
      ParticipantAudio

No speaker diarization is used.
"""


from backend.ai.asr_service import ASRService

from backend.ai.gd.audio_session import (
    ParticipantAudio,
)

from backend.ai.gd.vad_service import (
    VADService,
)


class GDParticipantPipeline:
    """
    Processes one known participant's audio.

    Identity comes from:
        participant_id

    NOT from:
        diarization
        IP address
        voice identification
    """

    def __init__(
        self,
        asr_model: str = "base",
    ) -> None:

        print(
            "Initializing GD participant pipeline..."
        )

        self.vad_service = VADService()

        self.asr_service = ASRService(
            model_name=asr_model
        )

        print(
            "GD participant pipeline ready."
        )

    def process(
        self,
        session_id: str,
        participant_id: str,
        audio_path: str,
    ) -> ParticipantAudio:
        """
        Process one participant's complete audio.

        Steps:

            1. Create ParticipantAudio
            2. Run VAD
            3. Store VAD segments
            4. Run ASR
            5. Store transcript segments
            6. Return ParticipantAudio
        """

        print()
        print("=" * 70)
        print("GD PARTICIPANT PIPELINE")
        print("=" * 70)

        print(
            f"Session: {session_id}"
        )

        print(
            f"Participant: {participant_id}"
        )

        print(
            f"Audio: {audio_path}"
        )

        # --------------------------------------------------
        # Create participant object
        # --------------------------------------------------

        participant = ParticipantAudio(
            session_id=session_id,
            participant_id=participant_id,
            audio_path=audio_path,
        )

        # --------------------------------------------------
        # VAD
        # --------------------------------------------------

        print()
        print("-" * 70)
        print("STEP 1: VAD")
        print("-" * 70)

        vad_result = self.vad_service.detect(
            audio_path=audio_path,
            session_id=session_id,
            participant_id=participant_id,
        )

        for segment in vad_result[
            "segments"
        ]:

            participant.add_vad_segment(
                start=segment["start"],
                end=segment["end"],
            )

        print(
            f"Speaking time: "
            f"{participant.total_speaking_time:.3f}s"
        )

        print(
            f"VAD segments: "
            f"{len(participant.vad_segments)}"
        )

        # --------------------------------------------------
        # ASR
        # --------------------------------------------------

        print()
        print("-" * 70)
        print("STEP 2: ASR")
        print("-" * 70)

        asr_result = self.asr_service.transcribe_segments(
           audio_path=audio_path,
           speech_segments=vad_result["segments"],
           language="en",
    )

        for segment in asr_result[
            "segments"
        ]:

            participant.add_transcript_segment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
            )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        participant.metadata[
            "asr_language"
        ] = asr_result["language"]

        participant.metadata[
            "asr_model"
        ] = asr_result["model"]

        participant.metadata[
            "asr_device"
        ] = asr_result["device"]

        participant.metadata[
            "audio_duration"
        ] = vad_result[
            "audio_duration"
        ]

        participant.metadata[
            "speech_ratio"
        ] = vad_result[
            "speech_ratio"
        ]

        print()
        print("-" * 70)
        print("PARTICIPANT RESULT")
        print("-" * 70)

        print(
            f"Session: "
            f"{participant.session_id}"
        )

        print(
            f"Participant: "
            f"{participant.participant_id}"
        )

        print(
            f"Speaking time: "
            f"{participant.total_speaking_time:.3f}s"
        )

        print(
            f"Transcript segments: "
            f"{len(participant.transcript_segments)}"
        )

        print(
            f"Word count: "
            f"{participant.word_count}"
        )

        print()
        print("Transcript:")
        print(
            participant.transcript
        )

        return participant