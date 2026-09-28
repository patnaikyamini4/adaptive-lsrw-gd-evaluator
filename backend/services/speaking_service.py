import os
from datetime import datetime

from backend.ai.asr_service import transcribe_audio


class SpeakingService:

    def process_audio(
        self,
        audio_path,
        session_id,
        participant_id,
        question_id
    ):
        """
        Process a candidate's speaking response.

        Current stage:
        Audio → ASR → Transcript

        Semantic evaluation and scoring will be
        connected after this pipeline is validated.
        """

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # ------------------------------------------
        # 1. Run ASR
        # ------------------------------------------

        asr_result = transcribe_audio(
            audio_path,
            language="en"
        )

        transcript = asr_result["transcript"]

        # ------------------------------------------
        # 2. Create response object
        # ------------------------------------------

        response = {
            "session_id": session_id,
            "participant_id": participant_id,
            "module": "SPEAKING",
            "question_id": question_id,

            "audio_path": audio_path,

            "transcript": transcript,

            "asr": asr_result,

            "started_at": None,
            "submitted_at": datetime.utcnow(),

            "evaluation": None,
            "score": None
        }

        return response