import os
from datetime import datetime

from backend.services.golden_answer_service import get_golden_answer
from backend.services.lsrw_evaluation_service import LSRWEvaluationService


class SpeakingService:

    def __init__(self, evaluation_service=None):
        self.evaluation_service = evaluation_service or LSRWEvaluationService()

    def process_audio(
        self,
        audio_path,
        session_id,
        participant_id,
        question_id,
    ):
        """
        Process a candidate's speaking response:
        1. Validate audio file exists.
        2. Retrieve pre-computed Golden Answer from MongoDB.
        3. Evaluate speaking response (ASR transcription + semantic similarity).
        4. Construct and return response document with evaluation results.
        """

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # ------------------------------------------
        # 1. Retrieve Golden Answer
        # ------------------------------------------

        golden_answer_doc = get_golden_answer(question_id)

        if not golden_answer_doc:
            raise ValueError(
                f"Golden Answer not found for question_id: {question_id}"
            )

        golden_answer_text = golden_answer_doc.get("golden_answer")

        if not golden_answer_text:
            raise ValueError(
                f"Golden Answer text missing for question_id: {question_id}"
            )

        # ------------------------------------------
        # 2. Run Evaluation Pipeline (ASR + Semantic)
        # ------------------------------------------

        evaluation = self.evaluation_service.evaluate_speaking(
            audio_path=audio_path,
            question_id=question_id,
            golden_answer=golden_answer_text,
        )

        # ------------------------------------------
        # 3. Create response object
        # ------------------------------------------

        response = {
            "session_id": session_id,
            "participant_id": participant_id,
            "module": "SPEAKING",
            "question_id": question_id,

            "audio_path": audio_path,

            "transcript": evaluation["transcript"],

            "asr": evaluation["asr"],

            "started_at": None,
            "submitted_at": datetime.utcnow(),

            "evaluation": evaluation,
            "score": None,
        }

        return response
