from backend.ai.asr_service import transcribe_audio


class LSRWEvaluationService:

    def evaluate_speaking(
        self,
        audio_path,
        question_id,
        golden_answer
    ):

        asr_result = transcribe_audio(audio_path)

        transcript = asr_result["transcript"]

        # Semantic evaluation will be connected here
        # Feedback will be connected here

        return {
            "module": "SPEAKING",
            "question_id": question_id,
            "transcript": transcript,
            "asr": asr_result
        }