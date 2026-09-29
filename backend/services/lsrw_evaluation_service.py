from backend.ai.asr_service import transcribe_audio
from backend.ai.semantic_agent import evaluate_semantic_similarity


class LSRWEvaluationService:

    def evaluate_speaking(
        self,
        audio_path,
        question_id,
        golden_answer
    ):
        # 1. Transcribe candidate audio
        asr_result = transcribe_audio(audio_path)

        transcript = asr_result["transcript"]

        # 2. Compare candidate transcript with Golden Answer
        semantic_result = evaluate_semantic_similarity(
            golden_answer,
            transcript
        )

        # 3. Return complete evaluation
        return {
            "module": "SPEAKING",
            "question_id": question_id,
            "transcript": transcript,
            "asr": asr_result,
            "semantic": semantic_result
        }