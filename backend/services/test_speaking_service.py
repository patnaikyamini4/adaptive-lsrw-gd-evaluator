import os
from unittest.mock import MagicMock, patch
import pytest

from backend.services.speaking_service import SpeakingService


def test_speaking_service_success(tmp_path):
    # Create temporary dummy audio file
    dummy_audio = tmp_path / "test_audio.wav"
    dummy_audio.write_bytes(b"dummy audio data")

    fake_golden = {
        "question_id": "speaking-q1",
        "golden_answer": "Online education offers flexibility and convenience.",
        "key_points": ["flexibility", "convenience"],
    }

    fake_evaluation = {
        "module": "SPEAKING",
        "question_id": "speaking-q1",
        "transcript": "Online education gives flexibility to students.",
        "asr": {
            "text": "Online education gives flexibility to students.",
            "transcript": "Online education gives flexibility to students.",
            "segments": [],
            "language": "en",
        },
        "semantic": {
            "model": "all-MiniLM-L6-v2",
            "similarity": 0.8520,
        },
    }

    mock_eval_service = MagicMock()
    mock_eval_service.evaluate_speaking.return_value = fake_evaluation

    service = SpeakingService(evaluation_service=mock_eval_service)

    with patch("backend.services.speaking_service.get_golden_answer", return_value=fake_golden) as mock_get_ga:
        response = service.process_audio(
            audio_path=str(dummy_audio),
            session_id="session-123",
            participant_id="candidate-456",
            question_id="speaking-q1",
        )

    # 1. Verify get_golden_answer was called with question_id
    mock_get_ga.assert_called_once_with("speaking-q1")

    # 2. Verify evaluate_speaking was called with audio_path, question_id, and golden_answer text
    mock_eval_service.evaluate_speaking.assert_called_once_with(
        audio_path=str(dummy_audio),
        question_id="speaking-q1",
        golden_answer="Online education offers flexibility and convenience.",
    )

    # 3. Verify response structure and fields
    assert response["session_id"] == "session-123"
    assert response["participant_id"] == "candidate-456"
    assert response["module"] == "SPEAKING"
    assert response["question_id"] == "speaking-q1"
    assert response["audio_path"] == str(dummy_audio)
    assert response["transcript"] == "Online education gives flexibility to students."
    assert response["asr"] == fake_evaluation["asr"]
    assert response["evaluation"] == fake_evaluation
    assert response["score"] is None
    assert response["started_at"] is None
    assert response["submitted_at"] is not None


def test_speaking_service_missing_audio():
    service = SpeakingService()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        service.process_audio(
            audio_path="non_existent_path.wav",
            session_id="session-123",
            participant_id="candidate-456",
            question_id="speaking-q1",
        )


def test_speaking_service_missing_golden_answer(tmp_path):
    dummy_audio = tmp_path / "test_audio.wav"
    dummy_audio.write_bytes(b"dummy audio data")

    service = SpeakingService()

    with patch("backend.services.speaking_service.get_golden_answer", return_value=None):
        with pytest.raises(ValueError, match="Golden Answer not found for question_id: speaking-q1"):
            service.process_audio(
                audio_path=str(dummy_audio),
                session_id="session-123",
                participant_id="candidate-456",
                question_id="speaking-q1",
            )
