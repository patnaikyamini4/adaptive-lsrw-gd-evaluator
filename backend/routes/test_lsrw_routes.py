import io
import os
from unittest.mock import MagicMock, patch
import pytest

from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_submit_response_session_not_found(client):
    with patch("backend.routes.lsrw_routes.get_session", return_value=None):
        resp = client.post(
            "/api/lsrw/sessions/nonexistent-session/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Session not found"


def test_submit_response_session_not_active(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "COMPLETED",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }
    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Session is not active"


def test_submit_response_wrong_module(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "WRITING",
        "current_module": "WRITING",
        "current_question_id": "WR001",
    }
    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "supports SPEAKING only" in data["message"]


def test_submit_response_missing_audio(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }
    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Audio file is required"


def test_submit_response_missing_golden_answer_with_cleanup(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.speaking_service.process_audio", side_effect=ValueError("Golden Answer not found for question_id: SP001")):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "Golden Answer not found" in data["message"]


def test_submit_response_evaluation_failure_with_cleanup(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.speaking_service.process_audio", side_effect=RuntimeError("ASR model crash")):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Speaking audio processing failed"


def test_submit_response_db_failure_with_cleanup(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }
    fake_response = {
        "session_id": "session-1",
        "participant_id": "p1",
        "module": "SPEAKING",
        "question_id": "SP001",
        "audio_path": "dummy/path.wav",
        "transcript": "hello",
        "asr": {},
        "evaluation": {},
        "score": None,
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.speaking_service.process_audio", return_value=fake_response), \
         patch("backend.routes.lsrw_routes.create_response", side_effect=Exception("DB Connection failed")):
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Failed to save speaking response"


def test_submit_response_success(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
    }
    fake_response = {
        "session_id": "session-1",
        "participant_id": "p1",
        "module": "SPEAKING",
        "question_id": "SP001",
        "audio_path": "data/lsrw/audio/speaking/sample.wav",
        "transcript": "Online education is flexible.",
        "asr": {"transcript": "Online education is flexible."},
        "evaluation": {
            "module": "SPEAKING",
            "question_id": "SP001",
            "transcript": "Online education is flexible.",
            "asr": {"transcript": "Online education is flexible."},
            "semantic": {"model": "all-MiniLM-L6-v2", "similarity": 0.85},
            "score": {
                "final_score": 85.0,
                "components": {
                    "semantic": {
                        "raw_similarity": 0.85,
                        "score": 85.0,
                        "weight": 1.0,
                    }
                },
                "scoring_method": "deterministic_semantic_v1",
                "version": "1.0",
            },
        },
        "score": 85.0,
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.speaking_service.process_audio", return_value=fake_response), \
         patch("backend.routes.lsrw_routes.create_response", return_value=fake_response) as mock_create_resp:
        resp = client.post(
            "/api/lsrw/sessions/session-1/response",
            data={"audio": (io.BytesIO(b"fake audio"), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["message"] == "Speaking response processed"
        res = data["response"]
        assert res["session_id"] == "session-1"
        assert res["participant_id"] == "p1"
        assert res["module"] == "SPEAKING"
        assert res["question_id"] == "SP001"
        assert res["transcript"] == "Online education is flexible."
        assert res["asr"] == {"transcript": "Online education is flexible."}
        assert res["evaluation"]["semantic"]["similarity"] == 0.85
        assert res["evaluation"]["score"]["final_score"] == 85.0
        assert res["score"] == 85.0
        assert 0.0 <= res["score"] <= 100.0
        mock_create_resp.assert_called_once_with(fake_response)


def test_finish_module_session_not_found(client):
    with patch("backend.routes.lsrw_routes.get_session", return_value=None):
        resp = client.post("/api/lsrw/sessions/nonexistent-session/module/finish")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Session not found"


def test_finish_module_no_active_module(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "ACTIVE",
        "current_module": None,
        "current_question_id": None,
    }
    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session):
        resp = client.post("/api/lsrw/sessions/session-1/module/finish")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "No active module"


def test_finish_module_success(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "SPEAKING",
        "current_module": "SPEAKING",
        "current_question_id": "SP001",
        "completed_modules": [],
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.update_session", side_effect=lambda sid, s: s) as mock_update:
        resp = client.post("/api/lsrw/sessions/session-1/module/finish")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["message"] == "SPEAKING module finished"
        session_data = data["session"]
        assert session_data["status"] == "ACTIVE"
        assert session_data["current_module"] is None
        assert session_data["current_question_id"] is None
        assert session_data["completed_modules"] == ["SPEAKING"]
        mock_update.assert_called_once()


def test_finish_session_not_found(client):
    with patch("backend.routes.lsrw_routes.get_session", return_value=None):
        resp = client.post("/api/lsrw/sessions/nonexistent-session/finish")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["message"] == "Session not found"


def test_finish_session_success(client):
    fake_session = {
        "session_id": "session-1",
        "participant_id": "p1",
        "status": "ACTIVE",
        "current_module": None,
        "completed_modules": ["SPEAKING"],
        "session_ended_at": None,
    }

    with patch("backend.routes.lsrw_routes.get_session", return_value=fake_session), \
         patch("backend.routes.lsrw_routes.update_session", side_effect=lambda sid, s: s) as mock_update:
        resp = client.post("/api/lsrw/sessions/session-1/finish")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["message"] == "LSRW session finished"
        session_data = data["session"]
        assert session_data["status"] == "COMPLETED"
        assert session_data["session_ended_at"] is not None
        mock_update.assert_called_once()
