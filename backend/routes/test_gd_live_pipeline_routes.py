"""
Unit and integration tests for new Live GD routes:
- GET  /api/gd/sessions/<session_id>/transcript
- POST /api/gd/sessions/<session_id>/audio/upload
- POST /api/gd/sessions/<session_id>/evaluate
- End-to-end Live GD session lifecycle integration test
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ==============================================================================
# 1. GET /api/gd/sessions/<session_id>/transcript TESTS
# ==============================================================================

def test_get_live_transcript_success(client):
    """Test successful retrieval of live transcript."""
    with patch("backend.routes.gd_routes.gd_live_service.get_transcript") as mock_get_t:
        mock_get_t.return_value = {
            "session_id": "GD-TEST-1",
            "topic": "Clean Energy",
            "group_transcript": "[P1] Hello team\n[P2] Great to be here",
            "group_segments": [
                {"participant_id": "P1", "start": 0.0, "end": 2.0, "text": "Hello team"},
                {"participant_id": "P2", "start": 2.5, "end": 4.5, "text": "Great to be here"},
            ],
            "participant_transcripts": {
                "P1": "Hello team",
                "P2": "Great to be here",
            },
            "segment_count": 2,
        }

        response = client.get("/api/gd/sessions/GD-TEST-1/transcript")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["session_id"] == "GD-TEST-1"
        assert data["segment_count"] == 2
        assert "[P1] Hello team" in data["group_transcript"]


def test_get_live_transcript_not_found(client):
    """Test 404 when session not found."""
    with patch("backend.routes.gd_routes.gd_live_service.get_transcript") as mock_get_t:
        mock_get_t.side_effect = ValueError("GD session 'NONEXISTENT' not found")

        response = client.get("/api/gd/sessions/NONEXISTENT/transcript")
        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()


# ==============================================================================
# 2. POST /api/gd/sessions/<session_id>/audio/upload TESTS
# ==============================================================================

def test_upload_gd_audio_file_success(client, tmp_path):
    """Test uploading an audio chunk via multipart/form-data."""
    with patch("backend.routes.gd_routes.gd_live_audio_service.process_audio_event") as mock_proc:
        mock_proc.return_value = {
            "session_id": "GD-TEST-2",
            "participant_id": "P1",
            "audio_duration": 3.0,
            "speech_duration": 2.5,
            "speech_ratio": 0.833,
            "transcript": "Renewable energy is important.",
            "word_count": 4,
            "turn_count": 1,
            "runtime": {"session_time": 10.0},
        }

        audio_bytes = io.BytesIO(b"RIFFdummydataWAVEfmt ")
        response = client.post(
            "/api/gd/sessions/GD-TEST-2/audio/upload",
            data={
                "participant_id": "P1",
                "file": (audio_bytes, "chunk_01.wav"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["transcript"] == "Renewable energy is important."
        assert data["participant_id"] == "P1"
        mock_proc.assert_called_once()


def test_upload_gd_audio_file_missing_participant_id(client):
    """Test 400 when participant_id is missing from form."""
    audio_bytes = io.BytesIO(b"dummy")
    response = client.post(
        "/api/gd/sessions/GD-TEST-2/audio/upload",
        data={
            "file": (audio_bytes, "chunk.wav"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "'participant_id' is required" in data["message"]


def test_upload_gd_audio_file_missing_file(client):
    """Test 400 when file is missing from form."""
    response = client.post(
        "/api/gd/sessions/GD-TEST-2/audio/upload",
        data={
            "participant_id": "P1",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "Audio file is required" in data["message"]


# ==============================================================================
# 3. POST /api/gd/sessions/<session_id>/evaluate TESTS
# ==============================================================================

def test_evaluate_live_session_route_success(client):
    """Test successful evaluation of an ENDED live GD session."""
    with patch("backend.routes.gd_routes.gd_live_service.evaluate_live_session") as mock_eval:
        mock_eval.return_value = {
            "session_id": "GD-TEST-3",
            "topic": "Clean Energy",
            "session_duration": 180.0,
            "group_transcript": "[P1] Hello [P2] Hi",
            "participants": [
                {"participant_id": "P1", "final_score": 85.0},
                {"participant_id": "P2", "final_score": 80.0},
            ],
            "final_scores": {"P1": 85.0, "P2": 80.0},
            "document_type": "gd_evaluation",
        }

        response = client.post("/api/gd/sessions/GD-TEST-3/evaluate")
        assert response.status_code == 201
        data = response.get_json()
        assert data["status"] == "success"
        assert data["evaluation"]["session_id"] == "GD-TEST-3"
        assert data["evaluation"]["final_scores"]["P1"] == 85.0


def test_evaluate_live_session_route_not_ended_error(client):
    """Test 400 when evaluating a session that is still ACTIVE or SCHEDULED."""
    with patch("backend.routes.gd_routes.gd_live_service.evaluate_live_session") as mock_eval:
        mock_eval.side_effect = ValueError("Cannot evaluate live GD session in 'ACTIVE' status. Must be 'ENDED'.")

        response = client.post("/api/gd/sessions/GD-ACTIVE/evaluate")
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "Must be 'ENDED'" in data["message"]


def test_evaluate_live_session_route_not_found(client):
    """Test 404 when evaluating a non-existent session."""
    with patch("backend.routes.gd_routes.gd_live_service.evaluate_live_session") as mock_eval:
        mock_eval.side_effect = ValueError("GD session 'NONEXISTENT' not found")

        response = client.post("/api/gd/sessions/NONEXISTENT/evaluate")
        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "error"
        assert "not found" in data["message"]
