"""
Unit tests for Group Discussion (GD) REST API routes (backend/routes/gd_routes.py).

All tests mock GDEvaluationService and GD repository functions to ensure
deterministic, offline execution without loading Whisper, VAD, or Qwen models.
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.app import app


@pytest.fixture
def client():
    """Create a Flask test client for API testing."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_post_payload():
    """Valid POST request payload for GD evaluation."""
    return {
        "session_id": "GD-SESSION-202",
        "topic": "Renewable Energy Adoption Challenges",
        "session_duration": 240.0,
        "participants": [
            {
                "participant_id": "P1",
                "audio_path": "uploads/p1.wav",
                "session_offset": 0.0,
                "metadata": {"role": "moderator"},
            },
            {
                "participant_id": "P2",
                "audio_path": "uploads/p2.wav",
                "session_offset": 1.5,
                "metadata": {},
            },
        ],
    }


@pytest.fixture
def sample_evaluation_result():
    """Sample output produced by GDEvaluationService.evaluate_session()."""
    return {
        "session_id": "GD-SESSION-202",
        "topic": "Renewable Energy Adoption Challenges",
        "session_duration": 240.0,
        "group_transcript": "[0.0s - 10.0s] P1: Solar is viable.\n[10.5s - 20.0s] P2: Storage is the bottleneck.",
        "group_segments": [
            {"participant_id": "P1", "start": 0.0, "end": 10.0, "text": "Solar is viable."},
            {"participant_id": "P2", "start": 10.5, "end": 20.0, "text": "Storage is the bottleneck."},
        ],
        "participants": [
            {
                "participant_id": "P1",
                "transcript": "Solar is viable.",
                "final_score": 8.0,
                "scorecard": {"final_score": 8.0},
            },
            {
                "participant_id": "P2",
                "transcript": "Storage is the bottleneck.",
                "final_score": 7.5,
                "scorecard": {"final_score": 7.5},
            },
        ],
        "participant_transcripts": {
            "P1": "Solar is viable.",
            "P2": "Storage is the bottleneck.",
        },
        "objective_features": {"P1": {"wpm": 120.0}, "P2": {"wpm": 115.0}},
        "interaction_features": {"P1": {"overlaps": 0}, "P2": {"overlaps": 1}},
        "agent_results": {
            "P1": [{"agent_name": "relevance", "score": 8.0}],
            "P2": [{"agent_name": "relevance", "score": 7.5}],
        },
        "scorecards": {
            "P1": {"final_score": 8.0},
            "P2": {"final_score": 7.5},
        },
        "final_scores": {
            "P1": 8.0,
            "P2": 7.5,
        },
    }


# ==============================================================================
# POST /api/gd/evaluations TESTS
# ==============================================================================

def test_post_evaluation_success(client, sample_post_payload, sample_evaluation_result):
    """Test successful evaluation orchestration and persistence."""
    mock_service = MagicMock()
    mock_service.evaluate_session.return_value = sample_evaluation_result

    persisted_doc = sample_evaluation_result.copy()
    persisted_doc["document_type"] = "gd_evaluation"

    with patch("backend.routes.gd_routes.gd_service", mock_service), \
         patch("backend.routes.gd_routes.save_evaluation", return_value=persisted_doc) as mock_save:

        response = client.post(
            "/api/gd/evaluations",
            json=sample_post_payload,
            content_type="application/json",
        )

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"
    assert "evaluation" in data
    assert data["evaluation"]["session_id"] == "GD-SESSION-202"
    assert data["evaluation"]["document_type"] == "gd_evaluation"

    # Verify service and repository calls
    mock_service.evaluate_session.assert_called_once_with(
        session_id="GD-SESSION-202",
        topic="Renewable Energy Adoption Challenges",
        participants=sample_post_payload["participants"],
        session_duration=240.0,
    )
    mock_save.assert_called_once_with(sample_evaluation_result)


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (None, "Request body must be a valid JSON object"),
        ({}, "'session_id' is required"),
        ({"session_id": "  ", "topic": "T", "participants": [{"participant_id": "P1"}]}, "'session_id' is required"),
        ({"session_id": "S1", "topic": "  ", "participants": [{"participant_id": "P1"}]}, "'topic' is required"),
        ({"session_id": "S1", "topic": "T", "participants": []}, "'participants' is required"),
        ({"session_id": "S1", "topic": "T", "participants": "not-a-list"}, "'participants' is required"),
        ({"session_id": "S1", "topic": "T", "participants": [{"p": 1}], "session_duration": "invalid"}, "'session_duration' must be a numeric value"),
        ({"session_id": "S1", "topic": "T", "participants": [{"p": 1}], "session_duration": -5.0}, "'session_duration' cannot be negative"),
    ],
)
def test_post_evaluation_invalid_payload(client, payload, expected_msg):
    """Test validation errors for malformed POST request payloads."""
    if payload is None:
        response = client.post(
            "/api/gd/evaluations",
            data="not-valid-json",
            content_type="application/json",
        )
    else:
        response = client.post(
            "/api/gd/evaluations",
            json=payload,
            content_type="application/json",
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert expected_msg in data["message"]


def test_post_evaluation_service_value_error(client, sample_post_payload):
    """Test handling of ValueError raised from GDEvaluationService."""
    mock_service = MagicMock()
    mock_service.evaluate_session.side_effect = ValueError("Duplicate participant_id found: 'P1'")

    with patch("backend.routes.gd_routes.gd_service", mock_service):
        response = client.post(
            "/api/gd/evaluations",
            json=sample_post_payload,
            content_type="application/json",
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Duplicate participant_id" in data["message"]


def test_post_evaluation_service_file_not_found(client, sample_post_payload):
    """Test handling of FileNotFoundError for missing audio file in service."""
    mock_service = MagicMock()
    mock_service.evaluate_session.side_effect = FileNotFoundError("Audio file not found: uploads/p1.wav")

    with patch("backend.routes.gd_routes.gd_service", mock_service):
        response = client.post(
            "/api/gd/evaluations",
            json=sample_post_payload,
            content_type="application/json",
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Audio file not found" in data["message"]


def test_post_evaluation_service_unexpected_exception(client, sample_post_payload):
    """Test 500 handling when an unexpected error occurs during evaluation."""
    mock_service = MagicMock()
    mock_service.evaluate_session.side_effect = RuntimeError("Unexpected GPU failure")

    with patch("backend.routes.gd_routes.gd_service", mock_service):
        response = client.post(
            "/api/gd/evaluations",
            json=sample_post_payload,
            content_type="application/json",
        )

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "GD evaluation pipeline failed" in data["message"]


def test_post_evaluation_repository_persistence_failure(client, sample_post_payload, sample_evaluation_result):
    """Test 500 handling when MongoDB repository persistence fails."""
    mock_service = MagicMock()
    mock_service.evaluate_session.return_value = sample_evaluation_result

    with patch("backend.routes.gd_routes.gd_service", mock_service), \
         patch("backend.routes.gd_routes.save_evaluation", side_effect=Exception("Database connection timeout")):

        response = client.post(
            "/api/gd/evaluations",
            json=sample_post_payload,
            content_type="application/json",
        )

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "Failed to persist GD evaluation" in data["message"]


# ==============================================================================
# GET /api/gd/evaluations/<session_id> TESTS
# ==============================================================================

def test_get_evaluation_success(client, sample_evaluation_result):
    """Test retrieving an existing GD evaluation document."""
    with patch("backend.routes.gd_routes.get_evaluation", return_value=sample_evaluation_result) as mock_get:
        response = client.get("/api/gd/evaluations/GD-SESSION-202")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["evaluation"]["session_id"] == "GD-SESSION-202"
    mock_get.assert_called_once_with("GD-SESSION-202")


def test_get_evaluation_not_found(client):
    """Test 404 response when GD evaluation does not exist."""
    with patch("backend.routes.gd_routes.get_evaluation", return_value=None):
        response = client.get("/api/gd/evaluations/NON-EXISTENT")

    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
    assert "GD evaluation not found" in data["message"]


def test_get_evaluation_server_error(client):
    """Test 500 response when database fails during GET."""
    with patch("backend.routes.gd_routes.get_evaluation", side_effect=Exception("DB down")):
        response = client.get("/api/gd/evaluations/GD-SESSION-202")

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "Failed to retrieve GD evaluation" in data["message"]


# ==============================================================================
# DELETE /api/gd/evaluations/<session_id> TESTS
# ==============================================================================

def test_delete_evaluation_success(client):
    """Test deleting an existing GD evaluation document."""
    with patch("backend.routes.gd_routes.delete_evaluation", return_value=True) as mock_delete:
        response = client.delete("/api/gd/evaluations/GD-SESSION-202")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "deleted successfully" in data["message"]
    mock_delete.assert_called_once_with("GD-SESSION-202")


def test_delete_evaluation_not_found(client):
    """Test 404 response when deleting non-existent GD evaluation."""
    with patch("backend.routes.gd_routes.delete_evaluation", return_value=False):
        response = client.delete("/api/gd/evaluations/NON-EXISTENT")

    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
    assert "GD evaluation not found" in data["message"]


def test_delete_evaluation_server_error(client):
    """Test 500 response when database fails during DELETE."""
    with patch("backend.routes.gd_routes.delete_evaluation", side_effect=Exception("DB write error")):
        response = client.delete("/api/gd/evaluations/GD-SESSION-202")

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "Failed to delete GD evaluation" in data["message"]


# ==============================================================================
# GET /api/gd/evaluations (LIST) TESTS
# ==============================================================================

def test_list_evaluations_success(client, sample_evaluation_result):
    """Test listing recent evaluations."""
    mock_evals = [sample_evaluation_result]
    with patch("backend.routes.gd_routes.get_all_evaluations", return_value=mock_evals) as mock_list:
        response = client.get("/api/gd/evaluations?limit=25")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert len(data["evaluations"]) == 1
    mock_list.assert_called_once_with(limit=25)


def test_list_evaluations_server_error(client):
    """Test 500 response when database fails during LIST."""
    with patch("backend.routes.gd_routes.get_all_evaluations", side_effect=Exception("Cursor error")):
        response = client.get("/api/gd/evaluations")

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "Failed to list GD evaluations" in data["message"]


def test_list_evaluations_invalid_limit(client):
    """Test 400 response when limit query parameter is invalid."""
    # Non-integer
    response = client.get("/api/gd/evaluations?limit=abc")
    assert response.status_code == 400
    assert "'limit' query parameter must be an integer" in response.get_json()["message"]

    # Non-positive integer
    response = client.get("/api/gd/evaluations?limit=0")
    assert response.status_code == 400
    assert "'limit' query parameter must be a positive integer" in response.get_json()["message"]

    response = client.get("/api/gd/evaluations?limit=-5")
    assert response.status_code == 400
    assert "'limit' query parameter must be a positive integer" in response.get_json()["message"]


def test_route_does_not_duplicate_scoring_logic():
    """Architectural test: Verify gd_routes delegates scoring directly to GDEvaluationService."""
    import inspect
    import backend.routes.gd_routes as gd_routes_module
    source = inspect.getsource(gd_routes_module)
    # Ensure route does not import or invoke GDScorer or individual GD agents directly
    assert "GDScorer" not in source
    assert "RelevanceAgent" not in source
    assert "CoherenceAgent" not in source
    assert "FluencyAgent" not in source
    assert "ParticipationAgent" not in source


# ==============================================================================
# POST /api/gd/sessions/<session_id>/audio TESTS
# ==============================================================================

@pytest.fixture
def sample_audio_event_result():
    """Sample output produced by GDLiveAudioService.process_audio_event()."""
    return {
        "session_id": "GD-SESSION-202",
        "participant_id": "P1",
        "audio_duration": 4.5,
        "speech_duration": 3.2,
        "speech_ratio": 0.7111,
        "segments": [{"start": 0.5, "end": 3.7, "duration": 3.2, "text": "Solar energy is essential."}],
        "vad_segments": [{"start": 0.5, "end": 3.7, "duration": 3.2}],
        "transcript": "Solar energy is essential.",
        "word_count": 4,
        "turn_count": 1,
        "runtime": {
            "session_id": "GD-SESSION-202",
            "topic": "Renewable Energy Adoption Challenges",
            "duration_seconds": 300,
            "session_time": 10.5,
            "started": True,
            "ended": False,
            "participants": {
                "P1": {
                    "connected": True,
                    "joined_at": 0.0,
                    "left_at": None,
                    "last_audio_at": 10.5,
                    "speaking": True,
                    "speaking_time": 3.2,
                    "turn_count": 1,
                    "word_count": 4,
                }
            },
        },
    }


def test_post_audio_event_success(client, sample_audio_event_result):
    """1, 2, 3, 4, 5. Test valid audio event returns 200, extracts parameters from URL/body, and calls service once."""
    mock_service = MagicMock()
    mock_service.process_audio_event.return_value = sample_audio_event_result

    payload = {
        "participant_id": "P1",
        "audio_path": "data/gd/audio/p1_test.wav",
    }

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json=payload,
            content_type="application/json",
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session_id"] == "GD-SESSION-202"
    assert data["participant_id"] == "P1"
    assert data["transcript"] == "Solar energy is essential."
    assert data["word_count"] == 4
    assert data["turn_count"] == 1
    assert "runtime" in data

    # 5. Service called exactly once with expected arguments
    mock_service.process_audio_event.assert_called_once_with(
        session_id="GD-SESSION-202",
        participant_id="P1",
        audio_path="data/gd/audio/p1_test.wav",
    )


def test_post_audio_event_session_id_from_url(client, sample_audio_event_result):
    """2. Test session_id is taken strictly from URL route parameter."""
    mock_service = MagicMock()
    mock_service.process_audio_event.return_value = sample_audio_event_result

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/CUSTOM-SESSION-999/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 200
    mock_service.process_audio_event.assert_called_once_with(
        session_id="CUSTOM-SESSION-999",
        participant_id="P1",
        audio_path="uploads/test.wav",
    )


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        ({"audio_path": "data/test.wav"}, "'participant_id' is required"),
        ({"participant_id": "", "audio_path": "data/test.wav"}, "'participant_id' is required"),
        ({"participant_id": "   ", "audio_path": "data/test.wav"}, "'participant_id' is required"),
        ({"participant_id": 123, "audio_path": "data/test.wav"}, "'participant_id' is required"),
        ({"participant_id": None, "audio_path": "data/test.wav"}, "'participant_id' is required"),
    ],
)
def test_post_audio_event_missing_participant_id(client, payload, expected_msg):
    """6. Test missing or invalid participant_id returns 400."""
    response = client.post(
        "/api/gd/sessions/GD-SESSION-202/audio",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert expected_msg in data["message"]


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        ({"participant_id": "P1"}, "'audio_path' is required"),
        ({"participant_id": "P1", "audio_path": ""}, "'audio_path' is required"),
        ({"participant_id": "P1", "audio_path": "   "}, "'audio_path' is required"),
        ({"participant_id": "P1", "audio_path": 123}, "'audio_path' is required"),
        ({"participant_id": "P1", "audio_path": None}, "'audio_path' is required"),
    ],
)
def test_post_audio_event_missing_audio_path(client, payload, expected_msg):
    """7. Test missing or invalid audio_path returns 400."""
    response = client.post(
        "/api/gd/sessions/GD-SESSION-202/audio",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert expected_msg in data["message"]


def test_post_audio_event_malformed_json(client):
    """8. Test malformed or non-dict JSON body returns 400."""
    # Non-JSON content
    res1 = client.post(
        "/api/gd/sessions/GD-SESSION-202/audio",
        data="invalid-raw-text",
        content_type="application/json",
    )
    assert res1.status_code == 400
    assert "Request body must be a valid JSON object" in res1.get_json()["message"]

    # JSON array instead of dict
    res2 = client.post(
        "/api/gd/sessions/GD-SESSION-202/audio",
        json=["item1", "item2"],
        content_type="application/json",
    )
    assert res2.status_code == 400
    assert "Request body must be a valid JSON object" in res2.get_json()["message"]


def test_post_audio_event_inactive_session_rejected(client):
    """9. Test inactive session rejection returns 400."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = ValueError(
        "Cannot process audio event in 'SCHEDULED' status. Must be 'ACTIVE'."
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 400
    assert "Cannot process audio event in 'SCHEDULED' status" in response.get_json()["message"]


def test_post_audio_event_unauthorized_participant_rejected(client):
    """10. Test unauthorized participant rejection returns 400."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = ValueError(
        "Participant 'P99' is not authorized for session 'GD-SESSION-202'"
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P99", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 400
    assert "Participant 'P99' is not authorized" in response.get_json()["message"]


def test_post_audio_event_disconnected_participant_rejected(client):
    """11. Test disconnected participant rejection returns 400."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = ValueError(
        "Participant 'P1' is not currently connected to session 'GD-SESSION-202'"
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 400
    assert "not currently connected" in response.get_json()["message"]


def test_post_audio_event_missing_runtime_rejected(client):
    """12. Test missing live runtime returns 400."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = RuntimeError(
        "Live runtime for active GD session 'GD-SESSION-202' is not available"
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 400
    assert "Live runtime for active GD session" in response.get_json()["message"]


def test_post_audio_event_missing_audio_file_handled_correctly(client):
    """13. Test missing audio file on filesystem returns 404."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = FileNotFoundError(
        "Audio file not found: non_existent.wav"
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P1", "audio_path": "non_existent.wav"},
        )

    assert response.status_code == 404
    assert "Audio file not found" in response.get_json()["message"]


def test_post_audio_event_session_not_found(client):
    """Test non-existent session returns 404."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = ValueError(
        "GD session 'NON-EXISTENT' not found"
    )

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/NON-EXISTENT/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 404
    assert "not found" in response.get_json()["message"].lower()


def test_post_audio_event_service_failure_handled_correctly(client):
    """14. Test unexpected service exception returns 500."""
    mock_service = MagicMock()
    mock_service.process_audio_event.side_effect = Exception("Internal hardware timeout")

    with patch("backend.routes.gd_routes.gd_live_audio_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-202/audio",
            json={"participant_id": "P1", "audio_path": "uploads/test.wav"},
        )

    assert response.status_code == 500
    assert "Failed to process live audio event" in response.get_json()["message"]


def test_route_does_not_directly_invoke_vad_asr_or_qwen():
    """15, 16, 17, 18. Architectural test: Verify route does not directly import or invoke VAD, ASR, Qwen, or LiveGDSession."""
    import inspect
    import backend.routes.gd_routes as gd_routes_module

    source = inspect.getsource(gd_routes_module)

    # 15. Route does not directly invoke VAD
    assert "VADService" not in source
    assert "vad_service.detect" not in source

    # 16. Route does not directly invoke ASR
    assert "ASRService" not in source
    assert "transcribe_segments" not in source
    assert "whisper" not in source.lower()

    # 17. Route does not invoke Qwen
    assert "ask_qwen" not in source
    assert "Qwen" not in source

    # 18. Route does not instantiate LiveGDSession directly
    assert "LiveGDSession(" not in source


def test_live_audio_route_integration_with_live_session(client, tmp_path):
    """
    Integration test: Flask route -> GDLiveAudioService -> existing LiveGDSession.
    Verifies actual route-to-service wiring and live runtime metric accumulation.
    """
    from datetime import datetime, timezone
    from backend.ai.gd.live_session import LiveGDSession
    from backend.services.gd_live_audio_service import GDLiveAudioService
    from backend.services.gd_live_service import GDLiveService
    from backend.services.gd_session_service import GDSessionService

    now = datetime.now(timezone.utc)
    session_id = "GD-INTEG-100"
    session_data = {
        "session_id": session_id,
        "topic": "Live Integration Test Topic",
        "coordinator_id": "COORD-1",
        "participant_ids": ["P1", "P2"],
        "status": "ACTIVE",
        "scheduled_start": now,
        "scheduled_end": None,
        "session_started_at": now,
        "session_ended_at": None,
        "session_duration": 300.0,
        "evaluation_id": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "document_type": "gd_session",
    }

    # Setup session service with mock session lookup
    mock_session_svc = MagicMock(spec=GDSessionService)
    mock_session_svc.get_session.return_value = session_data

    # Setup live service and real in-memory LiveGDSession
    live_svc = GDLiveService(session_service=mock_session_svc)
    runtime = LiveGDSession(
        session_id=session_id,
        topic=session_data["topic"],
        duration_seconds=300,
    )
    runtime.add_participant("P1")
    runtime.add_participant("P2")
    runtime.participant_join("P1")
    runtime.start()
    live_svc._runtimes[session_id] = runtime

    # Mock VAD & ASR for deterministic fast test execution
    mock_vad = MagicMock()
    mock_vad.detect.return_value = {
        "session_id": session_id,
        "participant_id": "P1",
        "audio_duration": 4.0,
        "speech_duration": 2.8,
        "speech_ratio": 0.7,
        "segments": [{"start": 0.5, "end": 3.3, "duration": 2.8}],
    }

    mock_asr = MagicMock()
    mock_asr.transcribe_segments.return_value = {
        "text": "Live route integration test verified.",
        "segments": [{"start": 0.5, "end": 3.3, "duration": 2.8, "text": "Live route integration test verified."}],
        "language": "en",
    }

    # Real GDLiveAudioService instance
    real_audio_svc = GDLiveAudioService(
        session_service=mock_session_svc,
        live_service=live_svc,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    # Temporary dummy audio file on disk
    audio_file = tmp_path / "p1_integ.wav"
    audio_file.write_bytes(b"RIFFdummydata")

    # Wire real service to route
    with patch("backend.routes.gd_routes.gd_live_audio_service", real_audio_svc):
        response = client.post(
            f"/api/gd/sessions/{session_id}/audio",
            json={"participant_id": "P1", "audio_path": str(audio_file)},
        )

    # 1. Verify HTTP response
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session_id"] == session_id
    assert data["participant_id"] == "P1"
    assert data["transcript"] == "Live route integration test verified."
    assert data["speech_duration"] == 2.8
    assert data["word_count"] == 5
    assert data["turn_count"] == 1

    # 2. Verify LiveGDSession state updated directly
    p1_state = runtime.participants["P1"]
    assert p1_state.speaking is True
    assert p1_state.speaking_time == 2.8
    assert p1_state.turn_count == 1
    assert p1_state.word_count == 5
    assert p1_state.last_audio_at is not None

    # Verify P2 is isolated and untouched
    p2_state = runtime.participants["P2"]
    assert p2_state.speaking_time == 0.0
    assert p2_state.turn_count == 0
    assert p2_state.word_count == 0
