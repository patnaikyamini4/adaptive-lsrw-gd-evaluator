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
