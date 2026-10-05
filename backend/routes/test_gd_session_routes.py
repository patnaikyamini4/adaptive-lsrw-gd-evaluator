"""
Unit tests for GD Session REST API routes (backend/routes/gd_routes.py).

All tests mock GDSessionService to run deterministically and offline without DB dependencies.
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_session():
    """Sample GD session dictionary."""
    return {
        "session_id": "GD-SESSION-500",
        "topic": "Ethics of Autonomous AI",
        "coordinator_id": "COORD-99",
        "participant_ids": ["P001", "P002", "P003"],
        "status": "SCHEDULED",
        "scheduled_start": "2026-10-10T10:00:00Z",
        "scheduled_end": None,
        "session_started_at": None,
        "session_ended_at": None,
        "session_duration": 180.0,
        "evaluation_id": None,
        "metadata": {"room": "Room-1"},
        "created_at": "2026-10-05T00:00:00Z",
        "updated_at": "2026-10-05T00:00:00Z",
        "document_type": "gd_session",
    }


# ==============================================================================
# 1. CREATE SESSION TESTS (POST /api/gd/sessions)
# ==============================================================================

def test_create_session_success(client, sample_session):
    """Test successful session creation."""
    mock_service = MagicMock()
    mock_service.create_session.return_value = sample_session

    payload = {
        "topic": "Ethics of Autonomous AI",
        "coordinator_id": "COORD-99",
        "participant_ids": ["P001", "P002", "P003"],
        "session_id": "GD-SESSION-500",
        "scheduled_start": "2026-10-10T10:00:00",
        "session_duration": 180.0,
        "metadata": {"room": "Room-1"},
    }

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.post("/api/gd/sessions", json=payload)

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session"]["session_id"] == "GD-SESSION-500"
    assert data["session"]["status"] == "SCHEDULED"


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (None, "Request body must be a valid JSON object"),
        ({}, "'topic' is required"),
        ({"topic": "  ", "coordinator_id": "C", "participant_ids": ["P1"]}, "'topic' is required"),
        ({"topic": "T", "coordinator_id": "  ", "participant_ids": ["P1"]}, "'coordinator_id' is required"),
        ({"topic": "T", "coordinator_id": "C", "participant_ids": []}, "'participant_ids' is required"),
        ({"topic": "T", "coordinator_id": "C", "participant_ids": "not-a-list"}, "'participant_ids' is required"),
        ({"topic": "T", "coordinator_id": "C", "participant_ids": ["P1"], "scheduled_start": "invalid-iso"}, "'scheduled_start' must be a valid ISO datetime string"),
        ({"topic": "T", "coordinator_id": "C", "participant_ids": ["P1"], "scheduled_end": "invalid-iso"}, "'scheduled_end' must be a valid ISO datetime string"),
    ],
)
def test_create_session_invalid_payload(client, payload, expected_msg):
    """Test validation errors for malformed create session requests."""
    if payload is None:
        response = client.post("/api/gd/sessions", data="invalid-json", content_type="application/json")
    else:
        response = client.post("/api/gd/sessions", json=payload)

    assert response.status_code == 400
    assert expected_msg in response.get_json()["message"]


# ==============================================================================
# 2. GET SESSION TESTS (GET /api/gd/sessions/<session_id>)
# ==============================================================================

def test_get_session_success(client, sample_session):
    """Test retrieving an existing session."""
    mock_service = MagicMock()
    mock_service.get_session.return_value = sample_session

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.get("/api/gd/sessions/GD-SESSION-500")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session"]["session_id"] == "GD-SESSION-500"


def test_get_session_not_found(client):
    """Test 404 response for non-existent session."""
    mock_service = MagicMock()
    mock_service.get_session.side_effect = ValueError("GD session 'NON-EXISTENT' not found")

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.get("/api/gd/sessions/NON-EXISTENT")

    assert response.status_code == 404
    assert "not found" in response.get_json()["message"].lower()


# ==============================================================================
# 3. START SESSION TESTS (POST /api/gd/sessions/<session_id>/start)
# ==============================================================================

def test_start_session_success(client, sample_session):
    """Test starting a session."""
    active_session = sample_session.copy()
    active_session["status"] = "ACTIVE"
    active_session["session_started_at"] = "2026-10-05T00:01:00Z"

    mock_service = MagicMock()
    mock_service.start_session.return_value = {
        "session": active_session,
        "runtime": {"started": True, "session_time": 0.0},
    }

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post("/api/gd/sessions/GD-SESSION-500/start")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session"]["status"] == "ACTIVE"


def test_start_session_invalid_transition(client):
    """Test error when starting a non-SCHEDULED session."""
    mock_service = MagicMock()
    mock_service.start_session.side_effect = ValueError("Cannot start GD session in 'ACTIVE' status. Must be 'SCHEDULED'.")

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post("/api/gd/sessions/GD-SESSION-500/start")

    assert response.status_code == 400
    assert "Cannot start GD session in 'ACTIVE' status" in response.get_json()["message"]


# ==============================================================================
# 4. END SESSION TESTS (POST /api/gd/sessions/<session_id>/end)
# ==============================================================================

def test_end_session_success(client, sample_session):
    """Test ending an active session."""
    ended_session = sample_session.copy()
    ended_session["status"] = "ENDED"
    ended_session["session_ended_at"] = "2026-10-05T00:10:00Z"
    ended_session["session_duration"] = 540.0

    mock_service = MagicMock()
    mock_service.end_session.return_value = {
        "session": ended_session,
        "runtime": {"ended": True, "session_time": 540.0},
    }

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post("/api/gd/sessions/GD-SESSION-500/end")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session"]["status"] == "ENDED"
    assert data["session"]["session_duration"] == 540.0


def test_end_session_invalid_transition(client):
    """Test error when ending a non-ACTIVE session."""
    mock_service = MagicMock()
    mock_service.end_session.side_effect = ValueError("Cannot end GD session in 'SCHEDULED' status. Must be 'ACTIVE'.")

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post("/api/gd/sessions/GD-SESSION-500/end")

    assert response.status_code == 400
    assert "Cannot end GD session in 'SCHEDULED' status" in response.get_json()["message"]


# ==============================================================================
# 5. CANCEL SESSION TESTS (POST /api/gd/sessions/<session_id>/cancel)
# ==============================================================================

def test_cancel_session_success(client, sample_session):
    """Test cancelling a scheduled or active session."""
    cancelled_session = sample_session.copy()
    cancelled_session["status"] = "CANCELLED"
    cancelled_session["metadata"] = {"cancellation_reason": "Low attendance"}

    mock_service = MagicMock()
    mock_service.cancel_session.return_value = cancelled_session

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-500/cancel",
            json={"reason": "Low attendance"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["session"]["status"] == "CANCELLED"
    assert data["session"]["metadata"]["cancellation_reason"] == "Low attendance"


# ==============================================================================
# 6. PARTICIPANT MANAGEMENT TESTS (POST/DELETE participants)
# ==============================================================================

def test_add_participant_success(client, sample_session):
    """Test adding a participant to a scheduled session."""
    updated = sample_session.copy()
    updated["participant_ids"] = ["P001", "P002", "P003", "P004"]

    mock_service = MagicMock()
    mock_service.add_participant.return_value = updated

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-500/participants",
            json={"participant_id": "P004"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "P004" in data["session"]["participant_ids"]


def test_remove_participant_success(client, sample_session):
    """Test removing a participant from a scheduled session."""
    updated = sample_session.copy()
    updated["participant_ids"] = ["P001", "P003"]

    mock_service = MagicMock()
    mock_service.remove_participant.return_value = updated

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.delete("/api/gd/sessions/GD-SESSION-500/participants/P002")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "P002" not in data["session"]["participant_ids"]


# ==============================================================================
# 7. LIST & DELETE SESSION TESTS
# ==============================================================================

def test_list_sessions_success(client, sample_session):
    """Test listing GD sessions."""
    mock_service = MagicMock()
    mock_service.list_sessions.return_value = [sample_session]

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.get("/api/gd/sessions?status=SCHEDULED&limit=10")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert len(data["sessions"]) == 1


def test_delete_session_success(client):
    """Test deleting a session."""
    mock_service = MagicMock()
    mock_service.delete_session.return_value = True

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.delete("/api/gd/sessions/GD-SESSION-500")

    assert response.status_code == 200
    assert response.get_json()["status"] == "success"


def test_delete_session_not_found(client):
    """Test deleting non-existent session returns 404."""
    mock_service = MagicMock()
    mock_service.delete_session.return_value = False

    with patch("backend.routes.gd_routes.gd_session_service", mock_service):
        response = client.delete("/api/gd/sessions/NON-EXISTENT")

    assert response.status_code == 404
    assert response.get_json()["status"] == "error"


# ==============================================================================
# 8. LIVE RUNTIME ROUTES (JOIN, LEAVE, RUNTIME STATUS)
# ==============================================================================

def test_join_participant_route_success(client):
    """Test POST /api/gd/sessions/<session_id>/join route success."""
    mock_service = MagicMock()
    mock_service.join_participant.return_value = {
        "session_id": "GD-SESSION-500",
        "participant_id": "P001",
        "connected": True,
        "runtime": {"participants": {"P001": {"connected": True}}},
    }

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-500/join",
            json={"participant_id": "P001"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["connected"] is True
    assert data["participant_id"] == "P001"


def test_join_participant_route_validation_errors(client):
    """Test validation errors for join route."""
    # Missing json body
    res1 = client.post("/api/gd/sessions/GD-SESSION-500/join", data="not-json", content_type="application/json")
    assert res1.status_code == 400

    # Missing participant_id
    res2 = client.post("/api/gd/sessions/GD-SESSION-500/join", json={})
    assert res2.status_code == 400
    assert "'participant_id' is required" in res2.get_json()["message"]


def test_join_participant_route_not_found(client):
    """Test join route 404 when session not found."""
    mock_service = MagicMock()
    mock_service.join_participant.side_effect = ValueError("GD session 'NON-EXISTENT' not found")

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post(
            "/api/gd/sessions/NON-EXISTENT/join",
            json={"participant_id": "P001"},
        )

    assert response.status_code == 404


def test_leave_participant_route_success(client):
    """Test POST /api/gd/sessions/<session_id>/leave route success."""
    mock_service = MagicMock()
    mock_service.leave_participant.return_value = {
        "session_id": "GD-SESSION-500",
        "participant_id": "P001",
        "connected": False,
        "runtime": {"participants": {"P001": {"connected": False}}},
    }

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.post(
            "/api/gd/sessions/GD-SESSION-500/leave",
            json={"participant_id": "P001"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["connected"] is False
    assert data["participant_id"] == "P001"


def test_leave_participant_route_validation_errors(client):
    """Test validation errors for leave route."""
    # Missing json body
    res1 = client.post("/api/gd/sessions/GD-SESSION-500/leave", data="not-json", content_type="application/json")
    assert res1.status_code == 400

    # Missing participant_id
    res2 = client.post("/api/gd/sessions/GD-SESSION-500/leave", json={})
    assert res2.status_code == 400
    assert "'participant_id' is required" in res2.get_json()["message"]


def test_get_runtime_status_route_success(client):
    """Test GET /api/gd/sessions/<session_id>/runtime route success."""
    mock_service = MagicMock()
    mock_service.get_runtime_status.return_value = {
        "session_id": "GD-SESSION-500",
        "status": "ACTIVE",
        "runtime_active": True,
        "elapsed_time": 45.2,
        "duration_seconds": 300,
        "participants": {"P001": {"connected": True}},
    }

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.get("/api/gd/sessions/GD-SESSION-500/runtime")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["runtime"]["runtime_active"] is True
    assert data["runtime"]["elapsed_time"] == 45.2


def test_get_runtime_status_route_not_found(client):
    """Test GET /api/gd/sessions/<session_id>/runtime returns 404 when session not found."""
    mock_service = MagicMock()
    mock_service.get_runtime_status.side_effect = ValueError("GD session 'NON-EXISTENT' not found")

    with patch("backend.routes.gd_routes.gd_live_service", mock_service):
        response = client.get("/api/gd/sessions/NON-EXISTENT/runtime")

    assert response.status_code == 404
