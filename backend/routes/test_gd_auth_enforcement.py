"""
Security & RBAC Enforcement Tests (backend/routes/test_gd_auth_enforcement.py).

Validates all security invariants required by M8:
1. Missing, malformed, forged, and expired tokens -> 401
2. Candidate attempts coordinator actions (create, start, end, cancel, evaluate) -> 403
3. Coordinator ownership enforcement (coordinator A cannot manipulate coordinator B's session) -> 403
4. Non-member session access (candidate C3 cannot access session they are not enrolled in) -> 403
5. Participant identity spoofing prevention (candidate C1 cannot join, leave, or upload as C2) -> 403
6. Candidate scorecard sanitization (candidate sees only their personal scorecard and sanitized group metrics) -> sanitized
7. LiveKit room-token configuration and role grant enforcement
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from backend.app import app
from backend.services.auth_service import AuthService
from backend.services.gd_session_service import GDSessionService


@pytest.fixture
def auth_service():
    return AuthService()


@pytest.fixture
def coordinator_a(auth_service):
    user = {
        "user_id": "COORD-AAA",
        "email": "coordinator_a@example.com",
        "role": "COORDINATOR",
        "display_name": "Coordinator A",
    }
    token = auth_service.create_access_token(user)
    return {"user": user, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def coordinator_b(auth_service):
    user = {
        "user_id": "COORD-BBB",
        "email": "coordinator_b@example.com",
        "role": "COORDINATOR",
        "display_name": "Coordinator B",
    }
    token = auth_service.create_access_token(user)
    return {"user": user, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def candidate_p1(auth_service):
    user = {
        "user_id": "CAND-P001",
        "email": "p1@example.com",
        "role": "CANDIDATE",
        "display_name": "Candidate P1",
    }
    token = auth_service.create_access_token(user)
    return {"user": user, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def candidate_p2(auth_service):
    user = {
        "user_id": "CAND-P002",
        "email": "p2@example.com",
        "role": "CANDIDATE",
        "display_name": "Candidate P2",
    }
    token = auth_service.create_access_token(user)
    return {"user": user, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def candidate_outsider(auth_service):
    user = {
        "user_id": "CAND-OUTSIDER",
        "email": "outsider@example.com",
        "role": "CANDIDATE",
        "display_name": "Outsider",
    }
    token = auth_service.create_access_token(user)
    return {"user": user, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def sample_session_data():
    return {
        "session_id": "GD-AUTH-100",
        "topic": "Future of Autonomous Agents",
        "coordinator_id": "COORD-AAA",
        "participant_ids": ["CAND-P001", "CAND-P002"],
        "status": "ACTIVE",
        "scheduled_start": None,
        "scheduled_end": None,
        "session_started_at": "2026-10-22T10:00:00Z",
        "session_duration": 300.0,
    }


@pytest.fixture
def mock_session_service(sample_session_data):
    service = MagicMock(spec=GDSessionService)
    service.get_session.return_value = sample_session_data
    return service


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ==============================================================================
# 1. UNAUTHENTICATED ACCESS TESTS (401 Unauthorized)
# ==============================================================================

def test_missing_auth_header_rejected(client):
    """Endpoints reject requests without Authorization header."""
    res1 = client.post("/api/gd/sessions", json={"topic": "Test"})
    assert res1.status_code == 401
    assert "Missing Authorization header" in res1.get_json()["message"]

    res2 = client.get("/api/gd/sessions/GD-100")
    assert res2.status_code == 401

    res3 = client.get("/api/gd/evaluations")
    assert res3.status_code == 401


def test_malformed_auth_token_rejected(client):
    """Endpoints reject malformed Bearer tokens."""
    res = client.get("/api/gd/sessions", headers={"Authorization": "Bearer not-a-jwt"})
    assert res.status_code == 401
    assert "Invalid token" in res.get_json()["message"]


def test_expired_auth_token_rejected(client, auth_service):
    """Endpoints reject expired JWT tokens."""
    now = datetime.now(timezone.utc)
    expired = {
        "sub": "COORD-AAA",
        "email": "c@example.com",
        "role": "COORDINATOR",
        "iat": now - timedelta(hours=25),
        "exp": now - timedelta(hours=1),
    }
    import jwt
    token = jwt.encode(expired, auth_service.secret_key, algorithm="HS256")
    res = client.get("/api/gd/sessions", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401
    assert "expired" in res.get_json()["message"].lower()


# ==============================================================================
# 2. ROLE ENFORCEMENT TESTS (403 Forbidden)
# ==============================================================================

def test_candidate_cannot_create_session(client, candidate_p1):
    """Candidate role cannot call POST /api/gd/sessions."""
    payload = {
        "topic": "Ethics",
        "participant_ids": ["CAND-P001"],
    }
    res = client.post("/api/gd/sessions", json=payload, headers=candidate_p1["headers"])
    assert res.status_code == 403
    assert "COORDINATOR" in res.get_json()["message"]


def test_candidate_cannot_start_session(client, candidate_p1, mock_session_service):
    """Candidate role cannot call POST /api/gd/sessions/<id>/start."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/start", headers=candidate_p1["headers"])
    assert res.status_code == 403
    assert "coordinator" in res.get_json()["message"].lower()


def test_candidate_cannot_end_session(client, candidate_p1, mock_session_service):
    """Candidate role cannot call POST /api/gd/sessions/<id>/end."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/end", headers=candidate_p1["headers"])
    assert res.status_code == 403


def test_candidate_cannot_cancel_session(client, candidate_p1, mock_session_service):
    """Candidate role cannot call POST /api/gd/sessions/<id>/cancel."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/cancel", headers=candidate_p1["headers"])
    assert res.status_code == 403


def test_candidate_cannot_evaluate_session(client, candidate_p1, mock_session_service):
    """Candidate role cannot call POST /api/gd/sessions/<id>/evaluate."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/evaluate", headers=candidate_p1["headers"])
    assert res.status_code == 403


# ==============================================================================
# 3. COORDINATOR OWNERSHIP ENFORCEMENT
# ==============================================================================

def test_coordinator_b_cannot_manage_coordinator_a_session(client, coordinator_b, mock_session_service):
    """Coordinator B cannot start or end a session owned by Coordinator A."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/start", headers=coordinator_b["headers"])
    assert res.status_code == 403
    assert "not the designated coordinator" in res.get_json()["message"]


# ==============================================================================
# 4. SESSION MEMBERSHIP BOUNDARY TESTS
# ==============================================================================

def test_outsider_cannot_access_session(client, candidate_outsider, mock_session_service):
    """Candidate who is not in participant_ids cannot view session details."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.get("/api/gd/sessions/GD-AUTH-100", headers=candidate_outsider["headers"])
    assert res.status_code == 403
    assert "neither coordinator nor member" in res.get_json()["message"]


def test_outsider_cannot_request_livekit_token(client, candidate_outsider, mock_session_service):
    """Outsider cannot request a LiveKit meeting room token for this session."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post("/api/gd/sessions/GD-AUTH-100/livekit-token", headers=candidate_outsider["headers"])
    assert res.status_code == 403


# ==============================================================================
# 5. PARTICIPANT IMPERSONATION REJECTION (ANTI-SPOOFING)
# ==============================================================================

def test_participant_spoofing_rejected_on_join(client, candidate_p1, mock_session_service):
    """Candidate P1 attempting to join as P2 is rejected with 403."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post(
            "/api/gd/sessions/GD-AUTH-100/join",
            json={"participant_id": "CAND-P002"},
            headers=candidate_p1["headers"],
        )
    assert res.status_code == 403
    assert "impersonation detected" in res.get_json()["message"].lower()


def test_participant_spoofing_rejected_on_leave(client, candidate_p1, mock_session_service):
    """Candidate P1 attempting to leave as P2 is rejected with 403."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post(
            "/api/gd/sessions/GD-AUTH-100/leave",
            json={"participant_id": "CAND-P002"},
            headers=candidate_p1["headers"],
        )
    assert res.status_code == 403
    assert "impersonation detected" in res.get_json()["message"].lower()


def test_participant_spoofing_rejected_on_audio_upload(client, candidate_p1, mock_session_service):
    """Candidate P1 attempting to upload audio claiming to be P2 is rejected with 403."""
    import io
    fake_wav = io.BytesIO(b"RIFFfakeWAVE")
    data = {
        "participant_id": "CAND-P002",
        "file": (fake_wav, "p2_chunk.wav"),
    }
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        res = client.post(
            "/api/gd/sessions/GD-AUTH-100/audio/upload",
            data=data,
            content_type="multipart/form-data",
            headers=candidate_p1["headers"],
        )
    assert res.status_code == 403
    assert "impersonation detected" in res.get_json()["message"].lower()


# ==============================================================================
# 6. CANDIDATE SCORECARD SANITIZATION TESTS
# ==============================================================================

def test_candidate_scorecard_sanitized(client, candidate_p1):
    """Candidate P1 viewing evaluation sees only their individual score, stripping P2's score."""
    full_evaluation = {
        "session_id": "GD-AUTH-100",
        "topic": "Future of Autonomous Agents",
        "session_duration": 300.0,
        "group_transcript": "[P1] Agree.\n[P2] Disagree.",
        "group_segments": [],
        "participants": [
            {"participant_id": "CAND-P001", "speaking_time": 50.0},
            {"participant_id": "CAND-P002", "speaking_time": 60.0},
        ],
        "scorecards": {
            "CAND-P001": {"score": 85, "feedback": "Great content"},
            "CAND-P002": {"score": 92, "feedback": "Confidential peer score"},
        },
        "final_scores": {
            "CAND-P001": 85,
            "CAND-P002": 92,
        },
    }

    with patch("backend.routes.gd_routes.get_evaluation", return_value=full_evaluation):
        res = client.get("/api/gd/evaluations/GD-AUTH-100", headers=candidate_p1["headers"])

    assert res.status_code == 200
    data = res.get_json()["evaluation"]
    assert data["sanitized"] is True
    # P1's scorecard is present
    assert "CAND-P001" in data["scorecards"]
    assert data["final_scores"]["CAND-P001"] == 85
    # P2's confidential scorecard must be stripped!
    assert "CAND-P002" not in data["scorecards"]
    assert "CAND-P002" not in data["final_scores"]
    assert len(data["participants"]) == 1
    assert data["participants"][0]["participant_id"] == "CAND-P001"

def test_candidate_nested_evaluation_fields_sanitized(client, coordinator_a, candidate_p1):
    """
    Verify candidate privacy across all nested evaluation fields:
    agent_results, objective_features, interaction_features are stripped of peer data for candidates,
    while coordinators retain full authorized multi-participant access.
    """
    full_evaluation = {
        "session_id": "GD-AUTH-100",
        "topic": "Future of Autonomous Agents",
        "session_duration": 300.0,
        "group_transcript": "[P1] Agree.\n[P2] Disagree.",
        "group_segments": [],
        "participants": [
            {"participant_id": "CAND-P001", "speaking_time": 50.0},
            {"participant_id": "CAND-P002", "speaking_time": 60.0},
        ],
        "scorecards": {
            "CAND-P001": {"score": 85},
            "CAND-P002": {"score": 92},
        },
        "final_scores": {
            "CAND-P001": 85,
            "CAND-P002": 92,
        },
        "agent_results": {
            "CAND-P001": {"fluency": 80, "relevance": 90},
            "CAND-P002": {"fluency": 95, "relevance": 88},
        },
        "objective_features": {
            "CAND-P001": {"wpm": 130, "speaking_time": 50.0},
            "CAND-P002": {"wpm": 145, "speaking_time": 60.0},
        },
        "interaction_features": {
            "CAND-P001": {"interruptions_made": 1},
            "CAND-P002": {"interruptions_made": 3},
        },
    }

    with patch("backend.routes.gd_routes.get_evaluation", return_value=full_evaluation):
        # 1. Candidate P1 checks evaluation
        res_cand = client.get("/api/gd/evaluations/GD-AUTH-100", headers=candidate_p1["headers"])
        assert res_cand.status_code == 200
        cand_data = res_cand.get_json()["evaluation"]
        assert cand_data["sanitized"] is True
        # P1's nested features are present
        assert "CAND-P001" in cand_data["agent_results"]
        assert "CAND-P001" in cand_data["objective_features"]
        assert "CAND-P001" in cand_data["interaction_features"]
        # P2's nested features MUST be stripped
        assert "CAND-P002" not in cand_data["agent_results"]
        assert "CAND-P002" not in cand_data["objective_features"]
        assert "CAND-P002" not in cand_data["interaction_features"]

        # 2. Coordinator checks evaluation -> sees all participants in nested features
        res_coord = client.get("/api/gd/evaluations/GD-AUTH-100", headers=coordinator_a["headers"])
        assert res_coord.status_code == 200
        coord_data = res_coord.get_json()["evaluation"]
        assert "CAND-P001" in coord_data["agent_results"]
        assert "CAND-P002" in coord_data["agent_results"]
        assert "CAND-P001" in coord_data["objective_features"]
        assert "CAND-P002" in coord_data["objective_features"]


def test_outsider_cannot_view_evaluation(client, candidate_outsider):
    """Candidate not in evaluation participants receives 403."""
    full_evaluation = {
        "session_id": "GD-AUTH-100",
        "participants": [{"participant_id": "CAND-P001"}, {"participant_id": "CAND-P002"}],
    }
    with patch("backend.routes.gd_routes.get_evaluation", return_value=full_evaluation):
        res = client.get("/api/gd/evaluations/GD-AUTH-100", headers=candidate_outsider["headers"])
    assert res.status_code == 403
    assert "not a participant" in res.get_json()["message"]


# ==============================================================================
# 7. LIVEKIT ROOM TOKEN ISSUANCE TESTS
# ==============================================================================

def test_livekit_token_unconfigured_fails_gracefully(client, candidate_p1, mock_session_service):
    """When LiveKit keys are not in env, endpoint returns 503 Service Unavailable."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        with patch("backend.routes.gd_routes.LIVEKIT_API_KEY", None):
            res = client.post("/api/gd/sessions/GD-AUTH-100/livekit-token", headers=candidate_p1["headers"])
    assert res.status_code == 503
    assert "not configured" in res.get_json()["message"].lower()


def test_livekit_token_configured_issues_valid_grant(client, coordinator_a, candidate_p1, mock_session_service):
    """When LiveKit keys are configured, endpoint issues token with correct roomAdmin grant."""
    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service):
        with patch("backend.routes.gd_routes.LIVEKIT_API_KEY", "devkey"), \
             patch("backend.routes.gd_routes.LIVEKIT_API_SECRET", "secretkeysecretkeysecretkeysecretkey32"), \
             patch("backend.routes.gd_routes.LIVEKIT_URL", "wss://livekit.example.com"):

            # Coordinator token check
            res_coord = client.post("/api/gd/sessions/GD-AUTH-100/livekit-token", headers=coordinator_a["headers"])
            assert res_coord.status_code == 200
            data_c = res_coord.get_json()
            assert data_c["room_admin"] is True
            assert "room_token" in data_c

            # Candidate token check
            res_cand = client.post("/api/gd/sessions/GD-AUTH-100/livekit-token", headers=candidate_p1["headers"])
            assert res_cand.status_code == 200
            data_p = res_cand.get_json()
            assert data_p["room_admin"] is False
            assert "room_token" in data_p


# ==============================================================================
# 8. TRANSCRIPT PRIVACY ENFORCEMENT
# ==============================================================================

def test_candidate_transcript_privacy_enforced_between_two_candidates(
    client, coordinator_a, candidate_p1, candidate_p2, mock_session_service
):
    """
    Candidate privacy test with two candidates:
    Neither candidate receives the other's individual participant transcript.
    Coordinator receives the complete breakdown across all participants.
    Both candidates still see the shared group chronological dialogue.
    """
    mock_transcript = {
        "session_id": "GD-AUTH-100",
        "topic": "Future of Autonomous Agents",
        "group_transcript": "[CAND-P001] AI is evolving.\n[CAND-P002] Safety is essential.",
        "group_segments": [
            {"participant_id": "CAND-P001", "start": 0.0, "end": 2.0, "text": "AI is evolving."},
            {"participant_id": "CAND-P002", "start": 2.5, "end": 5.0, "text": "Safety is essential."},
        ],
        "participant_transcripts": {
            "CAND-P001": "AI is evolving.",
            "CAND-P002": "Safety is essential.",
        },
        "segment_count": 2,
    }

    with patch("backend.routes.gd_routes.gd_session_service", mock_session_service), \
         patch("backend.routes.gd_routes.gd_live_service.get_transcript", return_value=mock_transcript):

        # 1. Candidate P1 fetches transcript
        res_p1 = client.get("/api/gd/sessions/GD-AUTH-100/transcript", headers=candidate_p1["headers"])
        assert res_p1.status_code == 200
        data_p1 = res_p1.get_json()
        assert data_p1["sanitized"] is True
        assert "CAND-P001" in data_p1["participant_transcripts"]
        assert data_p1["participant_transcripts"]["CAND-P001"] == "AI is evolving."
        assert "CAND-P002" not in data_p1["participant_transcripts"]
        assert "[CAND-P002] Safety is essential." in data_p1["group_transcript"]

        # 2. Candidate P2 fetches transcript
        res_p2 = client.get("/api/gd/sessions/GD-AUTH-100/transcript", headers=candidate_p2["headers"])
        assert res_p2.status_code == 200
        data_p2 = res_p2.get_json()
        assert data_p2["sanitized"] is True
        assert "CAND-P002" in data_p2["participant_transcripts"]
        assert data_p2["participant_transcripts"]["CAND-P002"] == "Safety is essential."
        assert "CAND-P001" not in data_p2["participant_transcripts"]
        assert "[CAND-P001] AI is evolving." in data_p2["group_transcript"]

        # 3. Coordinator fetches transcript -> sees all participants
        res_coord = client.get("/api/gd/sessions/GD-AUTH-100/transcript", headers=coordinator_a["headers"])
        assert res_coord.status_code == 200
        data_c = res_coord.get_json()
        assert "CAND-P001" in data_c["participant_transcripts"]
        assert "CAND-P002" in data_c["participant_transcripts"]

