"""
Unit tests for GD Live Session Integration Service (backend/services/gd_live_service.py).

Tests verify that persistent GDSession state coordinates with in-memory LiveGDSession
runtime state without introducing diarization, VAD/ASR execution, speaker recognition,
continuous Qwen inference, or redundant MongoDB connections.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from backend.ai.gd.live_session import LiveGDSession
from backend.services.gd_live_service import GDLiveService
from backend.services.gd_session_service import GDSessionService


@pytest.fixture
def mock_session_service():
    """Mock GDSessionService with in-memory state tracking."""
    service = MagicMock(spec=GDSessionService)
    sessions_db = {}

    def mock_get_session(session_id):
        if session_id not in sessions_db:
            raise ValueError(f"GD session '{session_id}' not found")
        return sessions_db[session_id].copy()

    def mock_start_session(session_id):
        if session_id not in sessions_db:
            raise ValueError(f"GD session '{session_id}' not found")
        s = sessions_db[session_id]
        if s["status"] != "SCHEDULED":
            raise ValueError(f"Cannot start GD session in '{s['status']}' status. Must be 'SCHEDULED'.")
        s["status"] = "ACTIVE"
        s["session_started_at"] = datetime.now(timezone.utc)
        return s.copy()

    def mock_end_session(session_id):
        if session_id not in sessions_db:
            raise ValueError(f"GD session '{session_id}' not found")
        s = sessions_db[session_id]
        if s["status"] != "ACTIVE":
            raise ValueError(f"Cannot end GD session in '{s['status']}' status. Must be 'ACTIVE'.")
        s["status"] = "ENDED"
        s["session_ended_at"] = datetime.now(timezone.utc)
        s["session_duration"] = 120.0
        return s.copy()

    service.get_session.side_effect = mock_get_session
    service.start_session.side_effect = mock_start_session
    service.end_session.side_effect = mock_end_session
    service._db = sessions_db  # helper for test fixtures

    return service


@pytest.fixture
def sample_scheduled_session(mock_session_service):
    """Create a sample SCHEDULED session in mock service."""
    now = datetime.now(timezone.utc)
    session = {
        "session_id": "GD-LIVE-100",
        "topic": "Future of Clean Energy",
        "coordinator_id": "COORD-01",
        "participant_ids": ["P001", "P002", "P003"],
        "status": "SCHEDULED",
        "scheduled_start": now,
        "scheduled_end": None,
        "session_started_at": None,
        "session_ended_at": None,
        "session_duration": 300.0,
        "evaluation_id": None,
        "metadata": {"room": "CleanEnergy-1"},
        "created_at": now,
        "updated_at": now,
        "document_type": "gd_session",
    }
    mock_session_service._db["GD-LIVE-100"] = session
    return session


@pytest.fixture
def live_service(mock_session_service):
    """Instantiate GDLiveService with mock GDSessionService."""
    return GDLiveService(session_service=mock_session_service)


# ==============================================================================
# 1. RUNTIME INITIALIZATION & LIFECYCLE TESTS
# ==============================================================================

def test_scheduled_session_has_no_runtime(live_service, sample_scheduled_session):
    """1. Test that a SCHEDULED session has no active in-memory runtime."""
    assert live_service.get_runtime("GD-LIVE-100") is None
    status = live_service.get_runtime_status("GD-LIVE-100")
    assert status["session_id"] == "GD-LIVE-100"
    assert status["status"] == "SCHEDULED"
    assert status["runtime_active"] is False
    assert status["live_status"] is None


def test_start_session_creates_runtime(live_service, sample_scheduled_session):
    """2, 3, 4. Test that starting a session transitions persistent state and creates runtime with correct attributes."""
    result = live_service.start_session("GD-LIVE-100")

    # Persistent session updated
    assert result["session"]["status"] == "ACTIVE"

    # Runtime created
    runtime = live_service.get_runtime("GD-LIVE-100")
    assert runtime is not None
    assert runtime.session_id == "GD-LIVE-100"
    assert runtime.topic == "Future of Clean Energy"
    assert runtime.duration_seconds == 300
    assert runtime.started_at is not None

    # Authorized participants populated in runtime roster
    assert set(runtime.participants.keys()) == {"P001", "P002", "P003"}
    for p in runtime.participants.values():
        assert p.connected is False


def test_start_session_fails_if_not_scheduled(live_service, sample_scheduled_session):
    """Test start_session rejects non-SCHEDULED session."""
    live_service.start_session("GD-LIVE-100")

    with pytest.raises(ValueError, match="Cannot start GD session in 'ACTIVE' status. Must be 'SCHEDULED'."):
        live_service.start_session("GD-LIVE-100")


# ==============================================================================
# 2. PARTICIPANT JOIN & LEAVE TESTS
# ==============================================================================

def test_unauthorized_participant_cannot_join(live_service, sample_scheduled_session):
    """5. Test that an unauthorized participant not in scheduled roster cannot join."""
    live_service.start_session("GD-LIVE-100")

    with pytest.raises(ValueError, match="Participant 'P999' is not authorized for session 'GD-LIVE-100'"):
        live_service.join_participant("GD-LIVE-100", "P999")


def test_authorized_participant_can_join_active_session(live_service, sample_scheduled_session):
    """6. Test authorized participant successfully connects to ACTIVE session."""
    live_service.start_session("GD-LIVE-100")

    join_res = live_service.join_participant("GD-LIVE-100", "P001")
    assert join_res["session_id"] == "GD-LIVE-100"
    assert join_res["participant_id"] == "P001"
    assert join_res["connected"] is True

    runtime = live_service.get_runtime("GD-LIVE-100")
    assert runtime.participants["P001"].connected is True
    assert runtime.participants["P001"].joined_at is not None


def test_duplicate_participant_join_is_deterministic(live_service, sample_scheduled_session):
    """7. Test duplicate participant joins update connection state idempotently and deterministically."""
    live_service.start_session("GD-LIVE-100")

    res1 = live_service.join_participant("GD-LIVE-100", "P001")
    assert res1["connected"] is True

    res2 = live_service.join_participant("GD-LIVE-100", "P001")
    assert res2["connected"] is True

    runtime = live_service.get_runtime("GD-LIVE-100")
    assert runtime.participants["P001"].connected is True


def test_participant_leave_works(live_service, sample_scheduled_session):
    """8. Test connected participant leave marks disconnected without mutating persistent roster."""
    live_service.start_session("GD-LIVE-100")
    live_service.join_participant("GD-LIVE-100", "P001")

    leave_res = live_service.leave_participant("GD-LIVE-100", "P001")
    assert leave_res["session_id"] == "GD-LIVE-100"
    assert leave_res["participant_id"] == "P001"
    assert leave_res["connected"] is False

    runtime = live_service.get_runtime("GD-LIVE-100")
    assert runtime.participants["P001"].connected is False
    assert runtime.participants["P001"].left_at is not None

    # Verify persistent roster in session service is untouched
    persistent_session = live_service.session_service.get_session("GD-LIVE-100")
    assert "P001" in persistent_session["participant_ids"]


def test_participant_cannot_join_scheduled_session(live_service, sample_scheduled_session):
    """9. Test participant cannot join a SCHEDULED session."""
    with pytest.raises(ValueError, match="Cannot join session in 'SCHEDULED' status. Must be 'ACTIVE'."):
        live_service.join_participant("GD-LIVE-100", "P001")


def test_participant_cannot_join_ended_session(live_service, sample_scheduled_session):
    """10. Test participant cannot join an ENDED session."""
    live_service.start_session("GD-LIVE-100")
    live_service.end_session("GD-LIVE-100")

    with pytest.raises(ValueError, match="Cannot join session in 'ENDED' status. Must be 'ACTIVE'."):
        live_service.join_participant("GD-LIVE-100", "P001")


def test_participant_leave_unconnected_fails(live_service, sample_scheduled_session):
    """Test leaving an unconnected participant raises ValueError."""
    live_service.start_session("GD-LIVE-100")

    with pytest.raises(ValueError, match="is not currently connected"):
        live_service.leave_participant("GD-LIVE-100", "P001")


def test_active_session_with_missing_runtime_cannot_join(live_service, mock_session_service, sample_scheduled_session):
    """Test that an ACTIVE session without an existing in-memory runtime raises RuntimeError on join."""
    mock_session_service._db["GD-LIVE-100"]["status"] = "ACTIVE"
    assert live_service.get_runtime("GD-LIVE-100") is None

    with pytest.raises(
        RuntimeError,
        match="Live runtime for active GD session 'GD-LIVE-100' is not available",
    ):
        live_service.join_participant("GD-LIVE-100", "P001")


def test_missing_runtime_does_not_create_new_live_session(live_service, mock_session_service, sample_scheduled_session):
    """Test that attempting to join an ACTIVE session with missing runtime does NOT invoke factory or create runtime."""
    mock_session_service._db["GD-LIVE-100"]["status"] = "ACTIVE"
    factory_mock = MagicMock()
    live_service.live_session_factory = factory_mock

    with pytest.raises(RuntimeError):
        live_service.join_participant("GD-LIVE-100", "P001")

    factory_mock.assert_not_called()
    assert live_service.get_runtime("GD-LIVE-100") is None


# ==============================================================================
# 3. SESSION END & RUNTIME STATUS TESTS
# ==============================================================================

def test_ending_session_stops_runtime(live_service, sample_scheduled_session):
    """11. Test ending session transitions persistent state to ENDED and stops runtime clock."""
    live_service.start_session("GD-LIVE-100")
    live_service.join_participant("GD-LIVE-100", "P001")

    end_res = live_service.end_session("GD-LIVE-100")
    assert end_res["session"]["status"] == "ENDED"

    runtime = live_service.get_runtime("GD-LIVE-100")
    assert runtime.ended_at is not None

    status = live_service.get_runtime_status("GD-LIVE-100")
    assert status["status"] == "ENDED"
    assert status["runtime_active"] is False


def test_runtime_status_reports_correct_session_and_clock(live_service, sample_scheduled_session):
    """12, 13. Test runtime status reports correct session attributes using LiveGDSession clock."""
    live_service.start_session("GD-LIVE-100")
    live_service.join_participant("GD-LIVE-100", "P001")

    status = live_service.get_runtime_status("GD-LIVE-100")
    assert status["session_id"] == "GD-LIVE-100"
    assert status["status"] == "ACTIVE"
    assert status["runtime_active"] is True
    assert status["duration_seconds"] == 300
    assert status["elapsed_time"] >= 0.0
    assert "P001" in status["participants"]
    assert status["participants"]["P001"]["connected"] is True
    assert status["participants"]["P002"]["connected"] is False


def test_stop_runtime(live_service, sample_scheduled_session):
    """Test stop_runtime cleans up active runtime instance."""
    live_service.start_session("GD-LIVE-100")
    assert live_service.stop_runtime("GD-LIVE-100") is True
    assert live_service.get_runtime("GD-LIVE-100") is None
    assert live_service.stop_runtime("GD-LIVE-100") is False


# ==============================================================================
# 4. ZERO AI/AUDIO/VAD/ASR/QWEN SIDE-EFFECT TESTS
# ==============================================================================

def test_runtime_operations_do_not_call_vad_asr_qwen(live_service, sample_scheduled_session):
    """14-19. Test that creating runtime, joining, leaving, and ending NEVER invokes VAD, ASR, or Qwen."""
    with patch("backend.ai.gd.vad_service.VADService") as mock_vad, \
         patch("backend.ai.asr_service.ASRService") as mock_asr, \
         patch("backend.ai.qwen_service.ask_qwen") as mock_qwen:

        # 14, 15, 16. Start session / create runtime
        live_service.start_session("GD-LIVE-100")

        # 17, 18, 19. Join participant
        live_service.join_participant("GD-LIVE-100", "P001")

        # Leave participant
        live_service.leave_participant("GD-LIVE-100", "P001")

        # End session
        live_service.end_session("GD-LIVE-100")

        # Assert zero calls to VAD, ASR, and Qwen
        mock_vad.assert_not_called()
        mock_asr.assert_not_called()
        mock_qwen.assert_not_called()


# ==============================================================================
# 5. ROLLBACK & ERROR RESILIENCE TESTS
# ==============================================================================

def test_start_session_runtime_error_rolls_back_persistent_state(sample_scheduled_session):
    """Test that if LiveGDSession instantiation fails, persistent state is rolled back cleanly."""
    mock_service = MagicMock(spec=GDSessionService)
    mock_repo = MagicMock()
    mock_service.repo = mock_repo

    mock_service.get_session.return_value = sample_scheduled_session.copy()
    mock_service.start_session.return_value = {**sample_scheduled_session, "status": "ACTIVE"}

    def exploding_factory(*args, **kwargs):
        raise RuntimeError("Runtime initialization exploded")

    live_service = GDLiveService(session_service=mock_service, live_session_factory=exploding_factory)

    with pytest.raises(RuntimeError, match="Failed to initialize live runtime: Runtime initialization exploded"):
        live_service.start_session("GD-LIVE-100")

    # Verify rollback was attempted on repository
    mock_repo.update_session.assert_called_once_with(
        "GD-LIVE-100",
        {"status": "SCHEDULED", "session_started_at": None},
    )


# ==============================================================================
# 6. LIVE TRANSCRIPT & LIVE EVALUATION TESTS
# ==============================================================================

def test_get_transcript_active_and_empty(live_service, sample_scheduled_session):
    """Test get_transcript returns structured transcript data."""
    # Before start (fallback)
    t1 = live_service.get_transcript("GD-LIVE-100")
    assert t1["session_id"] == "GD-LIVE-100"
    assert t1["segment_count"] == 0
    assert t1["group_segments"] == []

    # After start with segments added
    live_service.start_session("GD-LIVE-100")
    runtime = live_service.get_runtime("GD-LIVE-100")
    runtime.add_transcript_segment("P001", 1.0, 3.5, "Hello team")
    runtime.add_transcript_segment("P002", 4.0, 6.0, "I agree with you")

    t2 = live_service.get_transcript("GD-LIVE-100")
    assert t2["segment_count"] == 2
    assert len(t2["group_segments"]) == 2
    assert "[P001] Hello team" in t2["group_transcript"]
    assert "[P002] I agree with you" in t2["group_transcript"]
    assert t2["participant_transcripts"]["P001"] == "Hello team"
    assert t2["participant_transcripts"]["P002"] == "I agree with you"


def test_evaluate_live_session_requires_ended_status(live_service, sample_scheduled_session):
    """Test evaluate_live_session rejects non-ENDED sessions."""
    # SCHEDULED
    with pytest.raises(ValueError, match="Must be 'ENDED' or 'EVALUATED'"):
        live_service.evaluate_live_session("GD-LIVE-100")

    # ACTIVE
    live_service.start_session("GD-LIVE-100")
    with pytest.raises(ValueError, match="Must be 'ENDED' or 'EVALUATED'"):
        live_service.evaluate_live_session("GD-LIVE-100")


def test_evaluate_live_session_success(live_service, sample_scheduled_session):
    """Test evaluate_live_session orchestrates evaluation, saves to repository, and marks evaluated."""
    live_service.start_session("GD-LIVE-100")
    runtime = live_service.get_runtime("GD-LIVE-100")
    runtime.add_transcript_segment("P001", 1.0, 5.0, "We need to invest in solar infrastructure.")
    runtime.add_transcript_segment("P002", 5.5, 9.0, "Storage batteries are crucial for solar.")
    runtime.participants["P001"].speaking_time = 4.0
    runtime.participants["P001"].word_count = 7
    runtime.participants["P002"].speaking_time = 3.5
    runtime.participants["P002"].word_count = 6

    # End session
    live_service.end_session("GD-LIVE-100")

    # Mock orchestrator and gd_repo
    mock_orchestrator = MagicMock()
    mock_orchestrator.evaluate_session.return_value = [
        {
            "participant_id": "P001",
            "scorecard": {"final_score": 85.0, "scores": {"relevance": 85.0}},
            "agent_results": [{"agent": "relevance", "score": 85.0}],
        },
        {
            "participant_id": "P002",
            "scorecard": {"final_score": 80.0, "scores": {"relevance": 80.0}},
            "agent_results": [{"agent": "relevance", "score": 80.0}],
        },
        {
            "participant_id": "P003",
            "scorecard": {"final_score": 0.0, "scores": {"relevance": 0.0}},
            "agent_results": [{"agent": "relevance", "score": 0.0}],
        },
    ]

    mock_repo = MagicMock()
    mock_repo.save_evaluation.side_effect = lambda doc: {**doc, "_id": "mock_eval_id"}

    live_service._orchestrator = mock_orchestrator
    live_service._gd_repo = mock_repo

    eval_result = live_service.evaluate_live_session("GD-LIVE-100")

    assert eval_result["session_id"] == "GD-LIVE-100"
    assert eval_result["topic"] == "Future of Clean Energy"
    assert len(eval_result["participants"]) == 3
    assert eval_result["final_scores"]["P001"] == 85.0
    assert eval_result["final_scores"]["P002"] == 80.0
    assert eval_result["final_scores"]["P003"] == 0.0

    mock_repo.save_evaluation.assert_called_once()
    mock_orchestrator.evaluate_session.assert_called_once()
