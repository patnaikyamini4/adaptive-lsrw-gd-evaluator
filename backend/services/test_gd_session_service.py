"""
Unit tests for GD Session Application Service (backend/services/gd_session_service.py).

All tests use a mocked repository to run deterministically and offline.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from backend.services.gd_session_service import GDSessionService


@pytest.fixture
def mock_repo():
    """Mock repository with in-memory state tracking."""
    repo = MagicMock()
    storage = {}

    def mock_create(data):
        storage[data["session_id"]] = data.copy()
        return data.copy()

    def mock_get(session_id):
        doc = storage.get(session_id)
        return doc.copy() if doc else None

    def mock_update(session_id, updates):
        if session_id not in storage:
            return None
        storage[session_id].update(updates)
        return storage[session_id].copy()

    def mock_delete(session_id):
        if session_id in storage:
            del storage[session_id]
            return True
        return False

    def mock_list(limit=100, status=None, coordinator_id=None):
        res = list(storage.values())
        if status:
            res = [d for d in res if d.get("status") == status]
        if coordinator_id:
            res = [d for d in res if d.get("coordinator_id") == coordinator_id]
        return [d.copy() for d in res[:limit]]

    repo.create_session.side_effect = mock_create
    repo.get_session.side_effect = mock_get
    repo.update_session.side_effect = mock_update
    repo.delete_session.side_effect = mock_delete
    repo.list_sessions.side_effect = mock_list

    return repo


@pytest.fixture
def session_service(mock_repo):
    """Instance of GDSessionService using mock_repo."""
    return GDSessionService(repository=mock_repo)


# ==============================================================================
# CREATE & GET TESTS
# ==============================================================================

def test_create_session(session_service):
    """Test standard session creation."""
    session = session_service.create_session(
        topic="Climate Change Solutions",
        coordinator_id="COORD-01",
        participant_ids=["P001", "P002", "P003"],
        session_id="GD-TEST-001",
    )

    assert session["session_id"] == "GD-TEST-001"
    assert session["topic"] == "Climate Change Solutions"
    assert session["coordinator_id"] == "COORD-01"
    assert session["participant_ids"] == ["P001", "P002", "P003"]
    assert session["status"] == "SCHEDULED"
    assert session["session_started_at"] is None
    assert session["session_ended_at"] is None


def test_get_session_not_found(session_service):
    """Test getting non-existent session raises ValueError."""
    with pytest.raises(ValueError, match="GD session 'NON-EXISTENT' not found"):
        session_service.get_session("NON-EXISTENT")


# ==============================================================================
# LIFECYCLE TRANSITION TESTS
# ==============================================================================

def test_start_session_scheduled_to_active(session_service):
    """Test valid transition: SCHEDULED -> ACTIVE."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-101",
    )

    active_session = session_service.start_session("GD-101")
    assert active_session["status"] == "ACTIVE"
    assert active_session["session_started_at"] is not None


def test_end_session_active_to_ended(session_service):
    """Test valid transition: ACTIVE -> ENDED and session duration calculation."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-101",
    )
    started = session_service.start_session("GD-101")

    # Manually set started_at 60 seconds in the past
    past_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    session_service.repo.update_session("GD-101", {"session_started_at": past_time})

    ended_session = session_service.end_session("GD-101")
    assert ended_session["status"] == "ENDED"
    assert ended_session["session_ended_at"] is not None
    assert ended_session["session_duration"] >= 59.0


def test_mark_evaluated_ended_to_evaluated(session_service):
    """Test valid transition: ENDED -> EVALUATED."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-101",
    )
    session_service.start_session("GD-101")
    session_service.end_session("GD-101")

    evaluated_session = session_service.mark_evaluated("GD-101", evaluation_id="EVAL-DOC-999")
    assert evaluated_session["status"] == "EVALUATED"
    assert evaluated_session["evaluation_id"] == "EVAL-DOC-999"


def test_cancel_session_from_scheduled_and_active(session_service):
    """Test cancellation from SCHEDULED and ACTIVE states."""
    # From SCHEDULED
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-101",
    )
    cancelled1 = session_service.cancel_session("GD-101", reason="Lack of quorum")
    assert cancelled1["status"] == "CANCELLED"
    assert cancelled1["metadata"]["cancellation_reason"] == "Lack of quorum"

    # From ACTIVE
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-102",
    )
    session_service.start_session("GD-102")
    cancelled2 = session_service.cancel_session("GD-102", reason="Technical outage")
    assert cancelled2["status"] == "CANCELLED"
    assert cancelled2["metadata"]["cancellation_reason"] == "Technical outage"
    assert cancelled2["session_ended_at"] is not None


# ==============================================================================
# INVALID LIFECYCLE TRANSITION TESTS
# ==============================================================================

def test_invalid_transitions(session_service):
    """Test all invalid state transitions raise ValueError with clear explanation."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-101",
    )

    # 1. SCHEDULED -> end (Must be ACTIVE)
    with pytest.raises(ValueError, match="Cannot end GD session in 'SCHEDULED' status. Must be 'ACTIVE'."):
        session_service.end_session("GD-101")

    # 2. SCHEDULED -> evaluate (Must be ENDED)
    with pytest.raises(ValueError, match="Cannot mark GD session as evaluated in 'SCHEDULED' status. Must be 'ENDED'."):
        session_service.mark_evaluated("GD-101", "EVAL-1")

    # Transition to ACTIVE
    session_service.start_session("GD-101")

    # 3. ACTIVE -> start (Cannot start again)
    with pytest.raises(ValueError, match="Cannot start GD session in 'ACTIVE' status. Must be 'SCHEDULED'."):
        session_service.start_session("GD-101")

    # 4. ACTIVE -> evaluate (Must be ENDED)
    with pytest.raises(ValueError, match="Cannot mark GD session as evaluated in 'ACTIVE' status. Must be 'ENDED'."):
        session_service.mark_evaluated("GD-101", "EVAL-1")

    # Transition to ENDED
    session_service.end_session("GD-101")

    # 5. ENDED -> start (Cannot start ended session)
    with pytest.raises(ValueError, match="Cannot start GD session in 'ENDED' status. Must be 'SCHEDULED'."):
        session_service.start_session("GD-101")

    # 6. ENDED -> cancel (Cannot cancel ended session)
    with pytest.raises(ValueError, match="Cannot cancel GD session in 'ENDED' status."):
        session_service.cancel_session("GD-101")

    # Transition to EVALUATED
    session_service.mark_evaluated("GD-101", "EVAL-1")

    # 7. EVALUATED -> start
    with pytest.raises(ValueError, match="Cannot start GD session in 'EVALUATED' status."):
        session_service.start_session("GD-101")

    # 8. EVALUATED -> end
    with pytest.raises(ValueError, match="Cannot end GD session in 'EVALUATED' status."):
        session_service.end_session("GD-101")

    # 9. EVALUATED -> cancel
    with pytest.raises(ValueError, match="Cannot cancel GD session in 'EVALUATED' status."):
        session_service.cancel_session("GD-101")


def test_cancelled_session_cannot_start_or_end(session_service):
    """Test that a CANCELLED session cannot start or end."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001"],
        session_id="GD-CANCEL",
    )
    session_service.cancel_session("GD-CANCEL")

    with pytest.raises(ValueError, match="Cannot start GD session in 'CANCELLED' status."):
        session_service.start_session("GD-CANCEL")

    with pytest.raises(ValueError, match="Cannot end GD session in 'CANCELLED' status."):
        session_service.end_session("GD-CANCEL")


# ==============================================================================
# PARTICIPANT MODIFICATION TESTS
# ==============================================================================

def test_participant_operations_in_scheduled_state(session_service):
    """Test adding and removing participants in SCHEDULED state."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001", "P002"],
        session_id="GD-PART",
    )

    # Add participant
    updated = session_service.add_participant("GD-PART", "P003")
    assert updated["participant_ids"] == ["P001", "P002", "P003"]

    # Reject duplicate add
    with pytest.raises(ValueError, match="Participant 'P003' is already registered"):
        session_service.add_participant("GD-PART", "P003")

    # Remove participant
    updated = session_service.remove_participant("GD-PART", "P002")
    assert updated["participant_ids"] == ["P001", "P003"]

    # Reject remove non-existent participant
    with pytest.raises(ValueError, match="Participant 'P999' not found"):
        session_service.remove_participant("GD-PART", "P999")


def test_participant_modification_fails_after_active(session_service):
    """Test adding or removing participants once session is ACTIVE fails."""
    session_service.create_session(
        topic="Topic",
        coordinator_id="COORD-01",
        participant_ids=["P001", "P002"],
        session_id="GD-PART-ACTIVE",
    )
    session_service.start_session("GD-PART-ACTIVE")

    # Add in ACTIVE state must fail
    with pytest.raises(ValueError, match="Cannot add participant to session in 'ACTIVE' status. Must be 'SCHEDULED'."):
        session_service.add_participant("GD-PART-ACTIVE", "P003")

    # Remove in ACTIVE state must fail
    with pytest.raises(ValueError, match="Cannot remove participant from session in 'ACTIVE' status. Must be 'SCHEDULED'."):
        session_service.remove_participant("GD-PART-ACTIVE", "P001")


# ==============================================================================
# LIST & DELETE TESTS
# ==============================================================================

def test_list_and_delete_sessions(session_service):
    """Test listing and deleting sessions."""
    session_service.create_session("T1", "COORD-A", ["P1"], "S1")
    session_service.create_session("T2", "COORD-B", ["P2"], "S2")

    listed = session_service.list_sessions(coordinator_id="COORD-A")
    assert len(listed) == 1
    assert listed[0]["session_id"] == "S1"

    deleted = session_service.delete_session("S1")
    assert deleted is True

    deleted_again = session_service.delete_session("S1")
    assert deleted_again is False
