"""
Unit tests for GDSession data model (backend/models/gd_session.py).
"""

from datetime import datetime, timezone
import pytest

from backend.models.gd_session import GDSession


def test_valid_gd_session_creation():
    """Test standard initialization with default and custom values."""
    now = datetime.now(timezone.utc)
    session = GDSession(
        session_id="GD-101",
        topic="Future of Renewable Energy",
        coordinator_id="COORD-01",
        participant_ids=["P001", "P002", "P003"],
        scheduled_start=now,
    )

    assert session.session_id == "GD-101"
    assert session.topic == "Future of Renewable Energy"
    assert session.coordinator_id == "COORD-01"
    assert session.participant_ids == ["P001", "P002", "P003"]
    assert session.status == "SCHEDULED"
    assert session.document_type == "gd_session"
    assert session.scheduled_start == now
    assert session.session_started_at is None
    assert session.session_ended_at is None
    assert session.session_duration is None
    assert session.evaluation_id is None
    assert session.metadata == {}


def test_gd_session_to_dict():
    """Test serialization to dictionary."""
    now = datetime.now(timezone.utc)
    session = GDSession(
        session_id="GD-202",
        topic="AI Ethics in Medicine",
        coordinator_id="COORD-02",
        participant_ids=["P001", "P002"],
        status="ACTIVE",
        session_started_at=now,
        metadata={"track": "healthcare"},
    )

    d = session.to_dict()
    assert d["session_id"] == "GD-202"
    assert d["topic"] == "AI Ethics in Medicine"
    assert d["coordinator_id"] == "COORD-02"
    assert d["participant_ids"] == ["P001", "P002"]
    assert d["status"] == "ACTIVE"
    assert d["document_type"] == "gd_session"
    assert d["session_started_at"] == now
    assert d["metadata"] == {"track": "healthcare"}


def test_gd_session_participant_list_preservation():
    """Test that participant IDs are stripped, preserved, and copies returned in to_dict."""
    session = GDSession(
        session_id="GD-303",
        topic="Climate Action",
        coordinator_id="COORD-03",
        participant_ids=[" P001 ", "P002", "P003"],
    )

    assert session.participant_ids == ["P001", "P002", "P003"]
    d = session.to_dict()
    assert d["participant_ids"] == ["P001", "P002", "P003"]
    # Modifying the returned dict copy does not mutate the instance
    d["participant_ids"].append("P004")
    assert session.participant_ids == ["P001", "P002", "P003"]


@pytest.mark.parametrize(
    "field_name, kwargs, expected_err",
    [
        ("session_id", {"session_id": "", "topic": "T", "coordinator_id": "C", "participant_ids": ["P1"]}, "session_id is required"),
        ("session_id", {"session_id": "   ", "topic": "T", "coordinator_id": "C", "participant_ids": ["P1"]}, "session_id is required"),
        ("session_id", {"session_id": None, "topic": "T", "coordinator_id": "C", "participant_ids": ["P1"]}, "session_id is required"),
        ("topic", {"session_id": "S", "topic": "", "coordinator_id": "C", "participant_ids": ["P1"]}, "topic is required"),
        ("topic", {"session_id": "S", "topic": "   ", "coordinator_id": "C", "participant_ids": ["P1"]}, "topic is required"),
        ("coordinator_id", {"session_id": "S", "topic": "T", "coordinator_id": "", "participant_ids": ["P1"]}, "coordinator_id is required"),
        ("coordinator_id", {"session_id": "S", "topic": "T", "coordinator_id": "   ", "participant_ids": ["P1"]}, "coordinator_id is required"),
        ("participant_ids", {"session_id": "S", "topic": "T", "coordinator_id": "C", "participant_ids": "not-a-list"}, "participant_ids must be a list"),
        ("participant_ids", {"session_id": "S", "topic": "T", "coordinator_id": "C", "participant_ids": ["P1", ""]}, r"participant_ids\[1\] must be a non-empty string"),
        ("participant_ids", {"session_id": "S", "topic": "T", "coordinator_id": "C", "participant_ids": ["P1", "   "]}, r"participant_ids\[1\] must be a non-empty string"),
    ],
)
def test_gd_session_invalid_required_fields(field_name, kwargs, expected_err):
    """Test rejection of invalid required parameters."""
    with pytest.raises(ValueError, match=expected_err):
        GDSession(**kwargs)


def test_gd_session_duplicate_participant_ids_rejected():
    """Test rejection of duplicate participant IDs in participant list."""
    with pytest.raises(ValueError, match="Duplicate participant_id found: 'P001'"):
        GDSession(
            session_id="GD-101",
            topic="Topic",
            coordinator_id="C1",
            participant_ids=["P001", "P002", "P001"],
        )


def test_gd_session_invalid_status_rejected():
    """Test rejection of status not in ALLOWED_STATUSES."""
    with pytest.raises(ValueError, match="Invalid status 'UNKNOWN'"):
        GDSession(
            session_id="GD-101",
            topic="Topic",
            coordinator_id="C1",
            participant_ids=["P001"],
            status="UNKNOWN",
        )


@pytest.mark.parametrize("valid_status", ["SCHEDULED", "ACTIVE", "ENDED", "EVALUATED", "CANCELLED"])
def test_gd_session_all_valid_statuses(valid_status):
    """Test all allowed statuses can be set successfully."""
    session = GDSession(
        session_id="GD-101",
        topic="Topic",
        coordinator_id="C1",
        participant_ids=["P001"],
        status=valid_status,
    )
    assert session.status == valid_status
