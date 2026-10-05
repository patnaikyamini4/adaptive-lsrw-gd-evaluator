"""
Unit tests for GD Session MongoDB Repository (backend/services/gd_session_repository.py).

All tests use a mocked PyMongo collection to run deterministically and offline.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from backend.models.gd_session import GDSession
import backend.services.gd_session_repository as session_repo


@pytest.fixture
def sample_session_dict():
    """Sample GD session dictionary."""
    now = datetime.now(timezone.utc)
    return {
        "session_id": "GD-SESSION-101",
        "topic": "Sustainable Urban Mobility",
        "coordinator_id": "COORD-001",
        "participant_ids": ["P001", "P002", "P003", "P004"],
        "status": "SCHEDULED",
        "scheduled_start": now,
        "scheduled_end": None,
        "session_started_at": None,
        "session_ended_at": None,
        "session_duration": 180.0,
        "evaluation_id": None,
        "metadata": {"room": "Room-A"},
        "created_at": now,
        "updated_at": now,
        "document_type": "gd_session",
    }


def test_create_session_success(sample_session_dict):
    """Test creating a new session document."""
    mock_collection = MagicMock()
    stored_doc = sample_session_dict.copy()
    stored_doc["_id"] = "mock_oid_123"
    mock_collection.find_one.return_value = stored_doc

    with patch.object(session_repo, "collection", mock_collection):
        created = session_repo.create_session(sample_session_dict)

    mock_collection.insert_one.assert_called_once()
    inserted_arg = mock_collection.insert_one.call_args[0][0]
    assert inserted_arg["session_id"] == "GD-SESSION-101"
    assert inserted_arg["document_type"] == "gd_session"
    assert "_id" not in inserted_arg

    assert created["session_id"] == "GD-SESSION-101"
    assert "_id" not in created
    assert created["document_type"] == "gd_session"


def test_create_session_from_dataclass(sample_session_dict):
    """Test creating a session from a GDSession dataclass instance."""
    mock_collection = MagicMock()
    stored_doc = sample_session_dict.copy()
    stored_doc["_id"] = "mock_oid_123"
    mock_collection.find_one.return_value = stored_doc

    session_obj = GDSession(
        session_id="GD-SESSION-101",
        topic="Sustainable Urban Mobility",
        coordinator_id="COORD-001",
        participant_ids=["P001", "P002"],
    )

    with patch.object(session_repo, "collection", mock_collection):
        created = session_repo.create_session(session_obj)

    assert mock_collection.insert_one.called
    assert created["session_id"] == "GD-SESSION-101"


def test_create_session_invalid_input():
    """Test validation errors on session creation."""
    mock_collection = MagicMock()
    with patch.object(session_repo, "collection", mock_collection):
        with pytest.raises(ValueError, match="session_data must be a dictionary or GDSession"):
            session_repo.create_session("invalid-input")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="session_id is required"):
            session_repo.create_session({"topic": "T", "coordinator_id": "C"})

        with pytest.raises(ValueError, match="session_id is required"):
            session_repo.create_session({"session_id": "   ", "topic": "T"})


def test_get_session_found(sample_session_dict):
    """Test retrieving an existing session document."""
    mock_collection = MagicMock()
    stored_doc = sample_session_dict.copy()
    stored_doc["_id"] = "mock_oid_456"
    mock_collection.find_one.return_value = stored_doc

    with patch.object(session_repo, "collection", mock_collection):
        result = session_repo.get_session("GD-SESSION-101")

    mock_collection.find_one.assert_called_once_with({"session_id": "GD-SESSION-101"})
    assert result is not None
    assert "_id" not in result
    assert result["session_id"] == "GD-SESSION-101"
    assert result["topic"] == "Sustainable Urban Mobility"


def test_get_session_not_found():
    """Test retrieving a non-existent session returns None."""
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None

    with patch.object(session_repo, "collection", mock_collection):
        result = session_repo.get_session("NON-EXISTENT")

    assert result is None


def test_get_session_invalid_input():
    """Test get_session with invalid/empty input returns None without DB call."""
    mock_collection = MagicMock()
    with patch.object(session_repo, "collection", mock_collection):
        assert session_repo.get_session("") is None
        assert session_repo.get_session("   ") is None
        assert session_repo.get_session(None) is None  # type: ignore[arg-type]

    assert not mock_collection.find_one.called


def test_update_session_success(sample_session_dict):
    """Test updating session document protects protected fields and updates updated_at."""
    mock_collection = MagicMock()
    mock_update_res = MagicMock()
    mock_update_res.matched_count = 1
    mock_update_res.modified_count = 1
    mock_collection.update_one.return_value = mock_update_res

    updated_doc = sample_session_dict.copy()
    updated_doc["status"] = "ACTIVE"
    updated_doc["_id"] = "mock_oid_789"
    mock_collection.find_one.return_value = updated_doc

    updates = {
        "status": "ACTIVE",
        "session_started_at": datetime.now(timezone.utc),
        "_id": "illegal_id",
        "session_id": "illegal_change",
        "created_at": "illegal_change",
    }

    with patch.object(session_repo, "collection", mock_collection):
        res = session_repo.update_session("GD-SESSION-101", updates)

    mock_collection.update_one.assert_called_once()
    filter_arg, update_arg = mock_collection.update_one.call_args[0]
    assert filter_arg == {"session_id": "GD-SESSION-101"}
    set_dict = update_arg["$set"]

    assert set_dict["status"] == "ACTIVE"
    assert "_id" not in set_dict
    assert "session_id" not in set_dict
    assert "created_at" not in set_dict
    assert "updated_at" in set_dict

    assert res["status"] == "ACTIVE"
    assert "_id" not in res


def test_update_session_not_found():
    """Test updating a non-existent session returns None."""
    mock_collection = MagicMock()
    mock_update_res = MagicMock()
    mock_update_res.matched_count = 0
    mock_update_res.modified_count = 0
    mock_collection.update_one.return_value = mock_update_res
    mock_collection.find_one.return_value = None

    with patch.object(session_repo, "collection", mock_collection):
        res = session_repo.update_session("NON-EXISTENT", {"status": "ACTIVE"})

    assert res is None


def test_update_session_invalid_input():
    """Test update_session validation on inputs."""
    mock_collection = MagicMock()
    with patch.object(session_repo, "collection", mock_collection):
        assert session_repo.update_session("", {"status": "ACTIVE"}) is None
        assert session_repo.update_session(None, {"status": "ACTIVE"}) is None  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="updates must be a dictionary"):
            session_repo.update_session("GD-101", "not-a-dict")  # type: ignore[arg-type]


def test_delete_session_success():
    """Test deleting a session document."""
    mock_collection = MagicMock()
    mock_del_res = MagicMock()
    mock_del_res.deleted_count = 1
    mock_collection.delete_one.return_value = mock_del_res

    with patch.object(session_repo, "collection", mock_collection):
        deleted = session_repo.delete_session("GD-SESSION-101")

    mock_collection.delete_one.assert_called_once_with({"session_id": "GD-SESSION-101"})
    assert deleted is True


def test_delete_session_not_found():
    """Test deleting non-existent session returns False."""
    mock_collection = MagicMock()
    mock_del_res = MagicMock()
    mock_del_res.deleted_count = 0
    mock_collection.delete_one.return_value = mock_del_res

    with patch.object(session_repo, "collection", mock_collection):
        deleted = session_repo.delete_session("NON-EXISTENT")

    assert deleted is False


def test_delete_session_invalid_input():
    """Test delete_session with invalid session_id returns False."""
    mock_collection = MagicMock()
    with patch.object(session_repo, "collection", mock_collection):
        assert session_repo.delete_session("") is False
        assert session_repo.delete_session("   ") is False
        assert session_repo.delete_session(None) is False  # type: ignore[arg-type]

    assert not mock_collection.delete_one.called


def test_list_sessions_with_filters(sample_session_dict):
    """Test listing sessions with status and coordinator filters."""
    mock_collection = MagicMock()
    doc1 = sample_session_dict.copy()
    doc1["_id"] = "oid1"
    doc2 = sample_session_dict.copy()
    doc2["_id"] = "oid2"
    doc2["session_id"] = "GD-SESSION-102"

    mock_cursor = [doc1, doc2]
    mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

    with patch.object(session_repo, "collection", mock_collection):
        results = session_repo.list_sessions(limit=50, status="SCHEDULED", coordinator_id="COORD-001")

    mock_collection.find.assert_called_once_with({"status": "SCHEDULED", "coordinator_id": "COORD-001"})
    assert len(results) == 2
    assert "_id" not in results[0]
    assert "_id" not in results[1]
    assert results[0]["session_id"] == "GD-SESSION-101"
    assert results[1]["session_id"] == "GD-SESSION-102"
