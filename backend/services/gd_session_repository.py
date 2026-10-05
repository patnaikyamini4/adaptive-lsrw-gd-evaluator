"""
GD Session MongoDB Repository

Handles persistence, retrieval, updates, and deletion of Group Discussion (GD) session
documents in MongoDB using the unified client architecture.

Collection: gd_sessions
Document Type: gd_session
"""

from datetime import datetime, timezone
from typing import Any, Optional

from backend.models.gd_session import GDSession
from backend.services.mongodb import db


collection = db["gd_sessions"]
gd_sessions_collection = collection


def create_session(session_data: dict[str, Any] | GDSession) -> dict[str, Any]:
    """
    Persist a new GD session document to MongoDB.

    - Validates session_id presence.
    - Sets 'document_type' = 'gd_session'.
    - Sets 'created_at' and 'updated_at' timestamps (UTC).
    - Removes MongoDB's internal '_id' before returning.
    """
    if isinstance(session_data, GDSession):
        doc_to_save = session_data.to_dict()
    elif isinstance(session_data, dict):
        doc_to_save = session_data.copy()
    else:
        raise ValueError("session_data must be a dictionary or GDSession instance")

    session_id = doc_to_save.get("session_id")
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required to create a GD session")

    session_id = session_id.strip()
    doc_to_save["session_id"] = session_id
    doc_to_save["document_type"] = "gd_session"

    now = datetime.now(timezone.utc)
    if "created_at" not in doc_to_save or doc_to_save["created_at"] is None:
        doc_to_save["created_at"] = now
    doc_to_save["updated_at"] = now

    doc_to_save.pop("_id", None)

    collection.insert_one(doc_to_save)
    return get_session(session_id)  # type: ignore[return-value]


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve a GD session document by session_id from MongoDB.

    Returns the document dictionary without MongoDB's '_id', or None if not found.
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        return None

    document = collection.find_one({"session_id": session_id.strip()})
    if document is None:
        return None

    document_copy = document.copy()
    document_copy.pop("_id", None)
    return document_copy


def update_session(session_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Update an existing GD session document in MongoDB.

    - Protects '_id', 'session_id', and 'created_at' from mutation.
    - Automatically updates 'updated_at' to current UTC datetime.
    - Returns updated document without '_id', or None if session does not exist.
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        return None

    if not isinstance(updates, dict):
        raise ValueError("updates must be a dictionary")

    clean_session_id = session_id.strip()
    updates_to_save = updates.copy()
    updates_to_save.pop("_id", None)
    updates_to_save.pop("session_id", None)
    updates_to_save.pop("created_at", None)

    updates_to_save["updated_at"] = datetime.now(timezone.utc)

    result = collection.update_one(
        {"session_id": clean_session_id},
        {"$set": updates_to_save},
    )

    if result.matched_count == 0 and result.modified_count == 0:
        # Check if the document exists
        if collection.find_one({"session_id": clean_session_id}) is None:
            return None

    return get_session(clean_session_id)


def delete_session(session_id: str) -> bool:
    """
    Delete a GD session document by session_id from MongoDB.

    Returns True if a document was deleted, False otherwise.
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        return False

    result = collection.delete_one({"session_id": session_id.strip()})
    return result.deleted_count > 0


def list_sessions(
    limit: int = 100,
    status: Optional[str] = None,
    coordinator_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    List GD sessions with optional status and coordinator_id filters, ordered by created_at desc.
    """
    query: dict[str, Any] = {}
    if status and isinstance(status, str) and status.strip():
        query["status"] = status.strip()
    if coordinator_id and isinstance(coordinator_id, str) and coordinator_id.strip():
        query["coordinator_id"] = coordinator_id.strip()

    cursor = collection.find(query).sort("created_at", -1).limit(max(1, limit))
    sessions: list[dict[str, Any]] = []
    for doc in cursor:
        doc_copy = doc.copy()
        doc_copy.pop("_id", None)
        sessions.append(doc_copy)
    return sessions
