"""
GD Evaluation MongoDB Repository

Handles persistence, retrieval, and deletion of Group Discussion (GD) evaluation documents
in MongoDB using the unified MongoDB client architecture.

Collection: gd_evaluations
Document Type: gd_evaluation
"""

from datetime import datetime, timezone
from typing import Any

from backend.services.mongodb import db


collection = db["gd_evaluations"]
gd_evaluations_collection = collection


def save_evaluation(evaluation_data: dict[str, Any]) -> dict[str, Any]:
    """
    Save or upsert a GD evaluation document to MongoDB.

    - Validates that 'session_id' is present.
    - Preserves all evaluation sections (session_id, topic, session_duration,
      group_transcript, group_segments, participants, participant_transcripts,
      objective_features, interaction_features, agent_results, scorecards,
      final_scores).
    - Sets 'document_type' = 'gd_evaluation'.
    - Sets 'created_at' on initial insert (UTC) and 'updated_at' on save/upsert (UTC).
    - Performs an upsert on session_id so duplicate calls replace/update deterministically
      without creating duplicate documents.
    - Returns the saved evaluation dictionary without internal '_id'.
    """
    if not isinstance(evaluation_data, dict):
        raise ValueError("evaluation_data must be a dictionary")

    session_id = evaluation_data.get("session_id")
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required to persist a GD evaluation")

    now = datetime.now(timezone.utc)
    doc_to_save = evaluation_data.copy()
    doc_to_save.pop("_id", None)
    doc_to_save["session_id"] = session_id.strip()
    doc_to_save["document_type"] = "gd_evaluation"
    doc_to_save["updated_at"] = now

    created_at = doc_to_save.pop("created_at", None) or now

    collection.update_one(
        {"session_id": session_id.strip()},
        {
            "$set": doc_to_save,
            "$setOnInsert": {"created_at": created_at},
        },
        upsert=True,
    )

    return get_evaluation(session_id.strip())  # type: ignore[return-value]


def get_evaluation(session_id: str) -> dict[str, Any] | None:
    """
    Retrieve a GD evaluation document by session_id from MongoDB.

    Returns the document dictionary without MongoDB's '_id', or None if not found.
    """
    if not session_id or not isinstance(session_id, str):
        return None

    document = collection.find_one({"session_id": session_id.strip()})
    if document is None:
        return None

    document_copy = document.copy()
    document_copy.pop("_id", None)
    return document_copy


def delete_evaluation(session_id: str) -> bool:
    """
    Delete a GD evaluation document by session_id from MongoDB.

    Returns True if a document was deleted, False otherwise.
    """
    if not session_id or not isinstance(session_id, str):
        return False

    result = collection.delete_one({"session_id": session_id.strip()})
    return result.deleted_count > 0


def get_all_evaluations(limit: int = 100) -> list[dict[str, Any]]:
    """
    Retrieve all GD evaluation documents from MongoDB up to `limit`.

    Returns a list of documents without MongoDB's '_id'.
    """
    cursor = collection.find().sort("created_at", -1).limit(limit)
    evaluations: list[dict[str, Any]] = []
    for doc in cursor:
        doc_copy = doc.copy()
        doc_copy.pop("_id", None)
        evaluations.append(doc_copy)
    return evaluations
