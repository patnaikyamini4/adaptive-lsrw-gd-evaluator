"""
User MongoDB Repository

Handles persistence, retrieval, and updates for User documents in MongoDB.
Collection: users
Document Type: user
"""

from datetime import datetime, timezone
from typing import Any, Optional

from backend.models.user import User
from backend.services.mongodb import db


collection = db["users"]
users_collection = collection


def create_user(user_data: dict[str, Any] | User) -> dict[str, Any]:
    """
    Persist a new User document to MongoDB.
    Validates model constraints through User dataclass.
    """
    if isinstance(user_data, User):
        doc_to_save = user_data.to_dict()
    elif isinstance(user_data, dict):
        user_obj = User(
            user_id=user_data.get("user_id", ""),
            email=user_data.get("email", ""),
            role=user_data.get("role", "CANDIDATE"),
            display_name=user_data.get("display_name", ""),
            is_verified=user_data.get("is_verified", True),
            metadata=user_data.get("metadata", {}),
        )
        doc_to_save = user_obj.to_dict()
    else:
        raise ValueError("user_data must be a dictionary or User instance")

    now = datetime.now(timezone.utc)
    if "created_at" not in doc_to_save or doc_to_save["created_at"] is None:
        doc_to_save["created_at"] = now
    doc_to_save["updated_at"] = now
    doc_to_save.pop("_id", None)

    collection.insert_one(doc_to_save)
    return get_user_by_id(doc_to_save["user_id"])  # type: ignore[return-value]


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Retrieve user document by unique user_id."""
    if not user_id or not isinstance(user_id, str):
        return None
    doc = collection.find_one({"user_id": user_id.strip()})
    if doc:
        doc.pop("_id", None)
    return doc


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Retrieve user document by normalized email address."""
    if not email or not isinstance(email, str):
        return None
    doc = collection.find_one({"email": email.strip().lower()})
    if doc:
        doc.pop("_id", None)
    return doc


def update_user(user_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Apply partial update to an existing user document."""
    if not user_id or not isinstance(user_id, str):
        return None
    if not isinstance(updates, dict) or not updates:
        return get_user_by_id(user_id)

    clean_updates = updates.copy()
    clean_updates.pop("_id", None)
    clean_updates.pop("user_id", None)
    clean_updates["updated_at"] = datetime.now(timezone.utc)

    result = collection.update_one(
        {"user_id": user_id.strip()},
        {"$set": clean_updates},
    )
    if result.matched_count == 0:
        return None
    return get_user_by_id(user_id)


def delete_user(user_id: str) -> bool:
    """Delete a user document by user_id."""
    if not user_id or not isinstance(user_id, str):
        return False
    result = collection.delete_one({"user_id": user_id.strip()})
    return result.deleted_count > 0


def list_users(limit: int = 100) -> list[dict[str, Any]]:
    """List registered users ordered by creation date."""
    cursor = collection.find().sort("created_at", -1).limit(limit)
    users = []
    for doc in cursor:
        doc.pop("_id", None)
        users.append(doc)
    return users
