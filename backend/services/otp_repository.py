"""
OTP MongoDB Repository

Handles persistence, lookup, attempt-decrement, and cleanup for One-Time Passwords (OTPs).
Collection: auth_otps
"""

from datetime import datetime, timezone
from typing import Any, Optional

from backend.services.mongodb import db


collection = db["auth_otps"]
otps_collection = collection


def store_otp(
    email: str,
    otp_hash: str,
    expires_at: datetime,
    attempts_left: int = 3,
    purpose: str = "login",
) -> dict[str, Any]:
    """Store or update an active OTP challenge for an email."""
    if not email or not isinstance(email, str):
        raise ValueError("email is required")

    clean_email = email.strip().lower()
    now = datetime.now(timezone.utc)

    doc = {
        "email": clean_email,
        "otp_hash": otp_hash,
        "expires_at": expires_at,
        "attempts_left": int(attempts_left),
        "purpose": purpose,
        "created_at": now,
        "updated_at": now,
    }

    collection.update_one(
        {"email": clean_email},
        {
            "$set": doc,
        },
        upsert=True,
    )
    saved = get_otp(clean_email)
    return saved or doc


def store_otp_if_cooldown_elapsed(
    email: str,
    otp_hash: str,
    expires_at: datetime,
    attempts_left: int = 3,
    purpose: str = "login",
    cooldown_seconds: int = 60,
) -> tuple[bool, Optional[dict[str, Any]], int]:
    """
    Safely store an OTP challenge if no active challenge exists or if at least
    `cooldown_seconds` have elapsed since the previous request.

    Returns:
        (success, record_or_none, remaining_cooldown_seconds)
    """
    if not email or not isinstance(email, str):
        raise ValueError("email is required")

    clean_email = email.strip().lower()
    now = datetime.now(timezone.utc)

    existing = get_otp(clean_email)
    if existing:
        last_req = existing.get("updated_at") or existing.get("created_at")
        if last_req:
            if isinstance(last_req, str):
                try:
                    last_req = datetime.fromisoformat(last_req)
                except ValueError:
                    last_req = None
            if last_req and last_req.tzinfo is None:
                last_req = last_req.replace(tzinfo=timezone.utc)
            if last_req:
                elapsed = (now - last_req).total_seconds()
                if elapsed < cooldown_seconds:
                    remaining = max(1, int(cooldown_seconds - elapsed))
                    return False, existing, remaining

    saved = store_otp(
        email=clean_email,
        otp_hash=otp_hash,
        expires_at=expires_at,
        attempts_left=attempts_left,
        purpose=purpose,
    )
    return True, saved, 0


def get_otp(email: str) -> Optional[dict[str, Any]]:
    """Retrieve active OTP record for an email."""
    if not email or not isinstance(email, str):
        return None
    clean_email = email.strip().lower()
    doc = collection.find_one({"email": clean_email})
    if doc:
        doc.pop("_id", None)
    return doc


def decrement_attempts(email: str) -> int:
    """
    Decrement the allowed verification attempts for an active OTP.
    Returns remaining attempts count.
    """
    if not email or not isinstance(email, str):
        return 0
    clean_email = email.strip().lower()
    result = collection.find_one_and_update(
        {"email": clean_email},
        {
            "$inc": {"attempts_left": -1},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
        return_document=True,
    )
    if not result:
        return 0
    return max(0, int(result.get("attempts_left", 0)))


def delete_otp(email: str) -> bool:
    """Delete the OTP record once consumed or expired."""
    if not email or not isinstance(email, str):
        return False
    clean_email = email.strip().lower()
    result = collection.delete_one({"email": clean_email})
    return result.deleted_count > 0
