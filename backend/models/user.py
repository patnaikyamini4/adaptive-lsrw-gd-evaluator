"""
User Data Model

Represents an authenticated user within the Adaptive LSRW GD system,
differentiating between COORDINATOR and CANDIDATE roles with strict verification state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Optional


ALLOWED_ROLES = frozenset({"COORDINATOR", "CANDIDATE"})
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class User:
    user_id: str
    email: str
    role: str
    display_name: str

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_verified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    document_type: str = "user"

    def __post_init__(self) -> None:
        if not self.user_id or not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("user_id is required and must be a non-empty string")
        self.user_id = self.user_id.strip()

        if not self.email or not isinstance(self.email, str) or not self.email.strip():
            raise ValueError("email is required and must be a non-empty string")
        self.email = self.email.strip().lower()

        if not EMAIL_REGEX.match(self.email):
            raise ValueError(f"Invalid email address format: '{self.email}'")

        if not self.role or not isinstance(self.role, str) or self.role.strip().upper() not in ALLOWED_ROLES:
            raise ValueError(f"role must be one of: {sorted(ALLOWED_ROLES)}")
        self.role = self.role.strip().upper()

        if not self.display_name or not isinstance(self.display_name, str) or not self.display_name.strip():
            self.display_name = self.email.split("@")[0]
        else:
            self.display_name = self.display_name.strip()

        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

        if not isinstance(self.metadata, dict):
            self.metadata = {}

        if not self.document_type:
            self.document_type = "user"

    def to_dict(self) -> dict[str, Any]:
        """Convert User dataclass instance into a clean dictionary."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "display_name": self.display_name,
            "is_verified": self.is_verified,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_type": self.document_type,
        }
