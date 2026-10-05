"""
GDSession Data Model

Represents a Group Discussion session entity, tracking coordinator ownership,
participant membership, scheduling, timestamps, status lifecycle, and linked evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


ALLOWED_STATUSES = frozenset({"SCHEDULED", "ACTIVE", "ENDED", "EVALUATED", "CANCELLED"})


@dataclass
class GDSession:
    session_id: str
    topic: str
    coordinator_id: str
    participant_ids: list[str]

    status: str = "SCHEDULED"

    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None

    session_started_at: Optional[datetime] = None
    session_ended_at: Optional[datetime] = None
    session_duration: Optional[float] = None

    evaluation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    document_type: str = "gd_session"

    def __post_init__(self) -> None:
        if not self.session_id or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")
        self.session_id = self.session_id.strip()

        if not self.topic or not isinstance(self.topic, str) or not self.topic.strip():
            raise ValueError("topic is required and must be a non-empty string")
        self.topic = self.topic.strip()

        if not self.coordinator_id or not isinstance(self.coordinator_id, str) or not self.coordinator_id.strip():
            raise ValueError("coordinator_id is required and must be a non-empty string")
        self.coordinator_id = self.coordinator_id.strip()

        if not isinstance(self.participant_ids, list):
            raise ValueError("participant_ids must be a list")

        cleaned_pids: list[str] = []
        seen_pids: set[str] = set()
        for idx, pid in enumerate(self.participant_ids):
            if not pid or not isinstance(pid, str) or not pid.strip():
                raise ValueError(f"participant_ids[{idx}] must be a non-empty string")
            clean_pid = pid.strip()
            if clean_pid in seen_pids:
                raise ValueError(f"Duplicate participant_id found: '{clean_pid}'")
            seen_pids.add(clean_pid)
            cleaned_pids.append(clean_pid)
        self.participant_ids = cleaned_pids

        if self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {sorted(ALLOWED_STATUSES)}"
            )

        if not isinstance(self.metadata, dict):
            self.metadata = {}

        if not self.document_type or not isinstance(self.document_type, str):
            self.document_type = "gd_session"

    def to_dict(self) -> dict[str, Any]:
        """Convert GDSession dataclass instance into a clean dictionary."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "coordinator_id": self.coordinator_id,
            "participant_ids": list(self.participant_ids),
            "status": self.status,
            "scheduled_start": self.scheduled_start,
            "scheduled_end": self.scheduled_end,
            "session_started_at": self.session_started_at,
            "session_ended_at": self.session_ended_at,
            "session_duration": self.session_duration,
            "evaluation_id": self.evaluation_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_type": self.document_type,
        }
