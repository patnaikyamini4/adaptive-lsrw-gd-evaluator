"""
GD Session Application Service

Coordinates the lifecycle state machine and membership management for Group Discussion sessions:

    CREATED / SCHEDULED
           |
           v
        ACTIVE
           |
           v
         ENDED
           |
           v
       EVALUATED

    Cancellation:
    SCHEDULED / ACTIVE -> CANCELLED
"""

from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from backend.models.gd_session import GDSession
import backend.services.gd_session_repository as default_repo


class GDSessionService:
    """
    Business service managing the GD session lifecycle and participant roster.
    """

    def __init__(self, repository=None):
        self.repo = repository or default_repo

    def create_session(
        self,
        topic: str,
        coordinator_id: str,
        participant_ids: list[str],
        session_id: Optional[str] = None,
        scheduled_start: Optional[datetime] = None,
        scheduled_end: Optional[datetime] = None,
        session_duration: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Create and persist a new GD session in SCHEDULED state.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            session_id = f"GD-{uuid.uuid4().hex[:8].upper()}"
        else:
            session_id = session_id.strip()

        # Validate with GDSession model dataclass
        session = GDSession(
            session_id=session_id,
            topic=topic,
            coordinator_id=coordinator_id,
            participant_ids=participant_ids,
            status="SCHEDULED",
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            session_duration=session_duration,
            metadata=metadata or {},
        )

        return self.repo.create_session(session.to_dict())

    def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve a GD session by ID or raise ValueError if not found.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        session = self.repo.get_session(session_id.strip())
        if session is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return session

    def start_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition session: SCHEDULED -> ACTIVE.
        Records session_started_at.
        """
        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status != "SCHEDULED":
            raise ValueError(
                f"Cannot start GD session in '{current_status}' status. Must be 'SCHEDULED'."
            )

        now = datetime.now(timezone.utc)
        updates = {
            "status": "ACTIVE",
            "session_started_at": now,
        }
        updated = self.repo.update_session(session_id.strip(), updates)
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition session: ACTIVE -> ENDED.
        Records session_ended_at and computes session_duration.
        """
        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot end GD session in '{current_status}' status. Must be 'ACTIVE'."
            )

        now = datetime.now(timezone.utc)
        started_at = session.get("session_started_at")
        session_duration = session.get("session_duration")

        if started_at:
            if isinstance(started_at, str):
                try:
                    started_dt = datetime.fromisoformat(started_at)
                except ValueError:
                    started_dt = now
            else:
                started_dt = started_at

            if started_dt.tzinfo is None:
                started_dt = started_dt.replace(tzinfo=timezone.utc)

            duration_seconds = max(0.0, (now - started_dt).total_seconds())
            session_duration = round(duration_seconds, 3)

        updates = {
            "status": "ENDED",
            "session_ended_at": now,
            "session_duration": session_duration,
        }
        updated = self.repo.update_session(session_id.strip(), updates)
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def mark_evaluated(self, session_id: str, evaluation_id: str) -> dict[str, Any]:
        """
        Transition session: ENDED -> EVALUATED.
        Stores reference to evaluation document ID.
        """
        if not evaluation_id or not isinstance(evaluation_id, str) or not evaluation_id.strip():
            raise ValueError("evaluation_id is required and must be a non-empty string")

        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status != "ENDED":
            raise ValueError(
                f"Cannot mark GD session as evaluated in '{current_status}' status. Must be 'ENDED'."
            )

        updates = {
            "status": "EVALUATED",
            "evaluation_id": evaluation_id.strip(),
        }
        updated = self.repo.update_session(session_id.strip(), updates)
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def cancel_session(self, session_id: str, reason: Optional[str] = None) -> dict[str, Any]:
        """
        Transition session: SCHEDULED / ACTIVE -> CANCELLED.
        Stores cancellation reason in metadata if provided.
        """
        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status not in ("SCHEDULED", "ACTIVE"):
            raise ValueError(
                f"Cannot cancel GD session in '{current_status}' status. Must be 'SCHEDULED' or 'ACTIVE'."
            )

        now = datetime.now(timezone.utc)
        metadata = session.get("metadata", {}).copy()
        if reason and isinstance(reason, str) and reason.strip():
            metadata["cancellation_reason"] = reason.strip()

        updates: dict[str, Any] = {
            "status": "CANCELLED",
            "metadata": metadata,
        }
        if current_status == "ACTIVE" and not session.get("session_ended_at"):
            updates["session_ended_at"] = now

        updated = self.repo.update_session(session_id.strip(), updates)
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def add_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Add a new participant to a SCHEDULED GD session.
        """
        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_pid = participant_id.strip()
        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status != "SCHEDULED":
            raise ValueError(
                f"Cannot add participant to session in '{current_status}' status. Must be 'SCHEDULED'."
            )

        participant_ids = list(session.get("participant_ids", []))
        if clean_pid in participant_ids:
            raise ValueError(
                f"Participant '{clean_pid}' is already registered in session '{session_id.strip()}'"
            )

        participant_ids.append(clean_pid)
        updated = self.repo.update_session(
            session_id.strip(),
            {"participant_ids": participant_ids},
        )
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def remove_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Remove a participant from a SCHEDULED GD session.
        """
        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_pid = participant_id.strip()
        session = self.get_session(session_id)
        current_status = session.get("status")

        if current_status != "SCHEDULED":
            raise ValueError(
                f"Cannot remove participant from session in '{current_status}' status. Must be 'SCHEDULED'."
            )

        participant_ids = list(session.get("participant_ids", []))
        if clean_pid not in participant_ids:
            raise ValueError(
                f"Participant '{clean_pid}' not found in session '{session_id.strip()}'"
            )

        participant_ids.remove(clean_pid)
        updated = self.repo.update_session(
            session_id.strip(),
            {"participant_ids": participant_ids},
        )
        if updated is None:
            raise ValueError(f"GD session '{session_id.strip()}' not found")
        return updated

    def list_sessions(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        coordinator_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        List sessions filtered by status or coordinator.
        """
        return self.repo.list_sessions(limit=limit, status=status, coordinator_id=coordinator_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session document.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            return False
        return self.repo.delete_session(session_id.strip())
