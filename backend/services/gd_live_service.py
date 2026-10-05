"""
GD Live Session Integration Service

Coordinates persistent GDSession entities with in-memory LiveGDSession runtime state:
- Manages runtime lifecycle for active Group Discussion sessions.
- Validates participant authorization against persistent roster before allowing joins.
- Tracks live participant connection state without diarization or speaker recognition.
- Bridges persistent MongoDB session state with in-memory audio event clock.
"""

from typing import Any, Optional

from backend.ai.gd.live_session import LiveGDSession
from backend.services.gd_session_service import GDSessionService


class GDLiveService:
    """
    Coordinates persistent GD sessions and in-memory live runtime state.
    """

    def __init__(
        self,
        session_service: Optional[GDSessionService] = None,
        live_session_factory: Any = LiveGDSession,
    ) -> None:
        self.session_service = session_service or GDSessionService()
        self.live_session_factory = live_session_factory
        self._runtimes: dict[str, LiveGDSession] = {}

    def start_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition persistent session (SCHEDULED -> ACTIVE) and create in-memory LiveGDSession.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "SCHEDULED":
            raise ValueError(
                f"Cannot start GD session in '{current_status}' status. Must be 'SCHEDULED'."
            )

        # Transition persistent state
        updated_session = self.session_service.start_session(clean_session_id)

        # Instantiate live runtime
        try:
            duration_sec = int(session.get("session_duration") or 0)
            live_session = self.live_session_factory(
                session_id=clean_session_id,
                topic=session.get("topic", ""),
                duration_seconds=duration_sec,
            )
            for pid in session.get("participant_ids", []):
                live_session.add_participant(pid)

            live_session.start()
            self._runtimes[clean_session_id] = live_session
        except Exception as e:
            # Rollback persistent session state to SCHEDULED if runtime creation fails
            try:
                self.session_service.repo.update_session(
                    clean_session_id,
                    {"status": "SCHEDULED", "session_started_at": None},
                )
            except Exception:
                pass
            raise RuntimeError(f"Failed to initialize live runtime: {str(e)}") from e

        return {
            "session": updated_session,
            "runtime": live_session.get_status(),
        }

    def get_runtime(self, session_id: str) -> Optional[LiveGDSession]:
        """
        Retrieve active LiveGDSession instance by session ID.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            return None
        return self._runtimes.get(session_id.strip())

    def join_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Connect an authorized participant to an ACTIVE live session.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        clean_pid = participant_id.strip()

        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot join session in '{current_status}' status. Must be 'ACTIVE'."
            )

        participant_ids = session.get("participant_ids", [])
        if clean_pid not in participant_ids:
            raise ValueError(
                f"Participant '{clean_pid}' is not authorized for session '{clean_session_id}'"
            )

        live_session = self._runtimes.get(clean_session_id)
        if live_session is None:
            raise RuntimeError(
                f"Live runtime for active GD session '{clean_session_id}' is not available"
            )

        live_session.participant_join(clean_pid)

        return {
            "session_id": clean_session_id,
            "participant_id": clean_pid,
            "connected": True,
            "runtime": live_session.get_status(),
        }

    def leave_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Disconnect a participant from an ACTIVE live session.
        Does NOT remove participant from the persistent roster.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        clean_pid = participant_id.strip()

        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot leave session in '{current_status}' status. Must be 'ACTIVE'."
            )

        live_session = self._runtimes.get(clean_session_id)
        if live_session is None:
            raise ValueError(
                f"No active live runtime found for session '{clean_session_id}'"
            )

        if clean_pid not in live_session.participants:
            raise ValueError(
                f"Participant '{clean_pid}' not found in live session '{clean_session_id}'"
            )

        participant = live_session.participants[clean_pid]
        if not participant.connected:
            raise ValueError(
                f"Participant '{clean_pid}' is not currently connected to session '{clean_session_id}'"
            )

        live_session.participant_leave(clean_pid)

        return {
            "session_id": clean_session_id,
            "participant_id": clean_pid,
            "connected": False,
            "runtime": live_session.get_status(),
        }

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition persistent session (ACTIVE -> ENDED) and stop in-memory LiveGDSession clock.
        Does NOT automatically trigger post-session evaluation.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot end GD session in '{current_status}' status. Must be 'ACTIVE'."
            )

        live_session = self._runtimes.get(clean_session_id)
        runtime_status = None
        if live_session is not None:
            live_session.end()
            runtime_status = live_session.get_status()

        ended_session = self.session_service.end_session(clean_session_id)

        return {
            "session": ended_session,
            "runtime": runtime_status,
        }

    def get_runtime_status(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve unified status combining persistent lifecycle state and in-memory clock/connections.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        live_session = self._runtimes.get(clean_session_id)
        if live_session is not None:
            live_status = live_session.get_status()
            return {
                "session_id": clean_session_id,
                "status": current_status,
                "runtime_active": (
                    current_status == "ACTIVE"
                    and live_status.get("started", False)
                    and not live_status.get("ended", False)
                ),
                "elapsed_time": live_status.get("session_time", 0.0),
                "duration_seconds": live_status.get(
                    "duration_seconds", session.get("session_duration")
                ),
                "participants": live_status.get("participants", {}),
                "live_status": live_status,
            }

        # Fallback when runtime is inactive/not initialized
        return {
            "session_id": clean_session_id,
            "status": current_status,
            "runtime_active": False,
            "elapsed_time": session.get("session_duration") or 0.0,
            "duration_seconds": session.get("session_duration"),
            "participants": {
                pid: {
                    "connected": False,
                    "joined_at": None,
                    "left_at": None,
                    "last_audio_at": None,
                    "speaking": False,
                    "speaking_time": 0.0,
                    "turn_count": 0,
                    "word_count": 0,
                }
                for pid in session.get("participant_ids", [])
            },
            "live_status": None,
        }

    def stop_runtime(self, session_id: str) -> bool:
        """
        Stop and remove live runtime instance.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            return False

        clean_session_id = session_id.strip()
        live_session = self._runtimes.pop(clean_session_id, None)
        if live_session is not None:
            live_session.end()
            return True
        return False
