"""
Live GD session manager.

Maintains the shared session clock and
participant state.

No speaker diarization is used.

Participant identity comes from the
authenticated participant/session relationship.
"""

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class LiveParticipant:
    participant_id: str

    connected: bool = False

    joined_at: float | None = None
    left_at: float | None = None

    last_audio_at: float | None = None

    speaking: bool = False

    speaking_time: float = 0.0
    turn_count: int = 0
    word_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveGDSession:
    session_id: str
    topic: str
    duration_seconds: int

    started_at: float | None = None
    ended_at: float | None = None

    participants: dict[str, LiveParticipant] = field(
        default_factory=dict
    )

    def add_participant(self, participant_id: str) -> None:
        """
        Register a participant in this GD session.
        """

        if participant_id not in self.participants:
            self.participants[participant_id] = LiveParticipant(
                participant_id=participant_id
            )

    def start(self) -> None:
        """
        Start the shared GD session clock.
        """

        if self.started_at is None:
            self.started_at = time.time()

    def end(self) -> None:
        """
        End the GD session.
        """

        if self.ended_at is None:
            self.ended_at = time.time()

    def session_time(self) -> float:
        """
        Return elapsed session time in seconds.
        """

        if self.started_at is None:
            return 0.0

        current_time = (
            self.ended_at
            if self.ended_at is not None
            else time.time()
        )

        return max(
            0.0,
            current_time - self.started_at
        )

    def participant_join(self, participant_id: str) -> None:
        """
        Mark a participant as connected.
        """

        self.add_participant(participant_id)

        participant = self.participants[participant_id]

        participant.connected = True
        participant.left_at = None
        participant.joined_at = self.session_time()

    def participant_leave(self, participant_id: str) -> None:
        """
        Mark a participant as disconnected.
        """

        participant = self.participants.get(participant_id)

        if participant is None:
            return

        participant.connected = False
        participant.speaking = False
        participant.left_at = self.session_time()

    def update_audio_activity(
        self,
        participant_id: str,
        speaking: bool,
    ) -> None:
        """
        Update the participant's current speaking state.

        This does not perform VAD itself.
        VAD will provide the speaking state.
        """

        participant = self.participants.get(participant_id)

        if participant is None:
            return

        now = self.session_time()

        participant.last_audio_at = now
        participant.speaking = speaking

    def add_speaking_time(
        self,
        participant_id: str,
        duration: float,
    ) -> None:
        """
        Add objectively measured speaking duration.
        """

        participant = self.participants.get(participant_id)

        if participant is None:
            return

        participant.speaking_time += max(0.0, duration)

    def add_turn(
        self,
        participant_id: str,
    ) -> None:
        """
        Record one speaking turn.
        """

        participant = self.participants.get(participant_id)

        if participant is None:
            return

        participant.turn_count += 1

    def add_words(
        self,
        participant_id: str,
        count: int,
    ) -> None:
        """
        Add words obtained from ASR.
        """

        participant = self.participants.get(participant_id)

        if participant is None:
            return

        participant.word_count += max(0, count)

    def get_status(self) -> dict[str, Any]:
        """
        Return the current live session state.
        """

        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "duration_seconds": self.duration_seconds,
            "session_time": round(
                self.session_time(),
                3
            ),
            "started": self.started_at is not None,
            "ended": self.ended_at is not None,
            "participants": {
                participant_id: {
                    "connected": participant.connected,
                    "joined_at": participant.joined_at,
                    "left_at": participant.left_at,
                    "last_audio_at": participant.last_audio_at,
                    "speaking": participant.speaking,
                    "speaking_time": round(
                        participant.speaking_time,
                        3
                    ),
                    "turn_count": participant.turn_count,
                    "word_count": participant.word_count,
                }
                for participant_id, participant
                in self.participants.items()
            },
        }