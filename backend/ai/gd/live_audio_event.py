"""
Live GD audio event.

Every audio event belongs to an authenticated
participant and a GD session.

No diarization is required.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class LiveAudioEvent:

    session_id: str
    participant_id: str

    session_start: float
    session_end: float

    audio_path: str | None = None
    transcript: str = ""

    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> float:
        return max(
            0.0,
            self.session_end - self.session_start,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "duration": self.duration,
            "audio_path": self.audio_path,
            "transcript": self.transcript,
            "metadata": self.metadata,
        }