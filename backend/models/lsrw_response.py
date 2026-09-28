from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class LSRWResponse:
    session_id: str
    participant_id: str

    module: str
    question_id: str

    response_text: Optional[str] = None
    audio_path: Optional[str] = None
    transcript: Optional[str] = None

    started_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None

    evaluation: Optional[dict] = None
    score: Optional[float] = None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "module": self.module,
            "question_id": self.question_id,
            "response_text": self.response_text,
            "audio_path": self.audio_path,
            "transcript": self.transcript,
            "started_at": self.started_at,
            "submitted_at": self.submitted_at,
            "evaluation": self.evaluation,
            "score": self.score,
        }