from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LSRWSession:
    session_id: str
    participant_id: str

    status: str = "SCHEDULED"

    current_module: Optional[str] = None
    current_question_id: Optional[str] = None

    scheduled_start: Optional[datetime] = None
    session_started_at: Optional[datetime] = None
    session_ended_at: Optional[datetime] = None

    module_started_at: Optional[datetime] = None

    completed_modules: list = field(default_factory=list)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "status": self.status,
            "current_module": self.current_module,
            "current_question_id": self.current_question_id,
            "scheduled_start": self.scheduled_start,
            "session_started_at": self.session_started_at,
            "session_ended_at": self.session_ended_at,
            "module_started_at": self.module_started_at,
            "completed_modules": self.completed_modules,
        }