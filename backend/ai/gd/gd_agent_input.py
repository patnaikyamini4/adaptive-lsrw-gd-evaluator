"""
GD Agent Input Contract

Defines the common input structure passed to
the GD evaluation agents.

No LLM is used in this file.

This file only organizes the information
required by the GD evaluation layer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GDParticipantInput:
    """
    Complete evaluation input for one GD participant.
    """

    session_id: str
    participant_id: str

    topic: str

    participant_transcript: str

    group_transcript: str

    features: dict[str, Any] = field(
        default_factory=dict
    )

    interaction_features: dict[str, Any] = field(
        default_factory=dict
    )

    group_segments: list[dict[str, Any]] = field(
        default_factory=list
    )

    other_participants: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the agent input into a dictionary.
        """

        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "topic": self.topic,
            "participant_transcript": self.participant_transcript,
            "group_transcript": self.group_transcript,
            "features": self.features,
            "interaction_features": self.interaction_features,
            "group_segments": self.group_segments,
            "other_participants": self.other_participants,
            "metadata": self.metadata,
        }