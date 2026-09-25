"""
GD Agent Result Contract

Defines the common output structure returned by
all GD evaluation agents.

No LLM is used here.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GDAgentResult:
    """
    Standard result returned by a GD evaluation agent.
    """

    participant_id: str

    agent: str

    score: float

    reasoning: str = ""

    strengths: list[str] = field(
        default_factory=list
    )

    weaknesses: list[str] = field(
        default_factory=list
    )

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a dictionary.
        """

        return {
            "participant_id": self.participant_id,
            "agent": self.agent,
            "score": self.score,
            "reasoning": self.reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }