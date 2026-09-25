"""
Live GD transcript manager.

Maintains chronological transcript events
for a live GD session.
"""

from typing import Any


class LiveTranscriptManager:

    def __init__(self):
        self.segments: list[dict[str, Any]] = []

    def add_segment(
        self,
        session_id: str,
        participant_id: str,
        start: float,
        end: float,
        text: str,
    ) -> None:

        text = text.strip()

        if not text:
            return

        self.segments.append(
            {
                "session_id": session_id,
                "participant_id": participant_id,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "text": text,
            }
        )

        self.segments.sort(
            key=lambda segment: (
                segment["start"],
                segment["end"],
            )
        )

    def get_segments(self) -> list[dict[str, Any]]:
        return list(self.segments)

    def get_group_text(self) -> str:

        lines = []

        for segment in self.segments:

            lines.append(
                f"[{segment['participant_id']}] "
                f"{segment['text']}"
            )

        return "\n".join(lines)

    def get_participant_text(
        self,
        participant_id: str,
    ) -> str:

        return " ".join(
            segment["text"]
            for segment in self.segments
            if segment["participant_id"] == participant_id
        )