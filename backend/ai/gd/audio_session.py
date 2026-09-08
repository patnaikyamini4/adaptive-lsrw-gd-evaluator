"""
GD Participant Audio Session

Keeps participant identity attached to audio throughout
the GD processing pipeline.

IMPORTANT:

No speaker diarization is used.

The application already knows who owns the audio:

    session_id + participant_id
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParticipantAudio:
    """
    Represents the audio and processing data belonging
    to one authenticated GD participant.
    """

    session_id: str
    participant_id: str

    sample_rate: int = 16000

    audio_path: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    vad_segments: list[dict[str, Any]] = field(
        default_factory=list
    )

    transcript_segments: list[dict[str, Any]] = field(
        default_factory=list
    )

    def add_vad_segment(
        self,
        start: float,
        end: float,
    ) -> None:
        """
        Add one VAD speech segment.
        """

        start = float(start)
        end = float(end)

        if end < start:
            raise ValueError(
                "VAD segment end cannot be before start."
            )

        self.vad_segments.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )

    def add_transcript_segment(
        self,
        start: float,
        end: float,
        text: str,
    ) -> None:
        """
        Add one ASR transcript segment.
        """

        start = float(start)
        end = float(end)

        if end < start:
            raise ValueError(
                "Transcript segment end cannot be before start."
            )

        text = str(text).strip()

        if not text:
            return

        self.transcript_segments.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text,
            }
        )

    @property
    def total_speaking_time(self) -> float:
        """
        Total speech duration detected by VAD.
        """

        return sum(
            float(segment["duration"])
            for segment in self.vad_segments
        )

    @property
    def transcript(self) -> str:
        """
        Combine all transcript segments.
        """

        return " ".join(
            segment["text"]
            for segment in self.transcript_segments
            if segment.get("text")
        )

    @property
    def word_count(self) -> int:
        """
        Number of words in the participant transcript.
        """

        return len(
            self.transcript.split()
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert participant data to a JSON-friendly dictionary.
        """

        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "sample_rate": self.sample_rate,
            "audio_path": self.audio_path,
            "metadata": self.metadata,
            "vad_segments": self.vad_segments,
            "transcript_segments": self.transcript_segments,
            "total_speaking_time": round(
                self.total_speaking_time,
                3,
            ),
            "transcript": self.transcript,
            "word_count": self.word_count,
        }