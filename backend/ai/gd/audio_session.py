"""
GD Participant Audio Session

A participant's microphone belongs to:

    session_id
    participant_id

No speaker diarization is used.

The participant identity comes from authentication/session
information, not from voice recognition.

Timestamps:

    local audio time
        +
    session_offset
        =
    session time
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParticipantAudio:

    session_id: str
    participant_id: str

    sample_rate: int = 16000

    audio_path: str | None = None

    # Time information

    session_start_time: float | None = None
    session_end_time: float | None = None

    # Offset between participant-local audio clock
    # and the shared GD session clock.
    #
    # Example:
    #
    # local audio starts at 0.0s
    # participant joined GD at 5.0s
    #
    # session_offset = 5.0
    #
    # local timestamp 2.0s
    # becomes session timestamp 7.0s

    session_offset: float = 0.0

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
    ):
        """
        Add a participant-local VAD segment.
        """

        self.vad_segments.append(
            {
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
            }
        )

    def add_transcript_segment(
        self,
        start: float,
        end: float,
        text: str,
    ):
        """
        Add a participant-local transcript segment.
        """

        self.transcript_segments.append(
            {
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "text": text.strip(),
            }
        )

    def local_to_session_time(
        self,
        timestamp: float,
    ) -> float:
        """
        Convert participant-local time to
        shared GD session time.
        """

        return float(
            timestamp + self.session_offset
        )

    def get_session_segment(
        self,
        segment: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert one transcript segment from
        participant-local time to session time.
        """

        start = float(segment["start"])
        end = float(segment["end"])

        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,

            "start": self.local_to_session_time(
                start
            ),

            "end": self.local_to_session_time(
                end
            ),

            "duration": float(
                end - start
            ),

            "text": str(
                segment.get("text", "")
            ).strip(),
        }

    @property
    def total_speaking_time(self) -> float:
        """
        Total detected speaking time.
        """

        return sum(
            segment["duration"]
            for segment in self.vad_segments
        )

    @property
    def word_count(self) -> int:
        """
        Total transcript word count.
        """

        return len(
            self.transcript.split()
        )

    @property
    def transcript(self) -> str:
        """
        Combined participant transcript.
        """

        return " ".join(
            segment["text"]
            for segment in self.transcript_segments
            if segment.get("text")
        )

    @property
    def turn_count(self) -> int:
        """
        Number of transcript segments.
        """

        return len(
            self.transcript_segments
        )

    def to_dict(self):
        """
        Convert participant audio data into
        a serializable dictionary.
        """

        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,

            "sample_rate": self.sample_rate,

            "audio_path": self.audio_path,

            "session_start_time": (
                self.session_start_time
            ),

            "session_end_time": (
                self.session_end_time
            ),

            "session_offset": (
                self.session_offset
            ),

            "metadata": self.metadata,

            "vad_segments": (
                self.vad_segments
            ),

            "transcript_segments": (
                self.transcript_segments
            ),

            "total_speaking_time": (
                self.total_speaking_time
            ),

            "word_count": self.word_count,

            "turn_count": self.turn_count,

            "transcript": self.transcript,
        }