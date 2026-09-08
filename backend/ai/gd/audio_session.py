"""
GD Participant Audio Session

A participant's microphone belongs to:

    session_id
    participant_id

No speaker diarization is used.

The participant identity comes from authentication/session
information, not from voice recognition.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParticipantAudio:

    # --------------------------------------------------
    # IDENTITY
    # --------------------------------------------------

    session_id: str
    participant_id: str

    # --------------------------------------------------
    # AUDIO
    # --------------------------------------------------

    sample_rate: int = 16000

    audio_path: str | None = None

    # --------------------------------------------------
    # SESSION TIMING
    # --------------------------------------------------

    session_start_time: float | None = None

    session_end_time: float | None = None

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------
    # VAD
    # --------------------------------------------------

    vad_segments: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # ASR
    # --------------------------------------------------

    transcript_segments: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # VAD SEGMENT
    # --------------------------------------------------

    def add_vad_segment(
        self,
        start: float,
        end: float,
    ):

        self.vad_segments.append(
            {
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
            }
        )

    # --------------------------------------------------
    # TRANSCRIPT SEGMENT
    # --------------------------------------------------

    def add_transcript_segment(
        self,
        start: float,
        end: float,
        text: str,
    ):

        self.transcript_segments.append(
            {
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "text": text.strip(),
            }
        )

    # --------------------------------------------------
    # SPEAKING TIME
    # --------------------------------------------------

    @property
    def total_speaking_time(self) -> float:

        return sum(
            segment["duration"]
            for segment in self.vad_segments
        )

    # --------------------------------------------------
    # WORD COUNT
    # --------------------------------------------------

    @property
    def word_count(self) -> int:

        return len(
            self.transcript.split()
        )

    # --------------------------------------------------
    # FULL PARTICIPANT TRANSCRIPT
    # --------------------------------------------------

    @property
    def transcript(self) -> str:

        return " ".join(
            segment["text"]
            for segment in self.transcript_segments
            if segment.get("text")
        )

    # --------------------------------------------------
    # NUMBER OF TURNS
    # --------------------------------------------------

    @property
    def turn_count(self) -> int:

        return len(
            self.transcript_segments
        )

    # --------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------

    def to_dict(self):

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

            "metadata": self.metadata,

            "vad_segments": self.vad_segments,

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