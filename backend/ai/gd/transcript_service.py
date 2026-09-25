"""
GD Transcript Service

Combines individual participant transcripts
into a shared GD session timeline.

IMPORTANT:

No speaker diarization is used.

Participant identity comes from:

    session_id
    participant_id

Each participant has a local audio clock.

The shared GD timeline is calculated using:

    local timestamp + session_offset
"""


from typing import Any

from .audio_session import ParticipantAudio


class GDTranscriptService:
    """
    Handles participant transcripts and converts
    them into a shared GD session timeline.
    """

    def build_participant_transcript(
        self,
        participant: ParticipantAudio,
    ) -> dict[str, Any]:
        """
        Return transcript information for one participant.
        """

        return {
            "session_id": (
                participant.session_id
            ),

            "participant_id": (
                participant.participant_id
            ),

            "transcript": (
                participant.transcript
            ),

            "word_count": (
                participant.word_count
            ),

            "speaking_time": round(
                participant.total_speaking_time,
                3,
            ),

            "segments": (
                participant.transcript_segments
            ),

            "session_offset": (
                participant.session_offset
            ),
        }

    def build_group_transcript(
        self,
        participants: list[ParticipantAudio],
    ) -> list[dict[str, Any]]:
        """
        Combine participant transcript segments
        using the shared GD session clock.
        """

        group_segments: list[dict[str, Any]] = []

        for participant in participants:

            for segment in (
                participant.transcript_segments
            ):

                text = str(
                    segment.get(
                        "text",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                session_segment = (
                    participant.get_session_segment(
                        segment
                    )
                )

                group_segments.append(
                    session_segment
                )

        group_segments.sort(
            key=lambda segment: (
                segment["start"],
                segment["end"],
            )
        )

        return group_segments

    def build_group_text(
        self,
        participants: list[ParticipantAudio],
    ) -> str:
        """
        Create readable chronological GD text.
        """

        segments = (
            self.build_group_transcript(
                participants
            )
        )

        lines: list[str] = []

        for segment in segments:

            participant_id = (
                segment["participant_id"]
            )

            text = segment["text"]

            lines.append(
                f"[{participant_id}] {text}"
            )

        return "\n".join(lines)

    def get_participant_statistics(
        self,
        participant: ParticipantAudio,
    ) -> dict[str, Any]:
        """
        Calculate basic participant statistics.
        """

        speaking_time = (
            participant.total_speaking_time
        )

        word_count = (
            participant.word_count
        )

        return {
            "session_id": (
                participant.session_id
            ),

            "participant_id": (
                participant.participant_id
            ),

            "speaking_time": round(
                speaking_time,
                3,
            ),

            "word_count": (
                word_count
            ),

            "transcript_segments": len(
                participant.transcript_segments
            ),

            "session_offset": (
                participant.session_offset
            ),
        }