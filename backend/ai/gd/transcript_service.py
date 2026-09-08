"""
GD Transcript Service

Combines individual participant transcripts into a
chronological group discussion transcript.

IMPORTANT:

No speaker diarization is used.

Participant identity comes from:

    session_id
    participant_id
"""


from typing import Any

from .audio_session import ParticipantAudio


class GDTranscriptService:
    """
    Handles participant transcripts and combines
    them into a chronological GD transcript.
    """

    def build_participant_transcript(
        self,
        participant: ParticipantAudio,
    ) -> dict[str, Any]:
        """
        Return transcript information for one participant.
        """

        return {
            "session_id": participant.session_id,
            "participant_id": participant.participant_id,
            "transcript": participant.transcript,
            "word_count": participant.word_count,
            "speaking_time": round(
                participant.total_speaking_time,
                3,
            ),
            "segments": participant.transcript_segments,
        }

    def build_group_transcript(
        self,
        participants: list[ParticipantAudio],
    ) -> list[dict[str, Any]]:
        """
        Combine all participant transcript segments
        and sort them chronologically.
        """

        group_segments: list[dict[str, Any]] = []

        for participant in participants:

            for segment in participant.transcript_segments:

                start = float(
                    segment["start"]
                )

                end = float(
                    segment["end"]
                )

                text = str(
                    segment.get("text", "")
                ).strip()

                if not text:
                    continue

                # Calculate duration here instead of
                # depending on the source segment.
                duration = end - start

                group_segments.append(
                    {
                        "session_id": (
                            participant.session_id
                        ),
                        "participant_id": (
                            participant.participant_id
                        ),
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "text": text,
                    }
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

        Example:

        [P001] I think AI is useful.
        [P002] I agree with that point.
        [P003] There are also some risks.
        """

        segments = self.build_group_transcript(
            participants
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

        word_count = participant.word_count

        return {
            "session_id": participant.session_id,
            "participant_id": participant.participant_id,
            "speaking_time": round(
                speaking_time,
                3,
            ),
            "word_count": word_count,
            "transcript_segments": len(
                participant.transcript_segments
            ),
        }