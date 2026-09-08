"""
GD Feature Extraction

This module calculates objective, measurable features
from participant audio/transcript data.

It does NOT use an LLM.

It does NOT perform speaker identification.

Identity is already known from participant_id.
"""

from typing import Any

from .audio_session import ParticipantAudio


class GDFeatureExtractor:

    def __init__(
        self,
        session_duration: float | None = None,
    ):

        self.session_duration = session_duration

    # --------------------------------------------------
    # EXTRACT PARTICIPANT FEATURES
    # --------------------------------------------------

    def extract(
        self,
        participant: ParticipantAudio,
    ) -> dict[str, Any]:

        speaking_time = (
            participant.total_speaking_time
        )

        word_count = (
            participant.word_count
        )

        turn_count = (
            participant.turn_count
        )

        # ----------------------------------------------
        # SPEAKING RATIO
        # ----------------------------------------------

        speaking_ratio = 0.0

        if self.session_duration:
            speaking_ratio = (
                speaking_time
                / self.session_duration
            )

        # ----------------------------------------------
        # WORDS PER MINUTE
        # ----------------------------------------------

        words_per_minute = 0.0

        if speaking_time > 0:

            words_per_minute = (
                word_count
                / speaking_time
            ) * 60

        # ----------------------------------------------
        # TURN DURATIONS
        # ----------------------------------------------

        turn_durations = [
            segment["duration"]
            for segment
            in participant.transcript_segments
        ]

        average_turn_duration = 0.0

        if turn_durations:

            average_turn_duration = (
                sum(turn_durations)
                / len(turn_durations)
            )

        longest_turn = 0.0

        if turn_durations:

            longest_turn = max(
                turn_durations
            )

        shortest_turn = 0.0

        if turn_durations:

            shortest_turn = min(
                turn_durations
            )

        # ----------------------------------------------
        # BUILD RESULT
        # ----------------------------------------------

        return {

            "session_id": (
                participant.session_id
            ),

            "participant_id": (
                participant.participant_id
            ),

            # Audio/speaking
            "audio_duration": (
                self._audio_duration(
                    participant
                )
            ),

            "speaking_time": (
                round(
                    speaking_time,
                    3
                )
            ),

            "speaking_ratio": (
                round(
                    speaking_ratio,
                    4
                )
            ),

            # Transcript
            "word_count": (
                word_count
            ),

            "turn_count": (
                turn_count
            ),

            # Speaking behavior
            "words_per_minute": (
                round(
                    words_per_minute,
                    2
                )
            ),

            "average_turn_duration": (
                round(
                    average_turn_duration,
                    3
                )
            ),

            "longest_turn": (
                round(
                    longest_turn,
                    3
                )
            ),

            "shortest_turn": (
                round(
                    shortest_turn,
                    3
                )
            ),

            # Transcript
            "transcript": (
                participant.transcript
            ),
        }

    # --------------------------------------------------
    # AUDIO DURATION
    # --------------------------------------------------

    @staticmethod
    def _audio_duration(
        participant: ParticipantAudio,
    ) -> float:

        if (
            participant.session_start_time
            is not None
            and participant.session_end_time
            is not None
        ):

            return round(
                participant.session_end_time
                - participant.session_start_time,
                3,
            )

        if participant.vad_segments:

            return round(
                max(
                    segment["end"]
                    for segment
                    in participant.vad_segments
                ),
                3,
            )

        return 0.0

    # --------------------------------------------------
    # EXTRACT ALL PARTICIPANTS
    # --------------------------------------------------

    def extract_all(
        self,
        participants: list[ParticipantAudio],
    ) -> list[dict[str, Any]]:

        return [
            self.extract(participant)
            for participant in participants
        ]