"""
GD Interaction Features

Calculates objective interaction features
from a shared GD session timeline.

No LLM is used here.
No speaker diarization is used.

Participant identity comes from participant_id.
"""

from typing import Any


class GDInteractionFeatureExtractor:
    """
    Extract objective interaction features
    from the shared GD session timeline.
    """

    def extract(
        self,
        group_segments: list[dict[str, Any]],
        participant_id: str,
    ) -> dict[str, Any]:

        # Keep the timeline chronological.
        group_segments = sorted(
            group_segments,
            key=lambda segment: (
                float(segment["start"]),
                float(segment["end"]),
            ),
        )

        participant_segments = [
            segment
            for segment in group_segments
            if segment.get("participant_id") == participant_id
        ]

        if not participant_segments:
            return {
                "participant_id": participant_id,
                "turn_count": 0,
                "first_speaking_time": None,
                "last_speaking_time": None,
                "average_gap_between_turns": 0.0,
                "responses": 0,
                "overlap_events": 0,
                "other_speakers_before": 0,
                "other_speakers_after": 0,
            }

        # ---------------------------------------------------------
        # Basic speaking information
        # ---------------------------------------------------------

        first_speaking_time = min(
            float(segment["start"])
            for segment in participant_segments
        )

        last_speaking_time = max(
            float(segment["end"])
            for segment in participant_segments
        )

        # ---------------------------------------------------------
        # Gaps between this participant's own turns
        # ---------------------------------------------------------

        gaps: list[float] = []

        for previous, current in zip(
            participant_segments,
            participant_segments[1:],
        ):
            previous_end = float(previous["end"])
            current_start = float(current["start"])

            gap = current_start - previous_end

            if gap >= 0:
                gaps.append(gap)

        average_gap = 0.0

        if gaps:
            average_gap = sum(gaps) / len(gaps)

        # ---------------------------------------------------------
        # Interaction information
        # ---------------------------------------------------------

        responses = 0
        overlap_events = 0
        other_speakers_before = 0
        other_speakers_after = 0

        for index, segment in enumerate(group_segments):

            if segment.get("participant_id") != participant_id:
                continue

            # -----------------------------------------------------
            # Previous speaker
            # -----------------------------------------------------

            if index > 0:

                previous = group_segments[index - 1]

                previous_participant = previous.get(
                    "participant_id"
                )

                if previous_participant != participant_id:

                    other_speakers_before += 1

                    previous_end = float(previous["end"])
                    current_start = float(segment["start"])

                    gap = current_start - previous_end

                    # A response is only counted when the previous
                    # speaker finished before this participant began.
                    #
                    # We use a short conversational gap of <= 3 sec.
                    if 0 <= gap <= 3.0:
                        responses += 1

                    # Negative gap means temporal overlap.
                    if gap < 0:
                        overlap_events += 1

            # -----------------------------------------------------
            # Next speaker
            # -----------------------------------------------------

            if index < len(group_segments) - 1:

                next_segment = group_segments[index + 1]

                next_participant = next_segment.get(
                    "participant_id"
                )

                if next_participant != participant_id:
                    other_speakers_after += 1

        return {
            "participant_id": participant_id,
            "turn_count": len(participant_segments),
            "first_speaking_time": round(
                first_speaking_time,
                3,
            ),
            "last_speaking_time": round(
                last_speaking_time,
                3,
            ),
            "average_gap_between_turns": round(
                average_gap,
                3,
            ),
            "responses": responses,
            "overlap_events": overlap_events,
            "other_speakers_before": other_speakers_before,
            "other_speakers_after": other_speakers_after,
        }

    def extract_all(
        self,
        group_segments: list[dict[str, Any]],
        participant_ids: list[str],
    ) -> list[dict[str, Any]]:

        return [
            self.extract(
                group_segments,
                participant_id,
            )
            for participant_id in participant_ids
        ]