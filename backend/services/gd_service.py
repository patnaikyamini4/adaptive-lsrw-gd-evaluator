"""
GD Application Service

Coordinates the complete Group Discussion evaluation flow:

    1. Validates session inputs & audio file paths
    2. Processes each participant's audio (VAD -> shared ASR -> ParticipantAudio)
    3. Merges participant streams into a chronological group timeline
    4. Extracts objective speech features (speaking time, WPM, turn metrics)
    5. Extracts interaction features (responses, overlaps, turn-taking dynamics)
    6. Constructs standardized GDParticipantInput contracts
    7. Evaluates each participant via GDOrchestrator (which runs the 4 GD agents and GDScorer)
    8. Returns a complete, structured GD evaluation payload

Identity is strictly established from session authentication/participant_id.
No speaker diarization or IP-based identification is used.
"""

from pathlib import Path
from typing import Any

from backend.ai.gd.audio_session import ParticipantAudio
from backend.ai.gd.feature_extraction import GDFeatureExtractor
from backend.ai.gd.gd_agent_input import GDParticipantInput
from backend.ai.gd.gd_orchestrator import GDOrchestrator
from backend.ai.gd.interaction_features import GDInteractionFeatureExtractor
from backend.ai.gd.participant_pipeline import GDParticipantPipeline
from backend.ai.gd.transcript_service import GDTranscriptService


class GDEvaluationService:
    """
    Unified Group Discussion application service.
    """

    def __init__(
        self,
        participant_pipeline: GDParticipantPipeline | None = None,
        transcript_service: GDTranscriptService | None = None,
        interaction_extractor: GDInteractionFeatureExtractor | None = None,
        orchestrator: GDOrchestrator | None = None,
    ) -> None:
        self._participant_pipeline = participant_pipeline
        self._transcript_service = transcript_service or GDTranscriptService()
        self._interaction_extractor = (
            interaction_extractor or GDInteractionFeatureExtractor()
        )
        self._orchestrator = orchestrator or GDOrchestrator()

    @property
    def participant_pipeline(self) -> GDParticipantPipeline:
        """
        Lazily initialize the participant pipeline if not injected.
        This prevents loading Whisper and Silero VAD unnecessarily during testing.
        """
        if self._participant_pipeline is None:
            self._participant_pipeline = GDParticipantPipeline()
        return self._participant_pipeline

    def validate_inputs(
        self,
        session_id: str,
        topic: str,
        participants: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Validate evaluation session configuration and audio paths.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")

        if not isinstance(participants, list) or not participants:
            raise ValueError("participants must be a non-empty list")

        validated_participants: list[dict[str, Any]] = []
        seen_participant_ids: set[str] = set()

        for idx, item in enumerate(participants):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Participant entry at index {idx} must be a dictionary"
                )

            participant_id = item.get("participant_id")
            if not isinstance(participant_id, str) or not participant_id.strip():
                raise ValueError(
                    f"Participant entry at index {idx} missing valid 'participant_id'"
                )

            participant_id = participant_id.strip()
            if participant_id in seen_participant_ids:
                raise ValueError(
                    f"Duplicate participant_id found: '{participant_id}'"
                )
            seen_participant_ids.add(participant_id)

            audio_path_raw = item.get("audio_path")
            if not audio_path_raw:
                raise ValueError(
                    f"Participant '{participant_id}' missing required 'audio_path'"
                )

            audio_path = Path(audio_path_raw)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found for participant '{participant_id}': {audio_path}"
                )

            session_offset = item.get("session_offset", 0.0)
            if isinstance(session_offset, bool) or not isinstance(
                session_offset, (int, float)
            ):
                raise ValueError(
                    f"session_offset for participant '{participant_id}' must be numeric"
                )
            session_offset = float(session_offset)
            if session_offset < 0:
                raise ValueError(
                    f"session_offset for participant '{participant_id}' cannot be negative"
                )

            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            validated_participants.append(
                {
                    "participant_id": participant_id,
                    "audio_path": str(audio_path),
                    "session_offset": session_offset,
                    "metadata": metadata,
                }
            )

        return validated_participants

    def evaluate_session(
        self,
        session_id: str,
        topic: str,
        participants: list[dict[str, Any]],
        session_duration: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end GD evaluation for all participants in a session.
        """
        validated_participants = self.validate_inputs(
            session_id=session_id,
            topic=topic,
            participants=participants,
        )

        # ------------------------------------------------------------------
        # Step 1: Process Audio per Participant (VAD -> ASR -> ParticipantAudio)
        # ------------------------------------------------------------------
        processed_participants: list[ParticipantAudio] = []

        for p_config in validated_participants:
            participant_id = p_config["participant_id"]
            audio_path = p_config["audio_path"]
            session_offset = p_config["session_offset"]
            metadata = p_config["metadata"]

            participant = self.participant_pipeline.process(
                session_id=session_id,
                participant_id=participant_id,
                audio_path=audio_path,
            )
            participant.session_offset = session_offset
            participant.metadata.update(metadata)
            processed_participants.append(participant)

        # ------------------------------------------------------------------
        # Step 2: Build Synchronized Group Timeline & Text
        # ------------------------------------------------------------------
        group_segments = self._transcript_service.build_group_transcript(
            processed_participants
        )
        group_text = self._transcript_service.build_group_text(
            processed_participants
        )

        # ------------------------------------------------------------------
        # Step 3: Compute Session Duration & Extract Objective Features
        # ------------------------------------------------------------------
        if session_duration is None:
            computed_duration = 0.0
            for p in processed_participants:
                for seg in p.transcript_segments:
                    computed_duration = max(
                        computed_duration,
                        p.local_to_session_time(seg["end"]),
                    )
            session_duration = computed_duration

        feature_extractor = GDFeatureExtractor(session_duration=session_duration)
        features_by_participant: dict[str, dict[str, Any]] = {}

        for p in processed_participants:
            features_by_participant[p.participant_id] = feature_extractor.extract(p)

        # ------------------------------------------------------------------
        # Step 4: Extract Interaction Features
        # ------------------------------------------------------------------
        participant_ids = [p.participant_id for p in processed_participants]
        interaction_by_participant: dict[str, dict[str, Any]] = {}

        for pid in participant_ids:
            interaction_by_participant[pid] = self._interaction_extractor.extract(
                group_segments=group_segments,
                participant_id=pid,
            )

        # ------------------------------------------------------------------
        # Step 5: Build Agent Inputs Contract
        # ------------------------------------------------------------------
        agent_inputs: list[GDParticipantInput] = []

        for p in processed_participants:
            pid = p.participant_id
            other_participants = [other for other in participant_ids if other != pid]

            agent_input = GDParticipantInput(
                session_id=session_id,
                participant_id=pid,
                topic=topic,
                participant_transcript=p.transcript,
                group_transcript=group_text,
                features=features_by_participant[pid],
                interaction_features=interaction_by_participant[pid],
                group_segments=group_segments,
                other_participants=other_participants,
                metadata={
                    "audio_path": p.audio_path,
                    "session_offset": p.session_offset,
                    **p.metadata,
                },
            )
            agent_inputs.append(agent_input)

        # ------------------------------------------------------------------
        # Step 6: Multi-Agent Evaluation (Relevance, Coherence, Fluency, Participation)
        # ------------------------------------------------------------------
        orchestrator_reports = self._orchestrator.evaluate_session(agent_inputs)
        reports_by_participant = {
            report["participant_id"]: report for report in orchestrator_reports
        }

        # ------------------------------------------------------------------
        # Step 7: Assemble Final Structured Evaluation Result
        # ------------------------------------------------------------------
        participants_output: list[dict[str, Any]] = []
        participant_transcripts: dict[str, str] = {}
        agent_results_map: dict[str, list[dict[str, Any]]] = {}
        scorecards_map: dict[str, dict[str, Any]] = {}
        final_scores_map: dict[str, float] = {}

        for p in processed_participants:
            pid = p.participant_id
            report = reports_by_participant.get(pid, {})
            scorecard = report.get("scorecard", {})
            agent_results = report.get("agent_results", [])
            final_score = scorecard.get("final_score", 0.0)

            participant_transcripts[pid] = p.transcript
            agent_results_map[pid] = agent_results
            scorecards_map[pid] = scorecard
            final_scores_map[pid] = final_score

            participants_output.append(
                {
                    "participant_id": pid,
                    "audio_path": p.audio_path,
                    "session_offset": p.session_offset,
                    "transcript": p.transcript,
                    "transcript_segments": p.transcript_segments,
                    "vad_segments": p.vad_segments,
                    "speaking_time": p.total_speaking_time,
                    "word_count": p.word_count,
                    "turn_count": p.turn_count,
                    "features": features_by_participant[pid],
                    "interaction_features": interaction_by_participant[pid],
                    "agent_results": agent_results,
                    "scorecard": scorecard,
                    "final_score": final_score,
                }
            )

        return {
            "session_id": session_id,
            "topic": topic,
            "session_duration": round(session_duration, 3),
            "group_transcript": group_text,
            "group_segments": group_segments,
            "participants": participants_output,
            "participant_transcripts": participant_transcripts,
            "objective_features": features_by_participant,
            "interaction_features": interaction_by_participant,
            "agent_results": agent_results_map,
            "scorecards": scorecards_map,
            "final_scores": final_scores_map,
        }
