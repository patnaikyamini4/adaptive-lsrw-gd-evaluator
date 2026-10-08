"""
GD Live Session Integration Service

Coordinates persistent GDSession entities with in-memory LiveGDSession runtime state:
- Manages runtime lifecycle for active Group Discussion sessions.
- Validates participant authorization against persistent roster before allowing joins.
- Tracks live participant connection state without diarization or speaker recognition.
- Bridges persistent MongoDB session state with in-memory audio event clock.
"""

from typing import Any, Optional

from backend.ai.gd.audio_session import ParticipantAudio
from backend.ai.gd.feature_extraction import GDFeatureExtractor
from backend.ai.gd.gd_agent_input import GDParticipantInput
from backend.ai.gd.gd_orchestrator import GDOrchestrator
from backend.ai.gd.interaction_features import GDInteractionFeatureExtractor
from backend.ai.gd.live_session import LiveGDSession
import backend.services.gd_repository as default_gd_repo
from backend.services.gd_session_service import GDSessionService


class GDLiveService:
    """
    Coordinates persistent GD sessions and in-memory live runtime state.
    """

    def __init__(
        self,
        session_service: Optional[GDSessionService] = None,
        live_session_factory: Any = LiveGDSession,
        orchestrator: Optional[GDOrchestrator] = None,
        interaction_extractor: Optional[GDInteractionFeatureExtractor] = None,
        gd_repo: Any = None,
    ) -> None:
        self.session_service = session_service or GDSessionService()
        self.live_session_factory = live_session_factory
        self._orchestrator = orchestrator
        self._interaction_extractor = interaction_extractor
        self._gd_repo = gd_repo or default_gd_repo
        self._runtimes: dict[str, LiveGDSession] = {}

    @property
    def orchestrator(self) -> GDOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = GDOrchestrator()
        return self._orchestrator

    @property
    def interaction_extractor(self) -> GDInteractionFeatureExtractor:
        if self._interaction_extractor is None:
            self._interaction_extractor = GDInteractionFeatureExtractor()
        return self._interaction_extractor

    def start_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition persistent session (SCHEDULED -> ACTIVE) and create in-memory LiveGDSession.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "SCHEDULED":
            raise ValueError(
                f"Cannot start GD session in '{current_status}' status. Must be 'SCHEDULED'."
            )

        # Transition persistent state
        updated_session = self.session_service.start_session(clean_session_id)

        # Instantiate live runtime
        try:
            duration_sec = int(session.get("session_duration") or 0)
            live_session = self.live_session_factory(
                session_id=clean_session_id,
                topic=session.get("topic", ""),
                duration_seconds=duration_sec,
            )
            for pid in session.get("participant_ids", []):
                live_session.add_participant(pid)

            live_session.start()
            self._runtimes[clean_session_id] = live_session
        except Exception as e:
            # Rollback persistent session state to SCHEDULED if runtime creation fails
            try:
                self.session_service.repo.update_session(
                    clean_session_id,
                    {"status": "SCHEDULED", "session_started_at": None},
                )
            except Exception:
                pass
            raise RuntimeError(f"Failed to initialize live runtime: {str(e)}") from e

        return {
            "session": updated_session,
            "runtime": live_session.get_status(),
        }

    def get_runtime(self, session_id: str) -> Optional[LiveGDSession]:
        """
        Retrieve active LiveGDSession instance by session ID.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            return None
        return self._runtimes.get(session_id.strip())

    def join_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Connect an authorized participant to an ACTIVE live session.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        clean_pid = participant_id.strip()

        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot join session in '{current_status}' status. Must be 'ACTIVE'."
            )

        participant_ids = session.get("participant_ids", [])
        if clean_pid not in participant_ids:
            raise ValueError(
                f"Participant '{clean_pid}' is not authorized for session '{clean_session_id}'"
            )

        live_session = self._runtimes.get(clean_session_id)
        if live_session is None:
            raise RuntimeError(
                f"Live runtime for active GD session '{clean_session_id}' is not available"
            )

        live_session.participant_join(clean_pid)

        return {
            "session_id": clean_session_id,
            "participant_id": clean_pid,
            "connected": True,
            "runtime": live_session.get_status(),
        }

    def leave_participant(self, session_id: str, participant_id: str) -> dict[str, Any]:
        """
        Disconnect a participant from an ACTIVE live session.
        Does NOT remove participant from the persistent roster.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        clean_pid = participant_id.strip()

        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot leave session in '{current_status}' status. Must be 'ACTIVE'."
            )

        live_session = self._runtimes.get(clean_session_id)
        if live_session is None:
            raise ValueError(
                f"No active live runtime found for session '{clean_session_id}'"
            )

        if clean_pid not in live_session.participants:
            raise ValueError(
                f"Participant '{clean_pid}' not found in live session '{clean_session_id}'"
            )

        participant = live_session.participants[clean_pid]
        if not participant.connected:
            raise ValueError(
                f"Participant '{clean_pid}' is not currently connected to session '{clean_session_id}'"
            )

        live_session.participant_leave(clean_pid)

        return {
            "session_id": clean_session_id,
            "participant_id": clean_pid,
            "connected": False,
            "runtime": live_session.get_status(),
        }

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        Transition persistent session (ACTIVE -> ENDED) and stop in-memory LiveGDSession clock.
        Does NOT automatically trigger post-session evaluation.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot end GD session in '{current_status}' status. Must be 'ACTIVE'."
            )

        live_session = self._runtimes.get(clean_session_id)
        runtime_status = None
        if live_session is not None:
            live_session.end()
            runtime_status = live_session.get_status()

        ended_session = self.session_service.end_session(clean_session_id)

        return {
            "session": ended_session,
            "runtime": runtime_status,
        }

    def get_runtime_status(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve unified status combining persistent lifecycle state and in-memory clock/connections.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        live_session = self._runtimes.get(clean_session_id)
        if live_session is not None:
            live_status = live_session.get_status()
            return {
                "session_id": clean_session_id,
                "status": current_status,
                "runtime_active": (
                    current_status == "ACTIVE"
                    and live_status.get("started", False)
                    and not live_status.get("ended", False)
                ),
                "elapsed_time": live_status.get("session_time", 0.0),
                "duration_seconds": live_status.get(
                    "duration_seconds", session.get("session_duration")
                ),
                "participants": live_status.get("participants", {}),
                "live_status": live_status,
            }

        # Fallback when runtime is inactive/not initialized
        return {
            "session_id": clean_session_id,
            "status": current_status,
            "runtime_active": False,
            "elapsed_time": session.get("session_duration") or 0.0,
            "duration_seconds": session.get("session_duration"),
            "participants": {
                pid: {
                    "connected": False,
                    "joined_at": None,
                    "left_at": None,
                    "last_audio_at": None,
                    "speaking": False,
                    "speaking_time": 0.0,
                    "turn_count": 0,
                    "word_count": 0,
                }
                for pid in session.get("participant_ids", [])
            },
            "live_status": None,
        }

    def stop_runtime(self, session_id: str) -> bool:
        """
        Stop and remove live runtime instance.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            return False

        clean_session_id = session_id.strip()
        live_session = self._runtimes.pop(clean_session_id, None)
        if live_session is not None:
            live_session.end()
            return True
        return False

    def get_transcript(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve live transcript information for a session.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        live_session = self._runtimes.get(clean_session_id)

        if live_session is not None and hasattr(live_session, "get_transcript_data"):
            return live_session.get_transcript_data()

        # Fallback if runtime has been stopped or was not started
        participant_ids = session.get("participant_ids", [])
        return {
            "session_id": clean_session_id,
            "topic": session.get("topic", ""),
            "group_transcript": "",
            "group_segments": [],
            "participant_transcripts": {pid: "" for pid in participant_ids},
            "segment_count": 0,
        }

    def evaluate_live_session(self, session_id: str) -> dict[str, Any]:
        """
        Execute full end-to-end evaluation for an ENDED live GD session from in-memory runtime data
        and persist results to MongoDB.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status not in ("ENDED", "EVALUATED"):
            raise ValueError(
                f"Cannot evaluate live GD session in '{current_status}' status. Must be 'ENDED' or 'EVALUATED'."
            )

        live_session = self._runtimes.get(clean_session_id)
        participant_ids = list(session.get("participant_ids", []))
        topic = session.get("topic", "")

        # Determine session duration
        session_duration = session.get("session_duration")
        if session_duration is None or session_duration <= 0:
            if live_session is not None and hasattr(live_session, "session_time"):
                session_duration = live_session.session_time()
            else:
                session_duration = 0.0
        session_duration = float(session_duration)

        # Retrieve transcript data
        if live_session is not None and hasattr(live_session, "get_transcript_segments"):
            group_segments = live_session.get_transcript_segments()
            group_text = live_session.get_group_text()
            participant_transcripts = {
                pid: live_session.get_participant_text(pid)
                for pid in participant_ids
            }
        else:
            group_segments = []
            group_text = ""
            participant_transcripts = {pid: "" for pid in participant_ids}

        # Step 1: Reconstruct participant audio structures for feature extraction
        features_by_participant: dict[str, dict[str, Any]] = {}
        feature_extractor = GDFeatureExtractor(session_duration=session_duration)

        for pid in participant_ids:
            p_audio = ParticipantAudio(
                session_id=clean_session_id,
                participant_id=pid,
            )
            p_segs = [s for s in group_segments if s.get("participant_id") == pid]
            for s in p_segs:
                p_audio.add_transcript_segment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=str(s.get("text", "")),
                )
                p_audio.add_vad_segment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                )

            # If runtime has additional speaking time from VAD
            if live_session is not None and pid in live_session.participants:
                p_state = live_session.participants[pid]
                if p_state.speaking_time > p_audio.total_speaking_time:
                    extra_dur = p_state.speaking_time - p_audio.total_speaking_time
                    p_audio.add_vad_segment(start=0.0, end=extra_dur)

            features_by_participant[pid] = feature_extractor.extract(p_audio)

        # Step 2: Extract interaction features
        interaction_by_participant: dict[str, dict[str, Any]] = {}
        for pid in participant_ids:
            interaction_by_participant[pid] = self.interaction_extractor.extract(
                group_segments=group_segments,
                participant_id=pid,
            )

        # Step 3: Build GDParticipantInput contracts
        agent_inputs: list[GDParticipantInput] = []
        for pid in participant_ids:
            other_participants = [other for other in participant_ids if other != pid]
            agent_input = GDParticipantInput(
                session_id=clean_session_id,
                participant_id=pid,
                topic=topic,
                participant_transcript=participant_transcripts.get(pid, ""),
                group_transcript=group_text,
                features=features_by_participant[pid],
                interaction_features=interaction_by_participant[pid],
                group_segments=group_segments,
                other_participants=other_participants,
                metadata={
                    "live": True,
                    "session_duration": session_duration,
                    **session.get("metadata", {}),
                },
            )
            agent_inputs.append(agent_input)

        # Step 4: Multi-Agent Evaluation (Relevance, Coherence, Fluency, Participation)
        orchestrator_reports = self.orchestrator.evaluate_session(agent_inputs)
        reports_by_participant = {
            report["participant_id"]: report for report in orchestrator_reports
        }

        # Step 5: Assemble Final Structured Evaluation Result
        participants_output: list[dict[str, Any]] = []
        agent_results_map: dict[str, list[dict[str, Any]]] = {}
        scorecards_map: dict[str, dict[str, Any]] = {}
        final_scores_map: dict[str, float] = {}

        for pid in participant_ids:
            report = reports_by_participant.get(pid, {})
            scorecard = report.get("scorecard", {})
            agent_results = report.get("agent_results", [])
            final_score = scorecard.get("final_score", 0.0)

            agent_results_map[pid] = agent_results
            scorecards_map[pid] = scorecard
            final_scores_map[pid] = final_score

            p_state = live_session.participants.get(pid) if live_session else None
            speaking_time = (
                p_state.speaking_time
                if p_state
                else features_by_participant[pid].get("speaking_time", 0.0)
            )
            word_count = (
                p_state.word_count
                if p_state
                else features_by_participant[pid].get("word_count", 0)
            )
            turn_count = (
                p_state.turn_count
                if p_state
                else features_by_participant[pid].get("turn_count", 0)
            )

            participants_output.append(
                {
                    "participant_id": pid,
                    "audio_path": None,
                    "session_offset": 0.0,
                    "transcript": participant_transcripts.get(pid, ""),
                    "transcript_segments": [
                        s for s in group_segments if s.get("participant_id") == pid
                    ],
                    "vad_segments": [],
                    "speaking_time": round(speaking_time, 3),
                    "word_count": word_count,
                    "turn_count": turn_count,
                    "features": features_by_participant[pid],
                    "interaction_features": interaction_by_participant[pid],
                    "agent_results": agent_results,
                    "scorecard": scorecard,
                    "final_score": final_score,
                }
            )

        evaluation_data = {
            "session_id": clean_session_id,
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

        # Step 6: Persist Evaluation via GDEvaluationRepository (upsert)
        persisted_eval = self._gd_repo.save_evaluation(evaluation_data)

        # Step 7: Update GDSession status to EVALUATED and record evaluation_id
        try:
            self.session_service.mark_evaluated(
                clean_session_id, evaluation_id=clean_session_id
            )
        except Exception:
            pass

        return persisted_eval
