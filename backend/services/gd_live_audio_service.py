"""
GD Live Audio Event Service

Coordinates participant-scoped live audio event processing with in-memory LiveGDSession runtime:
- Enforces strict participant isolation (no diarization, no speaker recognition).
- Validates active session state and authorized participant roster.
- Requires participant to be actively connected in the live runtime.
- Runs CPU-light Silero VAD to detect speech segments.
- Transcribes speech segments using shared Whisper ASR only when speech is detected.
- Deterministically updates LiveGDSession runtime metrics (speaking, speaking_time, turn_count, word_count, last_audio_at).
- Does not persist raw audio events to MongoDB or trigger Qwen scoring.
"""

from pathlib import Path
from typing import Any, Optional

from backend.ai.asr_service import ASRService
from backend.ai.gd.live_audio_event import LiveAudioEvent
from backend.ai.gd.vad_service import VADService
from backend.services.gd_live_service import GDLiveService
from backend.services.gd_session_service import GDSessionService


class GDLiveAudioService:
    """
    Service responsible for processing individual participant audio events
    and updating in-memory LiveGDSession runtime metrics.
    """

    def __init__(
        self,
        session_service: Optional[GDSessionService] = None,
        live_service: Optional[GDLiveService] = None,
        vad_service: Optional[VADService] = None,
        asr_service: Optional[ASRService] = None,
    ) -> None:
        self.session_service = session_service or GDSessionService()
        self.live_service = live_service or GDLiveService(session_service=self.session_service)
        self._vad_service = vad_service
        self._asr_service = asr_service

    @property
    def vad_service(self) -> VADService:
        if self._vad_service is None:
            self._vad_service = VADService()
        return self._vad_service

    @property
    def asr_service(self) -> ASRService:
        if self._asr_service is None:
            self._asr_service = ASRService()
        return self._asr_service

    def process_event(self, event: LiveAudioEvent) -> dict[str, Any]:
        """
        Convenience wrapper to process a LiveAudioEvent instance directly.
        """
        if not isinstance(event, LiveAudioEvent):
            raise ValueError("event must be an instance of LiveAudioEvent")

        return self.process_audio_event(
            session_id=event.session_id,
            participant_id=event.participant_id,
            audio_path=event.audio_path,
            event=event,
        )

    def process_audio_event(
        self,
        session_id: str,
        participant_id: str,
        audio_path: Optional[str] = None,
        event: Optional[LiveAudioEvent] = None,
    ) -> dict[str, Any]:
        """
        Process one participant-scoped audio event:
        1. Validate parameters and persistent session/roster authorization.
        2. Verify active LiveGDSession runtime exists.
        3. Verify participant exists and is actively connected in runtime.
        4. Run VAD to isolate speech regions.
        5. Run shared ASR only when speech is detected.
        6. Update LiveGDSession runtime metrics for the participant.
        7. Return structured processing result.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required and must be a non-empty string")

        if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id is required and must be a non-empty string")

        clean_session_id = session_id.strip()
        clean_participant_id = participant_id.strip()

        # Handle event validation if provided
        if event is not None:
            if not isinstance(event, LiveAudioEvent):
                raise ValueError("event must be an instance of LiveAudioEvent")
            if event.session_id.strip() != clean_session_id:
                raise ValueError(
                    f"Event session_id '{event.session_id}' does not match target session_id '{clean_session_id}'"
                )
            if event.participant_id.strip() != clean_participant_id:
                raise ValueError(
                    f"Event participant_id '{event.participant_id}' does not match target participant_id '{clean_participant_id}'"
                )
            if audio_path is None and event.audio_path:
                audio_path = event.audio_path
            elif audio_path and event.audio_path and audio_path.strip() != event.audio_path.strip():
                raise ValueError(
                    f"Provided audio_path '{audio_path}' does not match event audio_path '{event.audio_path}'"
                )

        if not audio_path or not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError("audio_path is required and must be a non-empty string")

        clean_audio_path = audio_path.strip()
        audio_file = Path(clean_audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {clean_audio_path}")

        # Validate persistent session
        session = self.session_service.get_session(clean_session_id)
        current_status = session.get("status")

        if current_status != "ACTIVE":
            raise ValueError(
                f"Cannot process audio event in '{current_status}' status. Must be 'ACTIVE'."
            )

        participant_ids = session.get("participant_ids", [])
        if clean_participant_id not in participant_ids:
            raise ValueError(
                f"Participant '{clean_participant_id}' is not authorized for session '{clean_session_id}'"
            )

        # Validate live runtime
        live_session = self.live_service.get_runtime(clean_session_id)
        if live_session is None:
            raise RuntimeError(
                f"Live runtime for active GD session '{clean_session_id}' is not available"
            )

        if clean_participant_id not in live_session.participants:
            raise ValueError(
                f"Participant '{clean_participant_id}' not found in live session '{clean_session_id}'"
            )

        participant_state = live_session.participants[clean_participant_id]
        if not participant_state.connected:
            raise ValueError(
                f"Participant '{clean_participant_id}' is not currently connected to session '{clean_session_id}'"
            )

        # Step 1: VAD Detection
        vad_result = self.vad_service.detect(
            audio_path=clean_audio_path,
            session_id=clean_session_id,
            participant_id=clean_participant_id,
        )

        audio_duration = float(vad_result.get("audio_duration") or 0.0)
        speech_duration = float(vad_result.get("speech_duration") or 0.0)
        speech_ratio = float(vad_result.get("speech_ratio") or 0.0)
        vad_segments = vad_result.get("segments") or []

        # Step 2: VAD-Aware Shared ASR (run only when speech segments are detected)
        if vad_segments and speech_duration > 0:
            asr_result = self.asr_service.transcribe_segments(
                audio_path=clean_audio_path,
                speech_segments=vad_segments,
                language="en",
            )
            transcript_text = str(asr_result.get("text") or "").strip()
            transcript_segments = asr_result.get("segments") or []
        else:
            transcript_text = ""
            transcript_segments = []

        word_count = len(transcript_text.split()) if transcript_text else 0
        has_speech = bool(vad_segments) and speech_duration > 0

        was_speaking = participant_state.speaking

        # Step 3: Update LiveGDSession runtime metrics (isolated to clean_participant_id)
        live_session.update_audio_activity(clean_participant_id, speaking=has_speech)

        if speech_duration > 0:
            live_session.add_speaking_time(clean_participant_id, speech_duration)

        # Increment turn count only on speech onset (transition from non-speaking to speaking)
        if has_speech and not was_speaking:
            live_session.add_turn(clean_participant_id)

        if word_count > 0:
            live_session.add_words(clean_participant_id, word_count)

        # Step 4: Accumulate chronological transcript segments in LiveGDSession
        if has_speech and hasattr(live_session, "add_transcript_segment"):
            if event is not None and event.session_start > 0:
                base_start = float(event.session_start)
            else:
                current_time = live_session.session_time() if hasattr(live_session, "session_time") else 0.0
                base_start = max(0.0, float(current_time) - audio_duration)

            if transcript_segments:
                for seg in transcript_segments:
                    seg_text = str(seg.get("text") or "").strip()
                    if seg_text:
                        seg_start = base_start + float(seg.get("start") or 0.0)
                        seg_end = base_start + float(seg.get("end") or 0.0)
                        live_session.add_transcript_segment(
                            participant_id=clean_participant_id,
                            start=round(seg_start, 3),
                            end=round(seg_end, 3),
                            text=seg_text,
                        )
            elif transcript_text:
                live_session.add_transcript_segment(
                    participant_id=clean_participant_id,
                    start=round(base_start, 3),
                    end=round(base_start + audio_duration, 3),
                    text=transcript_text,
                )

        # Update event object if passed
        if event is not None:
            event.transcript = transcript_text
            if event.metadata is not None:
                event.metadata["vad_segments"] = vad_segments
                event.metadata["transcript_segments"] = transcript_segments
                event.metadata["speech_duration"] = speech_duration
                event.metadata["audio_duration"] = audio_duration

        return {
            "session_id": clean_session_id,
            "participant_id": clean_participant_id,
            "audio_duration": audio_duration,
            "speech_duration": speech_duration,
            "speech_ratio": speech_ratio,
            "segments": transcript_segments,
            "vad_segments": vad_segments,
            "transcript": transcript_text,
            "word_count": word_count,
            "turn_count": participant_state.turn_count,
            "runtime": live_session.get_status(),
        }
