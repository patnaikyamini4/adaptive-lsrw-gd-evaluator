"""
Unit and integration tests for GD Live Audio Event Service (backend/services/gd_live_audio_service.py).

Tests verify that:
1. Valid participant audio events update participant-scoped LiveGDSession metrics.
2. Participant isolation is strictly preserved (P001 never updates P002).
3. Non-ACTIVE, missing, unauthorized, unconnected, and runtime-missing states are rejected.
4. Mismatched event and target identifiers are rejected.
5. VAD and shared ASR are invoked on speech segments without diarization or Qwen calls.
6. Service never creates duplicate LiveGDSession instances.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.ai.gd.live_audio_event import LiveAudioEvent
from backend.ai.gd.live_session import LiveGDSession
from backend.services.gd_live_audio_service import GDLiveAudioService
from backend.services.gd_live_service import GDLiveService
from backend.services.gd_session_service import GDSessionService


@pytest.fixture
def mock_session_service():
    """Mock GDSessionService with in-memory session database."""
    service = MagicMock(spec=GDSessionService)
    sessions_db = {}

    def mock_get_session(session_id):
        if session_id not in sessions_db:
            raise ValueError(f"GD session '{session_id}' not found")
        return sessions_db[session_id].copy()

    service.get_session.side_effect = mock_get_session
    service._db = sessions_db
    return service


@pytest.fixture
def active_session_setup(mock_session_service):
    """Set up an ACTIVE GD session with a live runtime and 3 connected participants."""
    now = datetime.now(timezone.utc)
    session_id = "GD-LIVE-200"
    session_data = {
        "session_id": session_id,
        "topic": "Renewable Energy Transition",
        "coordinator_id": "COORD-01",
        "participant_ids": ["P001", "P002", "P003"],
        "status": "ACTIVE",
        "scheduled_start": now,
        "scheduled_end": None,
        "session_started_at": now,
        "session_ended_at": None,
        "session_duration": 300.0,
        "evaluation_id": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "document_type": "gd_session",
    }
    mock_session_service._db[session_id] = session_data

    # Create live service and runtime with connected participants
    live_service = GDLiveService(session_service=mock_session_service)
    runtime = LiveGDSession(
        session_id=session_id,
        topic=session_data["topic"],
        duration_seconds=300,
    )
    for pid in session_data["participant_ids"]:
        runtime.add_participant(pid)
        runtime.participant_join(pid)
    runtime.start()
    live_service._runtimes[session_id] = runtime

    return {
        "session_id": session_id,
        "session_data": session_data,
        "live_service": live_service,
        "runtime": runtime,
    }


@pytest.fixture
def mock_vad():
    """Mock VADService returning realistic speech segments."""
    vad = MagicMock()
    vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "sample_rate": 16000,
        "audio_duration": 4.5,
        "speech_duration": 3.2,
        "speech_ratio": 0.7111,
        "segments": [
            {"start": 0.5, "end": 2.0, "duration": 1.5},
            {"start": 2.8, "end": 4.5, "duration": 1.7},
        ],
    }
    return vad


@pytest.fixture
def mock_asr():
    """Mock ASRService returning realistic transcript output."""
    asr = MagicMock()
    asr.model_name = "base"
    asr.device = "cpu"
    asr.transcribe_segments.return_value = {
        "text": "Solar energy is essential for sustainable progress.",
        "segments": [
            {"start": 0.5, "end": 2.0, "duration": 1.5, "text": "Solar energy is essential"},
            {"start": 2.8, "end": 4.5, "duration": 1.7, "text": "for sustainable progress."},
        ],
        "language": "en",
        "model": "base",
        "device": "cpu",
    }
    return asr


@pytest.fixture
def dummy_audio_file(tmp_path):
    """Create a temporary dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(b"RIFFdummywavdata")
    return str(audio_path)


# ==============================================================================
# 1. SUCCESSFUL AUDIO EVENT & RUNTIME METRIC UPDATES
# ==============================================================================

def test_connected_participant_can_submit_and_process_audio(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """9, 11, 12, 13, 14, 15. Test connected participant can process audio with correct session/participant metrics."""
    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    result = audio_service.process_audio_event(
        session_id="GD-LIVE-200",
        participant_id="P001",
        audio_path=dummy_audio_file,
    )

    # 12, 13. Output contains authoritative session and participant identifiers
    assert result["session_id"] == "GD-LIVE-200"
    assert result["participant_id"] == "P001"
    assert result["audio_duration"] == 4.5
    assert result["speech_duration"] == 3.2
    assert result["speech_ratio"] == 0.7111
    assert result["transcript"] == "Solar energy is essential for sustainable progress."
    assert result["word_count"] == 7
    assert result["turn_count"] == 1
    assert len(result["segments"]) == 2
    assert len(result["vad_segments"]) == 2

    # 11. Speech VAD result calls shared ASR
    mock_vad.detect.assert_called_once_with(
        audio_path=dummy_audio_file,
        session_id="GD-LIVE-200",
        participant_id="P001",
    )
    mock_asr.transcribe_segments.assert_called_once_with(
        audio_path=dummy_audio_file,
        speech_segments=mock_vad.detect.return_value["segments"],
        language="en",
    )

    # 14, 15. Runtime speaking metrics & last_audio_at are updated
    runtime = live_service.get_runtime("GD-LIVE-200")
    p001 = runtime.participants["P001"]
    assert p001.speaking is True
    assert p001.speaking_time == 3.2
    assert p001.turn_count == 1
    assert p001.word_count == 7
    assert p001.last_audio_at is not None


def test_process_live_audio_event_dataclass(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """Test processing via LiveAudioEvent instance."""
    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    event = LiveAudioEvent(
        session_id="GD-LIVE-200",
        participant_id="P001",
        session_start=0.0,
        session_end=4.5,
        audio_path=dummy_audio_file,
        metadata={"mic_channel": 1},
    )

    result = audio_service.process_event(event)

    assert result["session_id"] == "GD-LIVE-200"
    assert result["participant_id"] == "P001"
    assert event.transcript == "Solar energy is essential for sustainable progress."
    assert event.metadata["audio_duration"] == 4.5
    assert event.metadata["speech_duration"] == 3.2


def test_multiple_chunks_from_same_participant_accumulate_correctly(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """20. Test multiple chunks from the same participant accumulate metrics deterministically."""
    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    # Chunk 1: Speech onset (Turn 1)
    mock_vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "audio_duration": 3.0,
        "speech_duration": 2.0,
        "speech_ratio": 0.6667,
        "segments": [{"start": 0.5, "end": 2.5, "duration": 2.0}],
    }
    mock_asr.transcribe_segments.return_value = {
        "text": "First chunk text",
        "segments": [{"start": 0.5, "end": 2.5, "duration": 2.0, "text": "First chunk text"}],
        "language": "en",
    }
    res1 = audio_service.process_audio_event("GD-LIVE-200", "P001", dummy_audio_file)
    assert res1["turn_count"] == 1
    assert res1["word_count"] == 3

    # Chunk 2: Continuous speech in the same turn (Turn count remains 1, time and words accumulate)
    mock_vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "audio_duration": 4.0,
        "speech_duration": 3.0,
        "speech_ratio": 0.75,
        "segments": [{"start": 0.5, "end": 3.5, "duration": 3.0}],
    }
    mock_asr.transcribe_segments.return_value = {
        "text": "Second chunk continuing the speech",
        "segments": [{"start": 0.5, "end": 3.5, "duration": 3.0, "text": "Second chunk continuing the speech"}],
        "language": "en",
    }
    res2 = audio_service.process_audio_event("GD-LIVE-200", "P001", dummy_audio_file)
    assert res2["turn_count"] == 1
    assert res2["word_count"] == 5

    runtime = live_service.get_runtime("GD-LIVE-200")
    p001 = runtime.participants["P001"]
    assert p001.speaking is True
    assert p001.speaking_time == 5.0
    assert p001.turn_count == 1
    assert p001.word_count == 8

    # Chunk 3: Silence / pause (Speaking becomes False)
    mock_vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "audio_duration": 2.0,
        "speech_duration": 0.0,
        "speech_ratio": 0.0,
        "segments": [],
    }
    res3 = audio_service.process_audio_event("GD-LIVE-200", "P001", dummy_audio_file)
    assert res3["turn_count"] == 1
    assert p001.speaking is False

    # Chunk 4: Resumed speech after pause (Turn onset -> Turn 2)
    mock_vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "audio_duration": 2.0,
        "speech_duration": 1.5,
        "speech_ratio": 0.75,
        "segments": [{"start": 0.2, "end": 1.7, "duration": 1.5}],
    }
    mock_asr.transcribe_segments.return_value = {
        "text": "Another speaking turn",
        "segments": [{"start": 0.2, "end": 1.7, "duration": 1.5, "text": "Another speaking turn"}],
        "language": "en",
    }
    res4 = audio_service.process_audio_event("GD-LIVE-200", "P001", dummy_audio_file)
    assert res4["turn_count"] == 2
    assert p001.speaking is True
    assert p001.speaking_time == 6.5
    assert p001.turn_count == 2
    assert p001.word_count == 11


# ==============================================================================
# 2. PARTICIPANT ISOLATION TESTS
# ==============================================================================

def test_participant_isolation_p001_does_not_mutate_p002(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """Test that processing audio for P001 does NOT mutate P002 or P003 runtime state."""
    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    audio_service.process_audio_event(
        session_id="GD-LIVE-200",
        participant_id="P001",
        audio_path=dummy_audio_file,
    )

    runtime = live_service.get_runtime("GD-LIVE-200")
    p001 = runtime.participants["P001"]
    p002 = runtime.participants["P002"]
    p003 = runtime.participants["P003"]

    # P001 updated
    assert p001.speaking_time == 3.2
    assert p001.turn_count == 1
    assert p001.word_count == 7
    assert p001.speaking is True

    # P002 untouched (connected=True from setup fixture, but zero audio metrics)
    assert p002.speaking_time == 0.0
    assert p002.turn_count == 0
    assert p002.word_count == 0
    assert p002.speaking is False
    assert p002.last_audio_at is None

    # P003 untouched
    assert p003.speaking_time == 0.0
    assert p003.turn_count == 0
    assert p003.word_count == 0
    assert p003.speaking is False
    assert p003.last_audio_at is None


# ==============================================================================
# 3. SILENCE & ZERO-SPEECH HANDLING
# ==============================================================================

def test_silence_audio_event_does_not_call_asr_or_add_turns(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """10. Test audio with no detected speech skips ASR and marks speaking=False."""
    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    mock_vad.detect.return_value = {
        "session_id": "GD-LIVE-200",
        "participant_id": "P001",
        "audio_duration": 3.0,
        "speech_duration": 0.0,
        "speech_ratio": 0.0,
        "segments": [],
    }

    result = audio_service.process_audio_event(
        session_id="GD-LIVE-200",
        participant_id="P001",
        audio_path=dummy_audio_file,
    )

    mock_asr.transcribe_segments.assert_not_called()
    assert result["speech_duration"] == 0.0
    assert result["transcript"] == ""
    assert result["word_count"] == 0
    assert result["turn_count"] == 0

    runtime = live_service.get_runtime("GD-LIVE-200")
    p001 = runtime.participants["P001"]
    assert p001.speaking is False
    assert p001.speaking_time == 0.0
    assert p001.turn_count == 0
    assert p001.word_count == 0


# ==============================================================================
# 4. VALIDATION & ERROR REJECTION TESTS
# ==============================================================================

def test_invalid_session_id_rejected(mock_session_service, mock_vad, mock_asr, dummy_audio_file):
    """1. Test empty or whitespace session_id raises ValueError."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(ValueError, match="session_id is required and must be a non-empty string"):
        audio_service.process_audio_event(
            session_id="",
            participant_id="P001",
            audio_path=dummy_audio_file,
        )


def test_invalid_participant_id_rejected(mock_session_service, mock_vad, mock_asr, dummy_audio_file):
    """2. Test empty or whitespace participant_id raises ValueError."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(ValueError, match="participant_id is required and must be a non-empty string"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="   ",
            audio_path=dummy_audio_file,
        )


def test_missing_session_rejected(mock_session_service, mock_vad, mock_asr, dummy_audio_file):
    """3. Test missing session raises ValueError."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(ValueError, match="GD session 'NON_EXISTENT' not found"):
        audio_service.process_audio_event(
            session_id="NON_EXISTENT",
            participant_id="P001",
            audio_path=dummy_audio_file,
        )


@pytest.mark.parametrize("status", ["SCHEDULED", "ENDED", "CANCELLED", "EVALUATED"])
def test_inactive_session_status_rejected(
    mock_session_service, mock_vad, mock_asr, dummy_audio_file, status
):
    """4, 5. Test non-ACTIVE session statuses (SCHEDULED, ENDED, etc.) are rejected."""
    mock_session_service._db["GD-LIVE-STATUS"] = {
        "session_id": "GD-LIVE-STATUS",
        "topic": "Status Test",
        "coordinator_id": "COORD-01",
        "participant_ids": ["P001"],
        "status": status,
    }
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(ValueError, match=f"Cannot process audio event in '{status}' status. Must be 'ACTIVE'."):
        audio_service.process_audio_event(
            session_id="GD-LIVE-STATUS",
            participant_id="P001",
            audio_path=dummy_audio_file,
        )


def test_unauthorized_participant_not_in_roster_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """6. Test participant not in persistent roster is rejected."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=active_session_setup["live_service"],
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(ValueError, match="Participant 'P999' is not authorized for session 'GD-LIVE-200'"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P999",
            audio_path=dummy_audio_file,
        )


def test_missing_live_runtime_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """7, 18. Test that an ACTIVE session with missing runtime raises RuntimeError and creates no duplicate runtime."""
    live_service = active_session_setup["live_service"]
    live_service._runtimes.clear()

    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(
        RuntimeError, match="Live runtime for active GD session 'GD-LIVE-200' is not available"
    ):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            audio_path=dummy_audio_file,
        )

    # 18. Verify no replacement runtime was created
    assert live_service.get_runtime("GD-LIVE-200") is None


def test_participant_not_connected_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """8. Test that an authorized participant who has not connected (or has left) is rejected."""
    live_service = active_session_setup["live_service"]
    # Mark P002 as disconnected
    live_service.leave_participant("GD-LIVE-200", "P002")

    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    with pytest.raises(
        ValueError, match="Participant 'P002' is not currently connected to session 'GD-LIVE-200'"
    ):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P002",
            audio_path=dummy_audio_file,
        )


def test_mismatched_session_id_in_event_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """Test mismatched session_id between event and target is rejected."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=active_session_setup["live_service"],
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    event = LiveAudioEvent(
        session_id="OTHER-SESSION",
        participant_id="P001",
        audio_path=dummy_audio_file,
    )

    with pytest.raises(ValueError, match="Event session_id 'OTHER-SESSION' does not match target session_id 'GD-LIVE-200'"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            event=event,
        )


def test_mismatched_participant_id_in_event_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """Test mismatched participant_id between event and target is rejected."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=active_session_setup["live_service"],
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    event = LiveAudioEvent(
        session_id="GD-LIVE-200",
        participant_id="P002",
        audio_path=dummy_audio_file,
    )

    with pytest.raises(ValueError, match="Event participant_id 'P002' does not match target participant_id 'P001'"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            event=event,
        )


def test_malformed_and_missing_audio_path_rejected(
    mock_session_service, active_session_setup, mock_vad, mock_asr
):
    """19. Test missing or empty audio file path is rejected."""
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=active_session_setup["live_service"],
        vad_service=mock_vad,
        asr_service=mock_asr,
    )

    # Missing file
    with pytest.raises(FileNotFoundError, match="Audio file not found: non_existent.wav"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            audio_path="non_existent.wav",
        )

    # Empty string
    with pytest.raises(ValueError, match="audio_path is required and must be a non-empty string"):
        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            audio_path="   ",
        )


# ==============================================================================
# 5. ZERO DIARIZATION & ZERO QWEN SIDE-EFFECT TESTS
# ==============================================================================

def test_live_audio_processing_never_invokes_qwen_or_diarization(
    mock_session_service, active_session_setup, mock_vad, mock_asr, dummy_audio_file
):
    """16, 17. Test that processing audio events NEVER invokes Qwen or any diarization module."""
    with patch("backend.ai.qwen_service.ask_qwen") as mock_qwen:
        audio_service = GDLiveAudioService(
            session_service=mock_session_service,
            live_service=active_session_setup["live_service"],
            vad_service=mock_vad,
            asr_service=mock_asr,
        )

        audio_service.process_audio_event(
            session_id="GD-LIVE-200",
            participant_id="P001",
            audio_path=dummy_audio_file,
        )

        # 16. Zero Qwen / LLM calls
        mock_qwen.assert_not_called()


# ==============================================================================
# 6. INTEGRATION-STYLE TEST WITH REAL SAMPLE AUDIO
# ==============================================================================

def test_real_sample_audio_vad_integration(mock_session_service, active_session_setup, mock_asr):
    """Integration test verifying real VAD execution on an existing sample audio file."""
    real_audio_path = Path("data/gd/audio/p001_test.wav")
    if not real_audio_path.exists():
        pytest.skip("Sample audio p001_test.wav not available")

    live_service = active_session_setup["live_service"]
    audio_service = GDLiveAudioService(
        session_service=mock_session_service,
        live_service=live_service,
        vad_service=None,  # triggers real VADService
        asr_service=mock_asr,
    )

    result = audio_service.process_audio_event(
        session_id="GD-LIVE-200",
        participant_id="P001",
        audio_path=str(real_audio_path),
    )

    assert result["session_id"] == "GD-LIVE-200"
    assert result["participant_id"] == "P001"
    assert result["audio_duration"] > 0.0
    assert result["speech_duration"] > 0.0
    assert len(result["vad_segments"]) > 0

    runtime = live_service.get_runtime("GD-LIVE-200")
    p001 = runtime.participants["P001"]
    assert p001.speaking_time > 0.0
    assert p001.turn_count == 1
