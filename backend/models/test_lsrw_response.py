from datetime import datetime, timezone
from backend.models.lsrw_response import LSRWResponse


def test_lsrw_response_dataclass_initialization():
    now = datetime.now(timezone.utc)
    asr_data = {
        "text": "Hello world",
        "transcript": "Hello world",
        "segments": [],
        "language": "en",
    }
    eval_data = {
        "module": "SPEAKING",
        "question_id": "SP001",
        "transcript": "Hello world",
        "asr": asr_data,
        "semantic": {"model": "all-MiniLM-L6-v2", "similarity": 0.9},
    }

    response = LSRWResponse(
        session_id="session-1",
        participant_id="user-1",
        module="SPEAKING",
        question_id="SP001",
        audio_path="/path/to/audio.wav",
        transcript="Hello world",
        asr=asr_data,
        started_at=now,
        submitted_at=now,
        evaluation=eval_data,
        score=None,
    )

    assert response.session_id == "session-1"
    assert response.participant_id == "user-1"
    assert response.module == "SPEAKING"
    assert response.question_id == "SP001"
    assert response.audio_path == "/path/to/audio.wav"
    assert response.transcript == "Hello world"
    assert response.asr == asr_data
    assert response.started_at == now
    assert response.submitted_at == now
    assert response.evaluation == eval_data
    assert response.score is None


def test_lsrw_response_to_dict_includes_asr():
    response = LSRWResponse(
        session_id="session-1",
        participant_id="user-1",
        module="SPEAKING",
        question_id="SP001",
        asr={"transcript": "test transcript"},
        evaluation={"semantic": {"similarity": 0.8}},
        score=None,
    )

    d = response.to_dict()
    assert d["session_id"] == "session-1"
    assert d["participant_id"] == "user-1"
    assert d["module"] == "SPEAKING"
    assert d["question_id"] == "SP001"
    assert d["asr"] == {"transcript": "test transcript"}
    assert d["evaluation"] == {"semantic": {"similarity": 0.8}}
    assert d["score"] is None

