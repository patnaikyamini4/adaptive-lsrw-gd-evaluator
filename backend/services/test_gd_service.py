from pathlib import Path
from unittest.mock import MagicMock
import pytest

from backend.ai.gd.audio_session import ParticipantAudio
from backend.services.gd_service import GDEvaluationService


def create_mock_participant_audio(
    session_id: str,
    participant_id: str,
    audio_path: str,
    session_offset: float = 0.0,
    transcript_segments: list[dict] | None = None,
    vad_segments: list[dict] | None = None,
) -> ParticipantAudio:
    participant = ParticipantAudio(
        session_id=session_id,
        participant_id=participant_id,
        audio_path=audio_path,
        session_offset=session_offset,
    )
    if vad_segments:
        for seg in vad_segments:
            participant.add_vad_segment(start=seg["start"], end=seg["end"])
    else:
        participant.add_vad_segment(start=1.0, end=4.0)

    if transcript_segments:
        for seg in transcript_segments:
            participant.add_transcript_segment(
                start=seg["start"], end=seg["end"], text=seg["text"]
            )
    else:
        participant.add_transcript_segment(
            start=1.0, end=4.0, text=f"Contribution by {participant_id}"
        )

    return participant


def create_mock_orchestrator_report(
    session_id: str,
    participant_id: str,
    topic: str,
    score: float = 85.0,
) -> dict:
    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "topic": topic,
        "agent_results": [
            {
                "participant_id": participant_id,
                "agent": "relevance",
                "score": score,
                "reasoning": "Relevant to topic",
                "strengths": ["On point"],
                "weaknesses": [],
                "evidence": ["Good example"],
                "metadata": {},
            },
            {
                "participant_id": participant_id,
                "agent": "coherence",
                "score": score,
                "reasoning": "Logically structured",
                "strengths": ["Clear progression"],
                "weaknesses": [],
                "evidence": ["Logical flow"],
                "metadata": {},
            },
            {
                "participant_id": participant_id,
                "agent": "fluency",
                "score": score,
                "reasoning": "Fluent English",
                "strengths": ["Smooth articulation"],
                "weaknesses": [],
                "evidence": ["Natural pace"],
                "metadata": {},
            },
            {
                "participant_id": participant_id,
                "agent": "participation",
                "score": score,
                "reasoning": "Active engagement",
                "strengths": ["Constructive turn"],
                "weaknesses": [],
                "evidence": ["Responded well"],
                "metadata": {},
            },
        ],
        "scorecard": {
            "participant_id": participant_id,
            "scores": {
                "relevance": score,
                "coherence": score,
                "fluency": score,
                "participation": score,
            },
            "weights": {
                "relevance": 0.25,
                "coherence": 0.25,
                "fluency": 0.25,
                "participation": 0.25,
            },
            "final_score": score,
        },
    }


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    return pipeline


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    return orchestrator


@pytest.fixture
def temp_audio_files(tmp_path):
    p1_file = tmp_path / "p001.wav"
    p2_file = tmp_path / "p002.wav"
    p3_file = tmp_path / "p003.wav"
    p1_file.write_bytes(b"RIFF" + b"\x00" * 100)
    p2_file.write_bytes(b"RIFF" + b"\x00" * 100)
    p3_file.write_bytes(b"RIFF" + b"\x00" * 100)
    return {
        "P001": str(p1_file),
        "P002": str(p2_file),
        "P003": str(p3_file),
    }


def test_one_valid_participant(mock_pipeline, mock_orchestrator, temp_audio_files):
    session_id = "GD_TEST_001"
    topic = "Artificial Intelligence in Education"

    mock_participant = create_mock_participant_audio(
        session_id=session_id,
        participant_id="P001",
        audio_path=temp_audio_files["P001"],
        transcript_segments=[
            {"start": 1.0, "end": 4.0, "text": "AI helps students learn faster."}
        ],
    )
    mock_pipeline.process.return_value = mock_participant

    mock_report = create_mock_orchestrator_report(
        session_id=session_id,
        participant_id="P001",
        topic=topic,
        score=88.0,
    )
    mock_orchestrator.evaluate_session.return_value = [mock_report]

    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    result = service.evaluate_session(
        session_id=session_id,
        topic=topic,
        participants=[
            {
                "participant_id": "P001",
                "audio_path": temp_audio_files["P001"],
            }
        ],
    )

    assert result["session_id"] == session_id
    assert result["topic"] == topic
    assert len(result["participants"]) == 1
    assert result["participants"][0]["participant_id"] == "P001"
    assert result["participants"][0]["transcript"] == "AI helps students learn faster."
    assert result["participants"][0]["final_score"] == 88.0
    assert result["final_scores"]["P001"] == 88.0
    assert "[P001] AI helps students learn faster." in result["group_transcript"]


def test_multiple_participants(mock_pipeline, mock_orchestrator, temp_audio_files):
    session_id = "GD_MULTI_001"
    topic = "Renewable Energy Transition"

    p1 = create_mock_participant_audio(
        session_id=session_id,
        participant_id="P001",
        audio_path=temp_audio_files["P001"],
        session_offset=0.0,
        transcript_segments=[
            {"start": 1.0, "end": 5.0, "text": "Solar energy is becoming cheaper."}
        ],
    )
    p2 = create_mock_participant_audio(
        session_id=session_id,
        participant_id="P002",
        audio_path=temp_audio_files["P002"],
        session_offset=6.0,
        transcript_segments=[
            {"start": 1.0, "end": 4.0, "text": "Wind power is also very reliable."}
        ],
    )
    p3 = create_mock_participant_audio(
        session_id=session_id,
        participant_id="P003",
        audio_path=temp_audio_files["P003"],
        session_offset=12.0,
        transcript_segments=[
            {"start": 1.0, "end": 5.0, "text": "Battery storage is critical."}
        ],
    )

    def process_side_effect(session_id, participant_id, audio_path):
        if participant_id == "P001":
            return p1
        elif participant_id == "P002":
            return p2
        else:
            return p3

    mock_pipeline.process.side_effect = process_side_effect

    reports = [
        create_mock_orchestrator_report(session_id, "P001", topic, score=80.0),
        create_mock_orchestrator_report(session_id, "P002", topic, score=84.0),
        create_mock_orchestrator_report(session_id, "P003", topic, score=90.0),
    ]
    mock_orchestrator.evaluate_session.return_value = reports

    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    result = service.evaluate_session(
        session_id=session_id,
        topic=topic,
        participants=[
            {"participant_id": "P001", "audio_path": temp_audio_files["P001"], "session_offset": 0.0},
            {"participant_id": "P002", "audio_path": temp_audio_files["P002"], "session_offset": 6.0},
            {"participant_id": "P003", "audio_path": temp_audio_files["P003"], "session_offset": 12.0},
        ],
    )

    assert len(result["participants"]) == 3
    assert result["final_scores"] == {
        "P001": 80.0,
        "P002": 84.0,
        "P003": 90.0,
    }

    # Verify chronological sequence in group transcript
    lines = result["group_transcript"].split("\n")
    assert len(lines) == 3
    assert "[P001]" in lines[0]
    assert "[P002]" in lines[1]
    assert "[P003]" in lines[2]


def test_participant_id_preservation(mock_pipeline, mock_orchestrator, temp_audio_files):
    session_id = "GD_ID_PRESERVE"
    topic = "Ethical AI"

    p1 = create_mock_participant_audio(session_id, "P_ALICE", temp_audio_files["P001"])
    p2 = create_mock_participant_audio(session_id, "P_BOB", temp_audio_files["P002"])

    mock_pipeline.process.side_effect = lambda session_id, participant_id, audio_path: (
        p1 if participant_id == "P_ALICE" else p2
    )
    mock_orchestrator.evaluate_session.return_value = [
        create_mock_orchestrator_report(session_id, "P_ALICE", topic, 75.0),
        create_mock_orchestrator_report(session_id, "P_BOB", topic, 85.0),
    ]

    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    result = service.evaluate_session(
        session_id=session_id,
        topic=topic,
        participants=[
            {"participant_id": "P_ALICE", "audio_path": temp_audio_files["P001"]},
            {"participant_id": "P_BOB", "audio_path": temp_audio_files["P002"]},
        ],
    )

    assert "P_ALICE" in result["participant_transcripts"]
    assert "P_BOB" in result["participant_transcripts"]
    assert "P_ALICE" in result["objective_features"]
    assert "P_BOB" in result["objective_features"]
    assert "P_ALICE" in result["interaction_features"]
    assert "P_BOB" in result["interaction_features"]
    assert "P_ALICE" in result["agent_results"]
    assert "P_BOB" in result["agent_results"]


def test_missing_audio_path_raises_file_not_found(mock_pipeline, mock_orchestrator):
    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        service.evaluate_session(
            session_id="GD001",
            topic="Valid Topic",
            participants=[
                {
                    "participant_id": "P001",
                    "audio_path": "non_existent_audio_file_123.wav",
                }
            ],
        )


def test_empty_participants_raises_value_error(mock_pipeline, mock_orchestrator):
    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    with pytest.raises(ValueError, match="participants must be a non-empty list"):
        service.evaluate_session(
            session_id="GD001",
            topic="Valid Topic",
            participants=[],
        )


def test_missing_topic_and_session_id(mock_pipeline, mock_orchestrator, temp_audio_files):
    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        service.evaluate_session(
            session_id="",
            topic="Valid Topic",
            participants=[{"participant_id": "P001", "audio_path": temp_audio_files["P001"]}],
        )

    with pytest.raises(ValueError, match="topic must be a non-empty string"):
        service.evaluate_session(
            session_id="GD001",
            topic="   ",
            participants=[{"participant_id": "P001", "audio_path": temp_audio_files["P001"]}],
        )


def test_duplicate_participant_id_rejected(mock_pipeline, mock_orchestrator, temp_audio_files):
    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    with pytest.raises(ValueError, match="Duplicate participant_id"):
        service.evaluate_session(
            session_id="GD001",
            topic="Valid Topic",
            participants=[
                {"participant_id": "P001", "audio_path": temp_audio_files["P001"]},
                {"participant_id": "P001", "audio_path": temp_audio_files["P002"]},
            ],
        )


def test_orchestrator_result_propagation_and_structured_format(
    mock_pipeline, mock_orchestrator, temp_audio_files
):
    session_id = "GD_FULL_STRUCT_001"
    topic = "Digital Economy"

    p1 = create_mock_participant_audio(
        session_id=session_id,
        participant_id="P001",
        audio_path=temp_audio_files["P001"],
        transcript_segments=[
            {"start": 1.0, "end": 4.0, "text": "Fintech fosters financial inclusion."}
        ],
    )
    mock_pipeline.process.return_value = p1

    report = create_mock_orchestrator_report(
        session_id=session_id,
        participant_id="P001",
        topic=topic,
        score=92.5,
    )
    mock_orchestrator.evaluate_session.return_value = [report]

    service = GDEvaluationService(
        participant_pipeline=mock_pipeline,
        orchestrator=mock_orchestrator,
    )

    result = service.evaluate_session(
        session_id=session_id,
        topic=topic,
        participants=[{"participant_id": "P001", "audio_path": temp_audio_files["P001"]}],
        session_duration=60.0,
    )

    # Verify top-level structure
    assert result["session_id"] == session_id
    assert result["topic"] == topic
    assert result["session_duration"] == 60.0
    assert isinstance(result["group_transcript"], str)
    assert isinstance(result["group_segments"], list)
    assert isinstance(result["participants"], list)
    assert isinstance(result["participant_transcripts"], dict)
    assert isinstance(result["objective_features"], dict)
    assert isinstance(result["interaction_features"], dict)
    assert isinstance(result["agent_results"], dict)
    assert isinstance(result["scorecards"], dict)
    assert isinstance(result["final_scores"], dict)

    # Verify participant details
    p_data = result["participants"][0]
    assert p_data["participant_id"] == "P001"
    assert p_data["speaking_time"] == 3.0
    assert p_data["word_count"] == 4
    assert p_data["turn_count"] == 1
    assert p_data["final_score"] == 92.5

    # Verify objective features
    features = result["objective_features"]["P001"]
    assert features["speaking_time"] == 3.0
    assert features["word_count"] == 4
    assert features["turn_count"] == 1
    assert features["speaking_ratio"] == 0.05

    # Verify agent results propagation
    agent_results = result["agent_results"]["P001"]
    assert len(agent_results) == 4
    agent_names = {res["agent"] for res in agent_results}
    assert agent_names == {"relevance", "coherence", "fluency", "participation"}

    # Verify scorecard propagation
    scorecard = result["scorecards"]["P001"]
    assert scorecard["final_score"] == 92.5
    assert scorecard["scores"]["relevance"] == 92.5
