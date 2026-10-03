"""
Real 4-Participant GD Audio Pipeline Validation

Validates the full GD audio preprocessing pipeline using:
- Real Silero VAD (CPU)
- Real Whisper ASR (GPU/CPU)
- Real 4-Participant test audio files
- Mocked GDOrchestrator (zero Qwen / LLM calls)

Proves:
1. All four participant IDs are preserved
2. All four audio files are processed
3. VAD produces participant speech segments
4. Shared ASR produces participant transcripts
5. Session offsets are respected
6. Group transcript is chronological
7. Objective features are generated for every participant
8. Interaction features are generated for every participant
9. GDParticipantInput is constructed correctly
10. Orchestrator result structure is propagated into final service result
11. Final output contains four participant records
"""

from pathlib import Path
import sys
import time
from unittest.mock import MagicMock
import pytest

# Ensure repository root is in sys.path for standalone script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.gd_service import GDEvaluationService


AUDIO_DIR = Path("data/gd/audio")
SESSION_ID = "GD_REAL_VAL_001"
TOPIC = "Regional Food Culture in India"

PARTICIPANTS_CONFIG = [
    {
        "participant_id": "P001",
        "audio_path": str(AUDIO_DIR / "p001_test.wav"),
        "session_offset": 0.0,
    },
    {
        "participant_id": "P002",
        "audio_path": str(AUDIO_DIR / "p002_test.wav"),
        "session_offset": 16.994,
    },
    {
        "participant_id": "P003",
        "audio_path": str(AUDIO_DIR / "p003_test.wav"),
        "session_offset": 39.028,
    },
    {
        "participant_id": "P004",
        "audio_path": str(AUDIO_DIR / "p004_test.wav"),
        "session_offset": 68.502,
    },
]


def run_real_four_audio_validation():
    # ------------------------------------------------------------------
    # Verify all 4 audio files exist
    # ------------------------------------------------------------------
    for p in PARTICIPANTS_CONFIG:
        assert Path(p["audio_path"]).exists(), f"Missing audio file: {p['audio_path']}"

    # ------------------------------------------------------------------
    # Mock only the orchestrator (no Qwen/LLM calls)
    # ------------------------------------------------------------------
    mock_orchestrator = MagicMock()
    captured_inputs = []

    def mock_evaluate_session(agent_inputs):
        captured_inputs.extend(agent_inputs)
        reports = []
        for inp in agent_inputs:
            reports.append(
                {
                    "session_id": inp.session_id,
                    "participant_id": inp.participant_id,
                    "topic": inp.topic,
                    "agent_results": [
                        {
                            "participant_id": inp.participant_id,
                            "agent": "relevance",
                            "score": 85.0,
                            "reasoning": "Participant stayed relevant to Indian food culture.",
                            "strengths": ["Clear focus on regional traditions"],
                            "weaknesses": [],
                            "evidence": ["Regional cuisine discussion"],
                            "metadata": {},
                        },
                        {
                            "participant_id": inp.participant_id,
                            "agent": "coherence",
                            "score": 85.0,
                            "reasoning": "Logically structured arguments.",
                            "strengths": ["Smooth flow of points"],
                            "weaknesses": [],
                            "evidence": ["Connected ideas logically"],
                            "metadata": {},
                        },
                        {
                            "participant_id": inp.participant_id,
                            "agent": "fluency",
                            "score": 85.0,
                            "reasoning": "Clear and articulate English.",
                            "strengths": ["Natural vocabulary"],
                            "weaknesses": [],
                            "evidence": ["Consistent tempo"],
                            "metadata": {},
                        },
                        {
                            "participant_id": inp.participant_id,
                            "agent": "participation",
                            "score": 85.0,
                            "reasoning": "Active and balanced participation.",
                            "strengths": ["Timely responses"],
                            "weaknesses": [],
                            "evidence": ["Constructive discussion"],
                            "metadata": {},
                        },
                    ],
                    "scorecard": {
                        "participant_id": inp.participant_id,
                        "scores": {
                            "relevance": 85.0,
                            "coherence": 85.0,
                            "fluency": 85.0,
                            "participation": 85.0,
                        },
                        "weights": {
                            "relevance": 0.25,
                            "coherence": 0.25,
                            "fluency": 0.25,
                            "participation": 0.25,
                        },
                        "final_score": 85.0,
                    },
                }
            )
        return reports

    mock_orchestrator.evaluate_session.side_effect = mock_evaluate_session

    # Initialize GDEvaluationService with real VAD and real Whisper ASR
    service = GDEvaluationService(orchestrator=mock_orchestrator)

    start_time = time.perf_counter()
    result = service.evaluate_session(
        session_id=SESSION_ID,
        topic=TOPIC,
        participants=PARTICIPANTS_CONFIG,
    )
    elapsed_time = time.perf_counter() - start_time

    # ------------------------------------------------------------------
    # 1. All four participant IDs are preserved
    # ------------------------------------------------------------------
    expected_pids = ["P001", "P002", "P003", "P004"]
    result_pids = [p["participant_id"] for p in result["participants"]]
    assert result_pids == expected_pids

    # ------------------------------------------------------------------
    # 2. All four audio files are processed
    # ------------------------------------------------------------------
    assert len(result["participants"]) == 4

    # ------------------------------------------------------------------
    # 3. VAD produces participant speech segments
    # ------------------------------------------------------------------
    for p in result["participants"]:
        assert len(p["vad_segments"]) > 0, f"No VAD segments for {p['participant_id']}"
        assert p["speaking_time"] > 0.0, f"Zero speaking time for {p['participant_id']}"

    # ------------------------------------------------------------------
    # 4. Shared ASR produces participant transcripts
    # ------------------------------------------------------------------
    for p in result["participants"]:
        assert len(p["transcript_segments"]) > 0, f"No transcript segments for {p['participant_id']}"
        assert len(p["transcript"].strip()) > 0, f"Empty transcript for {p['participant_id']}"
        assert p["word_count"] > 0, f"Zero word count for {p['participant_id']}"

    # ------------------------------------------------------------------
    # 5. Session offsets are respected
    # ------------------------------------------------------------------
    assert result["participants"][0]["session_offset"] == 0.0
    assert result["participants"][1]["session_offset"] == 16.994
    assert result["participants"][2]["session_offset"] == 39.028
    assert result["participants"][3]["session_offset"] == 68.502

    # ------------------------------------------------------------------
    # 6. Group transcript is chronological
    # ------------------------------------------------------------------
    group_segments = result["group_segments"]
    assert len(group_segments) > 0

    for i in range(len(group_segments) - 1):
        curr_start = group_segments[i]["start"]
        next_start = group_segments[i + 1]["start"]
        assert curr_start <= next_start, (
            f"Group segments out of chronological order at index {i}: "
            f"{curr_start} > {next_start}"
        )

    # ------------------------------------------------------------------
    # 7. Objective features are generated for every participant
    # ------------------------------------------------------------------
    for pid in expected_pids:
        features = result["objective_features"][pid]
        assert features["participant_id"] == pid
        assert features["speaking_time"] > 0
        assert features["word_count"] > 0
        assert features["turn_count"] > 0
        assert features["words_per_minute"] > 0
        assert features["average_turn_duration"] > 0

    # ------------------------------------------------------------------
    # 8. Interaction features are generated for every participant
    # ------------------------------------------------------------------
    for pid in expected_pids:
        interaction = result["interaction_features"][pid]
        assert interaction["participant_id"] == pid
        assert "turn_count" in interaction
        assert "responses" in interaction
        assert "overlap_events" in interaction
        assert "other_speakers_before" in interaction
        assert "other_speakers_after" in interaction

    # ------------------------------------------------------------------
    # 9. GDParticipantInput is constructed correctly and reaches orchestrator
    # ------------------------------------------------------------------
    assert len(captured_inputs) == 4
    for inp in captured_inputs:
        assert inp.session_id == SESSION_ID
        assert inp.topic == TOPIC
        assert inp.participant_id in expected_pids
        assert len(inp.participant_transcript) > 0
        assert len(inp.group_transcript) > 0
        assert len(inp.other_participants) == 3

    # ------------------------------------------------------------------
    # 10. Orchestrator result structure is propagated into the final service result
    # ------------------------------------------------------------------
    for pid in expected_pids:
        agent_res = result["agent_results"][pid]
        assert len(agent_res) == 4
        scorecard = result["scorecards"][pid]
        assert scorecard["final_score"] == 85.0
        assert result["final_scores"][pid] == 85.0

    # ------------------------------------------------------------------
    # 11. Final output contains four participant records
    # ------------------------------------------------------------------
    assert len(result["participants"]) == 4
    assert result["session_id"] == SESSION_ID
    assert result["topic"] == TOPIC

    return {
        "elapsed_time": elapsed_time,
        "result": result,
    }


def test_real_four_audio_pipeline():
    run_real_four_audio_validation()


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING REAL 4-PARTICIPANT GD AUDIO PIPELINE VALIDATION")
    print("=" * 80)
    output = run_real_four_audio_validation()
    res = output["result"]
    elapsed = output["elapsed_time"]

    print(f"\nProcessing Time: {elapsed:.2f} seconds")
    print(f"Session ID: {res['session_id']}")
    print(f"Topic: {res['topic']}")
    print(f"Calculated Session Duration: {res['session_duration']:.2f}s")
    print(f"Total Group Segments: {len(res['group_segments'])}")
    print("\nChronological Group Transcript:")
    print("-" * 80)
    print(res["group_transcript"])
    print("-" * 80)

    print("\nParticipant Metrics Summary:")
    print(f"{'Participant':<12} | {'Speaking Time':<14} | {'Word Count':<11} | {'Turns':<6} | {'WPM':<8} | {'Final Score'}")
    print("-" * 75)
    for p in res["participants"]:
        pid = p["participant_id"]
        feat = p["features"]
        print(
            f"{pid:<12} | "
            f"{p['speaking_time']:<14.3f} | "
            f"{p['word_count']:<11} | "
            f"{p['turn_count']:<6} | "
            f"{feat['words_per_minute']:<8.1f} | "
            f"{p['final_score']:.1f}"
        )
    print("\n" + "=" * 80)
    print("ALL REAL AUDIO PIPELINE VALIDATION CRITERIA PASSED SUCCESSFULLY!")
    print("=" * 80)
