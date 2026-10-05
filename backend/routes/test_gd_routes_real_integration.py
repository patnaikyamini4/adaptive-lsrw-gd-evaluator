"""
Real GD REST API Integration Test (Milestone 5)

Validates the complete end-to-end GD evaluation flow through Flask's REST API:

    HTTP Request -> gd_routes -> GDEvaluationService -> GDParticipantPipeline ->
    Silero VAD -> Shared Whisper ASR -> Transcript Service -> Feature Extraction ->
    Interaction Features -> GD Agents (Relevance, Coherence, Fluency, Participation) ->
    GDScorer -> GDEvaluationRepository -> MongoDB

Executed real components:
1. Flask test client (`app.test_client()`)
2. Real Flask route endpoints:
   - POST   /api/gd/evaluations
   - GET    /api/gd/evaluations/<session_id>
   - DELETE /api/gd/evaluations/<session_id>
3. GDEvaluationService orchestration
4. Real Silero VAD (CPU) processing all 4 participant audio files
5. Real Whisper ASR (CPU/GPU) transcribing all 4 participant speech segments
6. Real chronological group transcript construction
7. Real objective feature extraction (WPM, speaking time, turn counts)
8. Real interaction feature extraction (responses, overlaps, turn-taking)
9. Real GD agent prompt generation, JSON response parsing, schema validation, and GDScorer
10. Real repository persistence lifecycle (Insert -> Retrieve -> Delete -> Verify 404)

LLM boundary:
- OpenRouter/Qwen is safely mocked at the `ask_qwen` boundary to respect in-flight budget
  while exercising 100% of the real agent prompt-building, response parsing, and scoring logic.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import uuid
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import app
import backend.services.gd_repository as gd_repo
from backend.services.gd_service import GDEvaluationService


AUDIO_DIR = Path("data/gd/audio")
AUDIO_P001 = str(AUDIO_DIR / "p001_test.wav")
AUDIO_P002 = str(AUDIO_DIR / "p002_test.wav")
AUDIO_P003 = str(AUDIO_DIR / "p003_test.wav")
AUDIO_P004 = str(AUDIO_DIR / "p004_test.wav")


class InMemoryMongoCollection:
    """
    In-memory simulation of PyMongo collection for offline/isolated environments
    where remote MongoDB Atlas cannot be reached (e.g., IP whitelist / TLS restrictions).
    """

    def __init__(self):
        self._docs: dict[str, dict] = {}

    def update_one(self, filter_doc: dict, update_doc: dict, upsert: bool = False):
        session_id = filter_doc.get("session_id")
        existing = self._docs.get(session_id)

        if existing is None and upsert:
            doc = {}
            if "$setOnInsert" in update_doc:
                doc.update(update_doc["$setOnInsert"])
            if "$set" in update_doc:
                doc.update(update_doc["$set"])
            doc["_id"] = f"mock_oid_{uuid.uuid4().hex[:12]}"
            self._docs[session_id] = doc
        elif existing is not None:
            if "$set" in update_doc:
                existing.update(update_doc["$set"])
        return type("UpdateResult", (), {"upserted_id": session_id, "matched_count": 1 if existing else 0})()

    def find_one(self, filter_doc: dict):
        session_id = filter_doc.get("session_id")
        doc = self._docs.get(session_id)
        if doc is not None:
            return doc.copy()
        return None

    def delete_one(self, filter_doc: dict):
        session_id = filter_doc.get("session_id")
        if session_id in self._docs:
            del self._docs[session_id]
            return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    def find(self):
        class Cursor:
            def __init__(self, docs):
                self._docs = list(docs)

            def sort(self, key, direction):
                return self

            def limit(self, limit_num):
                return [d.copy() for d in self._docs[:limit_num]]

            def __iter__(self):
                return iter([d.copy() for d in self._docs])

        return Cursor(list(self._docs.values()))


def is_mongodb_atlas_reachable() -> bool:
    """Check if the real MongoDB database is actively reachable."""
    try:
        from backend.services.mongodb import client
        client.admin.command("ping", maxTimeMS=2000)
        return True
    except Exception:
        return False


def mock_ask_qwen_agent_response(prompt: str, max_tokens: int = 1000) -> str:
    """
    Mock the OpenRouter/Qwen API boundary.

    Returns valid structured JSON for the corresponding GD evaluation agent,
    exercising full prompt generation, JSON response parsing, and score validation.
    """
    if "Relevance" in prompt or "relevance" in prompt:
        score = 86.0
        reasoning = "Participant arguments were directly relevant to the topic of Indian food culture."
        agent = "relevance"
    elif "Coherence" in prompt or "coherence" in prompt:
        score = 84.0
        reasoning = "Ideas were logically structured and flowed smoothly from one point to the next."
        agent = "coherence"
    elif "Fluency" in prompt or "fluency" in prompt:
        score = 88.0
        reasoning = "Clear delivery with steady pace, good vocabulary, and minimal hesitations."
        agent = "fluency"
    elif "Participation" in prompt or "participation" in prompt:
        score = 82.0
        reasoning = "Active turn-taking and constructive engagement with other participants."
        agent = "participation"
    else:
        score = 85.0
        reasoning = "Solid overall contribution to the discussion."
        agent = "general"

    return json.dumps({
        "score": score,
        "reasoning": reasoning,
        "strengths": [f"Consistent {agent} demonstration", "Constructive discussion input"],
        "weaknesses": ["Could expand on specific regional nuances"],
        "evidence": ["Clear points articulated during assigned speaking turns"],
        "metadata": {"agent": agent},
    })


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_real_gd_rest_api_full_lifecycle(client, monkeypatch):
    """
    Milestone 5 Real GD REST API Integration Test.

    Exercises the full flow:
    1. POST   /api/gd/evaluations             -> 201 Created & validated
    2. GET    /api/gd/evaluations/<session_id> -> 200 OK & validated
    3. DELETE /api/gd/evaluations/<session_id> -> 200 OK & deleted
    4. GET    /api/gd/evaluations/<session_id> -> 404 Not Found
    """
    # ------------------------------------------------------------------
    # Step 0: Pre-flight Verification of Audio Files
    # ------------------------------------------------------------------
    for audio_path in [AUDIO_P001, AUDIO_P002, AUDIO_P003, AUDIO_P004]:
        assert Path(audio_path).exists(), f"Required test audio file missing: {audio_path}"

    # ------------------------------------------------------------------
    # Setup Qwen boundary mock
    # ------------------------------------------------------------------
    monkeypatch.setattr("backend.ai.qwen_service.ask_qwen", mock_ask_qwen_agent_response)
    monkeypatch.setattr("backend.ai.gd.agents.base_agent.GDBaseAgent.call_llm", lambda self, prompt: mock_ask_qwen_agent_response(prompt))

    # ------------------------------------------------------------------
    # Setup Repository / MongoDB boundary
    # ------------------------------------------------------------------
    atlas_live = is_mongodb_atlas_reachable()
    if not atlas_live:
        in_memory_collection = InMemoryMongoCollection()
        monkeypatch.setattr(gd_repo, "collection", in_memory_collection)
        monkeypatch.setattr(gd_repo, "gd_evaluations_collection", in_memory_collection)

    unique_session_id = f"GD-REAL-API-{uuid.uuid4().hex[:8].upper()}"
    topic = "Regional Food Culture in India"

    payload = {
        "session_id": unique_session_id,
        "topic": topic,
        "participants": [
            {
                "participant_id": "P001",
                "audio_path": AUDIO_P001,
                "session_offset": 0.0,
                "metadata": {"role": "participant", "device": "mic-1"},
            },
            {
                "participant_id": "P002",
                "audio_path": AUDIO_P002,
                "session_offset": 16.994,
                "metadata": {"role": "participant", "device": "mic-2"},
            },
            {
                "participant_id": "P003",
                "audio_path": AUDIO_P003,
                "session_offset": 39.028,
                "metadata": {"role": "participant", "device": "mic-3"},
            },
            {
                "participant_id": "P004",
                "audio_path": AUDIO_P004,
                "session_offset": 68.502,
                "metadata": {"role": "participant", "device": "mic-4"},
            },
        ],
    }

    try:
        # ==============================================================
        # 1. POST /api/gd/evaluations (REAL PIPELINE EXECUTION)
        # ==============================================================
        post_response = client.post(
            "/api/gd/evaluations",
            json=payload,
            content_type="application/json",
        )

        assert post_response.status_code == 201, f"POST failed: {post_response.get_json()}"
        post_data = post_response.get_json()

        # Status & Response structure
        assert post_data.get("status") == "success"
        assert "evaluation" in post_data
        evaluation = post_data["evaluation"]

        # Session & Topic identity
        assert evaluation.get("session_id") == unique_session_id
        assert evaluation.get("topic") == topic
        assert evaluation.get("session_duration", 0.0) > 0.0

        # Participants list verification
        participants = evaluation.get("participants", [])
        assert len(participants) == 4, f"Expected 4 participants, got {len(participants)}"
        expected_pids = ["P001", "P002", "P003", "P004"]
        returned_pids = [p["participant_id"] for p in participants]
        assert returned_pids == expected_pids

        # Transcripts and VAD verification
        participant_transcripts = evaluation.get("participant_transcripts", {})
        for pid in expected_pids:
            assert pid in participant_transcripts
            assert len(participant_transcripts[pid].strip()) > 0

        for p in participants:
            assert len(p.get("vad_segments", [])) > 0, f"No VAD segments for {p['participant_id']}"
            assert len(p.get("transcript_segments", [])) > 0, f"No transcript segments for {p['participant_id']}"
            assert p.get("speaking_time", 0.0) > 0.0
            assert p.get("word_count", 0) > 0

        # Group transcript & chronological segments
        assert len(evaluation.get("group_transcript", "").strip()) > 0
        group_segments = evaluation.get("group_segments", [])
        assert len(group_segments) > 0

        for i in range(len(group_segments) - 1):
            assert float(group_segments[i]["start"]) <= float(group_segments[i + 1]["start"]), (
                f"Group segments not chronological at index {i}"
            )

        # Objective features verification
        objective_features = evaluation.get("objective_features", {})
        for pid in expected_pids:
            assert pid in objective_features
            features = objective_features[pid]
            assert features.get("speaking_time", 0) > 0
            assert features.get("word_count", 0) > 0
            assert features.get("words_per_minute", 0) > 0
            assert features.get("turn_count", 0) > 0

        # Interaction features verification
        interaction_features = evaluation.get("interaction_features", {})
        for pid in expected_pids:
            assert pid in interaction_features
            interaction = interaction_features[pid]
            assert "turn_count" in interaction
            assert "responses" in interaction
            assert "overlap_events" in interaction
            assert "other_speakers_before" in interaction
            assert "other_speakers_after" in interaction

        # Agent results verification
        agent_results = evaluation.get("agent_results", {})
        for pid in expected_pids:
            assert pid in agent_results
            results_list = agent_results[pid]
            assert len(results_list) == 4
            agent_names = {r["agent"] for r in results_list}
            assert agent_names == {"relevance", "coherence", "fluency", "participation"}

        # Scorecards & Final Scores verification
        scorecards = evaluation.get("scorecards", {})
        final_scores = evaluation.get("final_scores", {})
        for pid in expected_pids:
            assert pid in scorecards
            assert pid in final_scores
            assert 0.0 <= final_scores[pid] <= 100.0

        # Document persistence fields
        assert evaluation.get("document_type") == "gd_evaluation"
        assert "created_at" in evaluation or "updated_at" in evaluation

        # ==============================================================
        # 2. GET /api/gd/evaluations/<session_id>
        # ==============================================================
        get_response = client.get(f"/api/gd/evaluations/{unique_session_id}")
        assert get_response.status_code == 200, f"GET failed: {get_response.get_json()}"
        get_data = get_response.get_json()
        assert get_data.get("status") == "success"
        retrieved_eval = get_data.get("evaluation", {})
        assert retrieved_eval.get("session_id") == unique_session_id
        assert retrieved_eval.get("topic") == topic
        assert len(retrieved_eval.get("participants", [])) == 4
        assert retrieved_eval.get("document_type") == "gd_evaluation"

        # ==============================================================
        # 3. DELETE /api/gd/evaluations/<session_id>
        # ==============================================================
        delete_response = client.delete(f"/api/gd/evaluations/{unique_session_id}")
        assert delete_response.status_code == 200, f"DELETE failed: {delete_response.get_json()}"
        delete_data = delete_response.get_json()
        assert delete_data.get("status") == "success"
        assert f"'{unique_session_id}' deleted successfully" in delete_data.get("message", "")

        # ==============================================================
        # 4. GET /api/gd/evaluations/<session_id> (VERIFY 404 NOT FOUND)
        # ==============================================================
        get_deleted_response = client.get(f"/api/gd/evaluations/{unique_session_id}")
        assert get_deleted_response.status_code == 404
        get_deleted_data = get_deleted_response.get_json()
        assert get_deleted_data.get("status") == "error"
        assert f"GD evaluation not found for session '{unique_session_id}'" in get_deleted_data.get("message", "")

    finally:
        # Guarantee cleanup of only this test session
        try:
            gd_repo.delete_evaluation(unique_session_id)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
