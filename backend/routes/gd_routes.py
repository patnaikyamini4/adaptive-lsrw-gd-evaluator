"""
GD REST API Routes

Provides HTTP endpoints for Group Discussion evaluation orchestration and retrieval:
- POST   /api/gd/evaluations             -> Trigger offline GD session evaluation & persistence
- GET    /api/gd/evaluations/<session_id> -> Retrieve saved GD evaluation by session ID
- DELETE /api/gd/evaluations/<session_id> -> Delete saved GD evaluation by session ID
- GET    /api/gd/evaluations             -> List recent saved GD evaluations

Architecture:
HTTP Request -> Route Validation -> GDEvaluationService -> GDEvaluationRepository -> MongoDB
"""

from typing import Any
from flask import Blueprint, jsonify, request

from backend.services.gd_repository import (
    delete_evaluation,
    get_all_evaluations,
    get_evaluation,
    save_evaluation,
)
from backend.services.gd_service import GDEvaluationService


gd_bp = Blueprint("gd", __name__, url_prefix="/api/gd")
gd_service = GDEvaluationService()


@gd_bp.route("/evaluations", methods=["POST"])
def evaluate_gd_session():
    """
    Trigger end-to-end evaluation for a Group Discussion session and persist results.

    Expected JSON body:
    {
        "session_id": "GD-101",
        "topic": "Discussion Topic",
        "participants": [
            {
                "participant_id": "P1",
                "audio_path": "/path/to/p1.wav",
                "session_offset": 0.0,
                "metadata": {}
            },
            ...
        ],
        "session_duration": 180.0 (optional)
    }
    """
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    session_id = data.get("session_id")
    topic = data.get("topic")
    participants = data.get("participants")
    session_duration = data.get("session_duration")

    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' is required and must be a non-empty string",
        }), 400

    if not topic or not isinstance(topic, str) or not topic.strip():
        return jsonify({
            "status": "error",
            "message": "'topic' is required and must be a non-empty string",
        }), 400

    if not isinstance(participants, list) or len(participants) == 0:
        return jsonify({
            "status": "error",
            "message": "'participants' is required and must be a non-empty list",
        }), 400

    if session_duration is not None:
        if isinstance(session_duration, bool) or not isinstance(session_duration, (int, float)):
            return jsonify({
                "status": "error",
                "message": "'session_duration' must be a numeric value",
            }), 400
        if session_duration < 0:
            return jsonify({
                "status": "error",
                "message": "'session_duration' cannot be negative",
            }), 400

    # ------------------------------------------------------------------
    # Execute Evaluation Pipeline via GDEvaluationService
    # ------------------------------------------------------------------
    try:
        evaluation_result = gd_service.evaluate_session(
            session_id=session_id.strip(),
            topic=topic.strip(),
            participants=participants,
            session_duration=float(session_duration) if session_duration is not None else None,
        )
    except (ValueError, FileNotFoundError) as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation pipeline failed: {str(e)}",
        }), 500

    # ------------------------------------------------------------------
    # Persist Result via GDEvaluationRepository
    # ------------------------------------------------------------------
    try:
        persisted_doc = save_evaluation(evaluation_result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to persist GD evaluation: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": "GD evaluation completed and persisted successfully",
        "evaluation": persisted_doc,
    }), 201


@gd_bp.route("/evaluations/<session_id>", methods=["GET"])
def get_gd_evaluation(session_id: str):
    """
    Retrieve a persisted GD evaluation by session ID.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        evaluation = get_evaluation(session_id.strip())
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve GD evaluation: {str(e)}",
        }), 500

    if evaluation is None:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation not found for session '{session_id.strip()}'",
        }), 404

    return jsonify({
        "status": "success",
        "evaluation": evaluation,
    }), 200


@gd_bp.route("/evaluations/<session_id>", methods=["DELETE"])
def delete_gd_evaluation(session_id: str):
    """
    Delete a persisted GD evaluation by session ID.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        deleted = delete_evaluation(session_id.strip())
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to delete GD evaluation: {str(e)}",
        }), 500

    if not deleted:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation not found for session '{session_id.strip()}'",
        }), 404

    return jsonify({
        "status": "success",
        "message": f"GD evaluation for session '{session_id.strip()}' deleted successfully",
    }), 200


@gd_bp.route("/evaluations", methods=["GET"])
def list_gd_evaluations():
    """
    List recent persisted GD evaluations.
    """
    limit_raw = request.args.get("limit")
    limit = 100
    if limit_raw is not None:
        try:
            limit = int(limit_raw)
            if limit <= 0:
                return jsonify({
                    "status": "error",
                    "message": "'limit' query parameter must be a positive integer",
                }), 400
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "'limit' query parameter must be an integer",
            }), 400

    try:
        evaluations = get_all_evaluations(limit=limit)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to list GD evaluations: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "evaluations": evaluations,
        "count": len(evaluations),
    }), 200
