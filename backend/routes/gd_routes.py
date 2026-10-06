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

from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request

from backend.services.gd_live_audio_service import GDLiveAudioService
from backend.services.gd_live_service import GDLiveService
from backend.services.gd_repository import (
    delete_evaluation,
    get_all_evaluations,
    get_evaluation,
    save_evaluation,
)
from backend.services.gd_service import GDEvaluationService
from backend.services.gd_session_service import GDSessionService


gd_bp = Blueprint("gd", __name__, url_prefix="/api/gd")
gd_service = GDEvaluationService()
gd_session_service = GDSessionService()
gd_live_service = GDLiveService(session_service=gd_session_service)
gd_live_audio_service = GDLiveAudioService(
    session_service=gd_session_service,
    live_service=gd_live_service,
)


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


# ==============================================================================
# GD SESSION MANAGEMENT ROUTES (/api/gd/sessions)
# ==============================================================================

@gd_bp.route("/sessions", methods=["POST"])
def create_gd_session():
    """
    Create a new scheduled or on-demand Group Discussion session.
    """
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    topic = data.get("topic")
    coordinator_id = data.get("coordinator_id")
    participant_ids = data.get("participant_ids")
    session_id = data.get("session_id")
    scheduled_start = data.get("scheduled_start")
    scheduled_end = data.get("scheduled_end")
    session_duration = data.get("session_duration")
    metadata = data.get("metadata")

    if not topic or not isinstance(topic, str) or not topic.strip():
        return jsonify({
            "status": "error",
            "message": "'topic' is required and must be a non-empty string",
        }), 400

    if not coordinator_id or not isinstance(coordinator_id, str) or not coordinator_id.strip():
        return jsonify({
            "status": "error",
            "message": "'coordinator_id' is required and must be a non-empty string",
        }), 400

    if not isinstance(participant_ids, list) or len(participant_ids) == 0:
        return jsonify({
            "status": "error",
            "message": "'participant_ids' is required and must be a non-empty list",
        }), 400

    start_dt = None
    if scheduled_start:
        if isinstance(scheduled_start, str):
            try:
                start_dt = datetime.fromisoformat(scheduled_start)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": "'scheduled_start' must be a valid ISO datetime string",
                }), 400

    end_dt = None
    if scheduled_end:
        if isinstance(scheduled_end, str):
            try:
                end_dt = datetime.fromisoformat(scheduled_end)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": "'scheduled_end' must be a valid ISO datetime string",
                }), 400

    try:
        session = gd_session_service.create_session(
            topic=topic.strip(),
            coordinator_id=coordinator_id.strip(),
            participant_ids=participant_ids,
            session_id=session_id.strip() if isinstance(session_id, str) and session_id.strip() else None,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            session_duration=float(session_duration) if session_duration is not None else None,
            metadata=metadata if isinstance(metadata, dict) else {},
        )
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to create GD session: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": "GD session created successfully",
        "session": session,
    }), 201


@gd_bp.route("/sessions/<session_id>", methods=["GET"])
def get_gd_session(session_id: str):
    """
    Retrieve a GD session by ID.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        session = gd_session_service.get_session(session_id.strip())
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve GD session: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "session": session,
    }), 200


@gd_bp.route("/sessions/<session_id>/start", methods=["POST"])
def start_gd_session(session_id: str):
    """
    Transition a GD session to ACTIVE state and initialize live runtime.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        result = gd_live_service.start_session(session_id.strip())
        session = result.get("session") if isinstance(result, dict) and "session" in result else result
        runtime = result.get("runtime") if isinstance(result, dict) else None
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to start GD session: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"GD session '{session_id.strip()}' is now ACTIVE",
        "session": session,
        "runtime": runtime,
    }), 200


@gd_bp.route("/sessions/<session_id>/end", methods=["POST"])
def end_gd_session(session_id: str):
    """
    Transition a GD session to ENDED state and stop live runtime.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        result = gd_live_service.end_session(session_id.strip())
        session = result.get("session") if isinstance(result, dict) and "session" in result else result
        runtime = result.get("runtime") if isinstance(result, dict) else None
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to end GD session: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"GD session '{session_id.strip()}' has ENDED",
        "session": session,
        "runtime": runtime,
    }), 200


@gd_bp.route("/sessions/<session_id>/cancel", methods=["POST"])
def cancel_gd_session(session_id: str):
    """
    Transition a GD session to CANCELLED state.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    data = request.get_json(silent=True) or {}
    reason = data.get("reason") if isinstance(data, dict) else None

    try:
        session = gd_session_service.cancel_session(session_id.strip(), reason=reason)
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to cancel GD session: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"GD session '{session_id.strip()}' has been CANCELLED",
        "session": session,
    }), 200


@gd_bp.route("/sessions/<session_id>/participants", methods=["POST"])
def add_gd_participant(session_id: str):
    """
    Add a participant to a SCHEDULED GD session.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    participant_id = data.get("participant_id")
    if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' is required and must be a non-empty string",
        }), 400

    try:
        session = gd_session_service.add_participant(session_id.strip(), participant_id.strip())
    except ValueError as e:
        if "gd session" in str(e).lower() and "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to add participant: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Participant '{participant_id.strip()}' added successfully",
        "session": session,
    }), 200


@gd_bp.route("/sessions/<session_id>/participants/<participant_id>", methods=["DELETE"])
def remove_gd_participant(session_id: str, participant_id: str):
    """
    Remove a participant from a SCHEDULED GD session.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    if not participant_id or not participant_id.strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' parameter is required",
        }), 400

    try:
        session = gd_session_service.remove_participant(session_id.strip(), participant_id.strip())
    except ValueError as e:
        if "gd session" in str(e).lower() and "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to remove participant: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Participant '{participant_id.strip()}' removed successfully",
        "session": session,
    }), 200


@gd_bp.route("/sessions", methods=["GET"])
def list_gd_sessions():
    """
    List GD sessions with optional status and coordinator_id filters.
    """
    limit_raw = request.args.get("limit")
    status = request.args.get("status")
    coordinator_id = request.args.get("coordinator_id")
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
        sessions = gd_session_service.list_sessions(
            limit=limit,
            status=status.strip() if status else None,
            coordinator_id=coordinator_id.strip() if coordinator_id else None,
        )
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to list GD sessions: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "sessions": sessions,
        "count": len(sessions),
    }), 200


@gd_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_gd_session(session_id: str):
    """
    Delete a GD session by ID.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        deleted = gd_session_service.delete_session(session_id.strip())
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to delete GD session: {str(e)}",
        }), 500

    if not deleted:
        return jsonify({
            "status": "error",
            "message": f"GD session not found for session '{session_id.strip()}'",
        }), 404

    return jsonify({
        "status": "success",
        "message": f"GD session for session '{session_id.strip()}' deleted successfully",
    }), 200


@gd_bp.route("/sessions/<session_id>/join", methods=["POST"])
def join_gd_participant(session_id: str):
    """
    Connect an authorized participant to an ACTIVE live GD session.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    participant_id = data.get("participant_id")
    if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' is required and must be a non-empty string",
        }), 400

    try:
        result = gd_live_service.join_participant(session_id.strip(), participant_id.strip())
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to join participant: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Participant '{participant_id.strip()}' joined live session",
        "session_id": session_id.strip(),
        "participant_id": participant_id.strip(),
        "connected": result.get("connected", True),
        "runtime": result.get("runtime"),
    }), 200


@gd_bp.route("/sessions/<session_id>/leave", methods=["POST"])
def leave_gd_participant(session_id: str):
    """
    Disconnect a participant from an ACTIVE live GD session.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    participant_id = data.get("participant_id")
    if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' is required and must be a non-empty string",
        }), 400

    try:
        result = gd_live_service.leave_participant(session_id.strip(), participant_id.strip())
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to leave participant: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Participant '{participant_id.strip()}' left live session",
        "session_id": session_id.strip(),
        "participant_id": participant_id.strip(),
        "connected": result.get("connected", False),
        "runtime": result.get("runtime"),
    }), 200


@gd_bp.route("/sessions/<session_id>/runtime", methods=["GET"])
def get_gd_session_runtime(session_id: str):
    """
    Retrieve live runtime status for a GD session.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    try:
        runtime_status = gd_live_service.get_runtime_status(session_id.strip())
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve runtime status: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "runtime": runtime_status,
    }), 200


@gd_bp.route("/sessions/<session_id>/audio", methods=["POST"])
def process_gd_audio_event(session_id: str):
    """
    Process one participant-scoped live audio event for an ACTIVE GD session.
    Delegates processing directly to GDLiveAudioService.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    participant_id = data.get("participant_id")
    if not participant_id or not isinstance(participant_id, str) or not participant_id.strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' is required and must be a non-empty string",
        }), 400

    audio_path = data.get("audio_path")
    if not audio_path or not isinstance(audio_path, str) or not audio_path.strip():
        return jsonify({
            "status": "error",
            "message": "'audio_path' is required and must be a non-empty string",
        }), 400

    try:
        result = gd_live_audio_service.process_audio_event(
            session_id=session_id.strip(),
            participant_id=participant_id.strip(),
            audio_path=audio_path.strip(),
        )
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 404
    except ValueError as e:
        if "not found" in str(e).lower():
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except RuntimeError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to process live audio event: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Audio event processed for participant '{participant_id.strip()}'",
        "result": result,
        **result,
    }), 200
