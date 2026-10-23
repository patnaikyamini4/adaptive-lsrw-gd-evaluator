"""
GD REST API Routes

Provides HTTP endpoints for Group Discussion evaluation orchestration, session lifecycle,
live audio ingestion, and LiveKit room token issuance, strictly guarded by authentication
and role-based authorization:

- POST   /api/gd/evaluations                      -> (COORDINATOR) Trigger offline GD evaluation
- GET    /api/gd/evaluations/<session_id>          -> (AUTH) Retrieve GD evaluation (sanitized for candidates)
- DELETE /api/gd/evaluations/<session_id>          -> (COORDINATOR) Delete GD evaluation
- GET    /api/gd/evaluations                      -> (COORDINATOR) List GD evaluations
- POST   /api/gd/sessions                         -> (COORDINATOR) Create new GD session
- GET    /api/gd/sessions/<session_id>             -> (MEMBER) Get session metadata
- POST   /api/gd/sessions/<session_id>/start       -> (COORDINATOR) Transition SCHEDULED -> ACTIVE
- POST   /api/gd/sessions/<session_id>/end         -> (COORDINATOR) Transition ACTIVE -> ENDED
- POST   /api/gd/sessions/<session_id>/cancel      -> (COORDINATOR) Transition session to CANCELLED
- POST   /api/gd/sessions/<session_id>/participants -> (COORDINATOR) Add participant
- DELETE /api/gd/sessions/<session_id>/participants/<participant_id> -> (COORDINATOR) Remove participant
- GET    /api/gd/sessions                         -> (AUTH) List sessions (role-filtered)
- DELETE /api/gd/sessions/<session_id>             -> (COORDINATOR) Delete GD session
- POST   /api/gd/sessions/<session_id>/join        -> (MEMBER) Join live session (anti-spoofing)
- POST   /api/gd/sessions/<session_id>/leave       -> (MEMBER) Leave live session (anti-spoofing)
- GET    /api/gd/sessions/<session_id>/runtime     -> (MEMBER) Get live runtime metrics
- POST   /api/gd/sessions/<session_id>/audio       -> (COORDINATOR) Feed audio file path
- GET    /api/gd/sessions/<session_id>/transcript  -> (MEMBER) Retrieve accumulated live transcript
- POST   /api/gd/sessions/<session_id>/audio/upload -> (MEMBER) Upload 4-second WAV chunk (anti-spoofing)
- POST   /api/gd/sessions/<session_id>/evaluate    -> (COORDINATOR) Post-session evaluation
- POST   /api/gd/sessions/<session_id>/livekit-token -> (MEMBER) Issue scoped LiveKit meeting room token
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import uuid

from flask import Blueprint, g, jsonify, request
import jwt
from werkzeug.utils import secure_filename

from backend.config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL
from backend.middleware.auth_middleware import (
    require_auth,
    require_role,
    require_session_coordinator,
    require_session_member,
)
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
@require_role("COORDINATOR")
def evaluate_gd_session():
    """
    Trigger end-to-end evaluation for a Group Discussion session and persist results.
    Only authorized COORDINATORS can trigger offline evaluation orchestration.
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

    seen_ids: set[str] = set()
    cleaned_participants: list[dict[str, Any]] = []

    for idx, p in enumerate(participants):
        if not isinstance(p, dict):
            return jsonify({
                "status": "error",
                "message": f"Participant at index {idx} must be an object",
            }), 400

        pid = p.get("participant_id")
        audio_path = p.get("audio_path")
        offset = p.get("session_offset", 0.0)
        metadata = p.get("metadata", {})

        if not pid or not isinstance(pid, str) or not pid.strip():
            return jsonify({
                "status": "error",
                "message": f"Participant at index {idx} missing valid 'participant_id'",
            }), 400

        clean_pid = pid.strip()
        if clean_pid in seen_ids:
            return jsonify({
                "status": "error",
                "message": f"Duplicate participant_id '{clean_pid}' at index {idx}",
            }), 400
        seen_ids.add(clean_pid)

        if not audio_path or not isinstance(audio_path, str) or not audio_path.strip():
            return jsonify({
                "status": "error",
                "message": f"Participant '{clean_pid}' missing valid 'audio_path'",
            }), 400

        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            return jsonify({
                "status": "error",
                "message": f"Participant '{clean_pid}' 'session_offset' must be numeric",
            }), 400

        if offset < 0:
            return jsonify({
                "status": "error",
                "message": f"Participant '{clean_pid}' 'session_offset' cannot be negative",
            }), 400

        cleaned_participants.append({
            "participant_id": clean_pid,
            "audio_path": audio_path.strip(),
            "session_offset": float(offset),
            "metadata": metadata if isinstance(metadata, dict) else {},
        })

    try:
        evaluation_result = gd_service.evaluate_session(
            session_id=session_id.strip(),
            topic=topic.strip(),
            participants=cleaned_participants,
            session_duration=float(session_duration) if session_duration is not None else None,
        )
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": f"Audio file not found: {str(e)}",
        }), 400
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid evaluation request: {str(e)}",
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation pipeline failed: {str(e)}",
        }), 500

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
@require_auth
def get_gd_evaluation(session_id: str):
    """
    Retrieve a persisted GD evaluation by session ID.
    - COORDINATOR: Receives the complete evaluation report across all participants and metrics.
    - CANDIDATE: Receives an explicitly sanitized evaluation containing only their individual
      scorecard and general group dynamics, preserving peer candidate confidentiality.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    clean_session_id = session_id.strip()
    try:
        evaluation = get_evaluation(clean_session_id)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve GD evaluation: {str(e)}",
        }), 500

    if evaluation is None:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation not found for session '{clean_session_id}'",
        }), 404

    user = getattr(g, "current_user", {})
    user_role = user.get("role", "CANDIDATE")
    user_id = user.get("user_id", "")

    # Restrict candidate access to their own scorecard and sanitized group summary
    if user_role == "CANDIDATE":
        participants = evaluation.get("participants", [])
        participated_ids = [
            p.get("participant_id")
            for p in participants
            if isinstance(p, dict)
        ]

        if user_id not in participated_ids:
            return jsonify({
                "status": "error",
                "message": f"Access forbidden: user '{user_id}' was not a participant in this evaluation",
            }), 403

        sanitized_participants = [
            p for p in participants
            if isinstance(p, dict) and p.get("participant_id") == user_id
        ]
        sanitized_scorecards = {
            user_id: evaluation.get("scorecards", {}).get(user_id)
        } if isinstance(evaluation.get("scorecards"), dict) and user_id in evaluation.get("scorecards", {}) else {}

        sanitized_final_scores = {
            user_id: evaluation.get("final_scores", {}).get(user_id)
        } if isinstance(evaluation.get("final_scores"), dict) and user_id in evaluation.get("final_scores", {}) else {}

        sanitized_transcripts = {
            user_id: evaluation.get("participant_transcripts", {}).get(user_id)
        } if isinstance(evaluation.get("participant_transcripts"), dict) and user_id in evaluation.get("participant_transcripts", {}) else {}

        evaluation = {
            "session_id": evaluation.get("session_id"),
            "topic": evaluation.get("topic"),
            "session_duration": evaluation.get("session_duration"),
            "document_type": evaluation.get("document_type"),
            "group_transcript": evaluation.get("group_transcript"),
            "group_segments": evaluation.get("group_segments", []),
            "participants": sanitized_participants,
            "participant_transcripts": sanitized_transcripts,
            "scorecards": sanitized_scorecards,
            "final_scores": sanitized_final_scores,
            "created_at": evaluation.get("created_at"),
            "updated_at": evaluation.get("updated_at"),
            "sanitized": True,
        }

    return jsonify({
        "status": "success",
        "evaluation": evaluation,
    }), 200


@gd_bp.route("/evaluations/<session_id>", methods=["DELETE"])
@require_role("COORDINATOR")
def delete_gd_evaluation(session_id: str):
    """
    Delete a persisted GD evaluation by session ID.
    Only COORDINATORS can delete evaluations.
    """
    if not session_id or not session_id.strip():
        return jsonify({
            "status": "error",
            "message": "'session_id' parameter is required",
        }), 400

    clean_session_id = session_id.strip()
    try:
        deleted = delete_evaluation(clean_session_id)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to delete GD evaluation: {str(e)}",
        }), 500

    if not deleted:
        return jsonify({
            "status": "error",
            "message": f"GD evaluation not found for session '{clean_session_id}'",
        }), 404

    return jsonify({
        "status": "success",
        "message": f"GD evaluation for session '{clean_session_id}' deleted successfully",
    }), 200


@gd_bp.route("/evaluations", methods=["GET"])
@require_role("COORDINATOR")
def list_gd_evaluations():
    """
    List recent persisted GD evaluations.
    Only COORDINATORS can list all stored evaluations.
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
@require_role("COORDINATOR")
def create_gd_session():
    """
    Create a new scheduled or on-demand Group Discussion session.
    Derives coordinator_id strictly from verified authentication context.
    """
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    topic = data.get("topic")
    participant_ids = data.get("participant_ids")
    session_id = data.get("session_id")
    scheduled_start = data.get("scheduled_start")
    scheduled_end = data.get("scheduled_end")
    session_duration = data.get("session_duration")
    metadata = data.get("metadata")

    # Coordinator identity derived strictly from verified JWT
    coordinator_id = g.current_user.get("user_id")

    coordinator_id_payload = data.get("coordinator_id")
    if coordinator_id_payload is not None:
        if not isinstance(coordinator_id_payload, str) or not coordinator_id_payload.strip():
            return jsonify({
                "status": "error",
                "message": "'coordinator_id' is required and must be a non-empty string",
            }), 400

    if not topic or not isinstance(topic, str) or not topic.strip():
        return jsonify({
            "status": "error",
            "message": "'topic' is required and must be a non-empty string",
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
            coordinator_id=coordinator_id,
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
@require_session_member
def get_gd_session(session_id: str):
    """
    Retrieve a GD session by ID.
    Enforces that caller is either the coordinator or an invited participant.
    """
    session = getattr(g, "current_session", None)
    if not session:
        try:
            session = gd_session_service.get_session(session_id.strip())
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404

    return jsonify({
        "status": "success",
        "session": session,
    }), 200


@gd_bp.route("/sessions/<session_id>/start", methods=["POST"])
@require_session_coordinator
def start_gd_session(session_id: str):
    """
    Transition a GD session to ACTIVE state and initialize live runtime.
    Only the designated session coordinator can start the session.
    """
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
@require_session_coordinator
def end_gd_session(session_id: str):
    """
    Transition a GD session to ENDED state and stop live runtime.
    Only the designated session coordinator can end the session.
    """
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
@require_session_coordinator
def cancel_gd_session(session_id: str):
    """
    Transition a GD session to CANCELLED state.
    Only the designated session coordinator can cancel the session.
    """
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
@require_session_coordinator
def add_gd_participant(session_id: str):
    """
    Add a participant to a SCHEDULED GD session.
    Only the designated session coordinator can add participants.
    """
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
@require_session_coordinator
def remove_gd_participant(session_id: str, participant_id: str):
    """
    Remove a participant from a SCHEDULED GD session.
    Only the designated session coordinator can remove participants.
    """
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
@require_auth
def list_gd_sessions():
    """
    List GD sessions filtered by role:
    - COORDINATOR: Sees all sessions they coordinate.
    - CANDIDATE: Sees all sessions to which they are assigned.
    """
    limit_raw = request.args.get("limit")
    status = request.args.get("status")
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
        user = g.current_user
        user_role = user.get("role", "CANDIDATE")
        user_id = user.get("user_id")

        if user_role == "COORDINATOR":
            sessions = gd_session_service.list_sessions(
                limit=limit,
                status=status.strip() if status else None,
                coordinator_id=user_id,
            )
        else:
            all_sessions = gd_session_service.list_sessions(
                limit=limit,
                status=status.strip() if status else None,
            )
            sessions = [s for s in all_sessions if user_id in s.get("participant_ids", [])]

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
@require_session_coordinator
def delete_gd_session(session_id: str):
    """
    Delete a GD session by ID.
    Only the designated session coordinator can delete the session.
    """
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
@require_session_member
def join_gd_participant(session_id: str):
    """
    Connect an authorized participant to an ACTIVE live GD session.
    Enforces anti-spoofing: claimed participant_id must match authenticated caller.
    """
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

    clean_pid = participant_id.strip()
    caller_id = g.current_user.get("user_id")

    # Anti-spoofing invariant
    if clean_pid != caller_id:
        return jsonify({
            "status": "error",
            "message": f"Participant impersonation detected: authenticated user '{caller_id}' cannot join as '{clean_pid}'",
        }), 403

    try:
        result = gd_live_service.join_participant(session_id.strip(), clean_pid)
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
        "message": f"Participant '{clean_pid}' joined live session",
        "session_id": session_id.strip(),
        "participant_id": clean_pid,
        "connected": result.get("connected", True),
        "runtime": result.get("runtime"),
    }), 200


@gd_bp.route("/sessions/<session_id>/leave", methods=["POST"])
@require_session_member
def leave_gd_participant(session_id: str):
    """
    Disconnect a participant from an ACTIVE live GD session.
    Enforces anti-spoofing: claimed participant_id must match authenticated caller.
    """
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

    clean_pid = participant_id.strip()
    caller_id = g.current_user.get("user_id")

    # Anti-spoofing invariant
    if clean_pid != caller_id:
        return jsonify({
            "status": "error",
            "message": f"Participant impersonation detected: authenticated user '{caller_id}' cannot leave as '{clean_pid}'",
        }), 403

    try:
        result = gd_live_service.leave_participant(session_id.strip(), clean_pid)
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
        "message": f"Participant '{clean_pid}' left live session",
        "session_id": session_id.strip(),
        "participant_id": clean_pid,
        "connected": result.get("connected", False),
        "runtime": result.get("runtime"),
    }), 200


@gd_bp.route("/sessions/<session_id>/runtime", methods=["GET"])
@require_session_member
def get_gd_session_runtime(session_id: str):
    """
    Retrieve live runtime status for a GD session.
    Only session coordinators and enrolled participants can view runtime telemetry.
    """
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
@require_role("COORDINATOR")
def process_gd_audio_event(session_id: str):
    """
    Process one participant-scoped live audio event for an ACTIVE GD session from a file path.
    Restricted to session coordinators for diagnostic or automated test feeds.
    """
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


@gd_bp.route("/sessions/<session_id>/transcript", methods=["GET"])
@require_session_member
def get_gd_session_transcript(session_id: str):
    """
    Retrieve live chronological transcript segments and speaker breakdown for a GD session.
    """
    try:
        transcript_data = gd_live_service.get_transcript(session_id.strip())
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
            "message": f"Failed to retrieve live transcript: {str(e)}",
        }), 500

    user = getattr(g, "current_user", {})
    user_role = user.get("role", "CANDIDATE")
    user_id = user.get("user_id")

    # Candidate privacy protection:
    # Coordinators retain authorized full view.
    # Candidates receive only their own transcript in participant_transcripts (or empty dict if absent).
    if user_role == "CANDIDATE":
        raw_pts = transcript_data.get("participant_transcripts", {})
        sanitized_pts = {
            user_id: raw_pts[user_id]
        } if (isinstance(raw_pts, dict) and user_id in raw_pts) else {}

        transcript_data = {
            **transcript_data,
            "participant_transcripts": sanitized_pts,
            "sanitized": True,
        }

    return jsonify({
        "status": "success",
        "session_id": session_id.strip(),
        **transcript_data,
    }), 200


@gd_bp.route("/sessions/<session_id>/audio/upload", methods=["POST"])
@require_session_member
def upload_gd_audio_file(session_id: str):
    """
    Upload and process a live audio chunk (multipart/form-data) for an ACTIVE GD session.
    Enforces anti-spoofing: claimed participant_id must strictly match the authenticated caller.
    """
    participant_id = request.form.get("participant_id") or request.args.get("participant_id")
    if not participant_id or not str(participant_id).strip():
        return jsonify({
            "status": "error",
            "message": "'participant_id' is required and must be a non-empty string",
        }), 400

    clean_pid = str(participant_id).strip()
    caller_id = g.current_user.get("user_id")

    # Anti-spoofing invariant
    if clean_pid != caller_id:
        return jsonify({
            "status": "error",
            "message": f"Participant impersonation detected: authenticated user '{caller_id}' cannot upload audio for '{clean_pid}'",
        }), 403

    audio_file = (
        request.files.get("file")
        or request.files.get("audio")
        or request.files.get("audio_file")
    )
    if audio_file is None or audio_file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Audio file is required in multipart/form-data ('file', 'audio', or 'audio_file')",
        }), 400

    clean_session_id = session_id.strip()
    upload_dir = Path("uploads/live_sessions") / clean_session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(audio_file.filename) or f"{uuid.uuid4().hex[:8]}.wav"
    staging_filename = f"{clean_pid}_{int(datetime.now().timestamp()*1000)}_{filename}"
    target_path = upload_dir / staging_filename

    try:
        audio_file.save(str(target_path))
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to save uploaded audio file: {str(e)}",
        }), 500

    try:
        result = gd_live_audio_service.process_audio_event(
            session_id=clean_session_id,
            participant_id=clean_pid,
            audio_path=str(target_path),
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
        "message": f"Audio file uploaded and processed for participant '{clean_pid}'",
        "result": result,
        **result,
    }), 200


@gd_bp.route("/sessions/<session_id>/evaluate", methods=["POST"])
@require_role("COORDINATOR")
def evaluate_live_gd_session(session_id: str):
    """
    Trigger post-session evaluation for an ENDED live Group Discussion session.
    Only the designated session coordinator can trigger evaluation.
    """
    try:
        persisted_doc = gd_live_service.evaluate_live_session(session_id.strip())
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
            "message": f"Live GD evaluation pipeline failed: {str(e)}",
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Live GD session '{session_id.strip()}' evaluated and persisted successfully",
        "evaluation": persisted_doc,
    }), 201


@gd_bp.route("/sessions/<session_id>/livekit-token", methods=["POST"])
@require_session_member
def get_livekit_room_token(session_id: str):
    """
    Issue a scoped LiveKit SFU room token.
    - User must be authenticated and an authorized member of the session.
    - Session must be in SCHEDULED or ACTIVE status.
    - Coordinators receive roomAdmin=True grant.
    - Candidates receive roomAdmin=False grant.
    - Returns 503 if LiveKit credentials are not configured in the environment.
    """
    clean_session_id = session_id.strip()
    session = getattr(g, "current_session", None)
    if not session:
        try:
            session = gd_session_service.get_session(clean_session_id)
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404

    status = session.get("status")
    if status not in ("SCHEDULED", "ACTIVE"):
        return jsonify({
            "status": "error",
            "message": f"Cannot issue meeting room token for session in '{status}' status. Must be 'SCHEDULED' or 'ACTIVE'.",
        }), 400

    # Fail closed if LiveKit is not configured in environment
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_URL):
        return jsonify({
            "status": "error",
            "message": "LiveKit integration is not configured. Set LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL.",
        }), 503

    user_id = g.current_user.get("user_id")
    display_name = g.current_user.get("display_name") or user_id
    is_coordinator = session.get("coordinator_id") == user_id

    # Construct standard LiveKit video grant
    now = datetime.now(timezone.utc)
    grant = {
        "room": clean_session_id,
        "roomJoin": True,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
        "roomAdmin": is_coordinator,
    }

    claims = {
        "iss": LIVEKIT_API_KEY,
        "sub": user_id,
        "name": display_name,
        "video": grant,
        "iat": now,
        "exp": now + timedelta(hours=6),
    }

    try:
        room_token = jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to generate room token: {e}",
        }), 500

    return jsonify({
        "status": "success",
        "session_id": clean_session_id,
        "livekit_url": LIVEKIT_URL,
        "room_name": clean_session_id,
        "participant_id": user_id,
        "display_name": display_name,
        "room_admin": is_coordinator,
        "room_token": room_token,
    }), 200
