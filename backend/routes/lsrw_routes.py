import os
import uuid

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from backend.services.lsrw_session_service import (
    LSRWSessionService
)

from backend.services.lsrw_session_repository import (
    create_session,
    get_session,
    update_session
)

from backend.services.speaking_service import (
    SpeakingService
)

from backend.services.lsrw_response_repository import (
    create_response
)


# ==================================================
# BLUEPRINT
# ==================================================

lsrw_bp = Blueprint(
    "lsrw",
    __name__,
    url_prefix="/api/lsrw"
)


# ==================================================
# SERVICES
# ==================================================

session_service = LSRWSessionService()

speaking_service = SpeakingService()


# ==================================================
# CREATE LSRW SESSION
# ==================================================

@lsrw_bp.route(
    "/sessions",
    methods=["POST"]
)
def create_lsrw_session():

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "Request body is required"
        }), 400

    participant_id = data.get("participant_id")

    if not participant_id:
        return jsonify({
            "status": "error",
            "message": "participant_id is required"
        }), 400

    session = session_service.create_session(
        participant_id=participant_id
    )

    create_session(session)

    return jsonify({
        "status": "success",
        "session": session
    }), 201


# ==================================================
# START LSRW SESSION
# ==================================================

@lsrw_bp.route(
    "/sessions/<session_id>/start",
    methods=["POST"]
)
def start_lsrw_session(session_id):

    session = get_session(session_id)

    if not session:
        return jsonify({
            "status": "error",
            "message": "Session not found"
        }), 404

    session = session_service.start_session(
        session
    )

    updated_session = update_session(
        session_id,
        session
    )

    return jsonify({
        "status": "success",
        "message": "LSRW session started",
        "session": updated_session
    }), 200


# ==================================================
# GET LSRW SESSION
# ==================================================

@lsrw_bp.route(
    "/sessions/<session_id>",
    methods=["GET"]
)
def get_lsrw_session(session_id):

    session = get_session(session_id)

    if not session:
        return jsonify({
            "status": "error",
            "message": "Session not found"
        }), 404

    return jsonify({
        "status": "success",
        "session": session
    }), 200


# ==================================================
# START LSRW MODULE
# ==================================================

@lsrw_bp.route(
    "/sessions/<session_id>/module/start",
    methods=["POST"]
)
def start_lsrw_module(session_id):

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "Request body is required"
        }), 400

    module = data.get("module")
    question_id = data.get("question_id")

    allowed_modules = [
        "LISTENING",
        "SPEAKING",
        "READING",
        "WRITING"
    ]

    if not module:
        return jsonify({
            "status": "error",
            "message": "module is required"
        }), 400

    module = module.upper()

    if module not in allowed_modules:
        return jsonify({
            "status": "error",
            "message": (
                "Invalid module. Allowed modules: "
                "LISTENING, SPEAKING, READING, WRITING"
            )
        }), 400

    if not question_id:
        return jsonify({
            "status": "error",
            "message": "question_id is required"
        }), 400

    session = get_session(session_id)

    if not session:
        return jsonify({
            "status": "error",
            "message": "Session not found"
        }), 404

    # Session must be active before starting a module
    if session.get("status") not in [
        "ACTIVE",
        "LISTENING",
        "SPEAKING",
        "READING",
        "WRITING"
    ]:
        return jsonify({
            "status": "error",
            "message": (
                "Session cannot start a module "
                "in its current state"
            ),
            "current_status": session.get("status")
        }), 400

    session = session_service.start_module(
        session,
        module,
        question_id
    )

    updated_session = update_session(
        session_id,
        session
    )

    return jsonify({
        "status": "success",
        "message": f"{module} module started",
        "session": updated_session
    }), 200


# ==================================================
# SUBMIT RESPONSE
# ==================================================
#
# Currently supports:
#
# SPEAKING → audio upload → ASR → transcript → MongoDB
#
# Writing / Reading / Listening will be added later.
#
# ==================================================

@lsrw_bp.route(
    "/sessions/<session_id>/response",
    methods=["POST"]
)
def submit_response(session_id):

    # ----------------------------------------------
    # 1. Find session
    # ----------------------------------------------

    session = get_session(session_id)

    if not session:
        return jsonify({
            "status": "error",
            "message": "Session not found"
        }), 404

    # ----------------------------------------------
    # 2. Check current module
    # ----------------------------------------------

    module = session.get("current_module")

    if module != "SPEAKING":

        return jsonify({
            "status": "error",
            "message": (
                "Response endpoint currently "
                "supports SPEAKING only"
            ),
            "current_module": module
        }), 400

    # ----------------------------------------------
    # 3. Get question ID
    # ----------------------------------------------

    question_id = session.get(
        "current_question_id"
    )

    if not question_id:

        return jsonify({
            "status": "error",
            "message": "No active question"
        }), 400

    # ----------------------------------------------
    # 4. Get participant ID
    # ----------------------------------------------

    participant_id = session.get(
        "participant_id"
    )

    if not participant_id:

        return jsonify({
            "status": "error",
            "message": "Participant ID not found"
        }), 400

    # ----------------------------------------------
    # 5. Check uploaded audio
    # ----------------------------------------------

    if "audio" not in request.files:

        return jsonify({
            "status": "error",
            "message": "Audio file is required"
        }), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":

        return jsonify({
            "status": "error",
            "message": "No audio file selected"
        }), 400

    # ----------------------------------------------
    # 6. Create audio directory
    # ----------------------------------------------

    audio_directory = os.path.join(
        "data",
        "lsrw",
        "audio",
        "speaking"
    )

    os.makedirs(
        audio_directory,
        exist_ok=True
    )

    # ----------------------------------------------
    # 7. Create unique filename
    # ----------------------------------------------

    original_filename = secure_filename(
        audio_file.filename
    )

    extension = os.path.splitext(
        original_filename
    )[1]

    if not extension:
        extension = ".audio"

    filename = (
        f"{session_id}_"
        f"{participant_id}_"
        f"{question_id}_"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    audio_path = os.path.join(
        audio_directory,
        filename
    )

    # ----------------------------------------------
    # 8. Save audio
    # ----------------------------------------------

    try:

        audio_file.save(audio_path)

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": "Failed to save audio file",
            "error": str(e)
        }), 500

    # ----------------------------------------------
    # 9. Process audio with Speaking Service
    # ----------------------------------------------

    try:

        response = speaking_service.process_audio(
            audio_path=audio_path,
            session_id=session_id,
            participant_id=participant_id,
            question_id=question_id
        )

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": (
                "Speaking audio processing failed"
            ),
            "error": str(e)
        }), 500

    # ----------------------------------------------
    # 10. Save response to MongoDB
    # ----------------------------------------------

    try:

        create_response(response)

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": (
                "Failed to save speaking response"
            ),
            "error": str(e)
        }), 500

    # ----------------------------------------------
    # 11. Return response
    # ----------------------------------------------

    return jsonify({
        "status": "success",
        "message": "Speaking response processed",
        "response": response
    }), 201