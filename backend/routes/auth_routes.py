"""
Authentication REST API Routes

Endpoints for Email OTP identity proof and JWT token retrieval:
- POST /api/auth/otp/request -> Request one-time code to verified email
- POST /api/auth/otp/verify  -> Verify OTP code, resolve role, and retrieve JWT
- GET  /api/auth/me          -> Get current authenticated user profile
"""

from flask import Blueprint, g, jsonify, request

from backend.middleware.auth_middleware import require_auth
from backend.services.auth_service import AuthService


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
auth_service = AuthService()


@auth_bp.route("/otp/request", methods=["POST"])
def request_otp():
    """
    Request a 6-digit cryptographic OTP verification code sent to an email.
    Expected JSON body:
    {
        "email": "user@example.com"
    }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    email = data.get("email")
    if not email or not isinstance(email, str) or not email.strip():
        return jsonify({
            "status": "error",
            "message": "'email' is required and must be a valid email address",
        }), 400

    try:
        result = auth_service.request_otp(email.strip())
        return jsonify(result), 200
    except ValueError as e:
        status_code = 429 if "please wait" in str(e).lower() else 400
        return jsonify({
            "status": "error",
            "message": str(e),
        }), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Could not dispatch verification code: {e}",
        }), 500


@auth_bp.route("/otp/verify", methods=["POST"])
def verify_otp():
    """
    Verify the 6-digit OTP code against the salted HMAC digest.
    Upon successful proof of email control, returns signed JWT access token.
    Expected JSON body:
    {
        "email": "user@example.com",
        "otp": "491823",
        "display_name": "Jane Doe" (optional)
    }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a valid JSON object",
        }), 400

    email = data.get("email")
    otp = data.get("otp")
    display_name = data.get("display_name")

    if not email or not isinstance(email, str) or not email.strip():
        return jsonify({
            "status": "error",
            "message": "'email' is required",
        }), 400

    if not otp or not isinstance(otp, str) or not otp.strip():
        return jsonify({
            "status": "error",
            "message": "'otp' is required",
        }), 400

    try:
        result = auth_service.verify_otp(
            email=email.strip(),
            otp=otp.strip(),
            display_name=display_name.strip() if isinstance(display_name, str) else None,
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 401
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Verification failed: {e}",
        }), 500


@auth_bp.route("/me", methods=["GET"])
@require_auth
def get_current_user():
    """Return the verified identity and role of the authenticated caller."""
    user = getattr(g, "current_user", {})
    return jsonify({
        "status": "success",
        "user": user,
    }), 200
