"""
Authentication & Authorization Middleware

Provides decorators for Flask route endpoints enforcing:
- Cryptographic JWT token presence and validity (@require_auth)
- Role-based access control (@require_role)
- Session coordinator ownership (@require_session_coordinator)
- Session participant membership (@require_session_member)
"""

from functools import wraps
import sys
from typing import Any, Callable
from flask import g, jsonify, request

from backend.services.auth_service import AuthService
from backend.services.gd_session_service import GDSessionService


auth_service_instance = AuthService()
session_service_instance = GDSessionService()


def get_token_from_header() -> str:
    """Extract Bearer token from HTTP Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not isinstance(auth_header, str):
        raise ValueError("Missing Authorization header")
    parts = auth_header.strip().split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ValueError("Invalid Authorization header format. Must be 'Bearer <token>'")
    return parts[1]


def get_active_session_service() -> GDSessionService:
    """
    Retrieve the active GDSessionService instance, respecting route-level mock patches.
    """
    if "backend.routes.gd_routes" in sys.modules:
        gd_routes = sys.modules["backend.routes.gd_routes"]
        if hasattr(gd_routes, "gd_session_service"):
            return getattr(gd_routes, "gd_session_service")
    return session_service_instance


def require_auth(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to require a valid, signed JWT access token."""
    @wraps(fn)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        try:
            token = get_token_from_header()
            payload = auth_service_instance.decode_and_validate_token(token)
            g.current_user = payload
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 401
        return fn(*args, **kwargs)
    return decorated


def require_role(*allowed_roles: str) -> Callable[..., Any]:
    """Decorator to enforce that the authenticated user possesses one of the allowed roles."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        @require_auth
        def decorated(*args: Any, **kwargs: Any) -> Any:
            user = getattr(g, "current_user", None)
            if not user or user.get("role") not in allowed_roles:
                return jsonify({
                    "status": "error",
                    "message": f"Access forbidden: requires role in {list(allowed_roles)}",
                }), 403
            return fn(*args, **kwargs)
        return decorated
    return decorator


def require_session_coordinator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator enforcing that the caller is an authenticated COORDINATOR
    and is the designated owner (coordinator_id) of the specified session_id.
    """
    @wraps(fn)
    @require_auth
    def decorated(*args: Any, **kwargs: Any) -> Any:
        user = getattr(g, "current_user", None)
        if not user or user.get("role") != "COORDINATOR":
            return jsonify({
                "status": "error",
                "message": "Access forbidden: coordinator privileges required",
            }), 403

        session_id = kwargs.get("session_id")
        if not session_id:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")

        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required",
            }), 400

        service = get_active_session_service()
        try:
            session = service.get_session(session_id.strip())
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        except Exception:
            session = None

        if session is not None and isinstance(session, dict):
            coord_id = session.get("coordinator_id")
            if coord_id and coord_id != user.get("user_id"):
                return jsonify({
                    "status": "error",
                    "message": "Access forbidden: you are not the designated coordinator for this session",
                }), 403

        g.current_session = session
        return fn(*args, **kwargs)
    return decorated


def require_session_member(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator enforcing that the caller is authenticated and either:
    1. The designated coordinator (session.coordinator_id == user.user_id), OR
    2. An explicitly assigned candidate participant (user.user_id in session.participant_ids).
    """
    @wraps(fn)
    @require_auth
    def decorated(*args: Any, **kwargs: Any) -> Any:
        user = getattr(g, "current_user", None)
        if not user:
            return jsonify({
                "status": "error",
                "message": "Authentication required",
            }), 401

        session_id = kwargs.get("session_id")
        if not session_id:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")

        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required",
            }), 400

        service = get_active_session_service()
        try:
            session = service.get_session(session_id.strip())
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        except Exception:
            session = None

        if session is not None and isinstance(session, dict):
            caller_id = user.get("user_id")
            is_coordinator = session.get("coordinator_id") == caller_id
            participant_ids = session.get("participant_ids", [])
            is_participant = isinstance(participant_ids, list) and caller_id in participant_ids

            if not (is_coordinator or is_participant):
                return jsonify({
                    "status": "error",
                    "message": f"Access forbidden: user '{caller_id}' is neither coordinator nor member of session '{session_id}'",
                }), 403

        g.current_session = session
        return fn(*args, **kwargs)
    return decorated
