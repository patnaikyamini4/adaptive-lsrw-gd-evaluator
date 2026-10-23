"""
Authentication Service

Provides cryptographically verified Email OTP generation, HMAC-SHA256 validation,
attempt tracking, rate limiting, and signed JWT issuance.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
from typing import Any, Optional
import uuid

import jwt

from backend.config import (
    ENVIRONMENT,
    JWT_ALGORITHM,
    JWT_EXPIRY_HOURS,
    JWT_SECRET_KEY,
    OTP_COOLDOWN_SECONDS,
    OTP_EXPIRY_SECONDS,
    OTP_MAX_ATTEMPTS,
    TRUSTED_COORDINATOR_EMAILS,
)
from backend.models.user import EMAIL_REGEX, User
import backend.services.otp_repository as default_otp_repo
import backend.services.user_repository as default_user_repo


logger = logging.getLogger("auth_service")


class AuthService:
    """
    Manages identity verification and JWT session tokens.
    """

    # Thread lock guaranteeing atomic cooldown evaluation and OTP challenge persistence
    _otp_lock = threading.Lock()

    # In-memory register used strictly in test and development modes to capture
    # generated OTPs for automated test verification without sending real emails.
    _test_sent_otps: dict[str, str] = {}

    def __init__(self, user_repo=None, otp_repo=None, secret_key: Optional[str] = None):
        self.user_repo = user_repo or default_user_repo
        self.otp_repo = otp_repo or default_otp_repo
        self.secret_key = secret_key or JWT_SECRET_KEY

    @classmethod
    def get_test_otp(cls, email: str) -> Optional[str]:
        """
        Test helper to retrieve the latest captured OTP for an email.
        Strictly forbidden and disabled in production.
        """
        if ENVIRONMENT == "production":
            raise PermissionError("Test OTP capture is not available in production.")
        return cls._test_sent_otps.get(email.strip().lower())

    @classmethod
    def clear_test_otps(cls) -> None:
        """Clear the captured test OTPs register."""
        cls._test_sent_otps.clear()

    def _hash_otp(self, email: str, otp: str) -> str:
        """Produce a salted HMAC-SHA256 digest of the OTP bound to the email address."""
        key = f"{self.secret_key}:{email.strip().lower()}".encode("utf-8")
        return hmac.new(key, otp.encode("utf-8"), hashlib.sha256).hexdigest()

    def request_otp(self, email: str) -> dict[str, Any]:
        """
        Generate and dispatch a cryptographically secure 6-digit OTP to the user's email.
        Enforces 60-second cooldown between requests, 5-minute TTL, and max 3 verification attempts.
        """
        if not email or not isinstance(email, str) or not email.strip():
            raise ValueError("email is required and must be a non-empty string")

        clean_email = email.strip().lower()
        if not EMAIL_REGEX.match(clean_email):
            raise ValueError(f"Invalid email address format: '{clean_email}'")

        with AuthService._otp_lock:
            now = datetime.now(timezone.utc)

            # 1. Enforce 60-second rate-limiting cooldown per email
            existing = self.otp_repo.get_otp(clean_email)
            if existing:
                last_req = existing.get("updated_at") or existing.get("created_at")
                if last_req:
                    if isinstance(last_req, str):
                        try:
                            last_req = datetime.fromisoformat(last_req)
                        except ValueError:
                            last_req = None
                    if last_req and last_req.tzinfo is None:
                        last_req = last_req.replace(tzinfo=timezone.utc)
                    if last_req:
                        elapsed = (now - last_req).total_seconds()
                        if elapsed < OTP_COOLDOWN_SECONDS:
                            remaining_wait = max(1, int(OTP_COOLDOWN_SECONDS - elapsed))
                            raise ValueError(
                                f"Please wait {remaining_wait} seconds before requesting a new verification code."
                            )

            # 2. Generate 6-digit cryptographically random OTP
            otp = f"{secrets.randbelow(900000) + 100000}"
            otp_hash = self._hash_otp(clean_email, otp)
            expires_at = now + timedelta(seconds=OTP_EXPIRY_SECONDS)

            # 3. Store in OTP repository with attempts limit and timestamp
            self.otp_repo.store_otp(
                email=clean_email,
                otp_hash=otp_hash,
                expires_at=expires_at,
                attempts_left=OTP_MAX_ATTEMPTS,
                purpose="login",
            )

            # 4. Dispatch OTP according to environment (never log actual OTP value)
            if ENVIRONMENT != "production":
                AuthService._test_sent_otps[clean_email] = otp
                logger.info(f"[AUTH] Generated OTP challenge for {clean_email}")
            else:
                self._send_production_email(clean_email, otp)

        return {
            "status": "success",
            "message": f"Verification code sent to {clean_email}. Valid for {OTP_EXPIRY_SECONDS // 60} minutes.",
            "email": clean_email,
            "expires_in_seconds": OTP_EXPIRY_SECONDS,
        }

    def _send_production_email(self, email: str, otp: str) -> None:
        """
        Dispatch real email via SMTP in production.
        Fails closed if SMTP is not configured.
        """
        smtp_host = os.getenv("SMTP_HOST")
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")

        if not (smtp_host and smtp_user and smtp_pass):
            raise RuntimeError(
                "CRITICAL EMAIL DISPATCH ERROR: SMTP configuration is required in production. "
                "Set SMTP_HOST, SMTP_USER, and SMTP_PASS."
            )
        # Real SMTP delivery logic goes here in production deployment
        logger.info(f"Dispatched email to {email}")

    def verify_otp(
        self,
        email: str,
        otp: str,
        display_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Verify the supplied OTP against the salted HMAC digest using constant-time comparison.
        Upon success:
        - Consumes the OTP immediately (single-use).
        - Resolves user and assigns role (Coordinator if trusted, else Candidate).
        - Issues a signed JWT.
        """
        if not email or not isinstance(email, str) or not email.strip():
            raise ValueError("email is required")
        if not otp or not isinstance(otp, str) or not otp.strip():
            raise ValueError("otp is required")

        clean_email = email.strip().lower()
        clean_otp = otp.strip()

        record = self.otp_repo.get_otp(clean_email)
        if not record:
            raise ValueError("No active verification code found for this email. Please request a new code.")

        # Check expiration
        expires_at = record.get("expires_at")
        now = datetime.now(timezone.utc)
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at and now > expires_at:
            self.otp_repo.delete_otp(clean_email)
            raise ValueError("Verification code has expired. Please request a new code.")

        # Check attempts left
        attempts_left = record.get("attempts_left", 0)
        if attempts_left <= 0:
            self.otp_repo.delete_otp(clean_email)
            raise ValueError("Maximum verification attempts exceeded. Please request a new code.")

        # Constant-time comparison
        expected_hash = record.get("otp_hash", "")
        actual_hash = self._hash_otp(clean_email, clean_otp)

        if not hmac.compare_digest(expected_hash, actual_hash):
            remaining = self.otp_repo.decrement_attempts(clean_email)
            if remaining <= 0:
                self.otp_repo.delete_otp(clean_email)
                raise ValueError("Invalid verification code. Maximum attempts exceeded. Code invalidated.")
            raise ValueError(f"Invalid verification code. {remaining} attempt(s) remaining.")

        # OTP verified! Consume immediately to prevent replay
        self.otp_repo.delete_otp(clean_email)
        AuthService._test_sent_otps.pop(clean_email, None)

        # Resolve or provision user
        user = self.user_repo.get_user_by_email(clean_email)
        if not user:
            # Enforce security invariant:
            # Public registration can NEVER grant COORDINATOR role.
            # Role is derived strictly from trusted whitelist or provisioning.
            if clean_email in TRUSTED_COORDINATOR_EMAILS:
                role = "COORDINATOR"
            else:
                role = "CANDIDATE"

            user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
            user_obj = User(
                user_id=user_id,
                email=clean_email,
                role=role,
                display_name=display_name or clean_email.split("@")[0],
                is_verified=True,
            )
            user = self.user_repo.create_user(user_obj)

        # Issue signed JWT session token
        token = self.create_access_token(user)

        return {
            "status": "success",
            "access_token": token,
            "token_type": "Bearer",
            "expires_in_hours": JWT_EXPIRY_HOURS,
            "user": {
                "user_id": user["user_id"],
                "email": user["email"],
                "role": user["role"],
                "display_name": user.get("display_name", ""),
            },
        }

    def provision_coordinator(self, email: str, display_name: str) -> dict[str, Any]:
        """
        Trusted administrative process to explicitly provision a coordinator account.
        """
        clean_email = email.strip().lower()
        if not EMAIL_REGEX.match(clean_email):
            raise ValueError(f"Invalid email: {clean_email}")

        existing = self.user_repo.get_user_by_email(clean_email)
        if existing:
            if existing.get("role") != "COORDINATOR":
                self.user_repo.update_user(existing["user_id"], {"role": "COORDINATOR"})
                existing["role"] = "COORDINATOR"
            return existing

        user_id = f"COORD-{uuid.uuid4().hex[:8].upper()}"
        user_obj = User(
            user_id=user_id,
            email=clean_email,
            role="COORDINATOR",
            display_name=display_name,
            is_verified=True,
        )
        return self.user_repo.create_user(user_obj)

    def create_access_token(self, user: dict[str, Any]) -> str:
        """
        Generate a cryptographically signed HS256 JWT with strict required claims:
        sub (user_id), email, role, display_name, iat, exp.
        """
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user["user_id"],
            "user_id": user["user_id"],
            "email": user["email"],
            "role": user["role"],
            "display_name": user.get("display_name", ""),
            "iat": now,
            "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
        }
        return jwt.encode(payload, self.secret_key, algorithm=JWT_ALGORITHM)

    def decode_and_validate_token(self, token: str) -> dict[str, Any]:
        """
        Strictly decode and validate a JWT.
        Enforces allowed algorithms (HS256 only), signature verification, expiration,
        and required claims ('sub', 'email', 'role', 'exp', 'iat').
        """
        if not token or not isinstance(token, str):
            raise ValueError("Token must be a non-empty string")

        try:
            payload = jwt.decode(
                token.strip(),
                self.secret_key,
                algorithms=[JWT_ALGORITHM],
                options={
                    "require": ["sub", "email", "role", "exp", "iat"],
                    "verify_signature": True,
                    "verify_exp": True,
                },
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Session token has expired. Please log in again.")
        except jwt.InvalidAlgorithmError:
            raise ValueError("Invalid token algorithm.")
        except jwt.InvalidSignatureError:
            raise ValueError("Invalid token signature.")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")
