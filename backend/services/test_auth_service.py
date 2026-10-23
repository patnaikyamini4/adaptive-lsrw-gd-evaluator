"""
Unit tests for AuthService (backend/services/auth_service.py).

Tests OTP generation, HMAC hashing, constant-time validation, attempt tracking,
expiration, and signed JWT issuance and verification without sending real emails.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import jwt
import pytest

from backend.config import JWT_ALGORITHM, JWT_SECRET_KEY
from backend.models.user import User
from backend.services.auth_service import AuthService


class InMemoryUserRepository:
    """Mock in-memory user repository for isolated testing."""

    def __init__(self):
        self.users_by_id = {}
        self.users_by_email = {}

    def create_user(self, user_obj):
        doc = user_obj.to_dict() if isinstance(user_obj, User) else user_obj.copy()
        self.users_by_id[doc["user_id"]] = doc
        self.users_by_email[doc["email"]] = doc
        return doc

    def get_user_by_id(self, user_id):
        return self.users_by_id.get(user_id)

    def get_user_by_email(self, email):
        return self.users_by_email.get(email.strip().lower())

    def update_user(self, user_id, updates):
        if user_id in self.users_by_id:
            self.users_by_id[user_id].update(updates)
            return self.users_by_id[user_id]
        return None


class InMemoryOtpRepository:
    """Mock in-memory OTP repository for isolated testing."""

    def __init__(self):
        self.otps = {}

    def store_otp(self, email, otp_hash, expires_at, attempts_left=3, purpose="login"):
        clean_email = email.strip().lower()
        now = datetime.now(timezone.utc)
        doc = {
            "email": clean_email,
            "otp_hash": otp_hash,
            "expires_at": expires_at,
            "attempts_left": attempts_left,
            "purpose": purpose,
            "created_at": self.otps.get(clean_email, {}).get("created_at", now),
            "updated_at": now,
        }
        self.otps[clean_email] = doc
        return doc

    def get_otp(self, email):
        return self.otps.get(email.strip().lower())

    def decrement_attempts(self, email):
        clean_email = email.strip().lower()
        if clean_email in self.otps:
            self.otps[clean_email]["attempts_left"] -= 1
            return max(0, self.otps[clean_email]["attempts_left"])
        return 0

    def delete_otp(self, email):
        clean_email = email.strip().lower()
        return self.otps.pop(clean_email, None) is not None


@pytest.fixture
def auth_service():
    user_repo = InMemoryUserRepository()
    otp_repo = InMemoryOtpRepository()
    AuthService.clear_test_otps()
    return AuthService(user_repo=user_repo, otp_repo=otp_repo, secret_key="test-secure-secret-key-for-unit-testing")


def test_request_otp_success(auth_service):
    """Test requesting an OTP successfully generates code without sending real email."""
    result = auth_service.request_otp("candidate@example.com")
    assert result["status"] == "success"
    assert result["email"] == "candidate@example.com"
    assert result["expires_in_seconds"] == 300

    otp = AuthService.get_test_otp("candidate@example.com")
    assert otp is not None
    assert len(otp) == 6
    assert otp.isdigit()


def test_request_otp_invalid_email(auth_service):
    """Test invalid email format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid email address format"):
        auth_service.request_otp("not-an-email")


def test_verify_otp_success_new_candidate(auth_service):
    """Test verifying valid OTP registers new candidate with CANDIDATE role and issues JWT."""
    auth_service.request_otp("candidate@example.com")
    otp = AuthService.get_test_otp("candidate@example.com")

    verify_res = auth_service.verify_otp("candidate@example.com", otp, display_name="Jane Candidate")
    assert verify_res["status"] == "success"
    assert "access_token" in verify_res
    assert verify_res["user"]["role"] == "CANDIDATE"
    assert verify_res["user"]["email"] == "candidate@example.com"
    assert verify_res["user"]["display_name"] == "Jane Candidate"

    # Token must decode cleanly
    payload = auth_service.decode_and_validate_token(verify_res["access_token"])
    assert payload["email"] == "candidate@example.com"
    assert payload["role"] == "CANDIDATE"


def test_verify_otp_trusted_coordinator_role(auth_service):
    """Test that whitelisted coordinator email is automatically assigned COORDINATOR role."""
    auth_service.request_otp("coordinator@example.com")
    otp = AuthService.get_test_otp("coordinator@example.com")

    verify_res = auth_service.verify_otp("coordinator@example.com", otp, display_name="Dr. Coordinator")
    assert verify_res["user"]["role"] == "COORDINATOR"


def test_verify_otp_wrong_code(auth_service):
    """Test incorrect OTP decrements attempts and rejects verification."""
    auth_service.request_otp("candidate@example.com")

    with pytest.raises(ValueError, match="Invalid verification code. 2 attempt"):
        auth_service.verify_otp("candidate@example.com", "000000")


def test_verify_otp_max_attempts_exceeded(auth_service):
    """Test that exceeding max attempts invalidates the OTP challenge."""
    auth_service.request_otp("candidate@example.com")

    # 1st fail
    with pytest.raises(ValueError, match="2 attempt"):
        auth_service.verify_otp("candidate@example.com", "000001")
    # 2nd fail
    with pytest.raises(ValueError, match="1 attempt"):
        auth_service.verify_otp("candidate@example.com", "000002")
    # 3rd fail (invalidates)
    with pytest.raises(ValueError, match="Maximum attempts exceeded"):
        auth_service.verify_otp("candidate@example.com", "000003")

    # 4th try -> no active code
    with pytest.raises(ValueError, match="No active verification code found"):
        auth_service.verify_otp("candidate@example.com", "000004")


def test_verify_otp_expired(auth_service):
    """Test expired OTP challenge is rejected and cleaned up."""
    auth_service.request_otp("candidate@example.com")
    otp = AuthService.get_test_otp("candidate@example.com")

    # Manually backdate expiration
    auth_service.otp_repo.otps["candidate@example.com"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=10)

    with pytest.raises(ValueError, match="Verification code has expired"):
        auth_service.verify_otp("candidate@example.com", otp)


def test_verify_otp_single_use(auth_service):
    """Test that an OTP cannot be reused after successful verification."""
    auth_service.request_otp("candidate@example.com")
    otp = AuthService.get_test_otp("candidate@example.com")

    auth_service.verify_otp("candidate@example.com", otp)

    with pytest.raises(ValueError, match="No active verification code found"):
        auth_service.verify_otp("candidate@example.com", otp)


def test_token_expired_rejection(auth_service):
    """Test expired JWT token is rejected."""
    now = datetime.now(timezone.utc)
    expired_payload = {
        "sub": "USR-100",
        "email": "user@example.com",
        "role": "CANDIDATE",
        "iat": now - timedelta(hours=5),
        "exp": now - timedelta(hours=1),
    }
    expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm=JWT_ALGORITHM)

    with pytest.raises(ValueError, match="Session token has expired"):
        auth_service.decode_and_validate_token(expired_token)


def test_token_tampered_signature_rejection(auth_service):
    """Test JWT with forged signature is rejected."""
    now = datetime.now(timezone.utc)
    forged_token = jwt.encode(
        {"sub": "USR-100", "email": "evil@attacker.com", "role": "COORDINATOR", "iat": now, "exp": now + timedelta(hours=1)},
        "wrong-secret-key-used-by-attacker",
        algorithm=JWT_ALGORITHM,
    )

    with pytest.raises(ValueError, match="Invalid token"):
        auth_service.decode_and_validate_token(forged_token)


def test_token_missing_required_claims(auth_service):
    """Test token missing role or sub claim is rejected."""
    now = datetime.now(timezone.utc)
    missing_role_payload = {
        "sub": "USR-100",
        "email": "user@example.com",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    token = jwt.encode(missing_role_payload, auth_service.secret_key, algorithm=JWT_ALGORITHM)

    with pytest.raises(ValueError, match="Invalid token"):
        auth_service.decode_and_validate_token(token)


def test_provision_coordinator(auth_service):
    """Test administrative coordinator provisioning."""
    coord = auth_service.provision_coordinator("new_coordinator@adaptive.edu", "Prof. Smith")
    assert coord["role"] == "COORDINATOR"
    assert coord["email"] == "new_coordinator@adaptive.edu"
    assert coord["user_id"].startswith("COORD-")


def test_request_otp_immediate_repeat_triggers_cooldown(auth_service):
    """Test immediate repeated OTP request violates 60-second cooldown."""
    res1 = auth_service.request_otp("cooldown_user@example.com")
    assert res1["status"] == "success"

    # Immediate second request must fail with cooldown error
    with pytest.raises(ValueError, match="Please wait .* seconds before requesting a new verification code"):
        auth_service.request_otp("cooldown_user@example.com")


def test_request_otp_after_cooldown_succeeds(auth_service):
    """Test OTP request succeeds once cooldown period has passed."""
    auth_service.request_otp("cooldown_ok@example.com")

    # Fast-forward updated_at timestamp by 65 seconds
    past_time = datetime.now(timezone.utc) - timedelta(seconds=65)
    auth_service.otp_repo.otps["cooldown_ok@example.com"]["updated_at"] = past_time

    res2 = auth_service.request_otp("cooldown_ok@example.com")
    assert res2["status"] == "success"


def test_request_otp_cooldown_timezone_naive_timestamp_safety(auth_service):
    """Test timezone-naive timestamps from legacy DB records are handled safely."""
    auth_service.request_otp("naive_time@example.com")

    # Provide a timezone-naive UTC datetime in the past (as PyMongo returns when tz_aware=False)
    naive_past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=70)
    auth_service.otp_repo.otps["naive_time@example.com"]["updated_at"] = naive_past

    res = auth_service.request_otp("naive_time@example.com")
    assert res["status"] == "success"


def test_actual_otp_never_logged(auth_service, caplog):
    """Test that actual OTP secret code is never printed in log messages."""
    import logging
    with caplog.at_level(logging.INFO):
        auth_service.request_otp("secret_log@example.com")

    test_otp = AuthService.get_test_otp("secret_log@example.com")
    assert test_otp is not None

    # Verify the actual 6-digit string does not appear in any log message
    for record in caplog.records:
        assert test_otp not in record.message
        assert f": {test_otp}" not in record.message


def test_test_otp_capture_rejected_in_production(monkeypatch):
    """Test AuthService.get_test_otp raises PermissionError when ENVIRONMENT is production."""
    monkeypatch.setattr("backend.services.auth_service.ENVIRONMENT", "production")

    with pytest.raises(PermissionError, match="Test OTP capture is not available in production"):
        AuthService.get_test_otp("anyone@example.com")


def test_test_otp_capture_available_in_development_and_testing():
    """Test AuthService.get_test_otp continues to work in development/testing."""
    AuthService._test_sent_otps["dev_test@example.com"] = "654321"
    otp = AuthService.get_test_otp("dev_test@example.com")
    assert otp == "654321"

