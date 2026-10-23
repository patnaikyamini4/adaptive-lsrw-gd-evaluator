"""
Integration tests for Authentication REST API routes (/api/auth).

Tests:
- POST /api/auth/otp/request
- POST /api/auth/otp/verify
- GET  /api/auth/me
"""

from unittest.mock import patch
import uuid
import pytest

from backend.app import app
import backend.services.otp_repository as otp_repo
from backend.services.auth_service import AuthService
import backend.services.user_repository as user_repo


class InMemoryCollection:
    """In-memory collection for isolated, network-free auth route testing."""

    def __init__(self, key_field="email"):
        self.key_field = key_field
        self._docs = {}

    def insert_one(self, doc):
        d = doc.copy()
        key = d.get(self.key_field)
        d["_id"] = f"oid_{uuid.uuid4().hex[:12]}"
        self._docs[key] = d
        return type("InsertResult", (), {"inserted_id": d["_id"]})()

    def update_one(self, filter_doc, update_doc, upsert=False):
        key = filter_doc.get(self.key_field)
        existing = self._docs.get(key)
        if existing is None and upsert:
            d = {}
            if "$setOnInsert" in update_doc:
                d.update(update_doc["$setOnInsert"])
            if "$set" in update_doc:
                d.update(update_doc["$set"])
            d[self.key_field] = key
            d["_id"] = f"oid_{uuid.uuid4().hex[:12]}"
            self._docs[key] = d
            return type("UpdateResult", (), {"upserted_id": key, "matched_count": 0})()
        elif existing is not None:
            if "$set" in update_doc:
                existing.update(update_doc["$set"])
            return type("UpdateResult", (), {"matched_count": 1})()
        return type("UpdateResult", (), {"matched_count": 0})()

    def find_one_and_update(self, filter_doc, update_doc, return_document=True):
        key = filter_doc.get(self.key_field)
        doc = self._docs.get(key)
        if not doc:
            return None
        if "$inc" in update_doc:
            for k, inc in update_doc["$inc"].items():
                doc[k] = doc.get(k, 0) + inc
        if "$set" in update_doc:
            doc.update(update_doc["$set"])
        return doc.copy()

    def find_one(self, filter_doc):
        key = filter_doc.get(self.key_field)
        if key and key in self._docs:
            return self._docs[key].copy()
        # Alternative search (e.g. user_id vs email)
        for d in self._docs.values():
            match = True
            for k, v in filter_doc.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                return d.copy()
        return None

    def delete_one(self, filter_doc):
        key = filter_doc.get(self.key_field)
        if key in self._docs:
            del self._docs[key]
            return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    def find(self):
        class Cursor:
            def __init__(self, docs):
                self._docs = list(docs)

            def sort(self, key, direction):
                return self

            def limit(self, num):
                return [d.copy() for d in self._docs[:num]]

            def __iter__(self):
                return iter([d.copy() for d in self._docs])

        return Cursor(list(self._docs.values()))


@pytest.fixture(autouse=True)
def isolated_auth_db(monkeypatch):
    """Isolate auth tests using in-memory mock collections to avoid remote Atlas timeouts."""
    mock_users = InMemoryCollection(key_field="email")
    mock_otps = InMemoryCollection(key_field="email")

    monkeypatch.setattr(user_repo, "collection", mock_users)
    monkeypatch.setattr(otp_repo, "collection", mock_otps)
    AuthService.clear_test_otps()
    yield
    AuthService.clear_test_otps()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_request_otp_route_success(client):
    """Test successful OTP request returns 200."""
    response = client.post(
        "/api/auth/otp/request",
        json={"email": "candidate1@example.com"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "Verification code sent" in data["message"]
    assert data["email"] == "candidate1@example.com"


def test_request_otp_route_invalid_payload(client):
    """Test OTP request rejects missing email or non-json."""
    res1 = client.post("/api/auth/otp/request", data="not-json")
    assert res1.status_code == 400

    res2 = client.post("/api/auth/otp/request", json={})
    assert res2.status_code == 400
    assert "'email' is required" in res2.get_json()["message"]

    res3 = client.post("/api/auth/otp/request", json={"email": "not-an-email"})
    assert res3.status_code == 400
    assert "Invalid email" in res3.get_json()["message"]


def test_verify_otp_route_success(client):
    """Test full flow: request OTP -> retrieve OTP from test register -> verify -> get JWT."""
    email = "candidate2@example.com"
    req_res = client.post("/api/auth/otp/request", json={"email": email})
    assert req_res.status_code == 200

    otp = AuthService.get_test_otp(email)
    assert otp is not None

    verify_res = client.post(
        "/api/auth/otp/verify",
        json={"email": email, "otp": otp, "display_name": "Test Candidate"},
    )
    assert verify_res.status_code == 200
    data = verify_res.get_json()
    assert data["status"] == "success"
    assert "access_token" in data
    assert data["token_type"] == "Bearer"
    assert data["user"]["email"] == email
    assert data["user"]["role"] == "CANDIDATE"
    assert data["user"]["display_name"] == "Test Candidate"


def test_verify_otp_route_invalid_code(client):
    """Test wrong OTP code returns 401."""
    email = "candidate3@example.com"
    client.post("/api/auth/otp/request", json={"email": email})

    verify_res = client.post(
        "/api/auth/otp/verify",
        json={"email": email, "otp": "999999"},
    )
    assert verify_res.status_code == 401
    assert "Invalid verification code" in verify_res.get_json()["message"]


def test_get_me_route_authenticated(client):
    """Test GET /api/auth/me with valid Bearer token."""
    email = "coordinator@example.com"
    client.post("/api/auth/otp/request", json={"email": email})
    otp = AuthService.get_test_otp(email)

    verify_res = client.post(
        "/api/auth/otp/verify",
        json={"email": email, "otp": otp},
    )
    token = verify_res.get_json()["access_token"]

    me_res = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_res.status_code == 200
    user_data = me_res.get_json()["user"]
    assert user_data["email"] == email
    assert user_data["role"] == "COORDINATOR"


def test_get_me_route_unauthenticated(client):
    """Test GET /api/auth/me without Authorization header returns 401."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert "Missing Authorization header" in response.get_json()["message"]


def test_get_me_route_malformed_token(client):
    """Test GET /api/auth/me with malformed token returns 401."""
    res1 = client.get("/api/auth/me", headers={"Authorization": "InvalidFormatToken"})
    assert res1.status_code == 401

    res2 = client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
    assert res2.status_code == 401


def test_request_otp_route_cooldown_returns_429(client):
    """Test rapid repeated OTP requests return HTTP 429 Too Many Requests."""
    email = "fast_requester@example.com"
    res1 = client.post("/api/auth/otp/request", json={"email": email})
    assert res1.status_code == 200

    # Immediate second request must return 429
    res2 = client.post("/api/auth/otp/request", json={"email": email})
    assert res2.status_code == 429
    data = res2.get_json()
    assert data["status"] == "error"
    assert "Please wait" in data["message"]

