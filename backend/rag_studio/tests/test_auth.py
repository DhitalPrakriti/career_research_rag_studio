"""Auth tests.

The endpoints serve someone's personal career documents, so the properties worth pinning
down are: nothing is readable without a session, a session cannot be forged, and guessing
the password is throttled.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from rag_studio.api import app as app_module
from rag_studio.api.app import create_app
from rag_studio.api.auth import (
    SESSION_COOKIE,
    LoginThrottle,
    auth_required,
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)

PASSWORD = "correct-horse-battery-staple"

PROTECTED = [
    ("get", "/api/health"),
    ("get", "/api/documents"),
    ("post", "/api/documents/reindex"),
    ("post", "/api/query"),
    ("post", "/api/tailor"),
]


@pytest.fixture
def secured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    monkeypatch.setenv("APP_SECRET_KEY", "test-signing-secret")
    monkeypatch.setattr(app_module.AgentService, "load", lambda self, docs_dir=None: None)
    with TestClient(create_app("docs")) as client:
        yield client


@pytest.fixture
def open_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module.AgentService, "load", lambda self, docs_dir=None: None)
    with TestClient(create_app("docs")) as client:
        yield client


class TestPasswordChecking:
    def test_auth_is_off_without_a_password(self) -> None:
        assert auth_required() is False
        assert verify_password("anything") is False

    def test_plaintext_password_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PASSWORD", PASSWORD)

        assert auth_required() is True
        assert verify_password(PASSWORD) is True
        assert verify_password(PASSWORD + "x") is False

    def test_hashed_password_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PASSWORD_HASH", hash_password(PASSWORD))

        assert auth_required() is True
        assert verify_password(PASSWORD) is True
        assert verify_password("wrong") is False

    def test_hash_is_salted(self) -> None:
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_hash_wins_over_plaintext(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PASSWORD", "ignored")
        monkeypatch.setenv("APP_PASSWORD_HASH", hash_password(PASSWORD))

        assert verify_password(PASSWORD) is True
        assert verify_password("ignored") is False

    def test_a_malformed_hash_refuses_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_PASSWORD_HASH", "not-a-real-hash")

        assert verify_password(PASSWORD) is False


class TestSessionTokens:
    def test_a_freshly_minted_token_verifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_SECRET_KEY", "secret")

        assert verify_session_token(create_session_token()) is True

    def test_garbage_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_SECRET_KEY", "secret")

        for candidate in ["", "nope", "a.b", "....", "eyJ9.zzzz"]:
            assert verify_session_token(candidate) is False

    def test_a_tampered_payload_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_SECRET_KEY", "secret")
        token = create_session_token()
        payload, _, signature = token.partition(".")

        assert verify_session_token(f"{payload}x.{signature}") is False

    def test_a_token_signed_with_another_secret_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rotating APP_SECRET_KEY must invalidate existing sessions."""
        monkeypatch.setenv("APP_SECRET_KEY", "first-secret")
        token = create_session_token()

        monkeypatch.setenv("APP_SECRET_KEY", "second-secret")

        assert verify_session_token(token) is False

    def test_an_expired_token_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_SECRET_KEY", "secret")

        assert verify_session_token(create_session_token(ttl_seconds=-1)) is False


class TestThrottle:
    def test_allows_attempts_below_the_limit(self) -> None:
        throttle = LoginThrottle(max_attempts=3, lockout_seconds=60)
        throttle.record_failure("1.2.3.4")
        throttle.record_failure("1.2.3.4")

        assert throttle.retry_after("1.2.3.4") == 0

    def test_locks_out_after_the_limit(self) -> None:
        throttle = LoginThrottle(max_attempts=3, lockout_seconds=60)
        for _ in range(3):
            throttle.record_failure("1.2.3.4")

        assert throttle.retry_after("1.2.3.4") > 0

    def test_lockout_is_per_client(self) -> None:
        throttle = LoginThrottle(max_attempts=2, lockout_seconds=60)
        for _ in range(2):
            throttle.record_failure("1.2.3.4")

        assert throttle.retry_after("5.6.7.8") == 0

    def test_lockout_expires(self) -> None:
        throttle = LoginThrottle(max_attempts=1, lockout_seconds=1)
        throttle.record_failure("1.2.3.4")
        assert throttle.retry_after("1.2.3.4") > 0

        time.sleep(1.05)

        assert throttle.retry_after("1.2.3.4") == 0

    def test_success_clears_the_record(self) -> None:
        throttle = LoginThrottle(max_attempts=2, lockout_seconds=60)
        throttle.record_failure("1.2.3.4")
        throttle.record_success("1.2.3.4")
        throttle.record_failure("1.2.3.4")

        assert throttle.retry_after("1.2.3.4") == 0


class TestProtectedEndpoints:
    @pytest.mark.parametrize("method,path", PROTECTED)
    def test_everything_needs_a_session(
        self, secured: TestClient, method: str, path: str
    ) -> None:
        call = getattr(secured, method)
        response = call(path, json={}) if method == "post" else call(path)

        assert response.status_code == 401

    def test_liveness_stays_open_for_probes(self, secured: TestClient) -> None:
        """Cloud load balancers cannot authenticate, and it reveals nothing.

        Path is /api/live, not /healthz, which Cloud Run reserves and never forwards.
        """
        response = secured.get("/api/live")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_session_endpoint_reports_the_state(self, secured: TestClient) -> None:
        body = secured.get("/api/auth/session").json()

        assert body == {"auth_required": True, "authenticated": False}

    def test_login_then_access(self, secured: TestClient) -> None:
        login = secured.post("/api/auth/login", json={"password": PASSWORD})

        assert login.status_code == 200
        assert login.json() == {"auth_required": True, "authenticated": True}
        assert secured.get("/api/health").status_code == 200

    def test_the_session_cookie_is_httponly(self, secured: TestClient) -> None:
        """An httpOnly cookie is not readable by injected scripts; localStorage is."""
        response = secured.post("/api/auth/login", json={"password": PASSWORD})

        header = response.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=lax" in header

    def test_a_wrong_password_is_rejected(self, secured: TestClient) -> None:
        response = secured.post("/api/auth/login", json={"password": "wrong"})

        assert response.status_code == 401
        assert secured.get("/api/health").status_code == 401

    def test_repeated_failures_are_throttled(self, secured: TestClient) -> None:
        for _ in range(5):
            secured.post("/api/auth/login", json={"password": "wrong"})

        response = secured.post("/api/auth/login", json={"password": "wrong"})

        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_logout_ends_the_session(self, secured: TestClient) -> None:
        secured.post("/api/auth/login", json={"password": PASSWORD})
        assert secured.get("/api/health").status_code == 200

        secured.post("/api/auth/logout")

        assert secured.get("/api/health").status_code == 401

    def test_an_empty_password_is_rejected_by_validation(self, secured: TestClient) -> None:
        assert secured.post("/api/auth/login", json={"password": ""}).status_code == 422


class TestOpenApi:
    def test_without_a_password_everything_is_reachable(self, open_api: TestClient) -> None:
        """Local development stays frictionless; create_app logs a warning about it."""
        assert open_api.get("/api/health").status_code == 200
        assert open_api.get("/api/auth/session").json() == {
            "auth_required": False,
            "authenticated": True,
        }

    def test_login_is_a_no_op_when_no_password_is_set(self, open_api: TestClient) -> None:
        body = open_api.post("/api/auth/login", json={"password": "anything"}).json()

        assert body == {"auth_required": False, "authenticated": True}
