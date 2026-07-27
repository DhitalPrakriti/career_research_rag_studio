"""Password login for the API.

Single-user by design: this serves one person's career documents, so there is no user
table, just a password and a signed session cookie.

The session token is an httpOnly cookie rather than a bearer token in localStorage,
because localStorage is readable by any injected script while an httpOnly cookie is not.
The cookie is SameSite=Lax and the app is served same-origin (FastAPI serves the built UI,
and the Vite dev server proxies /api), so cross-site requests never carry it.

Auth is off when APP_PASSWORD is unset, which keeps local development frictionless. That is
also the deployment footgun, so create_app logs a warning at startup when it happens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

SESSION_COOKIE = "rag_session"
SESSION_TTL_SECONDS = 12 * 60 * 60

# Login throttling. Small numbers are fine for a single-user app and make an online
# password guess impractical without needing a datastore.
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60

_PBKDF2_ROUNDS = 240_000


def auth_required() -> bool:
    return bool(_configured_password() or _configured_hash())


def _configured_password() -> str | None:
    value = os.getenv("APP_PASSWORD")
    return value or None


def _configured_hash() -> str | None:
    value = os.getenv("APP_PASSWORD_HASH")
    return value or None


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Produce a `pbkdf2_sha256$rounds$salt$digest` string for APP_PASSWORD_HASH."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return "$".join(
        [
            "pbkdf2_sha256",
            str(_PBKDF2_ROUNDS),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(candidate: str) -> bool:
    """Check a submitted password, preferring a stored hash over a plaintext one."""
    stored_hash = _configured_hash()
    if stored_hash:
        return _verify_against_hash(candidate, stored_hash)

    expected = _configured_password()
    if not expected:
        return False
    # Constant time, so a wrong password cannot be narrowed down by response timing.
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _verify_against_hash(candidate: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt_b64, digest_b64 = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", candidate.encode("utf-8"), salt, int(rounds)
        )
    except (ValueError, TypeError):
        logger.warning("APP_PASSWORD_HASH is malformed; refusing the login.")
        return False
    return hmac.compare_digest(actual, expected)


def _signing_secret() -> bytes:
    """Secret used to sign session cookies.

    APP_SECRET_KEY should be set in any deployment. Without one, fall back to deriving a
    key from the password so cookies are still signed — the consequence is that every
    process gets the same key from the same password, which is acceptable for a single
    instance but is why the deployment docs ask for an explicit secret.
    """
    explicit = os.getenv("APP_SECRET_KEY")
    if explicit:
        return explicit.encode("utf-8")
    seed = (_configured_hash() or _configured_password() or "insecure-development-secret")
    return hashlib.sha256(f"session-signing:{seed}".encode("utf-8")).digest()


def create_session_token(ttl_seconds: int = SESSION_TTL_SECONDS) -> str:
    payload = json.dumps({"exp": int(time.time()) + ttl_seconds}, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    signature = hmac.new(_signing_secret(), encoded.encode("ascii"), hashlib.sha256)
    return f"{encoded}.{base64.urlsafe_b64encode(signature.digest()).decode('ascii')}"


def verify_session_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    encoded, _, provided = token.partition(".")
    expected = hmac.new(_signing_secret(), encoded.encode("ascii"), hashlib.sha256)
    try:
        provided_bytes = base64.urlsafe_b64decode(provided)
    except (ValueError, TypeError):
        return False
    if not hmac.compare_digest(provided_bytes, expected.digest()):
        return False

    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded))
        expires_at = int(payload["exp"])
    except (ValueError, TypeError, KeyError):
        return False
    return time.time() < expires_at


class LoginThrottle:
    """Counts failed logins per client and locks the client out for a while."""

    def __init__(
        self,
        max_attempts: int = MAX_ATTEMPTS,
        lockout_seconds: int = LOCKOUT_SECONDS,
    ) -> None:
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, tuple[int, float]] = {}

    def retry_after(self, client: str) -> int:
        count, first_failure = self._failures.get(client, (0, 0.0))
        if count < self.max_attempts:
            return 0
        remaining = int(first_failure + self.lockout_seconds - time.time())
        if remaining <= 0:
            self._failures.pop(client, None)
            return 0
        return remaining

    def record_failure(self, client: str) -> None:
        count, first_failure = self._failures.get(client, (0, 0.0))
        # Start a fresh window once the previous lockout has expired.
        if count >= self.max_attempts and time.time() > first_failure + self.lockout_seconds:
            count, first_failure = 0, 0.0
        self._failures[client] = (count + 1, first_failure or time.time())

    def record_success(self, client: str) -> None:
        self._failures.pop(client, None)
