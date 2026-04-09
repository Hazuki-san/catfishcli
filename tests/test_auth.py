"""
Regression tests for dynamic auth-password reading in authenticate_user.

These tests verify that authenticate_user always reads GEMINI_AUTH_PASSWORD
from the environment at call time, so changes to os.environ (or .env before
process start) are honoured without requiring a module reload.
"""
import base64
import os

import pytest
from fastapi import HTTPException
from starlette.requests import Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(*, headers=None, query_string=b""):
    """Build a minimal Starlette Request with the given headers / query."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": query_string,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return Request(scope)


def _basic_header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


# ---------------------------------------------------------------------------
# Import the function under test AFTER helpers are defined so that env
# manipulation in individual tests is the only variable.
# ---------------------------------------------------------------------------
from src.auth import authenticate_user, _expected_password  # noqa: E402


# ---------------------------------------------------------------------------
# _expected_password tests
# ---------------------------------------------------------------------------

class TestExpectedPassword:
    def test_default_fallback(self):
        """Without env var, falls back to '123456'."""
        os.environ.pop("GEMINI_AUTH_PASSWORD", None)
        assert _expected_password() == "123456"

    def test_reads_current_env_value(self):
        """Returns whatever is currently set in os.environ."""
        os.environ["GEMINI_AUTH_PASSWORD"] = "MySecretPass!"
        try:
            assert _expected_password() == "MySecretPass!"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    def test_non_numeric_password(self):
        """Non-numeric passwords (letters, symbols) are supported."""
        os.environ["GEMINI_AUTH_PASSWORD"] = "abc!@#XYZ"
        try:
            assert _expected_password() == "abc!@#XYZ"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]


# ---------------------------------------------------------------------------
# authenticate_user tests
# ---------------------------------------------------------------------------

class TestAuthenticateUser:

    # -- query key ----------------------------------------------------------

    def test_query_key_correct_password(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "qwerty"
        try:
            req = _make_request(query_string=b"key=qwerty")
            result = authenticate_user(req)
            assert result == "api_key_user"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    def test_query_key_wrong_password_raises_401(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "correct"
        try:
            req = _make_request(query_string=b"key=wrong")
            with pytest.raises(HTTPException) as exc_info:
                authenticate_user(req)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    # -- x-goog-api-key -----------------------------------------------------

    def test_goog_api_key_correct(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "goog-secret"
        try:
            req = _make_request(headers={"x-goog-api-key": "goog-secret"})
            assert authenticate_user(req) == "goog_api_key_user"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    def test_goog_api_key_wrong_raises_401(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "correct"
        try:
            req = _make_request(headers={"x-goog-api-key": "wrong"})
            with pytest.raises(HTTPException) as exc_info:
                authenticate_user(req)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    # -- Bearer token -------------------------------------------------------

    def test_bearer_correct(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "BearerPass123"
        try:
            req = _make_request(headers={"authorization": "Bearer BearerPass123"})
            assert authenticate_user(req) == "bearer_user"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    def test_bearer_wrong_raises_401(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "correct"
        try:
            req = _make_request(headers={"authorization": "Bearer wrong"})
            with pytest.raises(HTTPException) as exc_info:
                authenticate_user(req)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    # -- Basic auth ---------------------------------------------------------

    def test_basic_auth_correct(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "BasicPass!"
        try:
            req = _make_request(headers={"authorization": _basic_header("user", "BasicPass!")})
            assert authenticate_user(req) == "user"
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    def test_basic_auth_wrong_raises_401(self):
        os.environ["GEMINI_AUTH_PASSWORD"] = "correct"
        try:
            req = _make_request(headers={"authorization": _basic_header("user", "wrong")})
            with pytest.raises(HTTPException) as exc_info:
                authenticate_user(req)
            assert exc_info.value.status_code == 401
        finally:
            del os.environ["GEMINI_AUTH_PASSWORD"]

    # -- Dynamic env update (core regression) --------------------------------

    def test_password_change_reflected_without_reload(self):
        """
        Core regression: changing os.environ after module import must be
        picked up by authenticate_user immediately (no reload required).
        """
        os.environ["GEMINI_AUTH_PASSWORD"] = "first_password"
        try:
            req1 = _make_request(headers={"authorization": "Bearer first_password"})
            assert authenticate_user(req1) == "bearer_user"

            # Simulate .env change before next request
            os.environ["GEMINI_AUTH_PASSWORD"] = "second_password"

            # Old password must now fail
            with pytest.raises(HTTPException) as exc_info:
                authenticate_user(req1)
            assert exc_info.value.status_code == 401

            # New password must succeed
            req2 = _make_request(headers={"authorization": "Bearer second_password"})
            assert authenticate_user(req2) == "bearer_user"
        finally:
            os.environ.pop("GEMINI_AUTH_PASSWORD", None)

    # -- 401 response shape -------------------------------------------------

    def test_401_response_shape(self):
        os.environ.pop("GEMINI_AUTH_PASSWORD", None)
        req = _make_request(headers={"authorization": "Bearer bad"})
        with pytest.raises(HTTPException) as exc_info:
            authenticate_user(req)
        exc = exc_info.value
        assert exc.status_code == 401
        assert "Invalid authentication credentials" in exc.detail
        assert exc.headers.get("WWW-Authenticate") == "Basic"

    # -- Default password ---------------------------------------------------

    def test_default_password_still_works(self):
        """Without any env var the default '123456' must still authenticate."""
        os.environ.pop("GEMINI_AUTH_PASSWORD", None)
        req = _make_request(headers={"authorization": "Bearer 123456"})
        assert authenticate_user(req) == "bearer_user"
