import os
import json
import base64
import time
import logging
import threading
from datetime import datetime, timezone

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBasic
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest

from .utils import get_user_agent, get_client_metadata
from .config import (
    CLIENT_ID, CLIENT_SECRET, SCOPES, CREDENTIAL_FILE,
    CODE_ASSIST_ENDPOINT, GEMINI_AUTH_PASSWORD
)

# --- Global State for Account Polling ---
ACCOUNTS = []
_account_index = 0
_account_lock = threading.Lock()
onboarding_complete_map = {}
file_lock = threading.Lock()

security = HTTPBasic()


def _load_accounts():
    """Loads all accounts from the credential file."""
    global ACCOUNTS
    if not os.path.exists(CREDENTIAL_FILE):
        logging.warning(
            f"Credential file not found at {CREDENTIAL_FILE}. "
            f"Server started - authentication will be required on first request."
        )
        return

    try:
        with open(CREDENTIAL_FILE, "r") as f:
            creds_data = json.load(f)

        # Normalize: always work with a list internally
        if isinstance(creds_data, dict):
            creds_data = [creds_data]

        if isinstance(creds_data, list) and creds_data:
            ACCOUNTS = creds_data
            logging.info(f"Successfully loaded {len(ACCOUNTS)} account(s).")
        else:
            logging.error("Credential file is not a valid JSON array or object.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse credentials file {CREDENTIAL_FILE}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading accounts: {e}")


# Load accounts when the module is first imported
_load_accounts()


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code = None
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>OAuth authentication successful!</h1><p>You can close this window.</p>")
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1><p>Please try again.</p>")
    def log_message(self, format, *args):
        pass  # Suppress default HTTP server logging


def authenticate_user(request: Request):
    """Authenticate the user with multiple methods."""
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD:
        return "api_key_user"

    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD:
        return "goog_api_key_user"

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if bearer_token == GEMINI_AUTH_PASSWORD:
            return "bearer_user"

    if auth_header.startswith("Basic "):
        try:
            encoded_credentials = auth_header[6:]
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8', "ignore")
            username, password = decoded_credentials.split(':', 1)
            if password == GEMINI_AUTH_PASSWORD:
                return username
        except Exception:
            pass

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )


def _get_next_account() -> dict | None:
    """Thread-safe round-robin account selection."""
    global _account_index
    with _account_lock:
        if not ACCOUNTS:
            return None
        account = ACCOUNTS[_account_index]
        _account_index = (_account_index + 1) % len(ACCOUNTS)
        return account


def _update_account_in_memory(creds, project_id=None):
    """Update the in-memory ACCOUNTS list after a token refresh."""
    if not creds or not creds.refresh_token:
        return
    with _account_lock:
        for acc in ACCOUNTS:
            if acc.get("refresh_token") == creds.refresh_token:
                acc["token"] = creds.token
                if creds.expiry:
                    expiry_utc = (
                        creds.expiry.astimezone(timezone.utc)
                        if creds.expiry.tzinfo
                        else creds.expiry.replace(tzinfo=timezone.utc)
                    )
                    acc["expiry"] = expiry_utc.isoformat()
                if project_id:
                    acc["project_id"] = project_id
                break


def save_credentials(creds, project_id=None):
    """Saves updated credentials for a specific account back to the file."""
    with file_lock:
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                current_accounts = json.load(f)
                if not isinstance(current_accounts, list):
                    current_accounts = [current_accounts]
        except (FileNotFoundError, json.JSONDecodeError):
            current_accounts = []

        account_found = False
        for i, acc in enumerate(current_accounts):
            if acc.get("refresh_token") == creds.refresh_token:
                current_accounts[i]["token"] = creds.token
                if creds.expiry:
                    expiry_utc = (
                        creds.expiry.astimezone(timezone.utc)
                        if creds.expiry.tzinfo
                        else creds.expiry.replace(tzinfo=timezone.utc)
                    )
                    current_accounts[i]["expiry"] = expiry_utc.isoformat()
                if project_id:
                    current_accounts[i]["project_id"] = project_id
                account_found = True
                break

        if not account_found:
            logging.warning("Could not find matching account to save refreshed credentials.")
            return

        try:
            with open(CREDENTIAL_FILE, "w") as f:
                # Always save as array for consistency
                json.dump(current_accounts, f, indent=2)
            logging.info(f"Successfully saved refreshed token for project {project_id or 'unknown'}.")
        except Exception as e:
            logging.error(f"Failed to write updated credentials to file: {e}")


def get_credentials(allow_oauth_flow=True):
    """Gets next available credentials with failover across accounts."""
    if not ACCOUNTS:
        return _manual_oauth_flow() if allow_oauth_flow else None

    attempts = len(ACCOUNTS)

    for attempt in range(attempts):
        selected_account = _get_next_account()
        if selected_account is None:
            break

        project_id = selected_account.get("project_id", "unknown")

        try:
            creds_info = selected_account.copy()
            if "access_token" in creds_info and "token" not in creds_info:
                creds_info["token"] = creds_info["access_token"]
            if "scope" in creds_info and "scopes" not in creds_info:
                creds_info["scopes"] = creds_info["scope"].split()

            credentials = Credentials.from_authorized_user_info(creds_info, SCOPES)

            if credentials.expired and credentials.refresh_token:
                logging.info(f"Token for project {project_id} expired. Refreshing...")
                try:
                    credentials.refresh(GoogleAuthRequest())
                    save_credentials(credentials, project_id)
                    _update_account_in_memory(credentials, project_id)
                    logging.info(f"Token refreshed for project {project_id}")
                except Exception as e:
                    logging.warning(
                        f"Refresh failed for project {project_id}: {e}. "
                        f"Trying next account ({attempt + 1}/{attempts})..."
                    )
                    continue

            if not credentials.token:
                logging.warning(f"No token for project {project_id}. Trying next...")
                continue

            logging.info(f"Using account: {project_id}")
            return credentials

        except Exception as e:
            logging.error(f"Credentials failed for {project_id}: {e}. Trying next...")
            continue

    logging.error("All accounts exhausted. No valid credentials available.")
    return None


def get_user_project_id(creds):
    """Gets the user's project ID. Env var only used for single-account setups."""
    # Only use env var for single-account setups
    if len(ACCOUNTS) <= 1:
        env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if env_project_id:
            logging.info(f"Using project ID from GOOGLE_CLOUD_PROJECT: {env_project_id}")
            return env_project_id

    # Match project_id from the account selected by rotation
    if creds and creds.refresh_token:
        with file_lock:
            for acc in ACCOUNTS:
                if acc.get("refresh_token") == creds.refresh_token:
                    if acc.get("project_id"):
                        logging.info(f"Using project_id for this request: {acc['project_id']}")
                        return acc["project_id"]
                    break

    # Fallback: API discovery
    logging.warning("No project_id found in account data, attempting API discovery.")
    try:
        import requests as req
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }
        probe_payload = {"metadata": get_client_metadata()}
        resp = req.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(probe_payload),
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        discovered_project_id = data.get("cloudaicompanionProject")
        if discovered_project_id:
            logging.info(f"Discovered project ID via API: {discovered_project_id}")
            save_credentials(creds, discovered_project_id)
            _update_account_in_memory(creds, discovered_project_id)
            return discovered_project_id
        else:
            raise ValueError("No 'cloudaicompanionProject' in response.")
    except Exception as e:
        raise Exception(f"Failed to discover project ID: {e}")


def onboard_user(creds, project_id):
    """Ensures the user is onboarded."""
    global onboarding_complete_map
    if onboarding_complete_map.get(project_id):
        return

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    load_assist_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }

    try:
        import requests as req
        resp = req.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps(load_assist_payload),
            headers=headers,
        )
        resp.raise_for_status()
        load_data = resp.json()

        if load_data.get("currentTier"):
            onboarding_complete_map[project_id] = True
            return

        tier = None
        for allowed_tier in load_data.get("allowedTiers", []):
            if allowed_tier.get("isDefault"):
                tier = allowed_tier
                break

        if not tier:
            tier = {"id": "legacy-tier"}

        onboard_req_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }

        onboard_resp = req.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            data=json.dumps(onboard_req_payload),
            headers=headers,
        )
        onboard_resp.raise_for_status()
        onboarding_complete_map[project_id] = True

    except Exception as e:
        raise Exception(f"User onboarding failed for project {project_id}: {str(e)}")


def get_accounts_status_snapshot():
    """Returns credential status snapshot for the dashboard."""
    items = []
    now_utc = datetime.now(timezone.utc)

    with file_lock:
        for acc in ACCOUNTS:
            project_id = acc.get("project_id") or "unknown_project"
            expiry_raw = acc.get("expiry")
            expiry_iso = None
            is_expired = None

            if expiry_raw:
                try:
                    expiry_dt = datetime.fromisoformat(str(expiry_raw).replace("Z", "+00:00"))
                    if not expiry_dt.tzinfo:
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                    expiry_iso = expiry_dt.astimezone(timezone.utc).isoformat()
                    is_expired = expiry_dt <= now_utc
                except Exception:
                    expiry_iso = str(expiry_raw)

            items.append({
                "project_id": project_id,
                "has_refresh_token": bool(acc.get("refresh_token")),
                "has_access_token": bool(acc.get("token") or acc.get("access_token")),
                "expiry": expiry_iso,
                "is_expired": is_expired,
                "onboarding_complete": bool(onboarding_complete_map.get(project_id)),
            })

    return {
        "total_accounts": len(items),
        "accounts": items,
    }


def _manual_oauth_flow():
    """Initiates the manual OAuth flow if no credentials file is found."""
    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri="http://localhost:8989")
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")

    print(f"\n{'='*80}\nAUTHENTICATION REQUIRED\n{'='*80}")
    print(f"Please open this URL in your browser to log in:\n{auth_url}\n{'='*80}\n")
    logging.info(f"Please open this URL in your browser to log in: {auth_url}")

    server = HTTPServer(("", 8989), _OAuthCallbackHandler)
    server.handle_request()

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    try:
        flow.fetch_token(code=auth_code)
        new_creds = flow.credentials

        try:
            proj_id = get_user_project_id(new_creds)
        except Exception as e:
            proj_id = None
            logging.error(f"Could not discover project ID during initial login: {e}")

        creds_data = [{
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token": new_creds.token,
            "refresh_token": new_creds.refresh_token,
            "scopes": list(new_creds.scopes) if new_creds.scopes else [],
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry": new_creds.expiry.isoformat() if new_creds.expiry else None,
            "project_id": proj_id,
        }]

        with open(CREDENTIAL_FILE, "w") as f:
            json.dump(creds_data, f, indent=2)

        logging.info("Authentication successful! Credentials saved.")
        _load_accounts()
        return new_creds
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return None

def add_account_via_oauth() -> dict | None:
    """Runs the OAuth flow to add a NEW account to the existing credentials file."""
    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri="http://localhost:8989")
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")

    print(f"\n{'='*80}")
    print(f"ADD ACCOUNT - Open this URL in your browser:")
    print(f"{auth_url}")
    print(f"{'='*80}\n")
    logging.info(f"Add account URL: {auth_url}")

    _OAuthCallbackHandler.auth_code = None
    server = HTTPServer(("", 8989), _OAuthCallbackHandler)
    server.handle_request()

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    try:
        flow.fetch_token(code=auth_code)
        new_creds = flow.credentials

        with file_lock:
            for acc in ACCOUNTS:
                if acc.get("refresh_token") == new_creds.refresh_token:
                    logging.warning("This account is already registered.")
                    return acc

        try:
            proj_id = get_user_project_id(new_creds)
        except Exception as e:
            proj_id = None
            logging.error(f"Could not discover project ID: {e}")

        new_account = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token": new_creds.token,
            "refresh_token": new_creds.refresh_token,
            "scopes": list(new_creds.scopes) if new_creds.scopes else [],
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry": new_creds.expiry.isoformat() if new_creds.expiry else None,
            "project_id": proj_id,
        }

        with file_lock:
            try:
                with open(CREDENTIAL_FILE, "r") as f:
                    current = json.load(f)
                    if not isinstance(current, list):
                        current = [current]
            except (FileNotFoundError, json.JSONDecodeError):
                current = []

            current.append(new_account)

            with open(CREDENTIAL_FILE, "w") as f:
                json.dump(current, f, indent=2)

        _load_accounts()
        logging.info(f"Successfully added new account. Project: {proj_id}. Total accounts: {len(ACCOUNTS)}")
        return new_account

    except Exception as e:
        logging.error(f"Failed to add account: {e}")
        return None