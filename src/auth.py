import os
import json
import base64
import time
import logging
import threading
import httpx

from datetime import datetime, timezone

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBasic
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest

from .utils import get_user_agent, get_client_metadata, GEMINI_API_CLIENT_HEADER
from .config import (
    CLIENT_ID, CLIENT_SECRET, SCOPES, CREDENTIAL_FILE,
    CODE_ASSIST_ENDPOINT, GEMINI_AUTH_PASSWORD
)

CLOUD_AI_SERVICE = "cloudaicompanion.googleapis.com"
SERVICE_USAGE_URL = "https://serviceusage.googleapis.com"

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
    """Gets next available credentials with failover. Skips unusable accounts."""
    if not ACCOUNTS:
        return _manual_oauth_flow() if allow_oauth_flow else None

    attempts = len(ACCOUNTS)

    for attempt in range(attempts):
        selected_account = _get_next_account()
        if selected_account is None:
            break

        project_id = selected_account.get("project_id", "unknown")

        # Skip accounts marked as unusable
        onboard_info = onboarding_complete_map.get(project_id, {})
        if isinstance(onboard_info, dict) and onboard_info.get("unusable"):
            logging.debug(
                f"Skipping unusable account {project_id}: "
                f"{onboard_info.get('status_message', '')}"
            )
            continue

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
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
        }
        probe_payload = {"metadata": get_client_metadata()}
        resp = httpx.post(
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

def _detect_tier(load_data: dict) -> str:
    """
    Detect account tier from loadCodeAssist response.
    Returns: 'paid', 'free', 'workspace', 'legacy', or 'unknown'
    """
    current_tier = load_data.get("currentTier", {})
    if isinstance(current_tier, dict):
        tier_id = (current_tier.get("id") or "").lower().strip()
    else:
        tier_id = ""

    if not tier_id:
        for tier in load_data.get("allowedTiers", []):
            if tier.get("isDefault"):
                tier_id = (tier.get("id") or "").lower().strip()
                break

    if "standard" in tier_id:
        return "paid"
    elif "free" in tier_id:
        return "free"
    elif "legacy" in tier_id:
        return "legacy"

    return "unknown"


def _is_free_tier(tier_id: str, load_data: dict) -> bool:
    """Determine if this is a free-tier account based on actual tier detection."""
    tier = _detect_tier(load_data)
    return tier in ("free", "legacy")

def onboard_user(creds, project_id):
    """
    Ensures the user is onboarded. Detects unusable accounts
    (missing license, warnings) and marks them accordingly.
    """
    global onboarding_complete_map

    existing = onboarding_complete_map.get(project_id)
    if existing:
        if isinstance(existing, dict):
            if existing.get("complete"):
                return
            if existing.get("unusable"):
                raise Exception(
                    f"Account {project_id} is unusable: "
                    f"{existing.get('status_message', 'unknown reason')}"
                )
        elif isinstance(existing, bool) and existing:
            return

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    metadata = {
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }

    try:
        # Step 1: loadCodeAssist
        load_payload = {
            "cloudaicompanionProject": project_id,
            "metadata": metadata,
        }

        resp = httpx.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json=load_payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        load_data = resp.json()

        tier = _detect_tier(load_data)
        logging.info(f"Account tier for {project_id}: {tier}")

        # Already onboarded
        if load_data.get("currentTier"):
            onboarding_complete_map[project_id] = {
                "complete": True,
                "tier": tier,
            }
            logging.info(f"Already onboarded: {project_id} (tier: {tier})")
            return

        # Find default tier ID
        tier_id = "legacy-tier"
        for allowed_tier in load_data.get("allowedTiers", []):
            if allowed_tier.get("isDefault"):
                tier_id = allowed_tier.get("id", tier_id)
                break

        # Step 2: onboardUser — poll until done
        onboard_payload = {
            "tierId": tier_id,
            "cloudaicompanionProject": project_id,
            "metadata": metadata,
        }

        max_polls = 6
        for poll in range(max_polls):
            resp = httpx.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                json=onboard_payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            onboard_data = resp.json()

            if onboard_data.get("done", False):
                response_data = onboard_data.get("response", {})

                logging.info(
                    f"Onboarding response for {project_id}: "
                    f"raw_response={json.dumps(response_data)}"
                )

                # Check for license/subscription warnings
                onboard_status = response_data.get("status", {})
                status_code = onboard_status.get("statusCode", "").upper()
                status_message = onboard_status.get("displayMessage", "")
                status_title = onboard_status.get("messageTitle", "")

                if _is_unusable_account(status_code, status_message, status_title):
                    logging.warning(
                        f"Account {project_id} is UNUSABLE: "
                        f"{status_title} — {status_message}"
                    )
                    onboarding_complete_map[project_id] = {
                        "complete": False,
                        "unusable": True,
                        "tier": tier,
                        "status_message": status_message or status_title,
                    }
                    raise Exception(
                        f"Account {project_id} unusable: {status_message or status_title}"
                    )

                # Check for project remapping
                response_project = _extract_response_project(onboard_data)
                if response_project and response_project != project_id:
                    if _is_free_tier(tier_id, load_data):
                        logging.info(
                            f"Free-tier project remapping: {project_id} → {response_project}"
                        )
                        _remap_project_id(creds, project_id, response_project)
                        project_id = response_project
                    elif project_id.startswith("gen-lang-client-"):
                        logging.info(
                            f"Auto-generated project remapping: "
                            f"{project_id} → {response_project}"
                        )
                        _remap_project_id(creds, project_id, response_project)
                        project_id = response_project
                    else:
                        logging.info(
                            f"Pro account with custom project: keeping {project_id} "
                            f"(Google suggested {response_project}, ignoring)"
                        )

                onboarding_complete_map[project_id] = {
                    "complete": True,
                    "tier": tier,
                }
                logging.info(f"Onboarding complete for project: {project_id} (tier: {tier})")

                _ensure_cloud_ai_enabled(creds, project_id)
                return

            logging.info(
                f"Onboarding in progress for {project_id}, "
                f"waiting... ({poll + 1}/{max_polls})"
            )
            import time
            time.sleep(5)

        logging.warning(
            f"Onboarding polling timed out for {project_id}, proceeding anyway."
        )
        onboarding_complete_map[project_id] = {
            "complete": True,
            "tier": tier,
        }

    except httpx.HTTPStatusError as e:
        raise Exception(
            f"Onboarding failed for project {project_id}: "
            f"{e.response.status_code} {e.response.text}"
        )
    except Exception as e:
        # Re-raise so the retry loop catches it and skips this account
        raise


def _is_unusable_account(status_code: str, message: str, title: str) -> bool:
    """
    Detect if an onboarding response indicates the account can't use the API.
    Catches missing licenses, subscription requirements, etc.
    """
    if status_code == "WARNING":
        combined = (message + " " + title).lower()
        unusable_signals = [
            "missing a valid license",
            "subscription needed",
            "purchase or assign a license",
            "not authorized",
            "access denied",
            "billing",
        ]
        return any(signal in combined for signal in unusable_signals)

    if status_code in ("ERROR", "DENIED", "BLOCKED"):
        return True

    return False

def _extract_response_project(onboard_data: dict) -> str | None:
    """Extract project ID from onboardUser response, handling both string and dict formats."""
    response = onboard_data.get("response", {})
    project = response.get("    ")

    if isinstance(project, str):
        return project.strip() or None
    elif isinstance(project, dict):
        return (project.get("id") or "").strip() or None

    return None


def _remap_project_id(creds, old_project: str, new_project: str):
    """
    Update the account's project ID in memory and on disk when Google
    remaps a free-tier project to a backend project.
    """
    with _account_lock:
        for acc in ACCOUNTS:
            if acc.get("refresh_token") == creds.refresh_token:
                acc["project_id"] = new_project
                logging.info(f"Remapped in-memory project: {old_project} → {new_project}")
                break

    save_credentials(creds, new_project)


def _ensure_cloud_ai_enabled(creds, project_id: str):
    """
    Check if cloudaicompanion.googleapis.com is enabled.
    Auto-enable it if not.
    """
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    check_url = (
        f"{SERVICE_USAGE_URL}/v1/projects/{project_id}"
        f"/services/{CLOUD_AI_SERVICE}"
    )

    try:
        resp = httpx.get(check_url, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("state") == "ENABLED":
                logging.info(f"Cloud AI API already enabled for {project_id}")
                return

        # Try to enable it
        enable_url = (
            f"{SERVICE_USAGE_URL}/v1/projects/{project_id}"
            f"/services/{CLOUD_AI_SERVICE}:enable"
        )
        resp = httpx.post(
            enable_url,
            json={},
            headers=headers,
            timeout=30,
        )

        if resp.status_code in (200, 201):
            logging.info(f"Cloud AI API enabled for {project_id}")
        elif resp.status_code == 400:
            body = resp.text.lower()
            if "already enabled" in body:
                logging.info(f"Cloud AI API already enabled for {project_id}")
            else:
                logging.warning(f"Could not enable Cloud AI API for {project_id}: {resp.text}")
        else:
            logging.warning(
                f"Could not enable Cloud AI API for {project_id}: "
                f"{resp.status_code} {resp.text}"
            )

    except Exception as e:
        logging.warning(f"Cloud AI API check failed for {project_id}: {e}")

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
                    expiry_dt = datetime.fromisoformat(
                        str(expiry_raw).replace("Z", "+00:00")
                    )
                    if not expiry_dt.tzinfo:
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                    expiry_iso = expiry_dt.astimezone(timezone.utc).isoformat()
                    is_expired = expiry_dt <= now_utc
                except Exception:
                    expiry_iso = str(expiry_raw)

            onboard_info = onboarding_complete_map.get(project_id, {})
            if isinstance(onboard_info, bool):
                onboard_info = {"complete": onboard_info, "tier": "unknown"}

            items.append({
                "project_id": project_id,
                "has_refresh_token": bool(acc.get("refresh_token")),
                "has_access_token": bool(acc.get("token") or acc.get("access_token")),
                "expiry": expiry_iso,
                "is_expired": is_expired,
                "onboarding_complete": onboard_info.get("complete", False),
                "tier": onboard_info.get("tier", "unknown"),
                "unusable": onboard_info.get("unusable", False),
                "status_message": onboard_info.get("status_message", ""),
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
            "token_type": "Bearer",
            "id_token": getattr(new_creds, "id_token", None),
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
            "token_type": "Bearer",
            "id_token": getattr(new_creds, "id_token", None),
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
