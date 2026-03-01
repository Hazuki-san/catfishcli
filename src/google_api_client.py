"""
Google API Client - Handles all communication with Google's Gemini API.
Includes retry logic, disconnect detection, keep-alive, and session management.
"""
import json
import logging
import datetime
import threading
import time
import requests
import asyncio
from fastapi import Response
from fastapi.responses import StreamingResponse

from .auth import get_credentials, save_credentials, get_user_project_id, onboard_user, ACCOUNTS
from .utils import (
    get_user_agent,
    sanitize_historical_signatures,
    apply_scorched_earth_thinking_config,
    clamp_top_k
)
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts
)

# --- Timeouts ---
CONNECT_TIMEOUT = 10       # seconds to establish connection
READ_TIMEOUT_STREAM = 300  # 5 min for streaming (long thinking models)
READ_TIMEOUT_NORMAL = 120  # 2 min for non-streaming

# --- Keep-alive interval ---
KEEPALIVE_INTERVAL = 15  # seconds between keep-alive pings during streaming

# --- Stats ---
_stats_lock = threading.Lock()
_daily_stats = {
    "last_reset_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "total_success": 0,
    "total_fail": 0,
    "total": 0,
    "accounts": {}
}


# ===========================================================================
#  Public API
# ===========================================================================

def send_gemini_request(
    payload: dict,
    is_streaming: bool = False,
    disconnect_event: threading.Event = None,
) -> Response:
    """
    Send a request to Google's Gemini API with automatic retry and account rotation.

    Args:
        payload: The request payload in Gemini format.
        is_streaming: Whether this is a streaming request.
        disconnect_event: Optional event that signals client disconnect.

    Returns:
        FastAPI Response object.
    """
    num_accounts = len(ACCOUNTS) if ACCOUNTS else 1
    max_retries = num_accounts * 2  # Two full passes through all accounts

    for attempt in range(max_retries):
        # Bail early if client is gone
        if disconnect_event and disconnect_event.is_set():
            logging.info("Client disconnected, aborting retry loop.")
            return _error_response("Client disconnected.", 499)

        # --- Get credentials (with built-in failover) ---
        creds = get_credentials()
        if not creds or not creds.token:
            return _error_response(
                "No valid credentials available. Please check accounts and restart if needed.",
                500,
            )

        proj_id = get_user_project_id(creds)
        if not proj_id:
            return _error_response("Failed to get user project ID.", 500)

        onboard_user(creds, proj_id)

        # --- Send single attempt ---
        response = _send_single_request(
            payload, proj_id, creds, is_streaming, disconnect_event
        )

        # --- Retry on 429 ---
        status = getattr(response, "status_code", None)
        if status == 429 and attempt < max_retries - 1:
            record_usage(proj_id, False)

            if attempt >= num_accounts:
                # Second pass — exponential backoff, capped at 10 s
                delay = min(2 ** (attempt - num_accounts), 10)
                logging.warning(
                    f"429 on project {proj_id}. "
                    f"Retry {attempt + 1}/{max_retries}, waiting {delay}s..."
                )
                # Interruptible sleep so we stop if client disconnects
                for _ in range(int(delay * 10)):
                    if disconnect_event and disconnect_event.is_set():
                        logging.info("Client disconnected during retry backoff.")
                        return _error_response("Client disconnected.", 499)
                    time.sleep(0.1)
            else:
                # First pass — instant switch to next account
                logging.warning(
                    f"429 on project {proj_id}. "
                    f"Trying next account ({attempt + 1}/{max_retries})..."
                )
            continue

        return response

    return _error_response("All retry attempts exhausted (429).", 429)


# ===========================================================================
#  Single-request logic (no retry here)
# ===========================================================================

def _send_single_request(
    payload: dict,
    proj_id: str,
    creds,
    is_streaming: bool,
    disconnect_event: threading.Event = None,
) -> Response:
    """Execute one HTTP request against the Gemini API."""

    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {}),
    }

    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/22.17.0",
        "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
    }

    session = requests.Session()

    try:
        if is_streaming:
            resp = session.post(
                target_url,
                data=json.dumps(final_payload),
                headers=request_headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_STREAM),
            )
            record_usage(proj_id, resp.status_code == 200)
            return _handle_streaming_response(resp, session, disconnect_event)
        else:
            resp = session.post(
                target_url,
                data=json.dumps(final_payload),
                headers=request_headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_NORMAL),
            )
            record_usage(proj_id, resp.status_code == 200)
            result = _handle_non_streaming_response(resp)
            session.close()
            return result

    except requests.exceptions.Timeout:
        record_usage(proj_id, False)
        session.close()
        logging.error(f"Request to Google API timed out for project {proj_id}")
        return _error_response("Request to Google API timed out.", 504)

    except requests.exceptions.RequestException as e:
        record_usage(proj_id, False)
        session.close()
        logging.error(f"Request to Google API failed: {str(e)}")
        return _error_response(f"Request failed: {str(e)}", 502)

    except Exception as e:
        record_usage(proj_id, False)
        session.close()
        logging.error(f"Unexpected error during Google API request: {str(e)}")
        return _error_response(f"Unexpected error: {str(e)}", 500)


# ===========================================================================
#  Streaming response handler (with keep-alive + disconnect detection)
# ===========================================================================

def _handle_streaming_response(
    resp,
    session: requests.Session,
    disconnect_event: threading.Event = None,
) -> StreamingResponse:
    """Handle streaming response with keep-alive pings and disconnect detection."""

    # --- HTTP error before streaming starts ---
    if resp.status_code != 200:
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        error_message = f"Google API error: {resp.status_code}"
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except Exception:
            pass

        session.close()

        async def error_generator():
            error_response = {
                "error": {
                    "message": error_message,
                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                    "code": resp.status_code,
                }
            }
            yield f"event: error\ndata: {json.dumps(error_response)}\n\n".encode("utf-8")

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers=_sse_headers(),
            status_code=resp.status_code,
        )

    # --- Success: stream with keep-alive via background thread + asyncio.Queue ---
    async def stream_generator():
        loop = asyncio.get_event_loop()
        chunk_queue: asyncio.Queue = asyncio.Queue()

        def _read_upstream():
            """Runs in a daemon thread — reads from Google and puts chunks into the queue."""
            try:
                with resp:
                    for line in resp.iter_lines():
                        if disconnect_event and disconnect_event.is_set():
                            break
                        if line:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, line)
                # Sentinel: stream finished normally
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
            finally:
                session.close()

        reader = threading.Thread(target=_read_upstream, daemon=True)
        reader.start()

        try:
            while True:
                # Check disconnect
                if disconnect_event and disconnect_event.is_set():
                    logging.info("Client disconnected, stopping stream.")
                    break

                try:
                    item = await asyncio.wait_for(
                        chunk_queue.get(), timeout=KEEPALIVE_INTERVAL
                    )
                except asyncio.TimeoutError:
                    # No data for KEEPALIVE_INTERVAL seconds — send keep-alive comment
                    yield ": keep-alive\n\n".encode("utf-8")
                    continue

                # Sentinel — upstream finished
                if item is None:
                    break

                # Exception from reader thread
                if isinstance(item, Exception):
                    logging.error(f"Upstream read error: {item}")
                    error_body = {
                        "error": {
                            "message": f"Upstream error: {item}",
                            "type": "api_error",
                            "code": 502,
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_body)}\n\n".encode(
                        "utf-8"
                    )
                    break

                # Normal chunk processing
                if not isinstance(item, str):
                    item = item.decode("utf-8", "ignore")

                if item.startswith("data: "):
                    item = item[len("data: "):]

                    try:
                        obj = json.loads(item)

                        if "response" in obj:
                            response_chunk = obj["response"]
                            response_json = json.dumps(
                                response_chunk, separators=(",", ":")
                            )
                            yield f"data: {response_json}\n\n".encode("utf-8", "ignore")
                        else:
                            yield f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode(
                                "utf-8", "ignore"
                            )

                        await asyncio.sleep(0)
                    except json.JSONDecodeError:
                        continue

        except asyncio.CancelledError:
            logging.info("Stream generator cancelled.")
        except Exception as e:
            logging.error(f"Unexpected error during streaming: {e}")
            error_body = {
                "error": {
                    "message": f"Unexpected error: {e}",
                    "type": "api_error",
                    "code": 500,
                }
            }
            yield f"event: error\ndata: {json.dumps(error_body)}\n\n".encode("utf-8")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ===========================================================================
#  Non-streaming response handler
# ===========================================================================

def _handle_non_streaming_response(resp) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            google_api_response = resp.text
            if google_api_response.startswith("data: "):
                google_api_response = google_api_response[len("data: "):]
            google_api_response = json.loads(google_api_response)
            standard_gemini_response = google_api_response.get("response")
            return Response(
                content=json.dumps(standard_gemini_response),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse Google API response: {str(e)}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type"),
            )
    else:
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")

        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get(
                    "message", f"API error: {resp.status_code}"
                )
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error"
                        if resp.status_code == 404
                        else "api_error",
                        "code": resp.status_code,
                    }
                }
                return Response(
                    content=json.dumps(error_response),
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type"),
        )


# ===========================================================================
#  Payload builders
# ===========================================================================

def build_gemini_payload_from_openai(openai_payload: dict, raw_openai_request: dict = None) -> dict:
    """Build a Gemini API payload from an OpenAI-transformed request."""
    model = openai_payload.get("model")
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)

    contents = openai_payload.get("contents")
    if contents:
        contents = sanitize_historical_signatures(contents)

    generation_config = openai_payload.get("generationConfig", {})

    reasoning_effort = None
    if raw_openai_request and "reasoning_effort" in raw_openai_request:
        reasoning_effort = raw_openai_request.get("reasoning_effort")

    generation_config = apply_scorched_earth_thinking_config(
        generation_config=generation_config,
        openai_reasoning_effort=reasoning_effort,
    )

    request_data = {
        "contents": contents,
        "systemInstruction": openai_payload.get("systemInstruction"),
        "cachedContent": openai_payload.get("cachedContent"),
        "tools": openai_payload.get("tools"),
        "toolConfig": openai_payload.get("toolConfig"),
        "safetySettings": safety_settings,
        "generationConfig": generation_config,
    }

    request_data = {k: v for k, v in request_data.items() if v is not None}

    # Clamp topK
    if "generationConfig" in request_data:
        request_data["generationConfig"] = clamp_top_k(request_data["generationConfig"])

    return {
        "model": model,
        "request": request_data,
    }


def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """Build a Gemini API payload from a native Gemini request."""
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}

    if "contents" in native_request:
        native_request["contents"] = sanitize_historical_signatures(
            native_request["contents"]
        )

    if "gemini-2.5-flash-image" not in model_from_path:
        thinking_budget = get_thinking_budget(model_from_path)
        include_thoughts = should_include_thoughts(model_from_path)

        native_request["generationConfig"] = apply_scorched_earth_thinking_config(
            generation_config=native_request["generationConfig"],
            fallback_budget=thinking_budget,
            fallback_include=include_thoughts,
        )

    if is_search_model(model_from_path):
        if "tools" not in native_request:
            native_request["tools"] = []
        if not any(tool.get("googleSearch") for tool in native_request["tools"]):
            native_request["tools"].append({"googleSearch": {}})

    # Clamp topK
    native_request["generationConfig"] = clamp_top_k(native_request["generationConfig"])

    return {
        "model": get_base_model_name(model_from_path),
        "request": native_request,
    }


# ===========================================================================
#  Helpers
# ===========================================================================

def _sse_headers() -> dict:
    """Standard SSE response headers."""
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Disposition": "attachment",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Vary": "Origin, X-Origin, Referer",
        "X-XSS-Protection": "0",
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Server": "ESF",
    }


def _error_response(message: str, status_code: int) -> Response:
    """Build a JSON error response."""
    return Response(
        content=json.dumps({"error": {"message": message, "code": status_code}}),
        status_code=status_code,
        media_type="application/json",
    )


# ===========================================================================
#  Usage stats
# ===========================================================================

def record_usage(project_id: str, success: bool):
    """Safely increments the usage statistics."""
    with _stats_lock:
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

        if _daily_stats["last_reset_date"] != current_date:
            _daily_stats["last_reset_date"] = current_date
            _daily_stats["total_success"] = 0
            _daily_stats["total_fail"] = 0
            _daily_stats["total"] = 0
            _daily_stats["accounts"] = {}

        _daily_stats["total"] += 1

        if project_id not in _daily_stats["accounts"]:
            _daily_stats["accounts"][project_id] = {"success": 0, "fail": 0, "total": 0}

        _daily_stats["accounts"][project_id]["total"] += 1

        if success:
            _daily_stats["total_success"] += 1
            _daily_stats["accounts"][project_id]["success"] += 1
        else:
            _daily_stats["total_fail"] += 1
            _daily_stats["accounts"][project_id]["fail"] += 1


def get_usage_stats_snapshot() -> dict:
    """Returns a snapshot of the current usage stats."""
    with _stats_lock:
        accounts_list = [
            {"project_id": pid, **stats}
            for pid, stats in _daily_stats["accounts"].items()
        ]
        return {
            "last_reset_date": _daily_stats["last_reset_date"],
            "total_success": _daily_stats["total_success"],
            "total_fail": _daily_stats["total_fail"],
            "total": _daily_stats["total"],
            "accounts": accounts_list,
        }


def get_formatted_stats() -> dict:
    """Returns the stats formatted for the root endpoint."""
    return get_usage_stats_snapshot()
