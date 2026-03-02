"""
Google API Client - Fully async using httpx.
No more threading gymnastics for streaming.
"""
import json
import logging
import datetime
import threading
import time
import asyncio
import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from .auth import get_credentials, save_credentials, get_user_project_id, onboard_user, ACCOUNTS
from .utils import (
    get_user_agent,
    sanitize_historical_signatures,
    apply_scorched_earth_thinking_config,
    clamp_top_k,
    scrub_headers,
    parse_retry_delay,
    get_model_fallback_order,
    GEMINI_API_CLIENT_HEADER
)
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts,
)

# --- Timeouts ---
CONNECT_TIMEOUT = 10
READ_TIMEOUT_STREAM = 300
READ_TIMEOUT_NORMAL = 120
KEEPALIVE_INTERVAL = 15

# --- Shared async client (connection pooling) ---
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Lazy-init a shared async client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT_STREAM,
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
            follow_redirects=True,
        )
    return _http_client

# --- Stats ---
_stats_lock = threading.Lock()
_daily_stats = {
    "last_reset_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "total_success": 0,
    "total_fail": 0,
    "total": 0,
    "accounts": {},
}


# ===========================================================================
#  Public API
# ===========================================================================

async def send_gemini_request(
    payload: dict,
    is_streaming: bool = False,
    disconnect_event: asyncio.Event = None,
) -> Response:
    """
    Send a request to Google's Gemini API with retry, account rotation,
    and parsed retry delays from 429 responses.
    """
    num_accounts = len(ACCOUNTS) if ACCOUNTS else 1
    max_retries = num_accounts * 2

    for attempt in range(max_retries):
        if disconnect_event and disconnect_event.is_set():
            logging.info("Client disconnected, aborting retry loop.")
            return _error_response("Client disconnected.", 499)

        creds = await asyncio.to_thread(get_credentials)
        if not creds or not creds.token:
            return _error_response("No valid credentials available.", 500)

        proj_id = await asyncio.to_thread(get_user_project_id, creds)
        if not proj_id:
            return _error_response("Failed to get user project ID.", 500)

        await asyncio.to_thread(onboard_user, creds, proj_id)

        response = await _send_single_request(
            payload, proj_id, creds, is_streaming, disconnect_event
        )

        status = getattr(response, "status_code", None)
        if status == 429 and attempt < max_retries - 1:
            record_usage(proj_id, False)

            # Try to parse Google's retry delay from the response body
            delay = None
            try:
                body = getattr(response, "body", None)
                if body:
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", "ignore")
                    delay = parse_retry_delay(body)
            except Exception:
                pass

            if delay is not None and delay > 0:
                # Use Google's suggested delay, but cap at 30s
                delay = min(delay, 30.0)
                logging.warning(
                    f"429 on project {proj_id}. "
                    f"Google says retry after {delay:.2f}s. "
                    f"Retry {attempt + 1}/{max_retries}..."
                )
            elif attempt >= num_accounts:
                # Second pass
                delay = min(2 ** (attempt - num_accounts), 10)
                logging.warning(
                    f"429 on project {proj_id}. "
                    f"Retry {attempt + 1}/{max_retries}, backoff {delay}s..."
                )
            else:
                # First pass
                delay = 0
                logging.warning(
                    f"429 on project {proj_id}. "
                    f"Trying next account ({attempt + 1}/{max_retries})..."
                )

            if delay and delay > 0:
                try:
                    await asyncio.wait_for(
                        _wait_for_disconnect(disconnect_event),
                        timeout=delay,
                    )
                    logging.info("Client disconnected during retry backoff.")
                    return _error_response("Client disconnected.", 499)
                except asyncio.TimeoutError:
                    pass

            continue

        return response

    return _error_response("All retry attempts exhausted (429).", 429)

async def _wait_for_disconnect(event: asyncio.Event | None):
    """Wait until client disconnects. If no event, wait forever."""
    if event is None:
        await asyncio.Future()  # never resolves
    else:
        await event.wait()


# ===========================================================================
#  Single-request logic
# ===========================================================================

async def _send_single_request(
    payload: dict,
    proj_id: str,
    creds,
    is_streaming: bool,
    disconnect_event: asyncio.Event = None,
) -> Response:
    """Execute one HTTP request against the Gemini API. Fully async."""

    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {}),
    }

    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    model_name = payload.get("model", "unknown")

    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(model_name),
        "X-Goog-Api-Client": GEMINI_API_CLIENT_HEADER,
        "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
        "Accept": "text/event-stream" if is_streaming else "application/json",
    }

    post_data = json.dumps(final_payload)
    client = _get_client()

    try:
        if is_streaming:
            req = client.build_request(
                "POST", target_url, content=post_data, headers=request_headers
            )
            resp = await client.send(req, stream=True)
            record_usage(proj_id, resp.status_code == 200)
            return _handle_streaming_response(resp, disconnect_event)

        else:
            non_stream_timeout = httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT_NORMAL,
                write=10.0,
                pool=10.0,
            )
            resp = await client.post(
                target_url,
                content=post_data,
                headers=request_headers,
                timeout=non_stream_timeout,
            )
            record_usage(proj_id, resp.status_code == 200)
            return _handle_non_streaming_response(resp)

    except httpx.TimeoutException:
        record_usage(proj_id, False)
        logging.error(f"Request to Google API timed out for project {proj_id}")
        return _error_response("Request to Google API timed out.", 504)

    except httpx.RequestError as e:
        record_usage(proj_id, False)
        logging.error(f"Request to Google API failed: {e}")
        return _error_response(f"Request failed: {e}", 502)

    except Exception as e:
        record_usage(proj_id, False)
        logging.error(f"Unexpected error during Google API request: {e}")
        return _error_response(f"Unexpected error: {e}", 500)


# ===========================================================================
#  Streaming response — pure async, no threads, no queues
# ===========================================================================

def _handle_streaming_response(
    resp: httpx.Response,
    disconnect_event: asyncio.Event = None,
) -> StreamingResponse:
    """Handle streaming response. Clean async iteration."""

    if resp.status_code != 200:
        error_message = f"Google API error: {resp.status_code}"
        try:
            body = resp.read()  # httpx sync read for error body
            error_data = json.loads(body)
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except Exception:
            pass

        async def error_generator():
            await resp.aclose()
            error_body = {
                "error": {
                    "message": error_message,
                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                    "code": resp.status_code,
                }
            }
            yield f"event: error\ndata: {json.dumps(error_body)}\n\n".encode("utf-8")

        logging.error(f"Google API returned status {resp.status_code}: {error_message}")

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers=_sse_headers(),
            status_code=resp.status_code,
        )

    async def stream_generator():
        last_data_time = asyncio.get_event_loop().time()

        try:
            async for line in resp.aiter_lines():
                # Check disconnect
                if disconnect_event and disconnect_event.is_set():
                    logging.info("Client disconnected, stopping stream.")
                    break

                if not line:
                    # Empty line — check if we need keep-alive
                    now = asyncio.get_event_loop().time()
                    if now - last_data_time > KEEPALIVE_INTERVAL:
                        yield ": keep-alive\n\n".encode("utf-8")
                        last_data_time = now
                    continue

                last_data_time = asyncio.get_event_loop().time()

                if line.startswith("data: "):
                    line = line[len("data: "):]

                    try:
                        obj = json.loads(line)

                        if "response" in obj:
                            response_chunk = obj["response"]
                            response_json = json.dumps(response_chunk, separators=(",", ":"))
                            yield f"data: {response_json}\n\n".encode("utf-8")
                        else:
                            yield f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")

                    except json.JSONDecodeError:
                        continue

        except httpx.StreamClosed:
            logging.info("Upstream stream closed.")
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
        finally:
            await resp.aclose()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ===========================================================================
#  Non-streaming response
# ===========================================================================

def _handle_non_streaming_response(resp: httpx.Response) -> Response:
    """Handle non-streaming response."""
    if resp.status_code == 200:
        try:
            text = resp.text
            if text.startswith("data: "):
                text = text[len("data: "):]
            parsed = json.loads(text)
            standard_response = parsed.get("response")
            return Response(
                content=json.dumps(standard_response),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse Google API response: {e}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
    else:
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": error_message,
                            "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                            "code": resp.status_code,
                        }
                    }),
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
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
        native_request["contents"] = sanitize_historical_signatures(native_request["contents"])

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
        "X-Accel-Buffering": "no",
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
