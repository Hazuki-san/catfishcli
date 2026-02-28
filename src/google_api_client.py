"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import json
import logging
import datetime
import threading
import requests
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest

from .auth import get_credentials, save_credentials, get_user_project_id, onboard_user, ACCOUNTS
from .utils import (
    get_user_agent,
    sanitize_historical_signatures,
    apply_scorched_earth_thinking_config
)
from .config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    is_search_model,
    get_thinking_budget,
    should_include_thoughts
)
import asyncio

_stats_lock = threading.Lock()
_daily_stats = {
    "last_reset_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "total_success": 0,
    "total_fail": 0,
    "total": 0,
    "accounts": {}
}

def send_gemini_request(payload: dict, is_streaming: bool = False) -> Response:
    """
    Send a request to Google's Gemini API.
    
    Args:
        payload: The request payload in Gemini format
        is_streaming: Whether this is a streaming request
        
    Returns:
        FastAPI Response object
    """
    max_retries = min(len(ACCOUNTS), 3) if ACCOUNTS else 1

    for attempt in range(max_retries):
        creds = get_credentials()
        if not creds or not creds.token:
            return Response(
                content=json.dumps({"error": {"message": "No valid credentials available."}}),
                status_code=500,
                media_type="application/json",
            )

        proj_id = get_user_project_id(creds)
        if not proj_id:
            return Response(content="Failed to get user project ID.", status_code=500)

        onboard_user(creds, proj_id)

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

        try:
            resp = requests.post(
                target_url,
                data=json.dumps(final_payload),
                headers=request_headers,
                stream=is_streaming,
            )

            # Retry on 429 with next account
            if resp.status_code == 429 and attempt < max_retries - 1:
                record_usage(proj_id, False)
                logging.warning(
                    f"429 rate limited on project {proj_id}. "
                    f"Retrying with next account ({attempt + 1}/{max_retries})..."
                )
                continue

            record_usage(proj_id, resp.status_code == 200)

            if is_streaming:
                return _handle_streaming_response(resp)
            else:
                return _handle_non_streaming_response(resp)

        except requests.exceptions.RequestException as e:
            record_usage(proj_id, False)
            if attempt < max_retries - 1:
                logging.warning(f"Request failed: {e}. Trying next account...")
                continue
            return Response(
                content=json.dumps({"error": {"message": f"Request failed: {e}"}}),
                status_code=500,
                media_type="application/json",
            )

    # Should never reach here, but just in case
    return Response(
        content=json.dumps({"error": {"message": "All accounts exhausted."}}),
        status_code=429,
        media_type="application/json",
    )

def _handle_streaming_response(resp) -> StreamingResponse:
    """Handle streaming response from Google API."""
    
    # Check for HTTP errors before starting to stream
    if resp.status_code != 200:
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        error_message = f"Google API error: {resp.status_code}"
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except:
            pass
        
        # Return error as a streaming response
        async def error_generator():
            error_response = {
                "error": {
                    "message": error_message,
                    "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                    "code": resp.status_code
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8')
        
        response_headers = {
            "Content-Type": "text/event-stream",
            "Content-Disposition": "attachment",
            "Vary": "Origin, X-Origin, Referer",
            "X-XSS-Protection": "0",
            "X-Frame-Options": "SAMEORIGIN",
            "X-Content-Type-Options": "nosniff",
            "Server": "ESF"
        }
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers=response_headers,
            status_code=resp.status_code
        )
    
    async def stream_generator():
        try:
            with resp:
                for chunk in resp.iter_lines():
                    if chunk:
                        if not isinstance(chunk, str):
                            chunk = chunk.decode('utf-8', "ignore")
                            
                        if chunk.startswith('data: '):
                            chunk = chunk[len('data: '):]
                            
                            try:
                                obj = json.loads(chunk)
                                
                                if "response" in obj:
                                    response_chunk = obj["response"]
                                    response_json = json.dumps(response_chunk, separators=(',', ':'))
                                    response_line = f"data: {response_json}\n\n"
                                    yield response_line.encode('utf-8', "ignore")
                                    await asyncio.sleep(0)
                                else:
                                    obj_json = json.dumps(obj, separators=(',', ':'))
                                    yield f"data: {obj_json}\n\n".encode('utf-8', "ignore")
                            except json.JSONDecodeError:
                                continue
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Streaming request failed: {str(e)}")
            error_response = {
                "error": {
                    "message": f"Upstream request failed: {str(e)}",
                    "type": "api_error",
                    "code": 502
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")
        except Exception as e:
            logging.error(f"Unexpected error during streaming: {str(e)}")
            error_response = {
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "api_error",
                    "code": 500
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8', "ignore")

    response_headers = {
        "Content-Type": "text/event-stream",
        "Content-Disposition": "attachment",
        "Vary": "Origin, X-Origin, Referer",
        "X-XSS-Protection": "0",
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Server": "ESF"
    }
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers=response_headers
    )


def _handle_non_streaming_response(resp) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            google_api_response = resp.text
            if google_api_response.startswith('data: '):
                google_api_response = google_api_response[len('data: '):]
            google_api_response = json.loads(google_api_response)
            standard_gemini_response = google_api_response.get("response")
            return Response(
                content=json.dumps(standard_gemini_response),
                status_code=200,
                media_type="application/json; charset=utf-8"
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse Google API response: {str(e)}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type")
            )
    else:
        # Log the error details
        logging.error(f"Google API returned status {resp.status_code}: {resp.text}")
        
        # Try to parse error response and provide meaningful error message
        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", f"API error: {resp.status_code}")
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "invalid_request_error" if resp.status_code == 404 else "api_error",
                        "code": resp.status_code
                    }
                }
                return Response(
                    content=json.dumps(error_response),
                    status_code=resp.status_code,
                    media_type="application/json"
                )
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback to original response if we can't parse the error
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type")
        )


def build_gemini_payload_from_openai(openai_payload: dict, raw_openai_request: dict = None) -> dict:
    """
    Build a Gemini API payload from an OpenAI-transformed request.
    This is used when OpenAI requests are converted to Gemini format.
    """
    model = openai_payload.get("model")
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)
    
    # Sanitize historical messages for the signature bypass
    contents = openai_payload.get("contents")
    if contents:
        contents = sanitize_historical_signatures(contents)
        
    generation_config = openai_payload.get("generationConfig", {})
    
    # Extract reasoning_effort if the proxy captured the original raw request
    reasoning_effort = None
    if raw_openai_request and "reasoning_effort" in raw_openai_request:
        reasoning_effort = raw_openai_request.get("reasoning_effort")
        
    # Apply the scorched earth thinking logic
    generation_config = apply_scorched_earth_thinking_config(
        generation_config=generation_config,
        openai_reasoning_effort=reasoning_effort
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
    
    # Remove any keys with None values
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    return {
        "model": model,
        "request": request_data
    }


def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request.
    This is used for direct Gemini API calls.
    """
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    
    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}
        
    # Sanitize historical messages so past thoughts/tools don't break the validator
    if "contents" in native_request:
        native_request["contents"] = sanitize_historical_signatures(native_request["contents"])
    
    if "gemini-2.5-flash-image" not in model_from_path:
        thinking_budget = get_thinking_budget(model_from_path)
        include_thoughts = should_include_thoughts(model_from_path)
        
        # Apply the scorched earth thinking logic
        native_request["generationConfig"] = apply_scorched_earth_thinking_config(
            generation_config=native_request["generationConfig"],
            fallback_budget=thinking_budget,
            fallback_include=include_thoughts
        )
    
    # Add Google Search grounding for search models
    if is_search_model(model_from_path):
        if "tools" not in native_request:
            native_request["tools"] = []
        # Add googleSearch tool if not already present
        if not any(tool.get("googleSearch") for tool in native_request["tools"]):
            native_request["tools"].append({"googleSearch": {}})
    
    return {
        "model": get_base_model_name(model_from_path),  # Use base model name for API call
        "request": native_request
    }

def record_usage(project_id: str, success: bool):
    """Safely increments the usage statistics for the dashboard."""
    with _stats_lock:
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        
        # Reset stats daily
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
    """Returns a snapshot of the current usage stats for the dashboard."""
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
            "accounts": accounts_list
        }

def get_formatted_stats() -> dict:
    """Returns the stats formatted for the root endpoint."""
    return get_usage_stats_snapshot()