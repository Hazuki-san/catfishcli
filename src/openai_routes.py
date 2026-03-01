"""
OpenAI API Routes - OpenAI-compatible endpoints with disconnect detection.
"""
import json
import uuid
import asyncio
import logging
import threading
from fastapi import APIRouter, Request, Response, Depends
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .models import OpenAIChatCompletionRequest
from .openai_transformers import (
    openai_request_to_gemini,
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
)
from .google_api_client import send_gemini_request, build_gemini_payload_from_openai

router = APIRouter()


async def _watch_disconnect(request: Request, event: threading.Event):
    """Background task that sets the event when the client disconnects."""
    try:
        while not event.is_set():
            if await request.is_disconnected():
                event.set()
                logging.info("Client disconnect detected (OpenAI route).")
                break
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    http_request: Request,
    username: str = Depends(authenticate_user),
):
    """OpenAI-compatible chat completions endpoint."""

    try:
        logging.info(f"OpenAI chat completion request: model={request.model}, stream={request.stream}")
        raw_body = await http_request.json()

        gemini_request_data = openai_request_to_gemini(request)
        gemini_payload = build_gemini_payload_from_openai(
            openai_payload=gemini_request_data,
            raw_openai_request=raw_body,
        )

    except Exception as e:
        logging.error(f"Error processing OpenAI request: {str(e)}")
        return Response(
            content=json.dumps({
                "error": {"message": f"Request processing failed: {str(e)}", "type": "invalid_request_error", "code": 400}
            }),
            status_code=400,
            media_type="application/json",
        )

    # Create disconnect event for this request
    disconnect_event = threading.Event()

    if request.stream:
        async def openai_stream_generator():
            watcher = asyncio.create_task(_watch_disconnect(http_request, disconnect_event))
            try:
                response = send_gemini_request(
                    gemini_payload,
                    is_streaming=True,
                    disconnect_event=disconnect_event,
                )

                if isinstance(response, StreamingResponse):
                    response_id = "chatcmpl-" + str(uuid.uuid4())
                    logging.info(f"Starting streaming response: {response_id}")

                    async for chunk in response.body_iterator:
                        if disconnect_event.is_set():
                            logging.info("Client disconnected, stopping OpenAI stream.")
                            break

                        if isinstance(chunk, bytes):
                            chunk = chunk.decode("utf-8", "ignore")

                        if chunk.startswith("data: "):
                            try:
                                chunk_data = chunk[6:]
                                gemini_chunk = json.loads(chunk_data)

                                if "error" in gemini_chunk:
                                    logging.error(f"Error in streaming response: {gemini_chunk['error']}")
                                    error_data = {
                                        "error": {
                                            "message": gemini_chunk["error"].get("message", "Unknown error"),
                                            "type": gemini_chunk["error"].get("type", "api_error"),
                                            "code": gemini_chunk["error"].get("code"),
                                        }
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                openai_chunk = gemini_stream_chunk_to_openai(
                                    gemini_chunk, request.model, response_id
                                )
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                await asyncio.sleep(0)

                            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                                logging.warning(f"Failed to parse streaming chunk: {str(e)}")
                                continue

                        elif chunk.startswith(": keep-alive"):
                            # Forward keep-alive as SSE comment
                            yield ": keep-alive\n\n"

                    yield "data: [DONE]\n\n"
                    logging.info(f"Completed streaming response: {response_id}")

                else:
                    error_msg = "Streaming request failed"
                    status_code = 500

                    if hasattr(response, "status_code"):
                        status_code = response.status_code
                        error_msg += f" (status: {status_code})"

                    if hasattr(response, "body"):
                        try:
                            error_body = response.body
                            if isinstance(error_body, bytes):
                                error_body = error_body.decode("utf-8", "ignore")
                            error_data = json.loads(error_body)
                            if "error" in error_data:
                                error_msg = error_data["error"].get("message", error_msg)
                        except Exception:
                            pass

                    logging.error(f"Streaming request failed: {error_msg}")
                    error_data = {
                        "error": {
                            "message": error_msg,
                            "type": "invalid_request_error" if status_code == 404 else "api_error",
                            "code": status_code,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"

            except Exception as e:
                logging.error(f"Streaming error: {str(e)}")
                error_data = {
                    "error": {"message": f"Streaming failed: {str(e)}", "type": "api_error", "code": 500}
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                watcher.cancel()

        return StreamingResponse(
            openai_stream_generator(), media_type="text/event-stream"
        )

    else:
        try:
            response = send_gemini_request(
                gemini_payload,
                is_streaming=False,
                disconnect_event=disconnect_event,
            )

            if isinstance(response, Response) and response.status_code != 200:
                logging.error(f"Gemini API error: status={response.status_code}")

                try:
                    error_body = response.body
                    if isinstance(error_body, bytes):
                        error_body = error_body.decode("utf-8", "ignore")
                    error_data = json.loads(error_body)
                    if "error" in error_data:
                        openai_error = {
                            "error": {
                                "message": error_data["error"].get("message", f"API error: {response.status_code}"),
                                "type": error_data["error"].get("type", "api_error"),
                                "code": error_data["error"].get("code", response.status_code),
                            }
                        }
                        return Response(
                            content=json.dumps(openai_error),
                            status_code=response.status_code,
                            media_type="application/json",
                        )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                return Response(
                    content=json.dumps({
                        "error": {
                            "message": f"API error: {response.status_code}",
                            "type": "api_error",
                            "code": response.status_code,
                        }
                    }),
                    status_code=response.status_code,
                    media_type="application/json",
                )

            try:
                gemini_response = json.loads(response.body)
                openai_response = gemini_response_to_openai(gemini_response, request.model)
                logging.info(f"Successfully processed non-streaming response for model: {request.model}")
                return Response(
                    content=json.dumps(openai_response), media_type="application/json"
                )
            except (json.JSONDecodeError, AttributeError) as e:
                logging.error(f"Failed to parse Gemini response: {str(e)}")
                return Response(
                    content=json.dumps({
                        "error": {"message": f"Failed to process response: {str(e)}", "type": "api_error", "code": 500}
                    }),
                    status_code=500,
                    media_type="application/json",
                )

        except Exception as e:
            logging.error(f"Non-streaming request failed: {str(e)}")
            return Response(
                content=json.dumps({
                    "error": {"message": f"Request failed: {str(e)}", "type": "api_error", "code": 500}
                }),
                status_code=500,
                media_type="application/json",
            )


@router.get("/v1/models")
async def openai_list_models(username: str = Depends(authenticate_user)):
    """OpenAI-compatible models endpoint."""
    try:
        logging.info("OpenAI models list requested")
        from .config import SUPPORTED_MODELS

        openai_models = []
        for model in SUPPORTED_MODELS:
            model_id = model["name"].replace("models/", "")
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": "google",
                "permission": [{
                    "id": "modelperm-" + model_id.replace("/", "-"),
                    "object": "model_permission",
                    "created": 1677610602,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": False,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }],
                "root": model_id,
                "parent": None,
            })

        logging.info(f"Returning {len(openai_models)} models")
        return {"object": "list", "data": openai_models}

    except Exception as e:
        logging.error(f"Failed to list models: {str(e)}")
        return Response(
            content=json.dumps({
                "error": {"message": f"Failed to list models: {str(e)}", "type": "api_error", "code": 500}
            }),
            status_code=500,
            media_type="application/json",
        )
