"""
Gemini API Routes - Native Gemini API endpoints with disconnect detection.
"""
import json
import logging
import asyncio
import threading
from fastapi import APIRouter, Request, Response, Depends
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .google_api_client import send_gemini_request, build_gemini_payload_from_native
from .config import SUPPORTED_MODELS

router = APIRouter()


async def _watch_disconnect(request: Request, event: threading.Event):
    """Background task that sets the event when the client disconnects."""
    try:
        while not event.is_set():
            if await request.is_disconnected():
                event.set()
                logging.info("Client disconnect detected (Gemini route).")
                break
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass


def _wrap_with_disconnect(
    response: StreamingResponse,
    request: Request,
    disconnect_event: threading.Event,
) -> StreamingResponse:
    """Wrap a StreamingResponse with client disconnect detection."""
    original_iterator = response.body_iterator

    async def wrapped_generator():
        watcher = asyncio.create_task(_watch_disconnect(request, disconnect_event))
        try:
            async for chunk in original_iterator:
                if disconnect_event.is_set():
                    logging.info("Stopping stream — client disconnected.")
                    break
                yield chunk
        finally:
            watcher.cancel()

    return StreamingResponse(
        wrapped_generator(),
        media_type=response.media_type,
        headers=dict(response.headers) if response.headers else None,
        status_code=response.status_code,
    )


@router.get("/v1beta/models")
async def gemini_list_models(request: Request, username: str = Depends(authenticate_user)):
    """Native Gemini models endpoint."""
    try:
        logging.info("Gemini models list requested")
        models_response = {"models": SUPPORTED_MODELS}
        logging.info(f"Returning {len(SUPPORTED_MODELS)} Gemini models")
        return Response(
            content=json.dumps(models_response),
            status_code=200,
            media_type="application/json; charset=utf-8",
        )
    except Exception as e:
        logging.error(f"Failed to list Gemini models: {str(e)}")
        return Response(
            content=json.dumps({"error": {"message": f"Failed to list models: {str(e)}", "code": 500}}),
            status_code=500,
            media_type="application/json",
        )


@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gemini_proxy(request: Request, full_path: str, username: str = Depends(authenticate_user)):
    """Native Gemini API proxy endpoint with disconnect detection."""
    try:
        post_data = await request.body()
        is_streaming = "stream" in full_path.lower()
        model_name = _extract_model_from_path(full_path)

        logging.info(f"Gemini proxy request: path={full_path}, model={model_name}, stream={is_streaming}")

        if not model_name:
            logging.error(f"Could not extract model name from path: {full_path}")
            return Response(
                content=json.dumps({"error": {"message": f"Could not extract model name from path: {full_path}", "code": 400}}),
                status_code=400,
                media_type="application/json",
            )

        try:
            incoming_request = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in request body: {str(e)}")
            return Response(
                content=json.dumps({"error": {"message": "Invalid JSON in request body", "code": 400}}),
                status_code=400,
                media_type="application/json",
            )

        gemini_payload = build_gemini_payload_from_native(incoming_request, model_name)

        # Create disconnect event and pass it through
        disconnect_event = threading.Event()
        response = send_gemini_request(
            gemini_payload,
            is_streaming=is_streaming,
            disconnect_event=disconnect_event,
        )

        # Wrap streaming responses with disconnect detection
        if is_streaming and isinstance(response, StreamingResponse):
            response = _wrap_with_disconnect(response, request, disconnect_event)

        if hasattr(response, "status_code"):
            if response.status_code != 200:
                logging.error(f"Gemini API returned error: status={response.status_code}")
            else:
                logging.info(f"Successfully processed Gemini request for model: {model_name}")

        return response

    except Exception as e:
        logging.error(f"Gemini proxy error: {str(e)}")
        return Response(
            content=json.dumps({"error": {"message": f"Proxy error: {str(e)}", "code": 500}}),
            status_code=500,
            media_type="application/json",
        )


def _extract_model_from_path(path: str) -> str:
    """Extract the model name from a Gemini API path."""
    parts = path.split("/")
    try:
        models_index = parts.index("models")
        if models_index + 1 < len(parts):
            model_name = parts[models_index + 1]
            if ":" in model_name:
                model_name = model_name.split(":")[0]
            return model_name
    except ValueError:
        pass
    return None


@router.get("/v1/models")
async def gemini_list_models_v1(request: Request, username: str = Depends(authenticate_user)):
    """Alternative models endpoint for v1 API version."""
    return await gemini_list_models(request, username)


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "geminicli2api"}
