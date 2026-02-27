"""
OpenAI Format Transformers - Handles conversion between OpenAI and Gemini API formats.
This module contains all the logic for transforming requests and responses between the two formats.
"""
import json
import time
import uuid
import logging
from typing import Dict, Any, Union


from .models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from .config import (
    DEFAULT_SAFETY_SETTINGS,
    is_search_model,
    get_base_model_name,
    get_thinking_budget,
    should_include_thoughts
)


def openai_request_to_gemini(openai_request: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """
    将 OpenAI 聊天请求转换为 Gemini 格式。
    此版本已精简，将 thinkingConfig 和 signature 逻辑交由 client builder 集中处理。
    """
    contents = []
    
    # ------------------------------------------------------------------
    #  1. 转换对话历史 (messages)
    # ------------------------------------------------------------------
    for message in openai_request.messages:
        role = message.role
        
        if role == "system":
            contents.append({"role": "user", "parts": [{"text": str(message.content)}]})

        elif role == "user":
            if isinstance(message.content, list):
                parts = []
                for part in message.content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if image_url:
                            try:
                                mime_type, base64_data = image_url.split(";")
                                _, mime_type = mime_type.split(":")
                                _, base64_data = base64_data.split(",")
                                parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": base64_data
                                    }
                                })
                            except ValueError:
                                continue
                if parts:
                    contents.append({"role": "user", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": str(message.content)}]})

        elif role == "assistant":
            parts = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.get("function", {}).get("name"),
                            "args": json.loads(tc.get("function", {}).get("arguments", "{}"))
                        }
                    })
                    # REMOVED: manual thoughtSignature injection (utils.py handles it now)
            
            if message.content:
                parts.append({"text": str(message.content)})
            
            if parts:
                contents.append({"role": "model", "parts": parts})
                
        elif role == "tool":
            func_name = message.name
            if not func_name:
                for i in range(len(openai_request.messages) - 1, -1, -1):
                    prev_msg = openai_request.messages[i]
                    if prev_msg.role == "assistant" and prev_msg.tool_calls:
                        for tc in prev_msg.tool_calls:
                            if tc.get("id") == message.tool_call_id:
                                func_name = tc.get("function", {}).get("name")
                                break
                    if func_name:
                        break

            new_part = {
                "functionResponse": {
                    "name": func_name or "function_name_not_found",
                    "response": {"content": message.content}
                }
            }

            if (contents and
                    contents[-1]["role"] == "user" and
                    "parts" in contents[-1] and
                    len(contents[-1]["parts"]) > 0 and
                    "functionResponse" in contents[-1]["parts"][-1]):
                contents[-1]["parts"].append(new_part)
            else:
                contents.append({
                    "role": "user",
                    "parts": [new_part]
                })

    # ------------------------------------------------------------------
    #  2. 转换 API 参数
    # ------------------------------------------------------------------
    generation_config = {}
    if openai_request.temperature is not None:
        generation_config["temperature"] = openai_request.temperature
    if openai_request.top_p is not None:
        generation_config["topP"] = openai_request.top_p
    if openai_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = openai_request.max_tokens
    if openai_request.stop is not None:
        if isinstance(openai_request.stop, str):
            generation_config["stopSequences"] = [openai_request.stop]
        elif isinstance(openai_request.stop, list):
            generation_config["stopSequences"] = openai_request.stop
    if openai_request.frequency_penalty is not None:
        generation_config["frequencyPenalty"] = openai_request.frequency_penalty
    if openai_request.presence_penalty is not None:
        generation_config["presencePenalty"] = openai_request.presence_penalty
    if openai_request.n is not None:
        generation_config["candidateCount"] = openai_request.n
    if openai_request.seed is not None:
        generation_config["seed"] = openai_request.seed
    if openai_request.response_format is not None:
        if openai_request.response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

    # ------------------------------------------------------------------
    #  3. 构建最终请求体
    # ------------------------------------------------------------------
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(openai_request.model)
    }
    
    if is_search_model(openai_request.model):
        request_payload["tools"] = [{"googleSearch": {}}]

    # 处理工具声明 (Function Declarations)
    function_declarations = []
    if openai_request.tools:
        for tool in openai_request.tools:
            if tool.get("type") == "function":
                func_def = tool.get("function")
                if func_def and func_def.get("name"):
                    declaration = {k: v for k, v in func_def.items() if v is not None}
                    function_declarations.append(declaration)

    if function_declarations:
        request_payload["tools"] = request_payload.get("tools", [])
        request_payload["tools"].append({"functionDeclarations": function_declarations})

    # 处理工具选择 (Tool Choice)
    tool_config = None
    if openai_request.tool_choice and function_declarations:
        choice = openai_request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none": mode = "NONE"
            elif choice == "auto": mode = "AUTO"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"
                allowed_functions = [func_name]
        
        if mode:
            config = {"mode": mode}
            if allowed_functions:
                config["allowedFunctionNames"] = allowed_functions
            tool_config = {"functionCallingConfig": config}

    if tool_config:
        request_payload["toolConfig"] = tool_config
    
    return request_payload

def gemini_response_to_openai(gemini_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Transform a Gemini API response to OpenAI chat completion format.
    
    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response
        
    Returns:
        Dictionary in OpenAI chat completion format
    """
    choices = []
    
    for candidate in gemini_response.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        
        # Map Gemini roles back to OpenAI roles
        if role == "model":
            role = "assistant"
        
        # Extract and separate thinking tokens from regular content
        parts = candidate.get("content", {}).get("parts", [])
        content = ""
        tool_calls = []
        reasoning_content = ""
        for part in parts:
            if 'functionCall' in part:
                fc = part['functionCall']
                tool_calls.append({
                    "id": f"call_{fc.get('name')}_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": fc.get('name'),
                        "arguments": json.dumps(fc.get('args', {})),
                    }
                })
            elif part.get("thought", False):
                 reasoning_content += part.get("text", "")
            elif 'text' in part:
                 content += part.get("text", "")

        message = {
            "role": "assistant",
            "content": content if not tool_calls else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        
        finish_reason = _map_finish_reason(candidate.get("finishReason"))
        if tool_calls:
            finish_reason = "tool_calls"
        
        
        choices.append({
            "index": candidate.get("index", 0),
            "message": message,
            "finish_reason": _map_finish_reason(candidate.get("finishReason")),
        })
    
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def gemini_stream_chunk_to_openai(gemini_chunk: Dict[str, Any], model: str, response_id: str) -> Dict[str, Any]:
    """
    Transform a Gemini streaming response chunk to OpenAI streaming format.
    
    Args:
        gemini_chunk: Single chunk from Gemini streaming response
        model: Model name to include in response
        response_id: Consistent ID for this streaming response
        
    Returns:
        Dictionary in OpenAI streaming format
    """
    choices = []
    
    for candidate in gemini_chunk.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        
        # Map Gemini roles back to OpenAI roles
        if role == "model":
            role = "assistant"
        
        # Extract and separate thinking tokens from regular content
        parts = candidate.get("content", {}).get("parts", [])
        delta = {}
        tool_calls_chunks = []
        content_chunk = ""
        reasoning_content = ""

        for i, part in enumerate(parts):
            if 'functionCall' in part:
                fc = part['functionCall']
                tool_calls_chunks.append({
                    "index": i, # 流式中需要 index
                    "id": f"call_{fc.get('name')}_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": fc.get('name'),
                        "arguments": json.dumps(fc.get('args', {})),
                    }
                })
            elif part.get("thought", False):
                reasoning_content += part.get("text", "")
            elif 'text' in part:
                content_chunk += part.get("text", "")
        
        if content_chunk:
            delta["content"] = content_chunk
        if tool_calls_chunks:
            delta["tool_calls"] = tool_calls_chunks
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        
        finish_reason = _map_finish_reason(candidate.get("finishReason"))
        if tool_calls_chunks:
            finish_reason = "tool_calls"
        
        
        choices.append({
            "index": candidate.get("index", 0),
            "delta": delta,
            "finish_reason": _map_finish_reason(candidate.get("finishReason")),
        })
    
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _map_finish_reason(gemini_reason: str) -> str:
    """
    Map Gemini finish reasons to OpenAI finish reasons.
    
    Args:
        gemini_reason: Finish reason from Gemini API
        
    Returns:
        OpenAI-compatible finish reason
    """
    if gemini_reason == "STOP":
        return "stop"
    elif gemini_reason == "MAX_TOKENS":
        return "length"
    elif gemini_reason in ["SAFETY", "RECITATION"]:
        return "content_filter"
    else:
        return None
