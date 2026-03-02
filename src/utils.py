import json
import re
import platform
from .config import CLI_VERSION

GEMINI_DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
GEMINI_API_CLIENT_HEADER = "google-genai-sdk/1.41.0 gl-node/v22.19.0"

def _get_nodejs_os():
    """Map Python platform to Node.js-style OS string."""
    system = platform.system().lower()
    if system == "windows":
        return "win32"
    return system  # "linux", "darwin" already match


def _get_nodejs_arch():
    """Map Python arch to Node.js-style arch string."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    elif machine in ("i386", "i686", "x86"):
        return "x86"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def get_user_agent(model: str = "unknown"):
    """Generate User-Agent string matching gemini-cli format."""
    return f"GeminiCLI/{CLI_VERSION}/{model} ({_get_nodejs_os()}; {_get_nodejs_arch()})"


def get_platform_string():
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()

    if system == "DARWIN":
        if arch in ["ARM64", "AARCH64"]:
            return "DARWIN_ARM64"
        else:
            return "DARWIN_AMD64"
    elif system == "LINUX":
        if arch in ["ARM64", "AARCH64"]:
            return "LINUX_ARM64"
        else:
            return "LINUX_AMD64"
    elif system == "WINDOWS":
        return "WINDOWS_AMD64"
    else:
        return "PLATFORM_UNSPECIFIED"


def get_client_metadata(project_id=None):
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": get_platform_string(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }


def sanitize_historical_signatures(contents: list) -> list:
    """
    Recursively injects a dummy thought signature into parts that typically
    cause validation errors when passing historical messages back to Gemini.
    """
    if not contents:
        return contents

    for message in contents:
        parts = message.get("parts", [])
        for part in parts:
            needs_signature = any(key in part for key in ["functionCall", "thought", "inlineData"])

            if needs_signature and "thoughtSignature" not in part:
                part["thoughtSignature"] = GEMINI_DUMMY_THOUGHT_SIGNATURE

    return contents


def apply_scorched_earth_thinking_config(
    generation_config: dict,
    fallback_budget: int = None,
    fallback_include: bool = True,
    openai_reasoning_effort: str = None
) -> dict:
    """
    Aggressively removes conflicting thinking parameters (snake_case and camelCase)
    to guarantee the API never throws a 400 error for mutually exclusive keys.
    """
    if "thinkingConfig" not in generation_config:
        generation_config["thinkingConfig"] = {}

    tc = generation_config["thinkingConfig"]

    if "include_thoughts" in tc:
        tc["includeThoughts"] = tc.pop("include_thoughts")

    if openai_reasoning_effort:
        effort = str(openai_reasoning_effort).lower().strip()

        tc.pop("thinkingLevel", None)
        tc.pop("thinking_level", None)
        tc.pop("thinkingBudget", None)
        tc.pop("thinking_budget", None)

        if effort == "auto":
            tc["thinkingBudget"] = -1
            tc["includeThoughts"] = True
        elif effort in ["low", "medium", "high"]:
            tc["thinkingLevel"] = effort.upper()
            tc["includeThoughts"] = True
        elif effort == "none":
            tc["includeThoughts"] = False
    else:
        has_level = "thinkingLevel" in tc or "thinking_level" in tc
        has_budget = "thinkingBudget" in tc or "thinking_budget" in tc

        if has_level:
            tc.pop("thinkingBudget", None)
            tc.pop("thinking_budget", None)
            if "thinking_level" in tc:
                tc["thinkingLevel"] = tc.pop("thinking_level")
        elif has_budget:
            tc.pop("thinkingLevel", None)
            tc.pop("thinking_level", None)
            if "thinking_budget" in tc:
                tc["thinkingBudget"] = tc.pop("thinking_budget")
        elif fallback_budget is not None:
            tc["thinkingBudget"] = fallback_budget
            tc.pop("thinkingLevel", None)
            tc.pop("thinking_level", None)

    if "includeThoughts" not in tc:
        tc["includeThoughts"] = fallback_include

    return generation_config


def clamp_top_k(generation_config: dict) -> dict:
    """
    Clamp topK to the cloudcode maximum of 64.
    Values at or below 64 are left untouched.
    """
    for key in ("topK", "top_k"):
        value = generation_config.get(key)
        if value is not None and isinstance(value, (int, float)):
            if value > 64:
                generation_config[key] = 64
            else:
                generation_config[key] = int(value)
    return generation_config

# Headers to scrub from incoming requests before forwarding upstream.
# Prevents leaking proxy infrastructure, client identity, and browser fingerprints.
SCRUB_HEADERS = [
    # Proxy tracing
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-port",
    "x-real-ip",
    "forwarded",
    "via",
    # Client identity (OpenAI SDK fingerprints)
    "x-title",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-os",
    "x-stainless-arch",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "http-referer",
    "referer",
    # Browser / Electron fingerprints
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "sec-fetch-mode",
    "sec-fetch-site",
    "sec-fetch-dest",
    "priority",
    # Encoding negotiation (zstd is an Electron fingerprint)
    "accept-encoding",
]


def scrub_headers(headers: dict) -> dict:
    """
    Remove proxy, fingerprint, and identity headers from a dict.
    Returns a clean copy safe to forward upstream.
    """
    if not headers:
        return headers
    lowered_scrub = set(SCRUB_HEADERS)
    return {k: v for k, v in headers.items() if k.lower() not in lowered_scrub}


def parse_retry_delay(error_body: bytes | str) -> float | None:
    """
    Extract the retry delay from a Google API 429 error response.

    Checks (in order):
      1. error.details[].retryDelay  (RetryInfo)
      2. error.details[].metadata.quotaResetDelay (ErrorInfo)
      3. "Your quota will reset after Xs." in error.message

    Returns delay in seconds, or None if not found.
    """
    try:
        if isinstance(error_body, bytes):
            error_body = error_body.decode("utf-8", "ignore")

        data = json.loads(error_body) if isinstance(error_body, str) else error_body
        error_obj = data if "error" not in data else data["error"]

        # If the top-level is a list (Google sometimes wraps errors in arrays)
        if isinstance(data, list) and len(data) > 0:
            error_obj = data[0].get("error", data[0])

        details = error_obj.get("details", [])

        # Priority 1: RetryInfo.retryDelay (e.g. "0.847655010s")
        for detail in details:
            if detail.get("@type", "").endswith("RetryInfo"):
                delay_str = detail.get("retryDelay", "")
                if delay_str:
                    parsed = _parse_duration_string(delay_str)
                    if parsed is not None:
                        return parsed

        # Priority 2: ErrorInfo.metadata.quotaResetDelay (e.g. "373.801628ms")
        for detail in details:
            if detail.get("@type", "").endswith("ErrorInfo"):
                quota_delay = detail.get("metadata", {}).get("quotaResetDelay", "")
                if quota_delay:
                    parsed = _parse_duration_string(quota_delay)
                    if parsed is not None:
                        return parsed

        # Priority 3: Parse from error.message "Your quota will reset after Xs."
        message = error_obj.get("message", "")
        if message:
            match = re.search(r"after\s+(\d+)s\.?", message)
            if match:
                return float(match.group(1))

    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    return None


def _parse_duration_string(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None

    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        elif s.endswith("us") or s.endswith("µs"):
            return float(s[:-2]) / 1_000_000.0
        elif s.endswith("ns"):
            return float(s[:-2]) / 1_000_000_000.0
        elif s.endswith("s"):
            return float(s[:-1])
        else:
            # Try as raw seconds
            return float(s)
    except (ValueError, TypeError):
        return None


# Model fallback candidates for 429 retry.
MODEL_FALLBACK_ORDER = {
    "gemini-2.5-pro": [
        # "gemini-2.5-pro-preview-05-06",
        # "gemini-2.5-pro-preview-06-05",
    ],
    "gemini-2.5-flash": [
        # "gemini-2.5-flash-preview-04-17",
        # "gemini-2.5-flash-preview-05-20",
    ],
    "gemini-2.5-flash-lite": [
        # "gemini-2.5-flash-lite-preview-06-17",
    ],
}


def get_model_fallback_order(model: str) -> list[str]:
    """
    Returns list of models to try, starting with the requested model.
    Falls back to preview variants on 429.
    """
    candidates = MODEL_FALLBACK_ORDER.get(model, [])
    result = [model]
    for c in candidates:
        if c != model:
            result.append(c)
    return result
