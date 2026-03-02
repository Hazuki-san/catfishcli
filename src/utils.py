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
