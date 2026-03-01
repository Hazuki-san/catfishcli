import platform
from .config import CLI_VERSION

GEMINI_DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

def get_user_agent():
    """Generate User-Agent string matching gemini-cli format."""
    version = CLI_VERSION
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{version} ({system}; {arch})"

def get_platform_string():
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()
    
    # Map to gemini-cli platform format
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
    cause validation errors when passing historical messages back to Gemini 
    (like function calls, thoughts, and inline images).
    """
    if not contents:
        return contents
        
    for message in contents:
        parts = message.get("parts", [])
        for part in parts:
            # If the part is a tool call, reasoning thought, or inline image data,
            # it technically requires a cryptographic signature from the model. 
            # We bypass this for historical messages to avoid 400 errors.
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
    
    # 1. Clean up snake_case include_thoughts immediately
    if "include_thoughts" in tc:
        tc["includeThoughts"] = tc.pop("include_thoughts")
        
    # 2. CASE: Translating an OpenAI 'reasoning_effort' parameter
    if openai_reasoning_effort:
        effort = str(openai_reasoning_effort).lower().strip()
        
        # Scorched earth: wipe all existing budget/level keys to start fresh
        tc.pop("thinkingLevel", None)
        tc.pop("thinking_level", None)
        tc.pop("thinkingBudget", None)
        tc.pop("thinking_budget", None)
        
        if effort == "auto":
            # Auto defaults to budget mode with -1
            tc["thinkingBudget"] = -1
            tc["includeThoughts"] = True
        elif effort in ["low", "medium", "high"]:
            # Low/Med/High maps to thinkingLevel
            tc["thinkingLevel"] = effort.upper()
            tc["includeThoughts"] = True
        elif effort == "none":
            tc["includeThoughts"] = False
            
    # 3. CASE: Native Gemini request validation
    else:
        has_level = "thinkingLevel" in tc or "thinking_level" in tc
        has_budget = "thinkingBudget" in tc or "thinking_budget" in tc
        
        if has_level:
            # If level exists, destroy all traces of budget
            tc.pop("thinkingBudget", None)
            tc.pop("thinking_budget", None)
            # Ensure proper casing
            if "thinking_level" in tc:
                tc["thinkingLevel"] = tc.pop("thinking_level")
        elif has_budget:
            # If budget exists, destroy all traces of level
            tc.pop("thinkingLevel", None)
            tc.pop("thinking_level", None)
            # Ensure proper casing
            if "thinking_budget" in tc:
                tc["thinkingBudget"] = tc.pop("thinking_budget")
        elif fallback_budget is not None:
            # If neither exists, safely apply the fallback budget
            tc["thinkingBudget"] = fallback_budget
            tc.pop("thinkingLevel", None)
            tc.pop("thinking_level", None)

    # 4. Standardize 'includeThoughts' fallback if it's completely missing
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
