import re

# Comprehensive list of regex patterns covering known prompt injections,
# system prompt leakage attempts, and jailbreaks.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|previous\s+|prior\s+)?instructions",
    r"disregard\s+(all\s+|previous\s+|prior\s+)?instructions",
    r"forget\s+(all\s+|previous\s+|prior\s+)?instructions",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|secret|hidden\s+prompt)",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|initial\s+instructions|system\s+instructions)",
    r"print\s+(your\s+)?(system\s+prompt|initial\s+prompt)",
    r"show\s+(me\s+)?(your\s+)?(system\s+prompt|instructions)",
    r"you\s+are\s+now\s+(an?\s+)?(unrestricted|dan|developer|jailbreak|evil)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"act\s+as\s+(a\s+)?(dan|developer\s+mode|unfiltered\s+ai|jailbreak)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[\/INST\]",
    r"system:\s*",
    r"developer\s+mode\s+enabled",
    r"bypass\s+(all\s+)?(safety\s+guidelines|restrictions|filters|content\s+policy)",
    r"override\s+(all\s+)?(safety|rules|instructions)"
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

def check_prompt_injection(user_input: str) -> tuple[bool, str | None]:
    """
    Scans user input for adversarial prompt injection, system prompt extraction,
    and jailbreak attempts.
    
    Returns:
        (is_injection: bool, matched_reason: str | None)
    """
    if not user_input or not user_input.strip():
        return False, None

    normalized_input = user_input.strip()

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(normalized_input)
        if match:
            return True, f"Suspicious prompt pattern detected: '{match.group(0)}'"

    return False, None
