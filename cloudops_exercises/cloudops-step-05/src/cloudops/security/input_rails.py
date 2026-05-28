import re

# Patterns that indicate prompt injection, SQL injection, or XSS attempts.
# These block at the gate — before any LLM call is made (zero cost to block).
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(drop|delete|truncate|alter)\s+table", "SQL injection"),
    (r"(?i)ignore.{0,30}(previous|above|prior|all).{0,30}instruction", "prompt injection"),
    (r"(?i)disregard.{0,30}(previous|above|prior|all)", "prompt injection"),
    (r"<script[\s>]", "XSS"),
    (r"(?i)system\s*prompt\s*:", "system prompt extraction"),
]

MAX_INPUT_CHARS = 2_000


def validate_input(text: str) -> str:
    """Return text unchanged if it passes all rails; raise ValueError otherwise.

    Called in the CLI before any routing decision or LLM call.
    """
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Input too long ({len(text)} chars). Maximum is {MAX_INPUT_CHARS} characters."
        )

    for pattern, label in _BLOCKED_PATTERNS:
        if re.search(pattern, text):
            raise ValueError(f"Input blocked by security rail ({label}).")

    return text
