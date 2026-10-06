"""Token accounting: API usage anchor + incremental estimation.

Borrowed from pi's accounting principle: the provider-reported usage of the most
recent successful response is an exact anchor for everything that was sent plus
the response itself; only messages appended since then need to be estimated.
Any compaction changes the view, which invalidates the anchor.
"""

import math

PROTOCOL_OVERHEAD_TOKENS = 8
"""Per-message overhead estimate for role/protocol framing (chat template, tool wrappers)."""

_MULTIMODAL_FALLBACK_CHARS = 256


def estimate_message_tokens(message: dict, chars_per_token: float = 4.0) -> int:
    """Estimate the token count of one message (content + tool call JSON)."""
    content = message.get("content")
    if isinstance(content, str):
        n_chars = len(content)
    elif content is None:
        n_chars = 0
    else:
        # Multimodal content blocks carry images etc.; assume a fixed allowance.
        n_chars = _MULTIMODAL_FALLBACK_CHARS
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        n_chars += len(function.get("name") or "") + len(function.get("arguments") or "")
    return math.ceil(n_chars / max(chars_per_token, 0.1)) + PROTOCOL_OVERHEAD_TOKENS


def extract_usage(message: dict) -> tuple[int, int] | None:
    """Return (prompt_tokens, completion_tokens) reported for a successful assistant response."""
    response = message.get("extra", {}).get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if prompt <= 0 and completion <= 0:
        return None
    return prompt, completion


class TokenAccountant:
    """Tracks the most recent API usage anchor and estimates the current view.

    The anchor is ``(view_len, prompt_tokens, completion_tokens)`` where
    ``view_len`` is the length of the view that produced the usage. Because the
    transcript is append-only and the view only grows between compactions, the
    estimate is ``prompt + completion + estimate(view[view_len:])``.
    """

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
        self.anchor: tuple[int, int, int] | None = None

    def update_anchor(self, message: dict, view_len: int) -> None:
        usage = extract_usage(message)
        if usage is not None:
            self.anchor = (view_len, usage[0], usage[1])

    def invalidate_anchor(self) -> None:
        self.anchor = None

    def estimate_tokens(self, view: list[dict]) -> int:
        if self.anchor is not None:
            anchor_len, prompt, completion = self.anchor
            if anchor_len <= len(view):
                incremental = sum(estimate_message_tokens(m, self.chars_per_token) for m in view[anchor_len:])
                return prompt + completion + incremental
        return sum(estimate_message_tokens(m, self.chars_per_token) for m in view)
