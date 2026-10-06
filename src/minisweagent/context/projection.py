"""Model view projection: derive the message list sent to the provider from the
append-only transcript.

The transcript (``agent.messages``) is the single source of truth and is never
mutated or truncated. ``build_model_view`` is a pure function of the transcript:
the same transcript prefix always yields the same view, which makes every
request replayable offline (the basis for M2's "training sees what inference
saw" guarantee).
"""

from minisweagent.context.accounting import estimate_message_tokens

INTERNAL_ROLES = frozenset({"compaction", "exit"})
"""Roles that never enter the provider payload."""

CHECKPOINT_LEADIN = (
    "<context-compaction>\n"
    "The earlier conversation was automatically compacted to fit the context window.\n"
    "Below is a deterministic summary of everything that happened before the recent steps.\n"
)


def _is_discarded(message: dict) -> bool:
    return bool(message.get("extra", {}).get("discarded"))


def is_hidden_from_view(message: dict) -> bool:
    """True for messages that must never be sent to the provider."""
    return message.get("role") in INTERNAL_ROLES or _is_discarded(message)


def fixed_prefix_len(messages: list[dict]) -> int:
    """Length of the pinned prefix (system + original task message).

    The pinned prefix is never compacted: the original task must reach the model
    verbatim at every step.
    """
    if len(messages) >= 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
        return 2
    if len(messages) >= 1 and messages[0].get("role") == "system":
        return 1
    return 0


def is_action_message(message: dict) -> bool:
    """True for assistant messages that produced actions (action group starters)."""
    return message.get("role") == "assistant" and bool(message.get("extra", {}).get("actions"))


def split_action_groups(messages: list[dict]) -> list[list[int]]:
    """Partition the compactable region into atomic action groups.

    A group starts at an action-bearing assistant message and contains every
    message up to the next one (its observations, any format-error corrections,
    internal records). Compaction cut points may only fall on group boundaries
    so that a tool call is never separated from its results -- in both the
    tool-call mode (assistant(tool_calls) + tool results) and the text-based
    mode (assistant + user observation).
    """
    start = fixed_prefix_len(messages)
    groups: list[list[int]] = []
    current: list[int] = []
    for i in range(start, len(messages)):
        if is_action_message(messages[i]) and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def find_last_compaction(messages: list[dict]) -> int:
    """Index of the most recent compaction record, or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "compaction":
            return i
    return -1


def checkpoint_message(compaction: dict) -> dict:
    """Wrap a compaction record's checkpoint as a user message visible to the model."""
    return {"role": "user", "content": CHECKPOINT_LEADIN + compaction["content"] + "\n</context-compaction>"}


def build_model_view(messages: list[dict]) -> list[dict]:
    """Project the append-only transcript into the model-visible message list.

    Without any compaction record this is the identity projection (minus
    internal/discarded messages). With one, the view is:
    pinned prefix -> checkpoint (as a user message) -> kept-recent region ->
    messages appended after the compaction.
    """
    last = find_last_compaction(messages)
    if last < 0:
        return [m for m in messages if not is_hidden_from_view(m)]
    meta = messages[last].get("extra", {}).get("compaction", {})
    kept_start = int(meta.get("first_kept_message_index", fixed_prefix_len(messages)))
    view: list[dict] = list(messages[: fixed_prefix_len(messages)])
    view.append(checkpoint_message(messages[last]))
    for i in [*range(kept_start, last), *range(last + 1, len(messages))]:
        if not is_hidden_from_view(messages[i]):
            view.append(messages[i])
    return view


def select_cut_index(messages: list[dict], keep_recent_tokens: int, chars_per_token: float = 4.0) -> int:
    """Choose the transcript index where the kept-recent region starts.

    Whole action groups are accumulated from the end until the keep budget is
    reached; at least one group is always kept. Everything before the returned
    index (and after the pinned prefix) is the foldable region.
    """
    groups = split_action_groups(messages)
    if not groups:
        return len(messages)
    kept_tokens = 0
    kept_groups = 0
    for group in reversed(groups):
        kept_tokens += sum(estimate_message_tokens(messages[i], chars_per_token) for i in group)
        kept_groups += 1
        if kept_tokens >= keep_recent_tokens:
            break
    first_kept_group = len(groups) - kept_groups
    return groups[first_kept_group][0]
