"""Deterministic folding (fold-v1): compact discarded action groups into a
fixed-format checkpoint without any model calls.

Rules: fixed field order, hard character caps applied Unicode-safely with
explicit omission counts, and byte-identical output for identical input. No
semantic judgement is involved.
"""

import math
import re

from minisweagent.context.projection import is_action_message

MAX_FILES_PER_LIST = 50
_STORAGE_COMMAND_CHARS = 400
"""Cap for the command stored in compaction records (bounds trajectory size)."""

_OMISSION_MARKER = "…[+{n} chars]"

_READ_COMMANDS = {
    "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "find",
    "ls", "diff", "stat", "file", "wc", "sed", "awk", "nl", "python", "python3",
}
_MODIFY_COMMANDS = {"mv", "cp", "rm", "truncate", "chmod", "touch", "tee", "patch"}
_GIT_WRITE_SUBCOMMANDS = {"apply", "checkout", "restore", "reset", "rebase", "merge", "cherry-pick", "stash", "clean"}
_PATHLIKE = re.compile(r"/|\.\w{1,4}$")
_REDIRECT_TARGET = re.compile(r"(?:^|[\s;&|(])(?:>{1,2})\s*([^\s;|&)]+)")
_STRIP_CHARS = "\"'`;,(){}[]"


def _truncate(text: str, limit: int) -> str:
    """Unicode-safe truncation with an explicit omission marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + _OMISSION_MARKER.format(n=len(text) - limit)


def _normalize_path(token: str) -> str | None:
    token = token.strip(_STRIP_CHARS)
    if not token or token.startswith("-") or token in ("&&", "||", "|", ">", ">>"):
        return None
    if not _PATHLIKE.search(token):
        return None
    return token


def _sed_inplace(args: list[str]) -> bool:
    """Token-level detection of sed's in-place flag (`-i`, `-i.bak`, `--in-place`)."""
    return any(t == "-i" or t == "--in-place" or t.startswith("-i.") for t in args)


def _extract_from_command(command: str) -> tuple[set[str], set[str], set[str]]:
    """Best-effort deterministic classification of one command into file lists."""
    read_, modified_, touched_ = set(), set(), set()
    tokens = [t for t in re.split(r"\s+", command) if t]
    if not tokens:
        return read_, modified_, touched_
    head = tokens[0].split("/")[-1]
    args = tokens[1:]
    if head == "sed" and _sed_inplace(args):
        # `sed -i <expr> <file>`: only the trailing operand is the file
        if args and (path := _normalize_path(args[-1])):
            modified_.add(path)
    elif head in _MODIFY_COMMANDS:
        modified_ |= {p for t in args if (p := _normalize_path(t))}
    elif head == "git" and len(tokens) > 1 and tokens[1] in _GIT_WRITE_SUBCOMMANDS:
        modified_ |= {p for t in args[1:] if (p := _normalize_path(t))}
    elif head in _READ_COMMANDS:
        read_ |= {p for t in args if (p := _normalize_path(t))}
    else:
        touched_ |= {p for t in args if (p := _normalize_path(t))}
    for match in _REDIRECT_TARGET.finditer(command):
        if path := _normalize_path(match.group(1)):
            modified_.add(path)
    return read_, modified_, touched_


def extract_files(commands: list[str]) -> dict[str, list[str]]:
    """Programmatically derive the read/modified/touched file lists from commands.

    These lists are convenience context for the model, not ground truth: the
    classification is intentionally conservative and fully deterministic.
    Results are sorted, deduplicated, and capped.
    """
    read_, modified_, touched_ = set(), set(), set()
    for command in commands:
        r, m, t = _extract_from_command(command)
        read_ |= r
        modified_ |= m
        touched_ |= t
    touched_ -= read_ | modified_

    def _cap(paths: set[str]) -> tuple[list[str], int]:
        ordered = sorted(paths)[:MAX_FILES_PER_LIST]
        return ordered, max(len(paths) - len(ordered), 0)

    result: dict[str, list[str]] = {}
    for label, paths in (("read", read_), ("modified", modified_), ("touched", touched_)):
        capped, omitted = _cap(paths)
        result[label] = capped
        if omitted:
            result[f"{label}_omitted"] = [omitted]
    return result


def _match_observations(actions: list[dict], observations: list[dict]) -> list[dict | None]:
    """Pair actions with their observation messages.

    Tool-call mode matches by ``tool_call_id``; text-based mode zips in order
    (observations are emitted one per action).
    """
    if (
        actions
        and observations
        and actions[0].get("tool_call_id")
        and observations[0].get("tool_call_id")
    ):
        by_id = {obs.get("tool_call_id"): obs for obs in observations}
        return [by_id.get(action.get("tool_call_id")) for action in actions]
    return [observations[j] if j < len(observations) else None for j in range(len(actions))]


def _is_observation(message: dict) -> bool:
    """Observation messages: tool results, or user messages carrying a returncode.

    Format-error correction messages (user role, no returncode) are excluded.
    """
    if message.get("role") == "tool":
        return True
    return message.get("role") == "user" and "returncode" in message.get("extra", {})


def extract_fold_entries(indices: list[int], messages: list[dict], config) -> list[dict]:
    """Build raw fold entries for the action groups being compacted.

    One entry per executed action: transcript index, command, return code, and
    a short tail of the observation output.
    """
    entries: list[dict] = []
    pending_actions: list[dict] | None = None
    pending_step: int = -1
    pending_observations: list[dict] = []

    def flush() -> None:
        nonlocal pending_actions, pending_step, pending_observations
        if pending_actions is None:
            return
        for action, obs in zip(pending_actions, _match_observations(pending_actions, pending_observations)):
            output = ""
            returncode = None
            if obs is not None:
                returncode = obs.get("extra", {}).get("returncode")
                content = obs.get("content")
                output = content if isinstance(content, str) else ""
            entries.append(
                {
                    "step": pending_step,
                    "command": _truncate(action.get("command", ""), _STORAGE_COMMAND_CHARS),
                    "returncode": returncode,
                    "tail": _output_tail(output, config),
                }
            )
        pending_actions, pending_step, pending_observations = None, -1, []

    for i in indices:
        message = messages[i]
        if is_action_message(message):
            flush()
            pending_actions = list(message["extra"]["actions"])
            pending_step = i
            pending_observations = []
        elif _is_observation(message):
            if pending_actions is not None:
                pending_observations.append(message)
    flush()
    return entries


def _output_tail(output: str, config) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    text = "\n".join(lines[-config.fold_tail_lines :]) if lines else ""
    return _truncate(text, config.fold_tail_chars)


def render_checkpoint(
    entries: list[dict],
    files: dict[str, list[str]],
    *,
    strategy_version: str,
    reason: str,
    config,
) -> str:
    """Render the checkpoint text. Deterministic: same input -> same bytes."""
    lines = [f"[compacted steps | strategy={strategy_version} | reason={reason}]"]
    for label in ("read", "modified", "touched"):
        paths = files.get(label) or []
        omitted = files.get(f"{label}_omitted", [0])[0] if files.get(f"{label}_omitted") else 0
        suffix = f" (+{omitted} more)" if omitted else ""
        lines.append(f"files {label}: " + (", ".join(paths) + suffix if paths else "-"))
    for entry in entries:
        returncode = entry.get("returncode")
        rc = "unknown" if returncode is None else str(returncode)
        command = _truncate(entry.get("command", ""), config.fold_command_chars)
        line = f"[step {entry['step']}] rc={rc} cmd: {command}"
        if tail := entry.get("tail"):
            line += "\n  out: " + tail.replace("\n", "\n  ")
        lines.append(line)
    return "\n".join(lines)


def fit_checkpoint_to_budget(
    entries: list[dict],
    files: dict[str, list[str]],
    *,
    budget_tokens: int,
    chars_per_token: float,
    strategy_version: str,
    reason: str,
    config,
) -> str:
    """Render, then deterministically shrink until the estimate fits the budget.

    Shrink order (oldest-first): drop output tails -> drop entries beyond 4 ->
    drop file lists -> drop entries beyond 1 -> header only. The header-only
    form is a few dozen tokens; if even it exceeds the budget the budget is
    smaller than the protocol floor and the caller must treat it as a config
    error.
    """
    entries = [dict(e) for e in entries]
    files = {k: list(v) for k, v in files.items()}

    def render() -> str:
        return render_checkpoint(entries, files, strategy_version=strategy_version, reason=reason, config=config)

    def estimate(text: str) -> int:
        return math.ceil(len(text) / max(chars_per_token, 0.1)) + 8

    text = render()
    while estimate(text) > budget_tokens:
        progressed = False
        for entry in entries:  # 1) drop the oldest output tail
            if entry.get("tail"):
                entry["tail"] = ""
                progressed = True
                break
        if not progressed and len(entries) > 4:  # 2) drop the oldest entry
            entries.pop(0)
            progressed = True
        if not progressed and any(files.get(k) for k in ("read", "modified", "touched")):  # 3) drop file lists
            files = {k: [] for k in files}
            progressed = True
        if not progressed and len(entries) > 1:  # 4) down to a single entry
            entries.pop(0)
            progressed = True
        if not progressed:  # 5) header only
            entries = []
        text = render()
        if not entries and not any(files.values()):
            break
    return text
