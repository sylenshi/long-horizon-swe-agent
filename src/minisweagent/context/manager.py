"""ContextManager: orchestrates projection, accounting, compaction and recovery.

Contract with the agent: all ``*_record``/``prepare_request`` methods return
compaction records that the CALLER must append to the transcript via
``add_messages``; the manager never mutates the transcript itself. This keeps
the manager usable for offline replay of recorded trajectories.
"""

import hashlib
import json
import time

from minisweagent.context.accounting import TokenAccountant, estimate_message_tokens
from minisweagent.context.config import ContextConfig
from minisweagent.context.events import EventLog
from minisweagent.context.folding import extract_files, extract_fold_entries, fit_checkpoint_to_budget
from minisweagent.context.projection import (
    build_model_view,
    find_last_compaction,
    fixed_prefix_len,
    is_hidden_from_view,
    select_cut_index,
)


def view_hash(view: list[dict]) -> str:
    """Stable hash of the view as the provider sees it (extras stripped).

    Stored per query so that any past request can be re-derived and verified
    offline from the trajectory alone.
    """
    payload = json.dumps(
        [{k: v for k, v in m.items() if k != "extra"} for m in view],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ContextManager:
    def __init__(self, config: ContextConfig, events: EventLog | None = None):
        self.config = config
        self.events = events
        self.accountant = TokenAccountant(config.chars_per_token)
        self.enabled = config.window_tokens > 0
        self.recovery_used = False
        self.n_compactions = 0

    # --- request preparation -------------------------------------------------

    @property
    def threshold(self) -> int:
        return self.config.trigger_threshold()

    def build_view(self, messages: list[dict]) -> list[dict]:
        """Current model view of the transcript (used to re-read after appending a record)."""
        return build_model_view(messages)

    def estimate_tokens(self, view: list[dict]) -> int:
        return self.accountant.estimate_tokens(view)

    def prepare_request(self, messages: list[dict]) -> tuple[list[dict], int, dict | None]:
        """Build the view and estimate its tokens; compact if over the waterline.

        Returns ``(view, estimated_tokens, compaction_record_or_None)``. When a
        record is returned the caller must append it and rebuild the view.
        """
        view = build_model_view(messages)
        est = self.accountant.estimate_tokens(view)
        if not self.enabled or est <= self.threshold:
            return view, est, None
        record = self.make_compaction_record(
            messages, reason="threshold", keep_recent_tokens=self.config.effective_keep_recent_tokens(), est_before=est
        )
        return view, est, record

    # --- anchors ---------------------------------------------------------------

    def update_anchor(self, message: dict, view_len: int) -> None:
        self.accountant.update_anchor(message, view_len)

    # --- compaction ------------------------------------------------------------

    def make_compaction_record(
        self, messages: list[dict], *, reason: str, keep_recent_tokens: int, est_before: int
    ) -> dict | None:
        """Fold everything before the kept-recent region into a checkpoint.

        Appends nothing: returns the record for the caller to append, or None
        when there is nothing left to fold.
        """
        cut = select_cut_index(messages, keep_recent_tokens, self.config.chars_per_token)
        prefix = fixed_prefix_len(messages)
        if cut <= prefix:
            return None  # only the pinned prefix + at most one already-folded group remain

        previous_meta: dict = {}
        last = find_last_compaction(messages)
        if last >= 0:
            previous_meta = messages[last].get("extra", {}).get("compaction", {})

        fold_indices = [i for i in range(prefix, cut) if messages[i].get("role") != "compaction"]
        entries: list[dict] = list(previous_meta.get("folded_entries", []))
        entries += extract_fold_entries(fold_indices, messages, self.config)

        all_commands = [
            action.get("command", "")
            for message in messages
            if message.get("role") == "assistant"
            for action in message.get("extra", {}).get("actions", [])
        ]
        files = extract_files(all_commands)

        cpt = self.config.chars_per_token
        prefix_tokens = sum(estimate_message_tokens(m, cpt) for m in messages[:prefix])
        kept_tokens = sum(
            estimate_message_tokens(messages[i], cpt)
            for i in range(cut, len(messages))
            if not is_hidden_from_view(messages[i])
        )
        checkpoint_budget = max(self.threshold - prefix_tokens - kept_tokens - 32, 64)

        text = fit_checkpoint_to_budget(
            entries,
            files,
            budget_tokens=checkpoint_budget,
            chars_per_token=cpt,
            strategy_version=self.config.strategy_version,
            reason=reason,
            config=self.config,
        )

        self.n_compactions += 1
        self.accountant.invalidate_anchor()
        record = {
            "role": "compaction",
            "content": text,
            "extra": {
                "compaction": {
                    "compaction_number": self.n_compactions,
                    "strategy_version": self.config.strategy_version,
                    "reason": reason,
                    "window_tokens": self.config.window_tokens,
                    "threshold": self.threshold,
                    "tokens_before": est_before,
                    "first_kept_message_index": cut,
                    "folded_entries": entries[-self.config.fold_max_step_entries :],
                    "files": files,
                    "timestamp": time.time(),
                }
            },
        }
        self._emit(
            "compaction",
            reason=reason,
            compaction_number=self.n_compactions,
            cut_index=cut,
            tokens_before=est_before,
            checkpoint_chars=len(text),
        )
        return record

    # --- recovery ----------------------------------------------------------------

    def make_recovery_record(self, messages: list[dict], *, reason: str, est_before: int) -> dict | None:
        """One-shot fold-all recovery for overflow / length-truncated responses.

        Keeps only the most recent action group verbatim. Consumes the single
        per-run recovery budget; returns the record to append, or None when
        recovery is unavailable or there is nothing left to fold.
        """
        if not self.enabled or not self.config.recovery_enabled or self.recovery_used:
            return None
        record = self.make_compaction_record(
            messages, reason=f"recovery:{reason}", keep_recent_tokens=0, est_before=est_before
        )
        if record is None:
            return None
        self.recovery_used = True
        self._emit("recovery", reason=reason, tokens_before=est_before, compaction_number=self.n_compactions)
        return record

    # --- helpers -------------------------------------------------------------------

    def _emit(self, event: str, **fields) -> None:
        if self.events is not None:
            self.events.emit(event, **fields)
