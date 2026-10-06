"""Context window management: projection, accounting, folding, events.

Design (ProctorSWE harness upgrade, replaces the old compaction design doc):

1. The transcript (``agent.messages``) is append-only and never mutated;
2. ``build_model_view`` is a pure projection from transcript to the message
   list actually sent to the provider;
3. Compaction appends a ``role="compaction"`` checkpoint record and folds older
   action groups deterministically (no LLM calls);
4. Token accounting uses the provider-reported usage of the last response as an
   anchor and estimates only what was appended since;
5. The effective context window comes exclusively from ``ContextConfig``
   (YAML / ``-c agent.context.window_tokens=...``) and is decoupled from the
   provider's real limit;
6. A single fold-all recovery per run absorbs provider context-window errors and
   length-truncated responses.
"""

from minisweagent.context.accounting import TokenAccountant, estimate_message_tokens, extract_usage
from minisweagent.context.config import ContextConfig
from minisweagent.context.events import EventLog
from minisweagent.context.manager import ContextManager, view_hash
from minisweagent.context.projection import build_model_view
from minisweagent.context.projection import (
    checkpoint_message,
    find_last_compaction,
    fixed_prefix_len,
    split_action_groups,
)

__all__ = [
    "ContextConfig",
    "ContextManager",
    "EventLog",
    "TokenAccountant",
    "build_model_view",
    "checkpoint_message",
    "estimate_message_tokens",
    "extract_usage",
    "find_last_compaction",
    "fixed_prefix_len",
    "split_action_groups",
    "view_hash",
]
