"""Context window management configuration.

The effective context window is fully config-driven and deliberately decoupled
from the provider's real limit: a 1M-token API model can be evaluated under an
8k/16k/32k budget simply by setting ``window_tokens`` (YAML or
``-c agent.context.window_tokens=...``). No window size is hardcoded anywhere.
"""

from pydantic import BaseModel


class ContextConfig(BaseModel):
    window_tokens: int = 0
    """Effective context window (tokens) used for compaction decisions.
    0 disables context management entirely (vanilla behavior).
    Decoupled from the provider's real limit on purpose: set it to
    8192 / 16384 / 32768 ... to test large-window API models under a small
    budget, or to the model's true window for small local models."""

    reserve_tokens: int = 0
    """Tokens held back for the model output and estimation error. Compaction
    triggers when the estimated context exceeds window_tokens - reserve_tokens.
    0 derives window_tokens // 5 (an ~80% waterline)."""

    keep_recent_tokens: int = 0
    """Token budget of recent messages (whole action groups) kept verbatim when
    compacting. 0 derives window_tokens // 4."""

    chars_per_token: float = 2.6
    """Fallback estimation coefficient for messages not covered by an API usage
    anchor. 2.6 fits code-heavy conversations measured against StepFun usage
    (smoke 2026-10: anchored deviation mean ~8%, worst -43% on checkpoint-dense
    views at 2.8); a lower value biases toward overestimation, which triggers
    compaction earlier and is the safe direction on real small-window models."""

    fold_max_step_entries: int = 32
    """Maximum number of folded step entries kept in a checkpoint."""

    fold_command_chars: int = 160
    """Maximum characters of the command shown per folded step entry."""

    fold_tail_lines: int = 2
    """Non-empty output lines kept per folded step entry."""

    fold_tail_chars: int = 300
    """Maximum total characters of output tail per folded step entry."""

    obs_max_chars: int = 12000
    """Observation display cap in characters (tail-priority). Exposed to
    observation_template as ``context.obs_max_chars``."""

    obs_max_lines: int = 400
    """Observation display cap in lines (tail-priority). Exposed to
    observation_template as ``context.obs_max_lines``."""

    strategy_version: str = "fold-v1"
    """Compression strategy version, frozen into trajectories and event logs."""

    recovery_enabled: bool = True
    """Allow a single fold-all recovery per run when the provider raises a
    context-window error or returns a length-truncated response."""

    def effective_reserve_tokens(self) -> int:
        return self.reserve_tokens or max(self.window_tokens // 5, 1)

    def effective_keep_recent_tokens(self) -> int:
        return self.keep_recent_tokens or max(self.window_tokens // 4, 1)

    def trigger_threshold(self) -> int:
        """Estimated-token waterline above which compaction triggers."""
        return max(self.window_tokens - self.effective_reserve_tokens(), 1)
