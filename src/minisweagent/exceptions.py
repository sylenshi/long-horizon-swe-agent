class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


class OutputTruncatedError(FormatError):
    """Raised when a response was cut off by the output token limit
    (finish_reason == "length") before a complete action could be parsed.

    Subclasses FormatError so that, when context recovery is unavailable, the
    existing consecutive-format-error correction channel still applies. Actions
    from a truncated response are never executed.
    """
