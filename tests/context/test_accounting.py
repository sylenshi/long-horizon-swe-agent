"""Accounting tests: estimation, usage extraction, anchor semantics."""

from minisweagent.context.accounting import TokenAccountant, estimate_message_tokens, extract_usage


def msg(content: str, **extra) -> dict:
    return {"role": "user", "content": content, "extra": extra}


def assistant_with_usage(prompt: int, completion: int) -> dict:
    return {
        "role": "assistant",
        "content": "ok",
        "extra": {
            "actions": [{"command": "ls"}],
            "response": {"usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}},
        },
    }


class TestEstimation:
    def test_estimate_scales_with_chars(self):
        assert estimate_message_tokens(msg("x" * 400), 4.0) == 108  # 100 + 8 overhead

    def test_tool_call_json_counted(self):
        base = estimate_message_tokens(msg(""), 4.0)
        with_calls = estimate_message_tokens(
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}]},
            4.0,
        )
        assert with_calls > base

    def test_chars_per_token_coefficient(self):
        short = estimate_message_tokens(msg("x" * 80), 4.0)
        longer = estimate_message_tokens(msg("x" * 80), 2.0)
        assert longer > short


class TestUsageExtraction:
    def test_extracts_usage(self):
        assert extract_usage(assistant_with_usage(100, 20)) == (100, 20)

    def test_missing_usage_returns_none(self):
        assert extract_usage(msg("no response")) is None
        assert extract_usage({"role": "assistant", "content": "x", "extra": {"response": {}}}) is None

    def test_zero_usage_returns_none(self):
        assert extract_usage(assistant_with_usage(0, 0)) is None


class TestAnchor:
    def test_anchor_increments_only_new_messages(self):
        accountant = TokenAccountant(chars_per_token=4.0)
        view = [msg("a" * 400)] * 5  # 5 x 108 tokens = 540 estimated
        accountant.update_anchor(assistant_with_usage(1000, 100), view_len=4)
        # anchor: 1000 + 100 = 1100 covers first 4 messages; only the 5th is estimated
        assert accountant.estimate_tokens(view) == 1100 + estimate_message_tokens(view[4], 4.0)

    def test_invalidated_anchor_falls_back_to_full_estimate(self):
        accountant = TokenAccountant(chars_per_token=4.0)
        view = [msg("a" * 400)] * 5
        accountant.update_anchor(assistant_with_usage(1000, 100), view_len=4)
        accountant.invalidate_anchor()
        assert accountant.estimate_tokens(view) == sum(estimate_message_tokens(m, 4.0) for m in view)

    def test_stale_anchor_view_shrunk_falls_back(self):
        accountant = TokenAccountant(chars_per_token=4.0)
        view = [msg("a" * 400)] * 5
        accountant.update_anchor(assistant_with_usage(1000, 100), view_len=6)  # view shrank (shouldn't happen)
        assert accountant.estimate_tokens(view) == sum(estimate_message_tokens(m, 4.0) for m in view)
