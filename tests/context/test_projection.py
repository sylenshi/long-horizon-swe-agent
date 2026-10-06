"""Projection tests: identity view, compaction view, atomic groups, cut points."""

from minisweagent.context.config import ContextConfig
from minisweagent.context.projection import (
    build_model_view,
    checkpoint_message,
    find_last_compaction,
    fixed_prefix_len,
    select_cut_index,
    split_action_groups,
)


def make_transcript(n_steps: int = 5, obs_chars: int = 40) -> list[dict]:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task"},
    ]
    for i in range(n_steps):
        messages.append({"role": "assistant", "content": f"thought {i}", "extra": {"actions": [{"command": f"cmd {i}"}]}})
        messages.append(
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>\n{'x' * obs_chars}\n</output>",
                "extra": {"returncode": 0, "raw_output": "x" * obs_chars},
            }
        )
    return messages


def make_toolcall_transcript(n_steps: int = 4) -> list[dict]:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task"},
    ]
    for i in range(n_steps):
        tool_call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": f"thought {i}",
                "tool_calls": [{"function": {"name": "bash", "arguments": f'{{"command": "cmd {i}"}}'}}],
                "extra": {"actions": [{"command": f"cmd {i}", "tool_call_id": tool_call_id}]},
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"<returncode>0</returncode>",
                "extra": {"returncode": 0, "raw_output": f"out {i}"},
            }
        )
    return messages


class TestIdentityProjection:
    def test_no_compaction_is_identity(self):
        messages = make_transcript()
        view = build_model_view(messages)
        assert view == messages

    def test_internal_roles_filtered(self):
        messages = make_transcript(2)
        messages.append({"role": "exit", "content": "Submitted", "extra": {}})
        view = build_model_view(messages)
        assert all(m["role"] != "exit" for m in view)

    def test_discarded_messages_filtered(self):
        messages = make_transcript(2)
        messages.append({"role": "user", "content": "", "extra": {"discarded": True, "discard_reason": "output_truncated"}})
        view = build_model_view(messages)
        assert all(not m.get("extra", {}).get("discarded") for m in view)


class TestCompactionProjection:
    def test_view_structure_after_compaction(self):
        messages = make_transcript(6)
        cut = 6  # fold steps 0..1 (messages 2..5), keep from message 6 on
        messages.insert(
            6 + 2,  # irrelevant where; append at end instead
            {"role": "compaction", "content": "", "extra": {}},
        )
        # rebuild cleanly: use a compaction record placed at the end
        messages = make_transcript(6)
        messages.append(
            {
                "role": "compaction",
                "content": "checkpoint text",
                "extra": {"compaction": {"first_kept_message_index": cut}},
            }
        )
        view = build_model_view(messages)
        assert view[0] is messages[0]  # pinned system prefix
        assert view[1] is messages[1]  # pinned original task
        assert view[2] == checkpoint_message(messages[-1])
        assert view[3] is messages[cut]  # kept region starts at the recorded index
        assert all(m["role"] != "compaction" for m in view)
        # the folded messages (2..cut) are absent — compare by identity:
        # the homogeneous fixtures make `in`/`==` value-equal a different message
        assert not any(m is messages[2] for m in view)
        assert not any(m is messages[cut - 1] for m in view)

    def test_original_task_always_preserved_verbatim(self):
        messages = make_transcript(6)
        messages.append(
            {
                "role": "compaction",
                "content": "x" * 500,
                "extra": {"compaction": {"first_kept_message_index": 8}},
            }
        )
        view = build_model_view(messages)
        assert view[1]["content"] == "original task"

    def test_find_last_compaction(self):
        messages = make_transcript(2)
        assert find_last_compaction(messages) == -1
        messages.append({"role": "compaction", "content": "a", "extra": {"compaction": {}}})
        messages.append({"role": "assistant", "content": "t", "extra": {"actions": [{"command": "c"}]}})
        messages.append({"role": "compaction", "content": "b", "extra": {"compaction": {}}})
        assert find_last_compaction(messages) == len(messages) - 1

    def test_projection_is_deterministic(self):
        messages = make_transcript(6)
        messages.append(
            {
                "role": "compaction",
                "content": "checkpoint",
                "extra": {"compaction": {"first_kept_message_index": 6}},
            }
        )
        assert build_model_view(messages) == build_model_view(messages)


class TestActionGroups:
    def test_text_mode_groups(self):
        messages = make_transcript(3)
        groups = split_action_groups(messages)
        assert len(groups) == 3
        assert [g[0] for g in groups] == [2, 4, 6]
        # each group: assistant + its observation
        assert all(len(g) == 2 for g in groups)

    def test_toolcall_mode_groups(self):
        messages = make_toolcall_transcript(3)
        groups = split_action_groups(messages)
        assert len(groups) == 3
        assert [g[0] for g in groups] == [2, 4, 6]
        assert all(len(g) == 2 for g in groups)

    def test_format_error_correction_attaches_to_previous_group(self):
        messages = make_transcript(2)
        messages.insert(
            4,
            {"role": "user", "content": "format error correction", "extra": {"interrupt_type": "FormatError"}},
        )
        groups = split_action_groups(messages)
        assert len(groups) == 2
        assert 4 in groups[0]  # correction belongs to the first group, groups stay atomic

    def test_prefix_detection(self):
        assert fixed_prefix_len(make_transcript(1)) == 2
        assert fixed_prefix_len([{"role": "system", "content": "s"}]) == 1
        assert fixed_prefix_len([]) == 0


class TestCutIndex:
    def test_cut_respects_keep_budget(self):
        config = ContextConfig(window_tokens=100000, keep_recent_tokens=500)
        messages = make_transcript(10, obs_chars=200)  # each group ~60 tokens
        cut = select_cut_index(messages, config.effective_keep_recent_tokens(), config.chars_per_token)
        prefix = fixed_prefix_len(messages)
        assert cut > prefix
        # everything from the cut on is kept verbatim
        assert build_model_view(messages)[prefix:]  # sanity: view non-empty
        groups = split_action_groups(messages)
        cut_is_group_boundary = any(g[0] == cut for g in groups)
        assert cut_is_group_boundary

    def test_cut_never_inside_group(self):
        config = ContextConfig(window_tokens=100000, keep_recent_tokens=1)
        messages = make_transcript(6, obs_chars=100)
        cut = select_cut_index(messages, 1, config.chars_per_token)
        groups = split_action_groups(messages)
        assert any(g[0] == cut for g in groups)
        assert cut == groups[-1][0]  # minimal budget keeps exactly the last group

    def test_keep_recent_zero_keeps_single_group(self):
        messages = make_transcript(6, obs_chars=100)
        cut = select_cut_index(messages, 0, 4.0)
        groups = split_action_groups(messages)
        assert cut == groups[-1][0]

    def test_no_groups_returns_len(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
        ]
        assert select_cut_index(messages, 1000, 4.0) == len(messages)
