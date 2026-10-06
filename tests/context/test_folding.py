"""Folding tests: entries, file extraction, deterministic rendering, budget shrink."""

import copy

from minisweagent.context.config import ContextConfig
from minisweagent.context.folding import (
    extract_files,
    extract_fold_entries,
    fit_checkpoint_to_budget,
    render_checkpoint,
)


def make_transcript(n_steps: int = 4, obs_lines: int = 5) -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_steps):
        messages.append(
            {
                "role": "assistant",
                "content": f"thought {i}",
                "extra": {"actions": [{"command": f"cat src/file{i}.py"}]},
            }
        )
        output = "\n".join(f"line {i}-{j}" for j in range(obs_lines))
        messages.append({"role": "user", "content": f"<output>\n{output}\n</output>", "extra": {"returncode": 0, "raw_output": output}})
    return messages


class TestFoldEntries:
    def test_one_entry_per_action(self):
        config = ContextConfig()
        entries = extract_fold_entries(range(2, 10), make_transcript(4), config)
        assert len(entries) == 4
        assert entries[0]["command"] == "cat src/file0.py"
        assert entries[0]["returncode"] == 0
        assert "line 0-4" in entries[0]["tail"]  # last non-empty lines kept

    def test_tail_capped(self):
        config = ContextConfig(fold_tail_lines=2, fold_tail_chars=50)
        entries = extract_fold_entries(range(2, 6), make_transcript(2, obs_lines=50), config)
        assert entries[0]["tail"].count("\n") <= 1  # at most 2 lines
        assert len(entries[0]["tail"]) <= 50 + len("…[+9999 chars]")

    def test_toolcall_mode_matches_by_id(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "t"},
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        {"command": "first", "tool_call_id": "a"},
                        {"command": "second", "tool_call_id": "b"},
                    ]
                },
            },
            {"role": "tool", "tool_call_id": "b", "content": "out-b", "extra": {"returncode": 1, "raw_output": "out-b"}},
            {"role": "tool", "tool_call_id": "a", "content": "out-a", "extra": {"returncode": 0, "raw_output": "out-a"}},
        ]
        entries = extract_fold_entries([2, 3, 4], messages, ContextConfig())
        assert entries[0]["command"] == "first"
        assert entries[0]["returncode"] == 0
        assert entries[1]["returncode"] == 1

    def test_format_error_corrections_not_treated_as_observations(self):
        messages = make_transcript(1)
        messages.append(
            {"role": "user", "content": "correction", "extra": {"interrupt_type": "FormatError"}}
        )
        messages.append({"role": "assistant", "content": "retry", "extra": {"actions": [{"command": "ls"}]}})
        entries = extract_fold_entries(range(2, len(messages)), messages, ContextConfig())
        assert len(entries) == 2
        assert entries[0]["tail"] != "correction"


class TestFileExtraction:
    def test_read_and_modified_classification(self):
        files = extract_files(
            [
                "cat src/main.py",
                "grep -rn TODO src/",
                "sed -i 's/old/new/g' lib/util.py",
                "echo hi > /tmp/out.txt",
                "python -c 'import os'",
                "mv a.py b.py",
            ]
        )
        assert "src/main.py" in files["read"]
        assert "src/" in files["read"]
        assert "lib/util.py" in files["modified"]
        assert "/tmp/out.txt" in files["modified"]
        assert "b.py" in files["modified"]
        assert files["touched"] == []

    def test_deterministic(self):
        commands = ["cat b.py", "cat a.py", "rm z.py"]
        assert extract_files(commands) == extract_files(list(reversed(commands)))

    def test_capped_lists(self):
        commands = [f"cat file{i}.py" for i in range(80)]
        files = extract_files(commands)
        assert len(files["read"]) <= 50
        assert files["read_omitted"] == [30]


class TestCheckpointRendering:
    def test_deterministic_bytes(self):
        config = ContextConfig()
        entries = extract_fold_entries(range(2, 10), make_transcript(4), config)
        files = extract_files(["cat a.py"])
        text1 = render_checkpoint(entries, files, strategy_version="fold-v1", reason="threshold", config=config)
        text2 = render_checkpoint(copy.deepcopy(entries), copy.deepcopy(files), strategy_version="fold-v1", reason="threshold", config=config)
        assert text1 == text2
        assert "[compacted steps | strategy=fold-v1 | reason=threshold]" in text1
        assert "[step 2]" in text1

    def test_budget_shrink_reduces_size(self):
        config = ContextConfig()
        entries = extract_fold_entries(range(2, 22), make_transcript(10, obs_lines=20), config)
        files = extract_files([f"cat file{i}.py" for i in range(10)])
        big = fit_checkpoint_to_budget(
            entries, files, budget_tokens=10000, chars_per_token=4.0, strategy_version="fold-v1", reason="threshold", config=config
        )
        small = fit_checkpoint_to_budget(
            entries, files, budget_tokens=40, chars_per_token=4.0, strategy_version="fold-v1", reason="threshold", config=config
        )
        assert len(small) < len(big)
        assert "[compacted steps" in small  # header survives every shrink level
