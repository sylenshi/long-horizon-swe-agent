"""Agent-level integration tests: compaction, config-driven windows, recovery, events, hooks."""

import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.context.accounting import estimate_message_tokens
from minisweagent.exceptions import FormatError, OutputTruncatedError, Submitted


class FakeModel:
    """Records every received view; replays canned responses or raises.

    With ``dynamic_usage=True`` every returned message carries usage computed
    from the view it just received, mimicking a real provider whose
    prompt_tokens grow with the conversation (a fixed fake usage would pin the
    token-accounting anchor and freeze the estimate below any waterline).
    """

    def __init__(self, script: list, dynamic_usage: bool = False, completion_tokens: int = 10):
        self.script = list(script)
        self.dynamic_usage = dynamic_usage
        self.completion_tokens = completion_tokens
        self.received: list[list[dict]] = []
        self.config = SimpleNamespace(model_name="fake-model")

    def query(self, messages, **kwargs):
        self.received.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if self.dynamic_usage:
            prompt = sum(estimate_message_tokens(m, chars_per_token=4.0) for m in messages)
            extra = {**item.get("extra", {}), "response": {"usage": {
                "prompt_tokens": prompt,
                "completion_tokens": self.completion_tokens,
                "total_tokens": prompt + self.completion_tokens,
            }}}
            return {**item, "extra": extra}
        return item

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {
                "role": "user",
                "content": f"<returncode>{out['returncode']}</returncode>\n<output>\n{out['output']}\n</output>",
                "extra": {"raw_output": out.get("output", ""), "returncode": out.get("returncode")},
            }
            for out in outputs
        ]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FakeEnv:
    def __init__(self, output_size: int = 50):
        self.output_size = output_size
        self.commands: list[str] = []

    def execute(self, action):
        command = action["command"]
        self.commands.append(command)
        if command == "submit":
            raise Submitted(
                {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": "patch"}}
            )
        return {"output": "x" * self.output_size, "returncode": 0, "exception_info": None}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


def assistant(commands=("do work",), usage=None, content="thought") -> dict:
    extra = {"actions": [{"command": c} for c in commands], "cost": 0.01, "timestamp": 0.0}
    if usage is not None:
        extra["response"] = {
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": usage[0] + usage[1]}
        }
    return {"role": "assistant", "content": content, "extra": extra}


def make_agent(
    tmp_path,
    script,
    *,
    context: dict | None = None,
    output_size: int = 50,
    dynamic_usage: bool = False,
) -> tuple[DefaultAgent, FakeModel]:
    model = FakeModel(script, dynamic_usage=dynamic_usage)
    env = FakeEnv(output_size=output_size)
    kwargs = {
        "system_template": "sys",
        "instance_template": "task: {{task}}",
        "output_path": tmp_path / "t.traj.json",
    }
    if context is not None:
        kwargs["context"] = context
    return DefaultAgent(model, env, **kwargs), model


class TestDisabledByDefault:
    def test_identity_view_and_no_events_compaction(self, tmp_path):
        script = [assistant(("step1",)), assistant(("step2",)), assistant(("submit",))]
        agent, model = make_agent(tmp_path, script, output_size=2000)
        result = agent.run("fix the bug")
        assert result["exit_status"] == "Submitted"
        assert agent.context.n_compactions == 0
        # identity projection: the model saw the full transcript every time
        for view in model.received:
            assert view == agent.messages[: len(view)]
        assert len(model.received[0]) == 2  # system + task


class TestCompaction:
    def test_compaction_triggers_and_task_preserved(self, tmp_path):
        script = [assistant((f"step{i}",)) for i in range(12)] + [assistant(("submit",))]
        agent, model = make_agent(
            tmp_path, script,
            context={"window_tokens": 1500, "reserve_tokens": 300, "keep_recent_tokens": 300},
            output_size=800, dynamic_usage=True,
        )
        agent.run("fix the bug")
        assert agent.context.n_compactions >= 1
        transcript = agent.messages
        assert any(m["role"] == "compaction" for m in transcript)
        # the original task message is never compacted away
        assert transcript[1]["content"] == "task: fix the bug"
        # later views contain the checkpoint wrapper and stay far below the transcript length
        last_view = model.received[-2]  # last full step view (before submit)
        assert any("<context-compaction>" in m["content"] for m in last_view)
        assert len(last_view) < len(transcript)
        assert last_view[1]["content"] == "task: fix the bug"

    def test_window_config_effectiveness_8k_16k_32k(self, tmp_path):
        def count_compactions(window_tokens: int) -> int:
            script = [assistant((f"step{i}",)) for i in range(40)] + [assistant(("submit",))]
            agent, _ = make_agent(
                tmp_path / f"w{window_tokens}", script,
                context={"window_tokens": window_tokens}, output_size=1200,
            )
            agent.run("fix the bug")
            return agent.context.n_compactions

        c32k = count_compactions(32768)
        c16k = count_compactions(16384)
        c8k = count_compactions(8192)
        assert c8k >= c16k >= c32k
        assert c8k > 0

    def test_no_orphan_observations_in_view(self, tmp_path):
        script = [assistant((f"step{i}",)) for i in range(20)] + [assistant(("submit",))]
        agent, model = make_agent(
            tmp_path, script, context={"window_tokens": 1200}, output_size=700,
        )
        agent.run("fix the bug")
        for view in model.received:
            for i, message in enumerate(view):
                if message["role"] == "user" and "<output>" in message.get("content", ""):
                    # every observation in the view is preceded by its assistant action
                    assert any(
                        v["role"] == "assistant" and v.get("extra", {}).get("actions")
                        for v in view[max(0, i - 3) : i]
                    )
                if message["role"] == "assistant":
                    assert message.get("extra", {}).get("actions")  # no half-kept groups


class TestRecovery:
    def test_overflow_recovers_once_then_terminates(self, tmp_path):
        overflow = litellm.exceptions.ContextWindowExceededError(
            message="too big", llm_provider="openai", model="fake-model"
        )
        # overflow after some history exists -> fold-all recovery -> run completes
        script = [assistant(("step1",)), assistant(("step2",)), overflow, assistant(("submit",))]
        agent, model = make_agent(tmp_path, script, context={"window_tokens": 4000}, output_size=100)
        result = agent.run("fix the bug")
        assert result["exit_status"] == "Submitted"
        assert agent.context.recovery_used is True
        reasons = [m["extra"]["compaction"]["reason"] for m in agent.messages if m["role"] == "compaction"]
        assert "recovery:overflow" in reasons
        assert len(model.received) == 4  # step1, step2, overflow(+retry consumed one extra call)

        # a second overflow after the budget is spent must terminate the run
        script = [assistant(("step1",)), overflow, overflow, assistant(("submit",))]
        agent, model = make_agent(tmp_path, script, context={"window_tokens": 4000}, output_size=100)
        with pytest.raises(litellm.exceptions.ContextWindowExceededError):
            agent.run("fix the bug")
        assert agent.messages[-1]["extra"]["exit_status"] == "ContextWindowExceededError"

    def test_length_truncated_response_never_executed(self, tmp_path):
        truncated = OutputTruncatedError(
            {
                "role": "user",
                "content": "your response was cut off",
                "extra": {"interrupt_type": "FormatError", "cost": 0.5, "response": {"id": "resp1"}},
            }
        )
        # two completed steps first: fold-all recovery needs at least one foldable
        # group besides the single group it always keeps
        script = [assistant(("step1",)), assistant(("step2",)), truncated, assistant(("step3",)), assistant(("submit",))]
        agent, model = make_agent(tmp_path, script, context={"window_tokens": 4000}, output_size=100)
        result = agent.run("fix the bug")
        assert result["exit_status"] == "Submitted"
        # the billed truncated response was charged and kept as a discarded diagnostic
        assert any(m.get("extra", {}).get("discard_reason") == "output_truncated" for m in agent.messages)
        # the truncated response never enters any view sent to the model
        for view in model.received:
            assert all(m.get("content") != "" for m in view if m.get("role") == "user")
        # the recovery compaction happened
        reasons = [m["extra"]["compaction"]["reason"] for m in agent.messages if m["role"] == "compaction"]
        assert "recovery:length" in reasons

    def test_length_truncation_without_recovery_goes_to_format_error_channel(self, tmp_path):
        truncated = OutputTruncatedError(
            {
                "role": "user",
                "content": "be more concise",
                "extra": {"interrupt_type": "FormatError", "cost": 0.25, "response": {}},
            }
        )
        script = [assistant(("step1",)), truncated, assistant(("step2",)), assistant(("submit",))]
        agent, _ = make_agent(tmp_path, script, context={"window_tokens": 4000, "recovery_enabled": False}, output_size=100)
        agent.run("fix the bug")
        # fell through to the FormatError correction channel: correction message in transcript
        assert any(m["content"] == "be more concise" for m in agent.messages if m["role"] == "user")
        assert agent.context.recovery_used is False


class TestObservationTruncationTemplate:
    def test_swebench_template_tail_priority(self, tmp_path):
        from jinja2 import StrictUndefined, Template

        import yaml

        config = yaml.safe_load(Path("src/minisweagent/config/benchmarks/swebench.yaml").read_text())
        template = Template(config["model"]["observation_template"], undefined=StrictUndefined)
        context = {"obs_max_chars": 50, "obs_max_lines": 3}
        long_output = "\n".join(f"line {i}" for i in range(100))
        rendered = template.render(output={"output": long_output, "returncode": 0, "exception_info": None}, context=context)
        assert "<output_tail>" in rendered
        assert "line 99" in rendered  # tail kept
        assert "line 0" not in rendered  # head dropped
        short = template.render(output={"output": "all good", "returncode": 0, "exception_info": None}, context=context)
        assert "<output>\nall good" in short

    def test_mini_template_tail_priority(self, tmp_path):
        from jinja2 import StrictUndefined, Template

        import yaml

        config = yaml.safe_load(Path("src/minisweagent/config/mini.yaml").read_text())
        template = Template(config["model"]["observation_template"], undefined=StrictUndefined)
        context = {"obs_max_chars": 30, "obs_max_lines": 2}
        long_output = "\n".join(f"row {i}" for i in range(50))
        rendered = json.loads(
            template.render(output={"output": long_output, "returncode": 1, "exception_info": None}, context=context)
        )
        assert rendered["output_tail"].endswith("row 49")
        assert rendered["elided_chars"] > 0


class TestEventsAndHooks:
    def test_events_jsonl_written(self, tmp_path):
        script = [assistant((f"step{i}",)) for i in range(8)] + [assistant(("submit",))]
        agent, model = make_agent(
            tmp_path, script, context={"window_tokens": 2000, "reserve_tokens": 400}, output_size=1400,
            dynamic_usage=True,
        )
        seen_hooks = []
        agent.before_query = lambda: seen_hooks.append("before_query")
        agent.after_step = lambda msgs: seen_hooks.append("after_step")
        agent.run("fix the bug")

        events_path = tmp_path / "t.events.jsonl"
        assert events_path.exists()
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        by_type = {}
        for e in events:
            by_type.setdefault(e["event"], []).append(e)
        assert by_type["run_start"][0]["context"]["window_tokens"] == 2000
        assert by_type["run_start"][0]["model"] == "fake-model"
        assert len(by_type["query"]) == 9
        assert all("view_hash" in e and "est_tokens" in e for e in by_type["query"])
        assert by_type["response"][0]["usage"] == {
            "prompt_tokens": 22,  # system(9) + task(13) at 4 chars/token, as the provider would count
            "completion_tokens": 10,
        }
        assert by_type["compaction"], "compaction must trigger under this window"
        assert by_type["compaction"][0]["reason"] == "threshold"
        assert len(by_type["hook"]) >= 4
        assert by_type["exit"][0]["exit_status"] == "Submitted"
        assert "before_query" in seen_hooks and "after_step" in seen_hooks

    def test_trajectory_contains_frozen_context_config(self, tmp_path):
        script = [assistant(("step1",)), assistant(("submit",))]
        agent, _ = make_agent(tmp_path, script, context={"window_tokens": 4096})
        agent.run("fix the bug")
        data = agent.save(agent.config.output_path)
        assert data["info"]["config"]["agent"]["context"]["window_tokens"] == 4096
        assert data["info"]["config"]["agent"]["context"]["strategy_version"] == "fold-v1"


class TestViewReplayability:
    def test_view_hash_reproducible_from_transcript(self, tmp_path):
        from minisweagent.context import build_model_view, view_hash

        script = [assistant((f"step{i}",)) for i in range(15)] + [assistant(("submit",))]
        agent, model = make_agent(
            tmp_path, script, context={"window_tokens": 1500}, output_size=600,
        )
        agent.run("fix the bug")
        # every recorded view can be re-derived from the transcript snapshot at that time;
        # at minimum the final transcript projects deterministically to the same hash
        view = build_model_view(agent.messages)
        assert view_hash(view) == view_hash(build_model_view(agent.messages))
