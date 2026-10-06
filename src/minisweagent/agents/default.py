"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

from minisweagent import Environment, Model, __version__
from minisweagent.context import ContextConfig, ContextManager, EventLog, extract_usage, view_hash
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, OutputTruncatedError, TimeExceeded
from minisweagent.utils.serialize import recursive_merge

try:
    from litellm.exceptions import ContextWindowExceededError
except ImportError:  # pragma: no cover - litellm is a hard dependency in practice
    ContextWindowExceededError = ()  # type: ignore[assignment,misc]


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    context: ContextConfig = Field(default_factory=ContextConfig)
    """Context window management (projection, compaction, recovery). Disabled by default."""
    events_output_path: Path | None = None
    """Where to write the append-only JSONL event log. Defaults to output_path with an .events.jsonl suffix."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        events_path = self.config.events_output_path
        if events_path is None and self.config.output_path is not None:
            name = self.config.output_path.name
            if name.endswith(".traj.json"):  # x.traj.json -> x.events.jsonl, not x.traj.events.jsonl
                events_path = self.config.output_path.with_name(name[: -len(".traj.json")] + ".events.jsonl")
            else:
                events_path = self.config.output_path.with_suffix(".events.jsonl")
        self.events = EventLog(events_path)
        self.context = ContextManager(self.config.context, events=self.events)

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        self.events.emit(
            "crash", exit_status=type(e).__name__, n_calls=self.n_calls, n_compactions=self.context.n_compactions
        )
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.events.emit(
            "run_start",
            context=self.config.context.model_dump(),
            model=getattr(self.model.config, "model_name", ""),
            instance_id=getattr(self, "instance_id", ""),
            step_limit=self.config.step_limit,
            cost_limit=self.config.cost_limit,
        )
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        last_extra = self.messages[-1].get("extra", {})
        self.events.emit(
            "exit",
            exit_status=last_extra.get("exit_status", ""),
            n_calls=self.n_calls,
            cost=round(self.cost, 6),
            n_compactions=self.context.n_compactions,
            elapsed_seconds=int(time.time() - self._start_time),
        )
        return last_extra

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        result = self.execute_actions(self.query())
        self.after_step(result)
        self.events.emit("hook", name="after_step", step=self.n_calls)
        return result

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks.

        Sends the projected model view (not the full transcript) and, when the
        estimated tokens exceed the configured waterline, compacts first. A
        single fold-all recovery per run absorbs provider context-window errors
        and length-truncated responses.
        """
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        self.before_query()
        self.events.emit("hook", name="before_query", step=self.n_calls)
        while True:
            _view, est, record = self.context.prepare_request(self.messages)
            if record is not None:
                self.add_messages(record)
                _view = self.context.build_view(self.messages)
                est = self.context.estimate_tokens(_view)
            self.events.emit(
                "query",
                step=self.n_calls,
                est_tokens=est,
                view_len=len(_view),
                view_hash=view_hash(_view),
                window_tokens=self.config.context.window_tokens or None,
                threshold=self.context.threshold if self.context.enabled else None,
            )
            try:
                message = self.model.query(_view)
            except ContextWindowExceededError:
                recovery = self.context.make_recovery_record(self.messages, reason="overflow", est_before=est)
                if recovery is None:
                    raise  # unrecoverable: propagates to the exit-with-traceback channel
                self.add_messages(recovery)
                continue
            except OutputTruncatedError as e:
                recovery = self.context.make_recovery_record(self.messages, reason="length", est_before=est)
                if recovery is None:
                    raise  # falls back to the regular FormatError correction channel in run()
                # The billed response must never be executed; keep it as a discarded diagnostic.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                diagnostic = {
                    "role": "user",
                    "content": "",
                    "extra": {
                        "discarded": True,
                        "discard_reason": "output_truncated",
                        "response": e.messages[0].get("extra", {}).get("response"),
                    },
                }
                self.add_messages(diagnostic, recovery)
                continue
            break
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.context.update_anchor(message, len(_view))
        self.add_messages(message)
        usage = extract_usage(message)
        self.events.emit(
            "response",
            step=self.n_calls,
            cost=round(message.get("extra", {}).get("cost", 0.0), 6),
            usage={"prompt_tokens": usage[0], "completion_tokens": usage[1]} if usage else None,
            est_tokens=est,
        )
        self.after_model_response(message)
        self.events.emit("hook", name="after_model_response", step=self.n_calls)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        self.before_execute(actions)
        self.events.emit("hook", name="before_execute", step=self.n_calls, n_actions=len(actions))
        outputs = [self.env.execute(action) for action in actions]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    # --- lifecycle hooks (override to add behavior, e.g. dynamic interception) ---

    def before_query(self) -> None:
        """Called once per step, before the context view is built."""

    def after_model_response(self, message: dict) -> None:
        """Called after a successful model response was appended."""

    def before_execute(self, actions: list[dict]) -> None:
        """Called before the actions of a response are executed."""

    def after_step(self, messages: list[dict]) -> None:
        """Called after a clean step (query + execution + observations)."""

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
