"""Conversation state belongs to a path, never to the shared teammate."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import re

_BOUND_SESSION = ContextVar("agent_session", default=None)

@dataclass
class AgentSession:
    session_id: str = "standalone"
    dialog_history: list = field(default_factory=list)
    initial_dialog_history: list = field(default_factory=list)
    system_prompt: str = ""
    last_prompt: str = ""
    workspace_path: str = ""
    _activated: bool = False

@dataclass
class PathRuntimeContext:
    """Conversation state for a single branch of the reasoning graph.

    ``agent_turns`` preserves the baseline order of selected agents.  Routing
    reconstructs their simplified dialog histories from the sessions in this
    context, so sibling paths cannot share dialog state.
    """
    path_uid: str
    sessions: dict = field(default_factory=dict)
    agent_turns: list = field(default_factory=list)

    def session(self, teammate_id):
        if teammate_id not in self.sessions:
            self.sessions[teammate_id] = AgentSession(f"{self.path_uid}/{teammate_id}")
        return self.sessions[teammate_id]

    def record_agent_turn(self, teammate_id):
        self.agent_turns.append(str(teammate_id))

    @staticmethod
    def _simplified_dialog_history(dialog_history):
        """Match Agent.simplified_dialog_history from the baseline runtime."""
        result = []
        for message in dialog_history:
            copied = deepcopy(message)
            if copied.get("role") == "user":
                copied["content"] = re.sub(r"\*.*?\*", "", copied.get("content", ""))
            result.append(copied)
        return result

    def baseline_orchestrator_messages(self, question):
        """Return the baseline policy input for this path only.

        The baseline concatenates the simplified conversation of every agent in
        its selected-role sequence. Repeated selections intentionally repeat
        that agent's current dialog history.
        """
        if not self.agent_turns:
            return [{
                "role": "system",
                "content": "You are an assistant. Your task is to {}".format(question),
            }]
        history = []
        for teammate_id in self.agent_turns:
            session = self.sessions.get(teammate_id)
            if session is not None:
                history.extend(self._simplified_dialog_history(session.dialog_history))
        return history or [{
            "role": "system",
            "content": "You are an assistant. Your task is to {}".format(question),
        }]

    def fork(self, path_uid):
        result = PathRuntimeContext(
            path_uid=path_uid,
            sessions=deepcopy(self.sessions),
            agent_turns=deepcopy(self.agent_turns),
        )
        for teammate, session in result.sessions.items():
            session.session_id = f"{path_uid}/{teammate}"
            session._activated = False
        return result

@contextmanager
def bind_session(agent, session):
    token = _BOUND_SESSION.set((agent, session))
    try:
        yield session
    finally:
        _BOUND_SESSION.reset(token)

def session_field(name):
    def current(agent):
        binding = _BOUND_SESSION.get()
        if binding is not None and binding[0] is agent:
            return binding[1]
        if not hasattr(agent, "_default_session"):
            agent._default_session = AgentSession()
        return agent._default_session
    return property(lambda agent: getattr(current(agent), name),
                    lambda agent, value: setattr(current(agent), name, value))
