"""Conversation state belongs to a path, never to the shared teammate."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field

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
    path_uid: str
    sessions: dict = field(default_factory=dict)

    def session(self, teammate_id):
        if teammate_id not in self.sessions:
            self.sessions[teammate_id] = AgentSession(f"{self.path_uid}/{teammate_id}")
        return self.sessions[teammate_id]

    def fork(self, path_uid):
        result = PathRuntimeContext(path_uid, deepcopy(self.sessions))
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
