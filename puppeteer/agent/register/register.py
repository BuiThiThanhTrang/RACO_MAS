from typing import Any
from agent.reasoning_agent import Reasoning_Agent
from agent.register.persona_loader import load_teammate_specs
from role_aware.schemas import TeammateSpec

class AgentRegister:
    def __init__(self):
        self.agents = {}
        self.unique_agents = {}
        self._ordered_agents = []
        self._agent_personas = []
        self._runtime_config = {}
        self._loaded = False

    def _register_agent(self, name, agent):
        if agent.hash in self.unique_agents:
            raise ValueError(f"Duplicate teammate_id: {agent.hash}")
        self.agents[agent.hash] = agent
        self.unique_agents[agent.hash] = agent
        self._ordered_agents.append(agent)
    
    def __call__(self, *args: Any, **kwds: Any):
        def decorator(cls):
            agent = cls(*args, **kwds)
            self._register_agent(agent.role, agent)
            return cls
        return decorator

    @property 
    def agent_config(self):
        return self._agent_personas
    
    @property
    def ordered_agents(self):
        return tuple(self._ordered_agents)

    @property
    def agent_num(self):
        return len(self._ordered_agents)
    
    @property
    def agent_names(self):
        return self.agents.keys()
    
    @property
    def agent_identifiers(self):
        return self.unique_agents.keys()
    
    def get_agent_from_name(self, name):
        if name in self.agents:
            return self.agents[name]
        return next((agent for agent in self._ordered_agents if agent.role == name), None)
    
    def get_agent_from_idx(self, idx):
        return self.unique_agents.get(idx) 

    def create_agent(self, name):
        raise RuntimeError("Agents are immutable run-scoped objects; load a new registry instead")

    def load(self, teammate_specs, runtime_config=None):
        if self._loaded:
            raise RuntimeError("Agent registry is already loaded for this run")
        self._agent_personas = list(teammate_specs)
        self._runtime_config = dict(runtime_config or {})
        for index, spec in enumerate(self._agent_personas):
            self._initialize_agent(index, spec, self._runtime_config)
        self._loaded = True

    def register_all_agents(self, personas_path, runtime_config=None):
        self.load(load_teammate_specs(personas_path), runtime_config=runtime_config)
    
    def reset_episode_state(self):
        for agent in self._ordered_agents:
            agent.reset()

    def reset_all_agents(self):
        self.reset_episode_state()

    def clear(self):
        self.agents.clear()
        self.unique_agents.clear()
        self._ordered_agents.clear()
        self._agent_personas = []
        self._runtime_config = {}
        self._loaded = False
            
    def _initialize_agent(self, index, spec: TeammateSpec, runtime_config):
        agent = Reasoning_Agent(spec=spec, index=index, runtime_config=runtime_config)
        self._register_agent(spec.teammate_id, agent)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)


agent_global_registry = AgentRegister()