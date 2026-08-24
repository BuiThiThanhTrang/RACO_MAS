import json
import re
from utils.other_utils import JsonFormat
from copy import deepcopy
from abc import ABC, abstractmethod
from model.query_manager import query_manager
from agent.agent_info.global_info import GlobalInfo
from role_aware.schemas import TeammateSpec

class Agent(ABC):
    def __init__(self, spec: TeammateSpec, index, runtime_config=None, policy=None, global_info:GlobalInfo =None, initial_dialog_history=None) -> None:
        """
        Initialize the Agent object.
        :param spec: Internal teammate definition. Only its RoleCard is exposed to routing.
        :param index: The index to distinguish different agent instances
        :param global_info: Global configuration info, default is None
        :param initial_dialog_history: Initial dialog history, default is None
        """
        super().__init__()
        self.runtime_config = deepcopy(runtime_config or {})

        # Initialize model query function
        self.spec = spec
        self.teammate_id = spec.teammate_id
        self.role_card = spec.role_card
        self.model = spec.backbone
        self.query_func = None
        self.query_func = self._get_query_function()
        
        if not self.query_func:
            raise ValueError(f"Model '{self.model}' not implemented")
        
        # Other basic settings
        self.json_format = JsonFormat(query_func=self.query_func)
        self.role = self.role_card.role_name
        self.role_prompt = self.role_card.to_prompt()
        self.system_prompt = self.role_prompt  # Initial system prompt
        self.policy = policy
        self.index = index
        self.hash = self.teammate_id

        # Tools and file path settings
        self.actions = list(self.role_card.allowed_actions)
        self.tools = list(self.role_card.tools)
        self.root_file_path = self.runtime_config.get("file_path", {}).get("root_file_path", ".")
        if global_info:
            self.workspace_path = global_info.workpath
        
        # Activation state and dialog history
        self._activated = False
        self.initial_dialog_history = initial_dialog_history or []
        self.dialog_history = deepcopy(self.initial_dialog_history)
    
    @property
    def simplified_dialog_history(self):
        self._simplified_dialog_history = []
        for h in self.dialog_history:
            if h.get("role") == "user":
                # Mask user input 
                # "*Your previous reasoning was {}*”
                masked_text = re.sub(r'\*.*?\*', '', h["content"])
                self._simplified_dialog_history.append({"role": h["role"], "content": masked_text})
            else:
                self._simplified_dialog_history.append(h)
        return self._simplified_dialog_history

    @property
    def unique_identifier(self):
        """Return a unique identifier for the Agent instance."""
        return {
            "index": self.index,
            "role": self.role,
            "hash": self.hash
        }
    def _get_query_function(self):
        def query_func(messages, system_prompt=None):
            return query_manager.query(self.model, messages, system_prompt)
        return query_func
    
    @abstractmethod
    def activate(self, global_info, initial_dialog_history=None):
        """Activate the agent, enabling it to perform actions."""
        pass

    @abstractmethod
    def deactivate(self):
        """Deactivate the agent."""
        self._activated = False

    def reset(self):
        """Reset the agent's state, clearing dialog history and deactivating it."""
        self.dialog_history = []
        self.initial_dialog_history = []
        self.deactivate()


    @abstractmethod
    def _build_current_action(self, format_action, flag, answer, step_data):
        """Build the current workflow guiding the agent's actions."""
        pass

    @abstractmethod
    def take_action(self, global_info, external_tools_enabled=True):
        """Let the agent take an action based on the current state."""
        pass
    
    @abstractmethod
    def _execute_action(self, action, global_info):
        """Execute a specific action."""
        pass
    
    @abstractmethod
    def _reasoning_operation(self, action, global_info) -> str:
        """Perform a reasoning operation."""
        pass

    @abstractmethod
    def _answer_operation(self, global_info) -> str:
        """Generate an answer based on the current state."""
        pass

    @abstractmethod
    def _tool_operation(self, action: json, global_info) -> str:
        """Perform an operation involving external tools."""
        pass

    @abstractmethod
    def _interaction_operation(self, code, env, global_info) -> str:
        """Handle operations related to agent interaction."""
        pass
