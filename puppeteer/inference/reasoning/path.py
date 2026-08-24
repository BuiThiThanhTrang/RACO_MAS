from enum import Enum
import uuid
from inference.graph.action_graph import ActionGraph
import os
import copy
from agent.agent_info.global_info import GlobalInfo
from inference.policy.role_aware_reinforce import ORCHESTRATOR_STOP
class ReasoningState(Enum):
    INITIALIZED = 1
    SPLITING = 2
    ANSWERING = 3
    FINALIZING = 4
    DISCARDING = 5
    AGGREGATING = 6

class GraphReasoningPath:
    def __init__(self, start_agent, max_parallel_paths, global_logger, workspace_path, action_graph:ActionGraph, registry, frontier=None, agent_sequence=None, index=None, global_info:GlobalInfo=None, state=ReasoningState.INITIALIZED, env=None, env_name=None, policy=None, max_step_num=None, external_tools_enabled=False, role_adherence=None):
        
        self.state = state
        self.index = index
        self.global_logger = global_logger
        self.workspace_path = workspace_path
        self.action_graph = action_graph
        self.registry = registry
        self.frontier = list(frontier or [])
        
        global_logger.create_logger('path{}_logger'.format(index), os.path.join(global_logger.folder_path, "path{}.log".format(index)), "INFO")
        self.logger = global_logger.get_logger('path{}_logger'.format(index))
        self.workflow_path = os.path.join(workspace_path, "path_{}.jsonl".format(index))
        self.workcode_path = os.path.join(workspace_path, "code_{}.py".format(index))

        self.start_agent = start_agent
        self.agent_sequence = list(agent_sequence or [])
        self.role_adherence = dict(role_adherence or {})
        if self.agent_sequence == []:
            self.agent_sequence.append(start_agent.unique_identifier)
        
        self.max_parallel_paths = max_parallel_paths
        self.max_step_num = max_step_num
        self.external_tools_enabled = external_tools_enabled
        
        self.current_agent = start_agent
        self.next_agents = []
        
        self.env = env
        self.env_name = env_name

        self.policy = policy

        self.global_info = global_info
        self.global_info.logger = self.logger
        self.global_info.workpath = self.workspace_path
        self.global_info.path_id = self.index

        self.logger.info("{}[Reasoning Path{} Start]{}".format("-"*30,self.index, "-"*30))
        self.logger.info("Reasoning Path{}:{}".format(self.index, state))
        self.logger.info("Start agent: {}".format(start_agent.role))
        self.logger.info("Previous Agent sequence: {}".format(self.print_agent_sequence()))

    def update_global_info(self, current_action):
        self.global_info.update(current_action)
        self.logger.info("Updated global_info: {}".format(self.global_info.__dict__))

    def step(self):
        current_action, terminated = self.current_agent.take_action(self.global_info, self.external_tools_enabled, self.env, self.env_name)
        self.current_agent.deactivate()
        self.update_global_info(current_action)
        action_name = current_action.action.get("action")
        card = self.current_agent.role_card
        adhered = (
            action_name in card.allowed_actions
            and action_name not in card.forbidden_actions
            and (
                action_name not in card.tools
                or (self.external_tools_enabled and action_name in card.tools)
            )
        )
        teammate_id = self.current_agent.hash
        self.role_adherence[teammate_id] = (
            self.role_adherence.get(teammate_id, True) and adhered
        )

        node_id = str(uuid.uuid4())
        self.action_graph.add_action(node_id, current_action.to_dict(), self.current_agent.role) 
        for successor in self.frontier:
            self.action_graph.add_dependency(successor, node_id)
        self.frontier = [node_id]

        # Deal with the case meeting the termination condition:
        # 1. The current agent is the terminator 
        # 2. The maximum number of steps is reached
        if terminated or len(self.agent_sequence) >= self.max_step_num:
            self.state = ReasoningState.FINALIZING
            self.last_agent = self.current_agent
            self.last_query_func = self.current_agent.query_func
            return self.state
        
        # Deal with the case where the current agent is the terminator
        next_agents_idx = self.policy.forward(self.global_info)
        if ORCHESTRATOR_STOP in next_agents_idx:
            self.state = ReasoningState.FINALIZING
            self.last_agent = self.current_agent
            self.last_query_func = self.current_agent.query_func
            return self.state
        self.next_agents = [
            self.registry.get_agent_from_idx(idx) for idx in next_agents_idx
        ]
        self.next_agents = [agent for agent in self.next_agents if agent is not None]
        if not self.next_agents:
            self.state = ReasoningState.FINALIZING
            self.last_agent = self.current_agent
            self.last_query_func = self.current_agent.query_func
            return self.state
        
        # Deal with the case where there is only one next agent
        if len(self.next_agents) == 1:
            self.current_agent = self.next_agents[0]
            self.current_agent.activate(global_info=self.global_info, initial_dialog_history=self.current_agent.initial_dialog_history)
            self.agent_sequence.append(self.current_agent.unique_identifier)
            self.state = ReasoningState.ANSWERING
            return self.state
        
        # Deal with the case where there are multiple next agents
        else:
            for agent in self.next_agents:
                agent.activate(global_info=self.global_info, initial_dialog_history=agent.initial_dialog_history)
            self.state = ReasoningState.SPLITING
            return self.state
        

    def split(self, current_path_num):
        split_reasoning_paths = []
        if current_path_num >= self.max_parallel_paths:
            self.current_agent = self.next_agents[0]
            self.agent_sequence.append(self.current_agent.unique_identifier)
            self.state = ReasoningState.ANSWERING
            return split_reasoning_paths
        
        for index, agent in enumerate(self.next_agents[1:self.max_parallel_paths-current_path_num+1]):
            agent_sequence = copy.deepcopy(self.agent_sequence)
            if self.env is not None:
                env = copy.deepcopy(self.env)
            else:
                env = None
            path_index = current_path_num + index
            reasoning_path = GraphReasoningPath(
                                    start_agent=agent, 
                                    max_parallel_paths=self.max_parallel_paths, 
                                    action_graph=self.action_graph,
                                    registry=self.registry,
                                    agent_sequence = agent_sequence,
                                    index=path_index,
                                    global_info=copy.deepcopy(self.global_info),
                                    state=ReasoningState.ANSWERING,
                                    global_logger=self.global_logger,
                                    workspace_path=self.workspace_path,
                                    env=env,
                                    frontier=self.frontier,
                                    policy=self.policy,
                                    max_step_num=self.max_step_num,
                                    external_tools_enabled=self.external_tools_enabled,
                                    role_adherence=copy.deepcopy(self.role_adherence),
                                    )
            reasoning_path.agent_sequence.append(agent.unique_identifier)
            reasoning_path.current_agent = agent
            reasoning_path.next_agents = []
            split_reasoning_paths.append(reasoning_path)
            print("\033[1;36mPath {} Initialized (split from path {})\033[0m".format(path_index,self.index))
        
        self.current_agent = self.next_agents[0]
        self.agent_sequence.append(self.current_agent.unique_identifier)
        self.state = ReasoningState.ANSWERING
        return split_reasoning_paths
    
    def print_agent_sequence(self):
        agent_sequence = "".join([agent.get("role") + "->" for agent in self.agent_sequence[:-1]] + [self.agent_sequence[-1].get("role")])
        return agent_sequence