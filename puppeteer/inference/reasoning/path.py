from enum import Enum
import copy
import os
from pathlib import Path
import shutil

from agent.agent_session import PathRuntimeContext, bind_session
from role_aware.audit_trace import digest, write_json
from dataclasses import asdict

class ReasoningState(Enum):
    INITIALIZED = 1
    SPLITING = 2
    ANSWERING = 3
    FINALIZING = 4
    DISCARDING = 5
    AGGREGATING = 6

class GraphReasoningPath:
    """A path executes one reserved action at a time; GraphReasoning owns routing."""
    def __init__(self, start_agent, max_parallel_paths, global_logger, workspace_path,
                 action_graph, registry, frontier=None, agent_sequence=None, index=None,
                 global_info=None, state=ReasoningState.INITIALIZED, env=None, env_name=None,
                 policy=None, max_step_num=None, external_tools_enabled=False, role_adherence=None,
                 audit=None, path_uid=None, parent_path_uid=None, context=None):
        self.state, self.index = state, index
        self.path_uid = path_uid or f"path-{index}"
        self.parent_path_uid = parent_path_uid
        self.context = context or PathRuntimeContext(self.path_uid)
        self.audit = audit
        self.global_logger, self.action_graph, self.registry = global_logger, action_graph, registry
        self.workspace_path = str(Path(workspace_path) / self.path_uid)
        Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
        name = f"path{index}_logger"
        global_logger.create_logger(name, os.path.join(self.workspace_path, "path.log"), "INFO")
        self.logger = global_logger.get_logger(name)
        self.frontier = list(frontier or [])
        self.agent_sequence = list(agent_sequence or [])
        self.role_adherence = dict(role_adherence or {})
        self.max_parallel_paths, self.max_step_num = max_parallel_paths, max_step_num
        self.external_tools_enabled = external_tools_enabled
        self.current_agent = self.start_agent = start_agent
        self.next_agents = []
        self.env, self.env_name, self.policy = env, env_name, policy
        self.global_info = global_info
        # The policy reconstructs baseline-like dialog input from this path only.
        global_info.path_context = self.context
        global_info.logger = self.logger
        global_info.workpath = self.workspace_path
        global_info.path_id, global_info.path_uid = index, self.path_uid
        global_info.workflow.path_id = index
        global_info.workflow.workpath = self.workspace_path
        self.pending_action_id = None
        self.pending_decision_id = None
        self.stop_reason = None
        self.completed_steps = len(global_info.workflow.workflow)

    def emit(self, event, **fields):
        if self.audit:
            return self.audit.emit(event, path_uid=self.path_uid, parent_path_uid=self.parent_path_uid, **fields)

    def finish(self, reason, decision_id=None, p_stop=None):
        if self.stop_reason is not None:
            return self.state
        if reason in {"execution_error", "cancelled"} and self.pending_action_id and self.audit:
            receipted = any(e["event_type"] == "action_finished" and e.get("action_id") == self.pending_action_id
                           for e in self.audit.events)
            if not receipted:
                self.emit("action_finished", action_id=self.pending_action_id,
                          decision_id=self.pending_decision_id, status="cancelled",
                          tokens=None, model_cost=None)
        self.stop_reason = reason
        self.state = ReasoningState.FINALIZING
        self.emit("path_finished", stop_reason=reason, decision_id=decision_id,
                  p_stop=p_stop, stop_probability_observed=p_stop is not None,
                  steps_completed=self.completed_steps, depth_limit=self.max_step_num)
        return self.state

    def reserve(self, agent, action_id, decision_id):
        self.current_agent = agent
        self.pending_action_id, self.pending_decision_id = action_id, decision_id
        self.state = ReasoningState.ANSWERING

    def step(self):
        if self.stop_reason:
            return self.state
        agent = self.current_agent
        session = self.context.session(agent.hash)
        session.workspace_path = self.workspace_path
        action_id, decision_id = self.pending_action_id, self.pending_decision_id
        if action_id is None:
            raise RuntimeError("Cannot execute an unreserved action")
        self.emit("action_started", action_id=action_id, decision_id=decision_id,
                  teammate_id=agent.hash, role=agent.role, backbone=agent.model,
                  session_id=session.session_id)
        try:
            from model.model_config import model_registry
            size = model_registry.get_model_size(agent.model) or 0
            from contextlib import nullcontext
            scope = (self.audit.call_scope(path_uid=self.path_uid, action_id=action_id,
                     decision_id=decision_id, purpose="agent", session_id=session.session_id,
                     model_size=size) if self.audit else nullcontext())
            with bind_session(agent, session), scope:
                agent.activate(self.global_info, initial_dialog_history=session.initial_dialog_history)
                try:
                    current_action, terminated = agent.take_action(
                        self.global_info, self.external_tools_enabled, self.env, self.env_name)
                finally:
                    agent.deactivate()
        except BaseException as error:
            self.emit("action_finished", action_id=action_id, decision_id=decision_id,
                      status="error", error_type=type(error).__name__)
            self.finish("cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "execution_error")
            raise
        self.agent_sequence.append(agent.unique_identifier)
        self.completed_steps += 1
        current_action.action_id = action_id
        self.global_info.update(current_action)
        self.context.record_agent_turn(agent.hash)
        action_name = current_action.action.get("action")
        card = agent.role_card
        adhered = (action_name in card.allowed_actions and action_name not in card.forbidden_actions
                   and (action_name not in card.tools or self.external_tools_enabled))
        self.role_adherence[agent.hash] = self.role_adherence.get(agent.hash, True) and adhered
        self.action_graph.add_action(action_id, current_action.to_dict(), agent.role)
        for previous in self.frontier:
            self.action_graph.add_dependency(previous, action_id)
        self.frontier = [action_id]
        self.last_agent, self.last_query_func = agent, agent.query_func
        self.emit("action_finished", action_id=action_id, decision_id=decision_id,
                  status="ok" if current_action.success == "Success" else "failed",
                  teammate_id=agent.hash, role=agent.role, backbone=agent.model,
                  tokens=current_action.tokens, model_cost=current_action.cost,
                  output=current_action.result, step=self.completed_steps,
                  candidate=self.global_info.state_answers[-1] if self.global_info.state_answers else None,
                  session_digest=digest(session.dialog_history))
        self.pending_action_id = None
        if self.completed_steps == 1 and self.audit and self.audit.enabled:
            write_json(Path(self.workspace_path) / "step_1.json", dict(
                schema_version="1.0", **self.audit.context,
                split=getattr(self.global_info, "audit_split", "unknown"),
                path_uid=self.path_uid, task=self.global_info.task,
                answers=self.global_info.answers, workflow=self.global_info.workflow.to_dict(),
                sessions={key: asdict(value) for key, value in self.context.sessions.items()},
                agent_sequence=self.agent_sequence, role_adherence=self.role_adherence,
                last_teammate=agent.hash))
        if terminated:
            return self.finish("agent_terminated")
        if self.completed_steps >= self.max_step_num:
            return self.finish("depth_limit")
        return self.state

    def fork(self, index, path_uid, agent, workspace_path):
        """Copy only mutable path data. Clients, audit sinks and autograd stay shared."""
        info = copy.copy(self.global_info)
        info.workflow = copy.copy(self.global_info.workflow)
        info.workflow.workflow = list(self.global_info.workflow.workflow)
        info.answers = list(self.global_info.answers)
        if self.env is not None:
            if not hasattr(self.env, "fork"):
                raise RuntimeError("Tool environment does not implement independent fork()")
            info.env = self.env.fork()
        child = GraphReasoningPath(agent, self.max_parallel_paths, self.global_logger,
                workspace_path, self.action_graph, self.registry, frontier=self.frontier,
                agent_sequence=self.agent_sequence, index=index, global_info=info,
                state=ReasoningState.ANSWERING, env=info.env, env_name=self.env_name,
                policy=self.policy, max_step_num=self.max_step_num,
                external_tools_enabled=self.external_tools_enabled, role_adherence=self.role_adherence,
                audit=self.audit, path_uid=path_uid, parent_path_uid=self.path_uid,
                context=self.context.fork(path_uid))
        if info.code_path:
            original = Path(info.code_path).resolve()
            source_root = Path(self.workspace_path).resolve()
            if not original.is_relative_to(source_root):
                raise RuntimeError("Cannot fork artifact outside the parent path")
            target = Path(child.workspace_path) / original.name
            shutil.copy2(original, target)
            info.code_path = str(target)
        child.emit("path_created", inherited_steps=child.completed_steps,
                   inherited_action_ids=[getattr(a, "action_id", None) for a in info.workflow.workflow])
        return child

    def print_agent_sequence(self):
        return "->".join(item.get("role", "") for item in self.agent_sequence)
