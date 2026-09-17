"""Controlled one-step continuation from a saved, label-free MMLU prefix."""
from pathlib import Path
import copy
from agent.agent_session import AgentSession, PathRuntimeContext
from agent.agent_info.workflow import Action
from inference.reasoning.path import GraphReasoningPath
from inference.reasoning.reasoning import GraphReasoning
from role_aware.audit_trace import AuditTrace

class ProbePolicy:
    def finalize_task(self, transition, info):
        pass
    def update(self):
        return {}
    def abort_task(self):
        pass

def restore_prefix(reasoning, snapshot):
    snapshot = copy.deepcopy(snapshot)
    registry=reasoning.registry
    last=registry.get_agent_from_idx(snapshot["last_teammate"])
    if last is None:
        raise ValueError("Snapshot teammate does not exist in supplied pool")
    info=reasoning._new_info(0)
    for record in snapshot["workflow"]:
        action=Action(record["action"],record["result"],record["success"],
                      record["agent"],record["backbone"])
        action.tokens=record["tokens"]; action.cost=record["cost"]
        action.action_id=record.get("action_id")
        info.workflow.workflow.append(action)
    info.answers=list(snapshot["answers"])
    uid=reasoning.audit.new_id("path")
    context=PathRuntimeContext(uid)
    context.sessions={key:AgentSession(**value) for key,value in snapshot["sessions"].items()}
    context=context.fork(uid)
    path=GraphReasoningPath(last,1,reasoning.global_logger,reasoning.workspace_path,
         reasoning.action_graph,registry,index=0,global_info=info,policy=reasoning.policy,
         max_step_num=len(snapshot["workflow"])+1,audit=reasoning.audit,path_uid=uid,
         context=context,agent_sequence=snapshot["agent_sequence"],
         role_adherence=snapshot["role_adherence"])
    path.last_agent=last;path.last_query_func=last.query_func
    path.frontier = [snapshot["workflow"][-1].get("action_id")] if snapshot["workflow"] else []
    reasoning.reasoning_paths=[path]
    path.emit("path_created",inherited_steps=len(snapshot["workflow"]),
              inherited_action_ids=[a.get("action_id") for a in snapshot["workflow"]])
    return path

def run_probe(snapshot, registry, graph, action_graph, runtime, output, teammate_id, continue_step):
    if snapshot["task"]["type"] != "MMLU-Pro":
        raise ValueError("Stop probe currently supports MMLU-Pro only")
    trace=AuditTrace(output,"stop_probe",snapshot["task_id"],
                     "continue" if continue_step else "stop")
    config=dict(runtime,audit_split=snapshot["split"])
    reasoning=GraphReasoning(snapshot["task"],graph,ProbePolicy(),action_graph,1,
                            len(snapshot["workflow"])+1,config,registry,audit=trace,
                            external_tools_enabled=bool(getattr(graph, "allowed_tools", ())))
    trace.emit("task_started",split=snapshot["split"],diagnostic="forced_continuation")
    for action in snapshot["workflow"]:
        aid = action.get("action_id", "")
        if aid and aid.startswith("action-"):
            trace.counters["action"] = max(trace.counters.get("action", 0), int(aid.split("-")[-1]))
    path=restore_prefix(reasoning,snapshot)
    if continue_step:
        agent=registry.get_agent_from_idx(teammate_id)
        if agent is None:
            raise ValueError("Unknown continuation teammate")
        decision=trace.new_id("decision");action=trace.new_id("action")
        trace.emit("routing_decision",path_uid=path.path_uid,decision_id=decision,
                   mode="forced_diagnostic",p_stop=None,selected=[teammate_id])
        trace.emit("allocation",path_uid=path.path_uid,decision_id=decision,
                   accepted=[dict(action_id=action,action=teammate_id,path_uid=path.path_uid)],
                   rejected=[],reason="forced_diagnostic")
        path.reserve(agent,action,decision)
        path.step()
    else:
        path.finish("policy_stop")
    prediction,_=reasoning.finalize()
    return dict(prediction=prediction,cost=trace.call_totals(),trace_path=str(trace.directory))
