"""Model-independent live routing loop. Budget boundary behavior is preserved."""
from __future__ import annotations
import json
from pathlib import Path
import time

from ..artifacts.schema import canonical_hash
from .adapters.common import write_json
from .shared_actor import RoleTransportError


def write_jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


class SharedTrajectoryRunner:
    def __init__(self, *, router, actor, max_calls=8):
        self.router,self.actor,self.max_calls=router,actor,max_calls

    def run(self, *, manifest_path, adapter, workspace, output_dir, roles, seed=42):
        env=adapter.materialize(manifest_path,workspace)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        self.router.reset(seed)
        role_map={r['agent_id']:r for r in roles}
        actions=list(role_map)+['STOP']
        events,states,decisions=[],[],[]
        calls=0
        reason='CALL_LIMIT'
        stopped=False
        def persist():
            write_jsonl(output_dir/'trajectory.jsonl',events)
            write_jsonl(output_dir/'decision_states.jsonl',states)
            write_jsonl(output_dir/'decisions.jsonl',decisions)
        def public_events():
            projected=[]
            for event in events:
                record={k:v for k,v in event.items() if k!='result'}
                # Domain visibility contracts govern raw outputs, independently of router identity.
                if 'result' in event and env.manifest.domain=='database_migration_integrity':
                    record['execution']={k:v for k,v in event['result'].items() if k not in ('requests','responses')}
                projected.append(record)
            return projected
        while calls<=self.max_calls:
            current=adapter.public_observation(env)
            identifier=f'{env.manifest.base_workflow_id}:dynamic-{len(decisions):03}'
            observation={'benchmark_version':'4.4-shared-environment-v1','task':env.manifest.task,
                'current_state':current,'agent_profiles':roles,'recent_events':public_events(),
                'budget':{'calls_remaining':max(self.max_calls-calls,0),'reserved_terminal_stop':True},
                'constraints':{'allowed_actions':actions,'private_reference_visible':False}}
            # Copy immutable JSON snapshots; no references to the mutable event list.
            states.append(json.loads(json.dumps({'decision_id':identifier,'packet_id':env.manifest.base_workflow_id,
                'domain':env.manifest.domain,'task':env.manifest.task,'agent_pool':roles,'candidate_actions':actions,
                'observation':observation,'valid_actions':None,'label_status':'pending_human_review',
                'router_visible_private_reference':False})))
            persist()
            try:
                decision=self.router.select(observation,actions)
            except Exception as error:
                events.append({'kind':'router_error','error_type':type(error).__name__,'origin':'router_transport'})
                reason='ROUTER_SERVICE_ERROR'
                break
            action=decision.action
            record={'decision_id':identifier,'agent_id':action,'budget_before':self.max_calls-calls,
                    'metadata':dict(decision.metadata),'dispatch_status':None}
            decisions.append(record)
            if action=='STOP':
                record['dispatch_status']='stop_handled'
                stopped=True
                reason='STOP' if adapter.terminal_status(env).get('terminal') else 'PREMATURE_STOP'
                break
            # Deliberately after router/STOP: preserve the original decision flow.
            if calls>=self.max_calls:
                record['dispatch_status']='suppressed_by_budget'
                break
            calls+=1
            precondition=adapter.public_preconditions(env,action,role_map.get(action))
            if not precondition.executable:
                record['dispatch_status']='precondition_rejected'
                events.append({'kind':'action_rejected','decision_id':identifier,'agent_id':action,'status':'ACTION_REJECTED',
                               'reasons':list(precondition.reasons),'origin':'environment','semantic_delta':'no_state_change'})
                persist()
                continue
            record['dispatch_status']='dispatched'
            before=current
            print(f'[action] {env.manifest.base_workflow_id} {calls}/{self.max_calls} {action}',flush=True)
            try:
                result=self.actor.execute(env=env,adapter=adapter,action=action,role_profile=role_map[action],task=env.manifest.task)
            except RoleTransportError as error:
                result={'status':'transport_error','error_type':error.cause_type,'http_status':error.http_status,
                        'failure_origin':'executor_service','raw_output':'','responses':[],'tool_events':[]}
            except Exception as error:
                result={'status':'tool_failure','error_type':type(error).__name__,'failure_origin':'tool_environment',
                        'raw_output':'','responses':[],'tool_events':[]}
            after=adapter.public_observation(env)
            events.append({'kind':'role_execution','decision_id':identifier,'agent_id':action,'status':result['status'],
                           'origin':'selected_agent','result':result,
                           'semantic_delta':'state_changed' if canonical_hash(before)!=canonical_hash(after) else 'no_state_change',
                           'state_before_hash':canonical_hash(before),'state_after_hash':canonical_hash(after)})
            persist()
        persist()
        terminal=bool(adapter.terminal_status(env).get('terminal'))
        report={'packet_id':env.manifest.base_workflow_id,'domain':env.manifest.domain,'passed':terminal and stopped,
                'terminal_contract_passed':terminal,'terminal_reason':reason,'router_stop_selected':stopped,
                'executor_calls':calls,'router_decision_count':len(decisions),
                'dispatched':sum(d['dispatch_status']=='dispatched' for d in decisions),
                'precondition_rejected':sum(d['dispatch_status']=='precondition_rejected' for d in decisions),
                'stop_handled':sum(d['dispatch_status']=='stop_handled' for d in decisions),
                'suppressed_by_budget':sum(d['dispatch_status']=='suppressed_by_budget' for d in decisions),
                'primary_routing_decisions':sum(d['dispatch_status']!='suppressed_by_budget' for d in decisions),
                'human_review_status':'pending'}
        write_json(output_dir/'result.json',report)
        return report
