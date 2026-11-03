import argparse
import json
from pathlib import Path
import sys
import sqlite3
import subprocess
from contextlib import closing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing_benchmark.v4.execution.final_evaluation import audit_domain
from routing_benchmark.v4.execution.shared_evaluation import classify_legacy,episode_metrics
from routing_benchmark.v4.execution.shared_runner import write_jsonl
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.artifacts.schema import sha256_file

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]

def infrastructure_failure(events):
    for event in events:
        if event.get('kind')=='router_error': return True
        result=event.get('result',{})
        if result.get('failure_origin') in ('executor_service','tool_environment'): return True
        for item in result.get('tool_events',[]):
            tool=item.get('tool_result',{})
            if tool.get('status') in ('tool_failure','timeout'): return True
            output=tool.get('output',{})
            for key in (None,'result','kernel_probe','lookup','test_state'):
                value=output if key is None else output.get(key,{})
                if isinstance(value,dict) and (value.get('failure_origin') in ('executor_service','tool_environment') or value.get('failure_kind')=='environment'):
                    return True
    return False

def audit_database(folder):
    workspace=folder/'workspace'
    database=workspace/'working.sqlite'
    if not database.exists(): return None
    state=json.loads((workspace/'_runtime/state.json').read_text(encoding='utf-8'))
    candidate=state['artifacts'].get('candidate')
    public=workspace/'public'
    contract=json.loads((public/'execution_contract.json').read_text(encoding='utf-8'))
    probe=folder/'evaluation/final_database.sqlite'
    probe.parent.mkdir(parents=True,exist_ok=True)
    before=sha256_file(database)
    with closing(sqlite3.connect(database)) as source,closing(sqlite3.connect(probe)) as target:
        source.backup(target)
    argv=[sys.executable,str((public/'sqlite_runner.py').resolve()),'verify','--database',str(probe.resolve())]
    needs_sql=any('verify' in command and '--sql-file' in command for command in contract['allowed_commands'])
    if needs_sql and not candidate:
        return {'passed':False,'reason':'No candidate SQL for the task checker','evaluator_only':True}
    if needs_sql: argv+=['--sql-file',str((workspace/candidate['path']).resolve())]
    try:
        run=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        report={'passed':run.returncode==0,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'evaluator_only':True}
    except subprocess.TimeoutExpired:
        report={'passed':False,'reason':'Evaluator timeout','evaluator_only':True}
    report['working_database_unchanged']=before==sha256_file(database)
    if not report['working_database_unchanged']: report['passed']=False
    write_json(folder/'evaluation/final_task_audit.json',report)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--legacy',action='store_true')
    args=parser.parse_args()
    if args.legacy:
        reports=[]
        for folder in sorted((args.run_dir/'trajectories').glob('*/*')):
            if (folder/'decision_states.jsonl').exists():
                reports.append({'packet_id':folder.name,**classify_legacy(rows(folder/'decision_states.jsonl'),rows(folder/'trajectory.jsonl'))})
        total={key:sum(r[key] for r in reports) for key in ('router_decisions','dispatched','precondition_rejected','stop_handled','suppressed_by_budget','primary_routing_decisions')}
        # New derived report only; the source trajectory/state files are untouched.
        write_json(args.run_dir/'derived_budget_accounting.json',{'total':total,'episodes':reports})
        print(json.dumps(total,indent=2))
        return
    config_path=args.run_dir/'resolved_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {'domain':'database_migration_integrity'}
    reports,queue=[],[]
    for folder in sorted((args.run_dir/'episodes').iterdir()):
        if not (folder/'result.json').exists(): continue
        decisions=rows(folder/'decisions.jsonl')
        events=rows(folder/'trajectory.jsonl')
        report=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        report['infrastructure_failure']=report.get('terminal_reason')=='ROUTER_SERVICE_ERROR' or infrastructure_failure(events)
        report['evaluation_eligible']=not report['infrastructure_failure']
        report['metrics']=episode_metrics(decisions,events)
        report['metrics']['router_input_tokens']=sum(d.get('metadata',{}).get('input_tokens',0) or 0 for d in decisions)
        report['metrics']['router_latency_ms']=sum(d.get('metadata',{}).get('latency_ms',0) or 0 for d in decisions)
        responses=[response for event in events for response in event.get('result',{}).get('responses',[])]
        report['metrics']['executor_llm_requests']=len(responses)
        report['metrics']['executor_llm_latency_ms']=sum(r['metadata'].get('latency_ms',0) for r in responses)
        report['metrics']['tool_calls']=sum(len(event.get('result',{}).get('tool_events',[])) for event in events)
        report['end_task_audit']=audit_database(folder) if config['domain']=='database_migration_integrity' else audit_domain(folder,config,Path(__file__).resolve().parents[1])
        reports.append(report)
        for state in rows(folder/'decision_states.jsonl'):
            decision=next((d for d in decisions if d['decision_id']==state['decision_id']),None)
            exclusion='infrastructure' if report['infrastructure_failure'] else 'no_decision' if not decision else 'budget' if decision['dispatch_status']=='suppressed_by_budget' else None
            queue.append({**state,'run_id':args.run_dir.name,'evaluation_eligible':exclusion is None,'evaluation_exclusion_reason':exclusion})
    result={'run_id':args.run_dir.name,'evaluation_code_sha256':sha256_file(Path(__file__)),
        'episodes':reports,'completed':len(reports),'passed':sum(r['passed'] for r in reports),
        'end_task_passed':sum(bool(r['end_task_audit'] and r['end_task_audit']['passed']) for r in reports),
        'end_task_unavailable':sum(r['end_task_audit'] is None or r['end_task_audit'].get('passed') is None for r in reports),
        'router_decisions':sum(r['metrics']['router_decisions'] for r in reports),
        'primary_routing_decisions':sum(r['metrics']['primary_routing_decisions'] for r in reports),
        'suppressed_by_budget':sum(r['metrics']['suppressed_by_budget'] for r in reports),
        'human_review_status':'pending','valid_action_accuracy':None}
    attempt=args.run_dir/'attempt_status.json'
    result['attempt_status']=json.loads(attempt.read_text(encoding='utf-8')) if attempt.exists() else {'accepted_for_evaluation':None,'reason':'Pending execution validation and infrastructure audit'}
    result['infrastructure_affected_episodes']=sum(r['infrastructure_failure'] for r in reports)
    result['end_task_evaluated']=sum(r['end_task_audit'] is not None and r['end_task_audit'].get('passed') is not None for r in reports)
    write_json(args.run_dir/'evaluation_report.json',result)
    write_jsonl(args.run_dir/'human_state_review_queue.jsonl',queue)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))

if __name__=='__main__': main()
