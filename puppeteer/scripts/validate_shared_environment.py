"""Verify trace linkage, public visibility, context consistency and source freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.execution.adapters.common import write_json

PROJECT=Path(__file__).resolve().parents[1]
def rows(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--allow-historical-source',action='store_true',help='Permit later checkout changes only if the run recorded an unchanged source freeze at completion.')
    args=parser.parse_args()
    config=json.loads((args.run_dir/'resolved_config.json').read_text(encoding='utf-8'))
    errors=[]
    counts={'episodes':0,'states':0,'decisions':0,'role_executions':0,'model_responses':0,'checked_role_contexts':0}
    for folder in sorted((args.run_dir/'episodes').iterdir()):
        if not (folder/'result.json').exists():
            errors.append(folder.name+': incomplete episode')
            continue
        counts['episodes']+=1
        states=rows(folder/'decision_states.jsonl')
        decisions=rows(folder/'decisions.jsonl')
        events=rows(folder/'trajectory.jsonl')
        mapping={s['decision_id']:s for s in states}
        counts['states']+=len(states)
        counts['decisions']+=len(decisions)
        if len(mapping)!=len(states) or len(states)!=len(decisions): errors.append(folder.name+': state/decision join mismatch')
        for d in decisions:
            state=mapping.get(d['decision_id'])
            if not state:
                errors.append(folder.name+': missing observation')
                continue
            if state['valid_actions'] is not None or state['router_visible_private_reference']:
                errors.append(folder.name+': unexpected label/reference visibility')
            if 'valid_actions' in state['observation'] or 'private_labels' in state['observation']:
                errors.append(folder.name+': evaluator labels in public observation')
            linked=[e for e in events if e.get('decision_id')==d['decision_id'] and e['kind']=='role_execution']
            if d['dispatch_status']=='dispatched' and len(linked)!=1: errors.append(folder.name+': dispatch lacks unique execution')
            if d['dispatch_status']!='dispatched' and linked: errors.append(folder.name+': nondispatched action executed')
            if d['dispatch_status']=='suppressed_by_budget' and (d['budget_before']!=0 or d['agent_id']=='STOP'):
                errors.append(folder.name+': incorrect budget suppression')
            for event in linked:
                counts['role_executions']+=1
                result=event['result']
                for response in result.get('responses',[]):
                    counts['model_responses']+=1
                    metadata=response['metadata']
                    if not all(key in metadata for key in ('finish_reason','usage','model','generation_id')):
                        errors.append(folder.name+': incomplete response metadata')
                for request in result.get('requests',[]):
                    if request['max_tokens']!=16384: errors.append(folder.name+': token cap mismatch')
                    payload=request.get('executor_payload') or json.loads(request['messages'][1]['content'].partition('\n')[2])
                    if payload['role']['agent_id']!=d['agent_id']: errors.append(folder.name+': role mismatch')
                    if state['domain']=='database_migration_integrity' and payload['public_context']['state']!=state['observation']['current_state']:
                        errors.append(folder.name+': actor/router evidence mismatch')
                    counts['checked_role_contexts']+=1
        if config['domain']=='database_migration_integrity' and (folder/'workspace/private').exists(): errors.append(folder.name+': private workspace data')
    frozen=json.loads((args.run_dir/'source_hashes.json').read_text(encoding='utf-8'))
    changed=[]
    for name,digest in frozen.items():
        path=Path(name) if Path(name).is_absolute() else PROJECT/name
        if not path.exists() or sha256_file(path)!=digest: changed.append(name)
    freeze_path=args.run_dir/'freeze_verification.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf-8')) if freeze_path.exists() else {}
    if not freeze.get('unchanged'): errors.append('Run did not verify an unchanged source freeze at completion')
    if changed and not args.allow_historical_source: errors.append('Current checkout differs from the run source revision')
    summary=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
    if counts['episodes']!=summary['requested']: errors.append('Missing requested episodes')
    report={'valid':not errors,'counts':counts,'frozen_source_files':len(frozen),'source_files_unchanged_during_run':bool(freeze.get('unchanged')),
        'current_source_matches_run':not changed,'historical_source_allowed':args.allow_historical_source,
        'changed_files':changed,'errors':errors,'human_review_status':'pending','validator_code_sha256':sha256_file(Path(__file__))}
    write_json(args.run_dir/'execution_validation_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
