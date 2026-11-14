"""Score human-confirmed action sets without rerunning models or task checkers."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from routing_benchmark.v4.execution.shared_evaluation import episode_metrics
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.artifacts.schema import sha256_file


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def score(run,labels_file):
    states={s['decision_id']:s for path in (run/'episodes').glob('*/decision_states.jsonl') for s in rows(path)}
    labels={}
    for row in rows(labels_file):
        if row.get('run_id')!=run.name:
            raise ValueError('Every label must identify this exact run_id')
        if row.get('label_status')!='human_verified' or not row.get('reviewer'):
            raise ValueError('Only human_verified labels with a reviewer may be scored')
        identifier=row['decision_id']
        if identifier not in states or identifier in labels:
            raise ValueError('Unknown or duplicate decision_id')
        actions=row.get('valid_actions')
        if not isinstance(actions,list) or any(not isinstance(a,str) or a not in states[identifier]['candidate_actions'] for a in actions):
            raise ValueError('valid_actions must list actions from the original candidate pool')
        labels[identifier]=actions
    episodes=[]; correct=count=eligible_total=0
    for folder in sorted((run/'episodes').iterdir()):
        if not (folder/'result.json').exists(): continue
        decisions=rows(folder/'decisions.jsonl')
        metrics=episode_metrics(decisions,rows(folder/'trajectory.jsonl'),labels)
        eligible_total+=metrics['primary_routing_decisions']
        labeled=metrics['human_labeled_decisions']
        metrics['human_review_status']='human_verified' if labeled and labeled==metrics['primary_routing_decisions'] else 'partial' if labeled else 'pending'
        selected=[d for d in decisions if d['dispatch_status']!='suppressed_by_budget' and d['decision_id'] in labels]
        correct+=sum(d['agent_id'] in labels[d['decision_id']] for d in selected)
        count+=len(selected)
        episodes.append({'packet_id':folder.name,'metrics':metrics})
    return {'run_id':run.name,'human_confirmed_label_rows':len(labels),'scored_decisions':count,
            'eligible_decisions':eligible_total,'human_review_status':'human_verified' if count and count==eligible_total else 'partial' if count else 'pending',
            'correct_decisions':correct,'valid_action_accuracy':correct/count if count else None,
            'labels_sha256':sha256_file(labels_file),'scoring_code_sha256':sha256_file(Path(__file__)),
            'scope':'Labeled eligible decisions only; this does not assert all states have been reviewed.',
            'episodes':episodes}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--labels',type=Path,required=True)
    args=parser.parse_args()
    report=score(args.run_dir,args.labels)
    write_json(args.run_dir/'human_action_evaluation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))


if __name__=='__main__': main()
