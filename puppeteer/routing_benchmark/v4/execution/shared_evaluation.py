"""Accounting and review export without modifying raw execution records."""
from collections import Counter


def decision_accounting(decisions):
    groups=Counter(d['dispatch_status'] for d in decisions)
    return {'router_decisions':len(decisions),'dispatched':groups['dispatched'],
        'precondition_rejected':groups['precondition_rejected'],'stop_handled':groups['stop_handled'],
        'suppressed_by_budget':groups['suppressed_by_budget'],
        'primary_routing_decisions':len(decisions)-groups['suppressed_by_budget']}


def classify_legacy(states,events):
    by_id={s['decision_id']:s for s in states}
    records=[]
    for event in events:
        if event['kind']!='router_decision': continue
        remaining=by_id[event['decision_id']]['observation']['budget']['calls_remaining']
        status='stop_handled' if event['agent_id']=='STOP' else 'suppressed_by_budget' if remaining==0 else 'dispatched'
        records.append({**event,'dispatch_status':status})
    return decision_accounting(records)


def episode_metrics(decisions,events,labels=None):
    output=decision_accounting(decisions)
    eligible=[d for d in decisions if d['dispatch_status']!='suppressed_by_budget']
    outcomes=Counter()
    roles=Counter()
    usage=Counter()
    transitions=Counter()
    executed=[d['agent_id'] for d in decisions if d['dispatch_status']=='dispatched']
    for left,right in zip(executed,executed[1:]): transitions[f'{left} -> {right}']+=1
    for event in events:
        if event.get('kind')!='role_execution': continue
        roles[event['agent_id']]+=1
        outcomes[event['status']]+=1
        for response in event['result'].get('responses',[]):
            metadata=response['metadata']
            for key,value in metadata.get('usage',{}).items():
                if isinstance(value,(int,float)): usage[key]+=value
    output.update({'role_calls':dict(roles),'role_transitions':dict(transitions),'executor_outcomes':dict(outcomes),
        'token_usage':dict(usage),'valid_action_accuracy':None,'human_review_status':'pending'})
    if labels is not None:
        scored=[d for d in eligible if d['decision_id'] in labels]
        output['human_labeled_decisions']=len(scored)
        output['valid_action_accuracy']=sum(d['agent_id'] in labels[d['decision_id']] for d in scored)/len(scored) if scored else None
    return output


def table_qa_metrics(states, decisions, events):
    """Measure public progress and stopping opportunities from recorded observations."""
    observations = [row["observation"] for row in states]
    ready = next((i for i, row in enumerate(observations) if row["current_state"]["terminal"]), None)
    final_stop = next((i for i, row in enumerate(decisions) if row["dispatch_status"] == "stop_handled"), None)
    seen = set()
    failure_before_ready = False
    for i, row in enumerate(observations):
        state = row["current_state"]
        seen.update(item["kind"] for item in state["artifacts"].values())
        if state["last_error"] and (ready is None or i < ready):
            failure_before_ready = True
    return {
        "milestones": sorted(seen),
        "ready_with_budget": any(row["current_state"]["terminal"] and row["budget"]["calls_remaining"] > 0
                                 for row in observations),
        "premature_stop_with_budget": bool(final_stop is not None and
            not observations[final_stop]["current_state"]["terminal"] and
            observations[final_stop]["budget"]["calls_remaining"] > 0),
        "calls_after_first_ready": (
            sum(row["dispatch_status"] in ("dispatched", "precondition_rejected") for row in decisions[ready:])
            if ready is not None else None
        ),
        "public_failure_observed_before_ready": failure_before_ready,
        "ready_after_public_failure": failure_before_ready and ready is not None,
        "no_progress_role_calls": sum(event.get("status") == "no_progress" for event in events),
    }
