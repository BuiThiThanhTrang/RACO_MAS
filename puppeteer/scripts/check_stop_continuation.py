"""Compare STOP versus one fixed teammate from the identical stored first-step prefix."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from role_aware.audit_trace import digest,write_json
from role_aware.aggregation import normalize_choice

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",help=".../path-.../step_1.json")
    parser.add_argument("--teammate-id",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=Path(args.snapshot).resolve()
    snapshot=json.loads(source.read_text(encoding="utf-8"))
    if snapshot["split"]!="dev":
        parser.error("This diagnostic requires a dev snapshot")
    target=Path(args.output)
    if target.exists():
        parser.error("Use a new output directory")
    manifest_file=next((p/"manifest.json" for p in source.parents
                        if p.name=="audit" and (p/"manifest.json").is_file()),None)
    if manifest_file is None:
        parser.error("Source run audit manifest is missing")
    if snapshot.get("manifest_hash"):
        manifest_file = manifest_file.parent / "manifests" / (snapshot["manifest_hash"] + ".json")
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    if not manifest.get("personas_path"):
        parser.error("Source manifest lacks pool location; regenerate an audited dev run")
    from agent.register.register import AgentRegister
    from inference.graph.agent_graph import AgentGraph
    from inference.graph.action_graph import ActionGraph
    from role_aware.profile_store import ProfileStore
    from role_aware.checkpointing import pool_fingerprint
    from role_aware.stop_probe import run_probe
    registry=AgentRegister()
    runtime=manifest["config"]["runtime"]
    registry.register_all_agents(manifest["personas_path"],runtime)
    if pool_fingerprint(registry.agent_config)!=manifest["pool_hash"]:
        parser.error("Pool differs from source snapshot")
    profiles=ProfileStore();profiles.initialize(registry.agent_config)
    allowed_tools = tuple(manifest.get("allowed_tools", ()))
    graph=AgentGraph(registry,profiles,allowed_tools=allowed_tools)
    outcomes={}
    for name,continuation in (("stop",False),("continue",True)):
        registry.reset_episode_state()
        outcomes[name]=run_probe(snapshot,registry,graph,ActionGraph(allowed_tools=allowed_tools),runtime,
                                target/name,args.teammate_id,continuation)
    # Labels are joined only after both predictions have been generated.
    evaluation=json.loads(source.parent.parent.joinpath("evaluation.json").read_text(encoding="utf-8"))
    gold=normalize_choice(evaluation["gold"])
    for outcome in outcomes.values():
        outcome["correct"]=normalize_choice(outcome["prediction"])==gold
    events=[json.loads(line) for line in source.parent.parent.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()]
    observed=[e.get("p_stop") for e in events if e["event_type"]=="routing_decision"
              and e.get("path_uid")==snapshot["path_uid"] and e.get("steps_completed")==1]
    report=dict(source_hash=digest(snapshot),source_path=str(source),task_id=snapshot["task_id"],
                teammate_id=args.teammate_id,p_stop=observed[0] if observed else None,
                outcomes=outcomes,accuracy_change=int(outcomes["continue"]["correct"])-int(outcomes["stop"]["correct"]),
                note="One-state diagnostic; never select continuation by its gold label.")
    write_json(target/"comparison.json",report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":
    main()
