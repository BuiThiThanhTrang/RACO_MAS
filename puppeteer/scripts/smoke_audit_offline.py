"""Run the real orchestration stack with deterministic fake model responses."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
from audit_fixtures import setup
from role_aware.audit_analysis import validate_events
from role_aware.audit_trace import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    root=Path(a.output)
    if root.exists():
        p.error("Use a new output directory to avoid appending to old fixtures")
    results=[]
    cases=[
       ("depth1",1,1,lambda ids:[[ids[0]]]),
       ("stop",1,2,lambda ids:[[ids[0]],["__orchestrator_stop__"]]),
       ("depth2",1,2,lambda ids:[[ids[0]],[ids[0]]]),
       ("fork",3,2,lambda ids:[[ids[0]],[ids[0],ids[1],ids[2]]]),
    ]
    for name,width,depth,schedule in cases:
        r,policy,trace,_=setup(root/name,width=width,depth=depth,schedule=schedule)
        trace.context["run_id"] = name
        r.start(None);r.n_step(depth)
        results.append(dict(case=name,**validate_events(trace.events)))
    for mode in ("legacy_threshold","categorical_set_v2"):
        r,policy,trace,_=setup(root/mode,width=3,depth=3,mode=mode,train=True)
        trace.context["run_id"] = mode
        r.start(None);r.n_step(3)
        results.append(dict(case=mode,**validate_events(trace.events)))
    r,policy,trace,_=setup(root/"error",schedule=lambda ids:[[ids[0]]])
    def fail(*args):
        raise RuntimeError("intentional offline fixture")
    r.registry.ordered_agents[0].take_action=fail
    r.start(None)
    try:
        r.n_step(2)
    except RuntimeError:
        pass
    results.append(dict(case="error",**validate_events(trace.events)))
    write_json(root/"smoke_report.json",dict(kind="offline_fixture_not_benchmark",results=results))
    if any(v["errors"] for v in results):
        raise SystemExit(1)

if __name__=="__main__":
    main()
