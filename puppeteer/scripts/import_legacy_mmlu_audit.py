"""Report historical MMLU traces; unknown provenance/STOP probabilities stay unknown."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from role_aware.audit_trace import write_json

def analyze(root):
    records=[]
    for file in sorted(Path(root).rglob("trajectory_rewards.jsonl")):
        meta=file.with_name("meta.log")
        if not meta.exists():
            continue
        text=meta.read_text(encoding="utf-8")
        if "multiple-choice question" not in text:
            continue
        rewards=[json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidates={int(pid):answer.strip() for pid,answer in
                    re.findall(r"\[Aggregated Answer From Path (\d+)\]: ([^\r\n]*)",text)}
        final=re.findall(r"\[Final Answer\]: ([^\r\n]*)",text)
        if not final or not candidates:
            continue
        correct_answers={candidates[r["path_id"]].upper() for r in rewards
                         if r["task_reward"]>0 and r["path_id"] in candidates}
        final_correct=final[-1].strip().upper() in correct_answers
        records.append(dict(trace_path=str(file.parent),run_id="unknown",task_id="unknown",
            split="unknown",checkpoint="unknown",p_stop=None,stop_reason="unknown",
            candidates=candidates,final=final[-1].strip(),any_correct=bool(correct_answers),
            final_correct=final_correct,lost_correct=bool(correct_answers) and not final_correct))
    return dict(tasks=len(records),any_correct=sum(r["any_correct"] for r in records),
                final_correct=sum(r["final_correct"] for r in records),
                lost_correct=sum(r["lost_correct"] for r in records),
                records=records,note="Historical diagnostics only; cannot establish matched runs or STOP causes.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root");p.add_argument("--output",required=True)
    a=p.parse_args();result=analyze(a.root);write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}))

if __name__=="__main__":
    main()
