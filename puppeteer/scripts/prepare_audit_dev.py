"""Freeze MMLU-Pro dev IDs and audit presets; check prerequisites without network calls."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.runtime import load_experiment_config
from role_aware.audit_trace import digest, write_json
from tasks.mmlu_pro import load_dataset

def prepare(config_path, output):
    experiment=load_experiment_config(config_path)
    data=load_dataset("dev",seed=experiment.dataset.split_seed)
    ids=[int(v) for v in data["question_id"]]
    source=Path(config_path)
    raw=yaml.safe_load(source.read_text(encoding="utf-8"))
    raw.pop("run_id",None)
    raw["dataset"]["mode"]="dev"
    raw["mode"]="initialized"
    raw.setdefault("policy",{})["policy_mode"]="initialized"
    raw["policy"].setdefault("training",{})["training"]=False
    raw["dataset"]["data_start"]=0
    raw["dataset"]["data_limit"]=None
    raw.setdefault("global_config",{}).setdefault("audit",{})["enabled"]=True
    raw["global_config"]["aggregation"]={"mode":"legacy","seed":42}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    presets={}
    for name,mode,k in (("legacy","legacy_threshold",1),("k1","categorical_set_v2",1),("k2","categorical_set_v2",2)):
        variant=copy.deepcopy(raw)
        variant.setdefault("policy",{})["routing"]={"mode":mode,"selection_count":k,
                                                  "threshold_multiplier":1.5}
        file=output/f"mmlu_audit_{name}.yaml"
        file.write_text(yaml.safe_dump(variant,sort_keys=False,allow_unicode=True),encoding="utf-8")
        presets[name]=str(file)
    with Path("config/global.yaml").open(encoding="utf-8") as stream:
        runtime=yaml.safe_load(stream)
    primary=runtime.get("task_analyzer",{}).get("primary",{})
    presence={name:bool(os.environ.get(name)) for name in
              (primary.get("base_url_env","TASK_ANALYZER_BASE_URL"),
               primary.get("api_key_env","TASK_ANALYZER_API_KEY"))}
    from agent.register.persona_loader import load_teammate_specs
    from model.query_manager import query_manager
    specs=load_teammate_specs(experiment.personas_path)
    unavailable=sorted({s.backbone for s in specs if s.backbone in query_manager.client_errors})
    report=dict(dataset="MMLU-Pro",split="dev",split_seed=experiment.dataset.split_seed,
                source_config_hash=digest(raw),dev_ids=ids,smoke_ids=ids[:20],smoke_count=min(20,len(ids)),
                presets=presets,encoder_env_present=presence,models_missing_credentials=unavailable,
                ready_for_network_smoke=all(presence.values()) and not unavailable,
                note="Presence check only; no API or endpoint request has been made.")
    write_json(output/"dev_manifest.json",report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="config/experiments/role_aware_mmlu_pro.yaml")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report=prepare(a.config,a.output)
    print(json.dumps({k:v for k,v in report.items() if k not in {"dev_ids","smoke_ids"}},ensure_ascii=False))

if __name__=="__main__":
    main()
