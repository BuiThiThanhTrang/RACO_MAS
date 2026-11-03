"""Collect live trajectories using a common runner and configurable decision adapter."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from routing_benchmark.routers import DecisionModelRouter, RandomRouter
from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.domain_packets.roles import role_cards_for
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.execution.registry import live_adapter_for_manifest
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.runtime import resolve_runtime
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner, write_jsonl


def source_hashes(packet_root):
    paths=list(packet_root.rglob('*'))+list((PROJECT/'routing_benchmark/v4/execution').rglob('*.py'))+list((PROJECT/'routing_benchmark/v4/domain_packets').glob('*.py'))
    paths += [PROJECT/'routing_benchmark/routers.py',PROJECT/'inference/policy/decision_model_client.py',Path(__file__)]
    return {str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p):sha256_file(p)
            for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts}


def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT/path).resolve()


def build_router(configuration,seed):
    """Shared benchmarks use raw choices by default; optional guards are explicit."""
    config=dict(configuration)
    kind=config.pop('type','decision')
    profile=config.pop('orchestration_profile','structured')
    guard=config.pop('observable_rejection_guard_enabled',False)
    if not isinstance(guard,bool):
        raise ValueError('observable_rejection_guard_enabled must be a boolean')
    if kind=='random':
        return RandomRouter(seed)
    if kind=='decision':
        return DecisionModelRouter(config,orchestration_profile=profile,observable_rejection_guard_enabled=guard)
    raise ValueError('Unknown router adapter type')


def main():
    import yaml
    from dotenv import load_dotenv
    from openai import OpenAI
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--packet',action='append',default=[])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    load_dotenv(PROJECT.parent/'.env')
    load_dotenv(PROJECT/'.env')
    packet_root=resolve(config['packet_root'])
    run=resolve(config.get('output_root','runs'))/args.run_id
    config['selected_packets']=args.packet
    digest=source_hashes(packet_root)
    if config.get('runtime_profiles'): digest[str(resolve(config['runtime_profiles']))]=sha256_file(resolve(config['runtime_profiles']))
    if config.get('role_cards'): digest[str(resolve(config['role_cards']))]=sha256_file(resolve(config['role_cards']))
    if run.exists():
        if not args.resume: raise SystemExit('Run exists; use a new ID or --resume')
        if json.loads((run/'source_hashes.json').read_text(encoding='utf-8'))!=digest:
            raise SystemExit('Frozen source changed; resume refused')
        if json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))!=config:
            raise SystemExit('Config changed; resume refused')
    else:
        run.mkdir(parents=True)
        write_json(run/'resolved_config.json',config)
        write_json(run/'source_hashes.json',digest)
    domain=config['domain']
    packets=sorted((packet_root/'packs'/domain).glob('*/manifest.json'))
    if args.packet:
        packets=[p for p in packets if p.parent.name in args.packet]
        if {p.parent.name for p in packets}!=set(args.packet): raise SystemExit('Unknown packet selection')
    if not packets: raise SystemExit('No packets selected')
    roles=(json.loads(resolve(config['role_cards']).read_text(encoding='utf-8'))
           if config.get('role_cards') else role_cards_for(domain))
    write_json(run/'role_cards.json',roles)
    def episode(index,manifest_path):
        manifest=load_manifest(manifest_path)
        folder=run/'episodes'/manifest.base_workflow_id
        if (folder/'result.json').exists():
            return json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if (folder/'workspace').exists():
            raise RuntimeError('Interrupted workspace cannot be silently reset; use a new run ID')
        adapter=live_adapter_for_manifest(manifest)
        if hasattr(adapter,'configure_runtime'):
            runtime=resolve_runtime(config,manifest_path,PROJECT)
            adapter.configure_runtime(runtime)
        write_json(folder/'runtime.json',runtime if hasattr(adapter,'configure_runtime') else {})
        router=build_router(config['router'],config.get('seed',42)+index)
        actor_config=config['executor']
        provider=actor_config.get('provider','openrouter')
        with OpenAI(api_key=os.environ[actor_config.get('api_key_env','OPENROUTER_API_KEY')],
                    base_url=actor_config.get('base_url','https://openrouter.ai/api/v1' if provider=='openrouter' else 'https://router.huggingface.co/v1'),
                    timeout=actor_config['timeout_seconds'],max_retries=0) as client:
            actor=SharedRoleActor(client,actor_config['model'],max_tokens=actor_config['max_tokens'],
                transport_retries=actor_config.get('transport_retries',5),max_tool_calls=actor_config.get('max_tool_calls',3))
            runner=SharedTrajectoryRunner(router=router,actor=actor,max_calls=config['max_calls'])
            print('[start] '+manifest.base_workflow_id,flush=True)
            result=runner.run(manifest_path=manifest_path,adapter=adapter,workspace=folder/'workspace',output_dir=folder,roles=roles,seed=config.get('seed',42)+index)
            print('[done] '+manifest.base_workflow_id+' '+result['terminal_reason']+' passed='+str(result['passed']),flush=True)
            return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('workers',3)) as pool:
        futures=[pool.submit(episode,i,p) for i,p in enumerate(packets)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            write_json(run/'summary.json',{'requested':len(packets),'completed':len(results),'passed':sum(r['passed'] for r in results),'episodes':results})
    after=source_hashes(packet_root)
    if config.get('runtime_profiles'): after[str(resolve(config['runtime_profiles']))]=sha256_file(resolve(config['runtime_profiles']))
    if config.get('role_cards'): after[str(resolve(config['role_cards']))]=sha256_file(resolve(config['role_cards']))
    write_json(run/'freeze_verification.json',{'unchanged':after==digest,'tracked_files':len(digest)})
    if after!=digest: raise RuntimeError('Source freeze mismatch')
    states=[]
    for file in sorted((run/'episodes').glob('*/decision_states.jsonl')):
        states += [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line]
    write_jsonl(run/'human_state_review_queue.jsonl',states)
    print('[complete] '+str(len(results))+' episodes; '+str(len(states))+' review states; sources unchanged',flush=True)


if __name__=='__main__': main()
