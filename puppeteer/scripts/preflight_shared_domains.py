"""Check actual domain tools and baselines without hosted model calls."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.execution.adapters.formal_proof import proof_template
from routing_benchmark.v4.execution.registry import adapter_for_manifest
from routing_benchmark.v4.execution.runtime import resolve_runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    root=PROJECT/config['packet_root']
    args.output.mkdir(parents=True,exist_ok=False)
    def check(path):
        packet=path.parent.name
        try:
            validation=validate_artifact_pack(path.parent)
            if not validation['valid']: raise ValueError(str(validation))
            manifest=load_manifest(path)
            adapter=adapter_for_manifest(manifest)
            runtime=resolve_runtime(config,path,PROJECT)
            adapter.configure_runtime(runtime)
            env=adapter.materialize(path,args.output/packet/'workspace')
            if config['domain']=='code_debugging':
                result=adapter.execute_tool(env,'produce_candidate',{})
                output=dict(result.output)
                ready=output.get('test_state',{}).get('outcome') in ('pass','fail')
                raw=output.get('raw_output','')
                if output.get('return_code') in (2,3,4,5) or 'ModuleNotFoundError' in raw or 'ImportError:' in raw: ready=False
            else:
                prefix,_=proof_template((env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8'))
                probe=env.workspace_dir/'_runtime/context_preflight.lean'
                probe.write_text(prefix+' by sorry\n',encoding='utf-8')
                output=adapter._compile(probe)
                output['scope']='immutable context/toolchain only; initial proof is task input'
                ready=output.get('returncode')==0 and not output.get('failure_origin')
            row={'packet_id':packet,'ready':ready,'runtime':runtime,'baseline':output}
        except Exception as error:
            row={'packet_id':packet,'ready':False,'reason':type(error).__name__+': '+str(error)}
        write_json(args.output/packet/'preflight.json',row)
        print(packet+' ready='+str(row['ready']),flush=True)
        return row
    paths=sorted((root/'packs'/config['domain']).glob('*/manifest.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(check,paths))
    write_json(args.output/'report.json',{'domain':config['domain'],'packets':len(rows),'ready':sum(r['ready'] for r in rows),'rows':rows})


if __name__=='__main__': main()
