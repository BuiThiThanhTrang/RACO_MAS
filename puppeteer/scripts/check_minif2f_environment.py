"""Exercise packet isolation, compiler, kernel, freshness and final checker without models."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import yaml

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.execution.adapters.formal_proof import FormalProofAdapter
from routing_benchmark.v4.execution.final_evaluation import audit_domain
from routing_benchmark.v4.execution.runtime import resolve_runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load((PROJECT/'config/shared_environment/formal_minif2f_20_jev_16k.yaml').read_text(encoding='utf-8'))
    root=PROJECT/config['packet_root']
    manifests=sorted((root/'packs/formal_proof').glob('*/manifest.json'))
    assert len(manifests)==20
    for path in manifests:
        assert validate_artifact_pack(path.parent)['valid']
        state=json.loads((path.parent/'public/proof_state.json').read_text(encoding='utf-8'))
        assert '⊢' in state['initial_proof_state'] and 'RB_BEGIN' not in state['initial_proof_state']
    def check(name):
        packet='proof-minif2f-'+name.replace('_','-')
        pack=root/'packs/formal_proof'/packet
        folder=args.output/packet
        adapter=FormalProofAdapter()
        adapter.configure_runtime(resolve_runtime(config,pack/'manifest.json',PROJECT))
        env=adapter.materialize(pack,folder/'workspace')
        manifest=load_manifest(pack)
        assert sorted(p.name for p in env.workspace_dir.iterdir())==['_runtime','public']
        assert not any((env.workspace_dir/f.path).exists() for f in (*manifest.private_files,*manifest.fixture_files))
        initial=adapter.versions(env)
        result=adapter.execute_tool(env,'write_source',{'content':'theorem changed : True := by trivial'})
        assert result.status=='invalid_output' and adapter.versions(env)==initial
        assert not adapter.public_preconditions(env,'Kernel and Assumption Verifier').executable
        reference=(pack/'private/Reference.lean').read_text(encoding='utf-8')
        assert adapter.execute_tool(env,'write_source',{'content':reference}).status=='success'
        compilation=adapter.execute_tool(env,'compile_proof')
        assert compilation.status=='success',compilation.output
        verification=adapter.execute_tool(env,'verify_proof')
        assert verification.output['passed'],verification.output
        assert adapter.terminal_status(env)['terminal']
        assert adapter.execute_tool(env,'write_source',{'content':reference+'\n-- freshness probe\n'}).status=='success'
        assert not adapter.public_preconditions(env,'Kernel and Assumption Verifier').executable
        assert not adapter.terminal_status(env)['terminal']
        final=audit_domain(folder,config,PROJECT)
        assert final['status']=='PASS',final
        result={'packet_id':packet,'visibility_isolated':True,'changed_statement_rejected':True,
            'candidate_compile_pass':True,'kernel_pass':True,'stale_verification_rejected':True,
            'independent_final_checker_pass':True}
        (folder/'smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(check,['mathd_algebra_24','mathd_algebra_188']))
    report={'passed':True,'packet_checks':20,'adapter_smokes':rows,'hosted_model_calls':0,
        'benchmark_trajectories':0,'note':'Private validation proofs used only by preparation checker, never by an agent.',
        'source_hashes':{str(p.relative_to(PROJECT)):sha256_file(p) for p in [
            PROJECT/'routing_benchmark/v4/execution/adapters/formal_proof.py',
            PROJECT/'scripts/prepare_minif2f_environment.py',Path(__file__),
            PROJECT/'config/shared_environment/formal_minif2f_20_jev_16k.yaml']}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
