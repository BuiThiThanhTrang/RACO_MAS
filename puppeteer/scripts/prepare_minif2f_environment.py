"""Build pinned miniF2F packets and validate Lean locally; never call hosted models."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack
from routing_benchmark.v4.domain_packets.builder import _Writer, _json_bytes
from routing_benchmark.v4.execution.adapters.formal_proof import FormalProofAdapter, proof_template, uncomment
from routing_benchmark.v4.execution.runtime import resolve_runtime


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT/'data/RoutingBench/v4_minif2f_20_20261101')
    parser.add_argument('--checks', type=Path, default=PROJECT/'runs/minif2f_environment_preparation_20261101')
    args = parser.parse_args()
    if args.output.exists() or args.checks.exists():
        raise SystemExit('Use fresh output/checks directories; preparation never overwrites a cohort.')
    source_dir = PROJECT/'data/RoutingBench/minif2f_sources_20261101'
    selection = json.loads((source_dir/'selection.json').read_text(encoding='utf-8'))
    raw = (source_dir/'minif2f.jsonl').read_bytes()
    rows = {r['name']: r for r in map(json.loads, raw.decode('utf-8').splitlines())}
    selected = selection['samples']
    assert Counter(p['difficulty'] for p in selected) == {'easy': 10, 'medium': 10}
    assert len({p['id'] for p in selected}) == 20
    args.checks.mkdir(parents=True)
    # The known runtime is reused; no global imports, tactics or options are added.
    toolchain = 'Lean 4.8.0-rc1 with Mathlib4 db651742f2c631e5b8525e9aabcf3d61ed094a4a'
    context_batch, reference_batch, packets = [], [], []
    common_context = None
    for pick in selected:
        record = rows[pick['id']]
        assert record['split'] == selection['source_split']
        source = record['lean4_code'].rstrip() + '\n'
        prefix, name = proof_template(source)
        declaration = re.search(r'\btheorem\s+', uncomment(prefix)).start()
        context = prefix[:declaration]
        # All selected upstream records share the same semantic imports/options.
        if common_context is None:
            common_context = context
        assert uncomment(context).split() == uncomment(common_context).split()
        header = prefix[declaration:]
        context_batch.append(header+' by\n  trace "RB_BEGIN_'+name+'"\n  trace_state\n  trace "RB_END_'+name+'"\n  sorry\n')
        proof = pick['private_validation_proof']
        reference = prefix+' by\n'+'\n'.join('  '+line for line in proof.splitlines())+'\n'
        reference_batch.append(header+' by\n'+'\n'.join('  '+line for line in proof.splitlines())+'\n')
        packet_id = 'proof-minif2f-'+name.replace('_','-')
        pack = args.output/'packs/formal_proof'/packet_id
        task = 'Complete the supplied Lean theorem and verify it in its original context without sorry, admit or additional axioms.'
        contract = {
            'adapter_id': 'formal_proof.lean4.v1', 'language': 'Lean 4', 'toolchain': toolchain,
            'network_during_execution': False, 'public_source': 'public/Main.lean',
            'allowed_validation_command': 'pinned lean public/Main.lean',
            'terminal_condition': 'kernel_accepts_theorem_and_no_forbidden_placeholders',
            'forbidden_placeholders': ['sorry','admit'], 'router_raw_source': False,
            'router_observation': {'proof_state':'available_after_proof_state_inspection',
                'compiler_diagnostic':'available_after_compiler_execution','candidate_status':'available_after_candidate_checking'},
            'executor_visible_files': ['public/task.md','public/Main.lean','public/proof_state.json'],
            'evaluator_oracle_visibility': 'private'}
        provenance = {k:selection[k] for k in ('source_repository','source_revision','source_file','source_split')}
        provenance.update(source_record_id=name, source_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            record_sha256=hashlib.sha256(_json_bytes(record)).hexdigest(), statement_port_changes=[],
            actual_toolchain=toolchain, difficulty=pick['difficulty'], difficulty_rationale=pick['rationale'],
            difficulty_status='project_authored_pending_human_review', private_reference_origin='locally authored validation witness; not upstream solution')
        packets.append((pack, packet_id, source, reference, task, contract, provenance, pick))
    runtime_config = {'domain':'formal_proof','lean_cache':'../.routingbench_lean_cache'}
    # Resolve using a temporary contract in the checks directory, not a benchmark packet.
    runtime_pack = args.checks/'runtime_probe'
    write_json(runtime_pack/'public/execution_contract.json', packets[0][5])
    runtime = resolve_runtime(runtime_config, runtime_pack/'manifest.json', PROJECT)
    adapter = FormalProofAdapter()
    adapter.configure_runtime(runtime)
    results = {}
    for label, bodies in [('context',context_batch),('reference',reference_batch)]:
        path = args.checks/(label+'_batch.lean')
        suffix = '' if label=='context' else '\n'+'\n'.join('#print axioms '+p['id'] for p in selected)+'\n'
        path.write_text(common_context+'\n'+'\n'.join(bodies)+suffix, encoding='utf-8')
        print('Checking '+label+' batch of 20 statements', flush=True)
        results[label] = adapter._compile(path)
        write_json(args.checks/(label+'_compile.json'), results[label])
    if any(r['returncode'] != 0 for r in results.values()):
        write_json(args.checks/'report.json', {'ready':False,'runtime':runtime,'results':results})
        raise SystemExit('Lean checks failed; inspect diagnostics. Selection is retained unchanged.')
    stdout = results['reference']['stdout']
    axioms = re.findall(r'depends on axioms:\s*\[([^\]]*)\]', stdout)
    assert len(axioms)==20 and all(a.strip() in ('','propext','Classical.choice','Quot.sound') for group in axioms for a in group.split(','))
    for pack, packet_id, source, reference, task, contract, provenance, pick in packets:
        name = pick['id']
        # Lean flushes trace_state after the trace markers and sorry warning.
        segment = results['context']['stdout'].split('RB_END_'+name+'\n',1)[1].split('RB_BEGIN_',1)[0]
        goal = '\n'.join(line for line in segment.splitlines() if ': warning:' not in line and not line.startswith('note:')).strip()
        assert '⊢' in goal
        writer = _Writer(pack)
        writer.add('public/Main.lean',source,'public')
        writer.add('public/task.md',task+'\n','public')
        writer.add('public/proof_state.json',_json_bytes({'initial_proof_state':goal,'origin':'pinned Lean trace_state; by sorry input only'}),'public')
        writer.add('public/execution_contract.json',_json_bytes(contract),'public')
        writer.add('private/Reference.lean',reference,'private')
        writer.add('private/runtime_gate.json',_json_bytes({'context_compiles':True,'private_witness_compiles':True,
            'kernel_axioms_allowed':True,'validation_mode':'shared-context batch; theorem headers unchanged',
            'reference_exposed_to_agents':False}),'private')
        writer.add('fixtures/upstream_provenance.json',_json_bytes(provenance),'fixture')
        writer.add('fixtures/LICENSE.miniF2F',(source_dir/'LICENSE.miniF2F').read_text(encoding='utf-8'),'fixture')
        manifest = writer.manifest(packet_id=packet_id,domain='formal_proof',task=task,
            adapter_id='formal_proof.lean4.v1',source_type='minif2f_goedel_evaluation_export',license_name='MIT (miniF2F statements)',
            execution_readiness='runtime_verified',metadata=provenance)
        manifest=replace(manifest,generator_version='routingbench-minif2f-packets-v1')
        write_json(pack/'manifest.json',manifest.to_dict())
        assert validate_artifact_pack(pack)['valid']
    public_selection = [{k:v for k,v in p.items() if k!='private_validation_proof'} for p in selected]
    write_json(args.output/'selection.json', {**{k:v for k,v in selection.items() if k!='samples'},'samples':public_selection})
    write_json(args.output/'source_lock.json',{'source_repository':selection['source_repository'],
        'source_revision':selection['source_revision'],'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
        'selection_sha256':hashlib.sha256((source_dir/'selection.json').read_bytes()).hexdigest(),
        'selected_tasks_sha256':hashlib.sha256(_json_bytes(public_selection)).hexdigest(),
        'source_records':len(rows),'selected_records':20,'toolchain':toolchain})
    report = {'ready':True,'packets':20,'difficulty_counts':dict(Counter(p['difficulty'] for p in selected)),
        'context_checks_passed':20,'private_witness_checks_passed':20,'kernel_axiom_checks_passed':20,
        'packet_manifests_valid':20,'runtime':runtime,'live_trajectories_run':0,'human_review_status':'pending',
        'validation_mode':'Batch compilation with common upstream context verified equal for all records; not executor performance.'}
    write_json(args.checks/'report.json',report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
