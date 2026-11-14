"""Write-once source/data/evidence release; verify without model calls or runtime mutation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack

RELEASE=ROOT/'benchmarks/three_domain_v1_20261101'
DOMAINS={
    'database_migration_integrity':('data/RoutingBench/v4_domain_packets_rc1',12,'config/shared_environment/database_jev_16k.yaml'),
    'code_debugging':('data/RoutingBench/v4_code_debugging_rc1',20,'config/shared_environment/code_jev_16k.yaml'),
    'formal_proof':('data/RoutingBench/v4_formal_proof_30_20261101',30,'config/shared_environment/formal_jev_16k.yaml')}
RUNS=['database_shared_jev_16k_full_20261031','code_shared_jev_gemini_v3_20261101',
      'formal_shared_jev_prover_v4_20261031','formal_minif2f_20_jev_live_20261101']

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    archive=folder/manifest['archive']
    assert sha(archive.read_bytes())==manifest['archive_sha256'],'Archive checksum mismatch'
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={e['path'] for e in manifest['files']}
        for entry in manifest['files']:assert sha(z.read(entry['path']))==entry['sha256'],entry['path']
        assert sha(z.read('database_reference/database_v1.zip'))==manifest['database_reference_sha256']
    report={'valid':True,'release_id':manifest['release_id'],'files':len(manifest['files']),
        'packets':sum(x['packets'] for x in manifest['domains'].values()),'archive_sha256':manifest['archive_sha256'],
        'database_reference_unchanged':True,'verification_scope':'Archived bytes and entry set; no live model/runtime test.'}
    write(folder/'verification_report.json',report)
    print(json.dumps(report),flush=True)

def build():
    if RELEASE.exists():raise SystemExit('Release exists. Use --verify; changes require a new release ID.')
    files={};domains={}
    def add(path,name=None):
        assert path.is_file(),str(path)
        assert path.name not in {'.env','.env.local','.env.production'},'Credential file excluded'
        files[name or 'project/'+path.relative_to(ROOT).as_posix()]=path
    for domain,(bank,count,config) in DOMAINS.items():
        packs=sorted((ROOT/bank/'packs'/domain).glob('*/manifest.json'))
        assert len(packs)==count,(domain,len(packs))
        for path in packs:
            check=validate_artifact_pack(path.parent);assert check['valid'],check
            for file in path.parent.rglob('*'):
                if file.is_file():add(file)
        # Preserve bank-level selection/runtime locks, but never caches/workspaces.
        for path in (ROOT/bank).glob('*'):
            if path.is_file() and (domain=='formal_proof' or path.suffix in ('.json','.md','.jsonl')):add(path)
        add(ROOT/config)
        domains[domain]={'packets':count,'packet_ids':[p.parent.name for p in packs],'packet_root':bank,'config':config,
            'authority':'database_reference/database_v1.zip' if domain=='database_migration_integrity' else 'project/ current source snapshot',
            'runtime_portable':False}
    db=ROOT/'benchmarks/database_v1_20261031'
    database_sha=sha((db/'database_v1.zip').read_bytes())
    assert database_sha=='be3891b46e0065855af3a29a2ea86346c9d84a91085fa2202584b90e66c97fb4'
    add(db/'database_v1.zip','database_reference/database_v1.zip')
    add(db/'manifest.json','database_reference/manifest.json')
    # Current shared implementation belongs to code/formal. It does not replace DB v1.
    for suffix in ('*.py','*.json','*.yaml'):
        for path in (ROOT/'routing_benchmark').rglob(suffix):
            if '__pycache__' not in path.parts:add(path)
    for rel in ['inference/__init__.py','inference/policy/decision_model_client.py',
        'config/shared_environment/code_roles.json','config/shared_environment/code_runtime_profiles.json',
        'requirements-shared-environment.txt','docs/domain_extension_packet_adapter_artifact_standard_vi.md',
        'docs/code_formal_environment_design_vi.md','docs/database_benchmark_protocol_vi.md',
        'docs/formal_proof_domain_integration_vi.md','docs/formal_minif2f_live_report_vi.md',
        'docs/formal_rejected_output_audit_vi.md','docs/minif2f_environment_preparation_vi.md',
        'scripts/run_shared_environment.py','scripts/evaluate_shared_environment.py','scripts/validate_shared_environment.py',
        'scripts/export_shared_review.py','scripts/score_shared_reviews.py','scripts/preflight_shared_domains.py',
        'scripts/check_shared_environment.py','scripts/prepare_minif2f_environment.py','scripts/check_minif2f_environment.py',
        'tests/test_routing_benchmark_v4_artifact_execution.py',
        'scripts/freeze_three_domain_environment.py']:
        add(ROOT/rel)
    for path in (ROOT/'docs/templates/domain_extension').glob('*'):add(path)
    for path in (ROOT/'tests').glob('test_shared*.py'):add(path)
    for rel in ['runs/formal_minif2f_live_gate_20261101/tests.json','runs/formal_minif2f_live_gate_20261101/tests.log',
                'runs/formal_minif2f_live_gate_20261101/integration_integrity.json',
                'runs/non_database_release_final_20261101/report.json']:
        add(ROOT/rel)
    for directory in ['minif2f_sources_20261101']:
        for path in (ROOT/'data/RoutingBench'/directory).glob('*'):
            if path.is_file():add(path)
    historical={}
    allowed_root={'attempt_status.json','evaluation_report.json','execution_health.json','execution_validation_report.json',
        'freeze_verification.json','resolved_config.json','role_cards.json','source_hashes.json','summary.json',
        'difficulty_summary.json','launch_gate.json','source_snapshot.zip','human_state_review_queue.jsonl'}
    for run in RUNS:
        folder=ROOT/'runs'/run
        for path in folder.iterdir():
            if path.is_file() and path.name in allowed_root:add(path)
        for ep in (folder/'episodes').iterdir():
            if not ep.is_dir():continue
            for name in ['trajectory.jsonl','decisions.jsonl','decision_states.jsonl','result.json','runtime.json']:
                path=ep/name
                if path.exists():add(path)
            for path in (ep/'evaluation').glob('*.json'):add(path)
        evaluation=json.loads((folder/'evaluation_report.json').read_text(encoding='utf-8'))
        historical[run]={k:evaluation[k] for k in ['completed','passed','end_task_passed','router_decisions','primary_routing_decisions'] if k in evaluation}
        historical[run]['source_version']='Its own source_hashes/source_snapshot; not reassigned to current checkout.'
    runtime={'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,
        'installed_packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},
        'external_runtime_note':'Code venvs, Lean/Mathlib builds, repo mirrors and provider services are not bundled; provision and preflight on restore.'}
    lean=ROOT.parent/'.routingbench_lean_cache'
    runtime['lean_projects']={}
    for project in sorted(lean.glob('mathlib-*')):
        if not project.is_dir():continue
        paths=[project/'lean-toolchain',project/'lake-manifest.json',project/'.git/HEAD']
        runtime['lean_projects'][str(project)]={str(p.relative_to(project)):{'sha256':sha(p.read_bytes()),
            'content':p.read_text(encoding='utf-8')} for p in paths if p.is_file()}
    RELEASE.mkdir(parents=True,exist_ok=False)
    write(RELEASE/'runtime_inventory.json',runtime)
    add(RELEASE/'runtime_inventory.json','release/runtime_inventory.json')
    entries=[];archive=RELEASE/'three_domain_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):
            raw=path.read_bytes();z.writestr(name,raw);entries.append({'path':name,'bytes':len(raw),'sha256':sha(raw)})
    manifest={'release_id':RELEASE.name,'date':'2026-11-01','archive':archive.name,'archive_sha256':sha(archive.read_bytes()),
        'scope':'Write-once source/config/62 packet/evidence freeze; not a runtime container or a newly rerun 62-task experiment.',
        'domains':domains,'historical_runs':historical,'database_reference_sha256':database_sha,
        'policy':'Do not overwrite this release. Domain/packet/adapter/visibility/tool/role/budget/checker changes require a new environment version. Human labels remain external.',
        'files':entries}
    write(RELEASE/'manifest.json',manifest)
    (RELEASE/'README.md').write_text('# Three-domain freeze v1 — 2026-11-01\n\n62 packets: database 12, code debugging 20, formal proof 30.\n\nDatabase authority remains database_reference/database_v1.zip inside the archive; current project source is the code/formal snapshot and does not retroactively replace DB v1 or historical run versions.\n\nDesign standard: project/docs/domain_extension_packet_adapter_artifact_standard_vi.md.\n\nVerify from puppeteer: python scripts/freeze_three_domain_environment.py --verify benchmarks/three_domain_v1_20261101\n\nThis is a source/data/evidence freeze. Provision external runtimes and preflight after restoring. No credentials, full venvs, Lean cache binaries or repository working copies are included.\n',encoding='utf-8')
    verify(RELEASE)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path);args=parser.parse_args()
    verify(args.verify.resolve()) if args.verify else build()
