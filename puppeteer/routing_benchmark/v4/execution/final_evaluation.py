"""Independent task checks on disposable workspaces, never agent-visible feedback."""
import json
from pathlib import Path
import shutil

from .adapters.common import read_json, write_json
from .registry import adapter_for_manifest
from ..artifacts.manifest import load_manifest
from ..artifacts.schema import sha256_file


def _spreadsheetbench_audit(adapter, env, manifest_path, episode_folder):
    """Run the submitted candidate on trusted cases without exposing feedback to the episode."""
    episode_state=read_json(episode_folder/'workspace/_runtime/state.json')
    candidate=episode_state.get('artifacts',{}).get('candidate_solution')
    if not candidate:
        return {'passed':False,'status':'FAIL','reason':'No submitted candidate solution','evaluator_only':True}
    candidate_path=episode_folder/'workspace'/candidate['path']
    if not candidate_path.is_file() or sha256_file(candidate_path)!=candidate['sha256']:
        return {'passed':False,'status':'FAIL','reason':'Submitted candidate hash mismatch','evaluator_only':True}
    pack=Path(manifest_path).parent
    target=read_json(pack/'private/evaluation_target.json')
    validation_contract=read_json(pack/'public/validation_contract.json')
    timeout=int(read_json(pack/'public/execution_contract.json')['case_timeout_seconds'])
    case_reports=[]
    for case_id in target['trusted_evaluator_case_ids']:
        input_path=(pack/'public/cases/case_1/input.xlsx') if case_id==1 else (pack/f'private/cases/case_{case_id}/input.xlsx')
        output_path=env.workspace_dir/f'_runtime/evaluator/case_{case_id}/output.xlsx'
        execution=adapter._execute_candidate(env,candidate_path,input_path,output_path,timeout)
        report={'case_id':case_id,'candidate_sha256':candidate['sha256'],'execution':execution}
        if execution.get('failure_origin')=='tool_environment':
            report.update(status='UNAVAILABLE',passed=None)
        elif execution.get('status')!='success':
            report.update(status='FAIL',passed=False)
        else:
            comparison=adapter.validate_output(input_path,output_path,validation_contract)
            report.update(status='PASS' if comparison['passed'] else 'FAIL',passed=comparison['passed'],comparison=comparison,
                          recalculation={'status':'semantic_checker','reason':'The task-specific checker accepts equivalent stored values or explicitly allowed formulas.'})
        case_reports.append(report)
    unavailable=any(item['status']=='UNAVAILABLE' for item in case_reports)
    hard_pass=all(item['status']=='PASS' for item in case_reports)
    available=[item for item in case_reports if item['status']!='UNAVAILABLE']
    report={
        'passed':None if unavailable else hard_pass,
        'status':'UNAVAILABLE' if unavailable else ('PASS' if hard_pass else 'FAIL'),
        'hard_pass':hard_pass if not unavailable else None,
        'soft_score':None if unavailable else sum(item['status']=='PASS' for item in case_reports)/len(case_reports),
        'candidate_sha256':candidate['sha256'],
        'same_candidate_hash_all_cases':all(item['candidate_sha256']==candidate['sha256'] for item in case_reports),
        'case_results':case_reports,
        'available_cases':len(available),
        'evaluator_only':True,
    }
    return report


def audit_domain(folder, config, project):
    from .runtime import resolve_runtime
    domain=config['domain']
    root=Path(config['packet_root'])
    if not root.is_absolute(): root=project/root
    manifest_path=root/'packs'/domain/folder.name/'manifest.json'
    if not manifest_path.exists(): return {'passed':None,'status':'UNAVAILABLE','reason':'Manifest not found'}
    manifest=load_manifest(manifest_path)
    adapter=adapter_for_manifest(manifest)
    adapter.configure_runtime(resolve_runtime(config,manifest_path,project))
    target=folder/'evaluation/checker_workspace'
    # Re-evaluation is read-only with respect to the trajectory. Reuse a unique target.
    ordinal=0
    while target.exists():
        ordinal+=1; target=folder/f'evaluation/checker_workspace_{ordinal}'
    try:
        env=adapter.materialize(manifest_path,target)
        if domain=='formal_proof':
            content=(folder/'workspace/public/Main.lean').read_text(encoding='utf-8')
            result=adapter.execute_tool(env,'write_source',{'content':content})
            if result.status!='success': return {'passed':False,'status':'FAIL','reason':'Immutable context/edit scope violated'}
            compiled=adapter.execute_tool(env,'compile_proof')
            if compiled.output['result'].get('failure_origin')=='tool_environment':
                return {'passed':None,'status':'UNAVAILABLE','evidence':compiled.output}
            result=adapter.execute_tool(env,'verify_proof')
            report={'passed':result.output['passed'],'evidence':result.output}
        elif domain=='code_debugging':
            adapter._materialize_checkout(env)
            original=folder/'workspace/_external/repository'
            changed=adapter._run(['git','diff','--name-only','HEAD'],cwd=original).stdout.splitlines() if original.exists() else []
            changed=[p for p in changed if adapter.patch_path_allowed(p)]
            if changed:
                diff=adapter._run(['git','diff','--binary','HEAD','--',*changed],cwd=original)
                patch=target/'_runtime/final.patch'
                patch.write_text(diff.stdout,encoding='utf-8',newline='\n')
                applied=adapter._run(['git','apply',str(patch.resolve())],cwd=adapter._repository(env))
                if applied.returncode: return {'passed':False,'status':'FAIL','reason':'Final implementation cannot be reconstructed','stderr':applied.stderr}
            for name in (adapter._run(['git','ls-files','--others','--exclude-standard'],cwd=original).stdout.splitlines() if original.exists() else []):
                if adapter.patch_path_allowed(name):
                    source=(original/name).resolve()
                    if not source.is_relative_to(original.resolve()) or source.is_symlink(): raise ValueError('Unsafe source path')
                    destination=adapter._repository(env)/name
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,destination)
            result=adapter._run_tests(env,{},scope='candidate')
            evidence=dict(result.output)
            if result.status in ('timeout','tool_failure') or evidence.get('return_code') in (3,4,5):
                return {'passed':None,'status':'UNAVAILABLE','evidence':evidence}
            report={'passed':evidence.get('return_code')==0,'evidence':evidence}
        elif domain=='structured_data_processing' and manifest.adapter_id=='spreadsheetbench.python_oj.v1':
            report=_spreadsheetbench_audit(adapter,env,manifest_path,folder)
        else: return {'passed':None,'status':'UNAVAILABLE','reason':'No checker for domain'}
        if report.get('status')!='UNAVAILABLE':
            report.update(status='PASS' if report['passed'] else 'FAIL')
        report['evaluator_only']=True
        write_json(folder/'evaluation/final_task_audit.json',report)
        return report
    except Exception as error:
        report={'passed':None,'status':'UNAVAILABLE','reason':type(error).__name__+': '+str(error),'evaluator_only':True}
        write_json(folder/'evaluation/final_task_audit.json',report)
        return report
