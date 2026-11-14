from contextlib import closing
import difflib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from routing_benchmark.routers import RoutingDecision
from routing_benchmark.v4.artifacts.schema import ArtifactManifest,ArtifactFile,sha256_file
from routing_benchmark.v4.execution.adapters.common import write_json
from routing_benchmark.v4.execution.adapters.formal_proof import FormalProofAdapter
from routing_benchmark.v4.execution.adapters.external_code_debugging import ExternalCodeDebuggingAdapter
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner
from routing_benchmark.v4.domain_packets.roles import role_cards_for

PROJECT=Path(__file__).resolve().parents[1]
CANONICAL_RELEASE=PROJECT/'data/RoutingBench/v4_beta_4d_82_20261102'
CODE_PACK=CANONICAL_RELEASE/'packs/code_debugging/code-quixbugs-flatten'
CODE_ROLES=CANONICAL_RELEASE/'agent_pools/code_debugging.json'


class SequenceRouter:
    def __init__(self,actions): self.actions=actions
    def reset(self,seed): self.index=0
    def select(self,observation,candidates):
        action=self.actions[self.index]
        self.index+=1
        return RoutingDecision(action)


class SharedDomainTests(unittest.TestCase):
    def test_code_role_tools_match_cards(self):
        adapter=ExternalCodeDebuggingAdapter()
        roles=json.loads(CODE_ROLES.read_text())
        for role in roles:
            self.assertEqual([t.name for t in adapter.tool_spec(None,role['agent_id'],role)],role['tools'])

    def test_code_missing_checkout_is_action_rejection(self):
        pack=CODE_PACK
        adapter=ExternalCodeDebuggingAdapter()
        roles=json.loads(CODE_ROLES.read_text())
        result=SharedTrajectoryRunner(router=SequenceRouter(['Patch Author','STOP']),actor=SharedRoleActor(None,'unused')).run(
            manifest_path=pack,adapter=adapter,workspace=self.root/'not-ready',output_dir=self.root/'not-ready-run',roles=roles)
        self.assertEqual(result['precondition_rejected'],1)
        self.assertEqual(result['dispatched'],0)

    def test_code_actor_stops_after_work_product_not_after_read(self):
        from routing_benchmark.v4.execution.base import ToolDefinition,ToolResult
        class Adapter:
            def tool_spec(self,*args):
                schema={'type':'object','properties':{},'additionalProperties':False}
                return [ToolDefinition('inspect_inputs','read',schema),ToolDefinition('diagnose_failure','write',schema)]
            def executor_context(self,*args): return {}
            def execute_tool(self,env,name,args): return ToolResult(name,'success',{},name)
        class Actor(SharedRoleActor):
            def _query(self,messages):
                self.calls+=1
                return json.dumps({'tool':'inspect_inputs' if self.calls==1 else 'diagnose_failure','arguments':{}}),{}
        actor=Actor(None,'fake'); actor.calls=0
        result=actor.execute(env=None,adapter=Adapter(),action='Failure Diagnostician',role_profile={},task='diagnose')
        self.assertEqual(actor.calls,2)
        self.assertEqual(len(result['tool_events']),2)
        self.assertEqual(result['status'],'success')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PROJECT/'tests')
        self.root=Path(self.temp.name)

    def tearDown(self):
        # Git can retain a checkout CWD briefly after background maintenance on Windows.
        for attempt in range(20):
            try:
                self.temp.cleanup()
                return
            except PermissionError:
                if attempt==19: raise
                time.sleep(.1)

    def formal_pack(self):
        pack=self.root/'pack'
        public=pack/'public'
        public.mkdir(parents=True)
        (public/'Main.lean').write_text('import Mathlib\n\ntheorem shared_smoke (n : Nat) : n = n := by sorry\n',encoding='utf-8')
        write_json(public/'proof_state.json',{'goal':'n = n','hypotheses':['n : Nat']})
        write_json(public/'execution_contract.json',{'adapter_id':'formal_proof.lean4.v1','forbidden_placeholders':['sorry','admit'],'router_raw_source':False})
        (public/'task.md').write_text('Prove reflexivity while preserving the theorem statement.',encoding='utf-8')
        files=tuple(ArtifactFile(p.name,'public/'+p.name,'application/json' if p.suffix=='.json' else 'text/plain',sha256_file(p),'public') for p in public.iterdir())
        manifest=ArtifactManifest('formal-shared-smoke','formal_proof','Prove reflexivity.','formal_proof.lean4.v1',files,())
        write_json(pack/'manifest.json',manifest.to_dict())
        return pack

    def formal_adapter(self):
        adapter=FormalProofAdapter()
        cache=PROJECT.parent/'.routingbench_lean_cache'
        lean=cache/'elan-home/toolchains/leanprover--lean4---v4.8.0-rc1/bin/lean.exe'
        self.assertTrue(lean.exists(),'Pinned Lean runtime is required for the real smoke test')
        project=cache/'mathlib-4.8'
        roots=[project]+[p for p in (project/'.lake/packages').iterdir() if p.is_dir()]
        libraries=[p/'.lake/build/lib' for p in roots]
        libraries=[p/'lean' if (p/'lean').is_dir() else p for p in libraries]
        adapter.configure_runtime({'lean_executable':str(lean),'project_dir':str(project),'lean_path':os.pathsep.join(str(p) for p in libraries)})
        return adapter

    def test_formal_real_compiler_and_kernel_in_shared_runner(self):
        adapter=self.formal_adapter()
        class Actor(SharedRoleActor):
            def execute(self,**kwargs):
                if kwargs['action']=='Tactic Synthesizer':
                    result=kwargs['adapter'].execute_tool(kwargs['env'],'write_source',{'content':'import Mathlib\n\ntheorem shared_smoke (n : Nat) : n = n := by rfl\n'})
                    return {'status':result.status,'raw_output':result.observable_message,'tool_events':[result.to_dict()],'responses':[]}
                return super().execute(**kwargs)
        router=SequenceRouter(['Proof State Inspector','Tactic Synthesizer','Lean Compiler Executor','Kernel and Assumption Verifier','STOP'])
        result=SharedTrajectoryRunner(router=router,actor=Actor(None,'unused')).run(manifest_path=self.formal_pack(),adapter=adapter,
            workspace=self.root/'formal-work',output_dir=self.root/'formal-run',roles=role_cards_for('formal_proof'))
        self.assertTrue(result['passed'])
        state=json.loads((self.root/'formal-work/_runtime/state.json').read_text(encoding='utf-8'))
        verification=state['artifacts']['verification']
        checks=json.loads((self.root/'formal-work'/verification['path']).read_text(encoding='utf-8'))['checks']
        self.assertTrue(checks['kernel_assumptions'])
        snapshots=[json.loads(line) for line in (self.root/'formal-run/decision_states.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertNotIn('by rfl',json.dumps([s['observation'] for s in snapshots]))

    def test_formal_rejects_statement_change_and_sorry(self):
        adapter=self.formal_adapter()
        env=adapter.materialize(self.formal_pack(),self.root/'bad-proof')
        result=adapter.execute_tool(env,'write_source',{'content':'import Mathlib\n\ntheorem shared_smoke (n : Nat) : True := by trivial\n'})
        self.assertEqual(result.status,'invalid_output')
        self.assertIn('by sorry',(env.workspace_dir/'public/Main.lean').read_text())
        self.assertIn('immutable',adapter.public_observation(env)['latest_tool_feedback']['error'])
        rejected=adapter.execute_tool(env,'read_artifact',{'artifact':'rejected_candidate'})
        self.assertIn('True := by trivial',rejected.output['content'])

    def test_formal_accepts_original_declaration_without_requiring_repeated_imports(self):
        adapter=self.formal_adapter()
        env=adapter.materialize(self.formal_pack(),self.root/'declaration-proof')
        result=adapter.execute_tool(env,'write_source',{'content':'```lean4\ntheorem wrong : True := by trivial\n```\n```lean4\ntheorem shared_smoke(n:Nat):n=n:= by rfl\n```'})
        self.assertEqual(result.status,'success')
        self.assertEqual(result.output['input_normalization']['selected_lean_block'],1)
        self.assertTrue((env.workspace_dir/'public/Main.lean').read_text().startswith('import Mathlib\n\ntheorem shared_smoke'))
        self.assertEqual(adapter.execute_tool(env,'compile_proof').status,'success')
        self.assertTrue(adapter.execute_tool(env,'verify_proof').output['passed'])

    def test_formal_normalizes_declaration_name_and_verifies_original_theorem(self):
        adapter=self.formal_adapter()
        env=adapter.materialize(self.formal_pack(),self.root/'renamed-proof')
        result=adapter.execute_tool(env,'write_source',{'content':'```lean4\ntheorem generated_name (n : Nat) : n = n := by rfl\n```'})
        self.assertEqual(result.status,'success')
        normalization=result.output['input_normalization']
        self.assertTrue(normalization['declaration_name_normalized'])
        self.assertEqual(normalization['submitted_declaration_name'],'generated_name')
        self.assertEqual(normalization['original_declaration_name'],'shared_smoke')
        source=(env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8')
        self.assertIn('theorem shared_smoke',source)
        self.assertNotIn('generated_name',source)
        self.assertEqual(adapter.execute_tool(env,'compile_proof').status,'success')
        self.assertTrue(adapter.execute_tool(env,'verify_proof').output['passed'])

    def test_formal_name_normalization_does_not_relax_statement_or_context(self):
        from routing_benchmark.v4.execution.adapters.formal_proof import constrained_source
        original='import Mathlib\ntheorem original (n : Nat) : n = n := by sorry\n'
        for candidate in ['theorem renamed (n : Nat) : True := by trivial',
                          'theorem renamed (n : Int) : n = n := by rfl',
                          'import Mathlib\naxiom cheat : False\ntheorem renamed (n : Nat) : n = n := by rfl']:
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                constrained_source(original,candidate)

    def test_formal_declaration_adapter_cannot_change_original_definitions(self):
        from routing_benchmark.v4.execution.adapters.formal_proof import constrained_source
        original='def Goal (n : Nat) : Prop := n = 37\ntheorem audit (n : Nat) (h : n = 37) : Goal n := by sorry\n'
        changed=original.replace(':= n = 37',':= True').replace('by sorry','by trivial')
        with self.assertRaises(ValueError): constrained_source(original,changed)
        string_original='def value : String := "a b"\ntheorem audit : value = value := by sorry\n'
        with self.assertRaises(ValueError): constrained_source(string_original,string_original.replace('"a b"','"ab"'))

    def test_formal_rejects_context_bypass_and_added_axiom(self):
        adapter=self.formal_adapter()
        pack=self.formal_pack()
        env=adapter.materialize(pack,self.root/'context-proof')
        for content in ['import Mathlib\naxiom injected : False\ntheorem shared_smoke (n : Nat) : n = n := False.elim injected\n',
                        'by\n  trivial\ndef injected := True']:
            self.assertEqual(adapter.execute_tool(env,'write_source',{'content':content}).status,'invalid_output')
        result=adapter.execute_tool(env,'write_premises',{'symbols':['Nat.add_comm']})
        self.assertEqual(result.output['lookup']['returncode'],0)
        self.assertIn('Nat.add_comm',result.output['lookup']['stdout'])

    def test_formal_large_artifact_is_readable_without_automatic_prompt_expansion(self):
        adapter=self.formal_adapter()
        env=adapter.materialize(self.formal_pack(),self.root/'large-artifact')
        text='analysis '+('long evidence\n'*10000)
        adapter.execute_tool(env,'write_diagnosis',{'content':text})
        context=adapter.executor_context(env,'Proof Repair Specialist')
        self.assertNotIn('content',context['state']['artifacts']['diagnosis'])
        self.assertLess(len(json.dumps(context)),10000)
        fragments=[]
        for offset in range(0,len(text),8000):
            result=adapter.execute_tool(env,'read_artifact',{'artifact':'diagnosis','offset':offset,'max_chars':8000})
            fragments.append(result.output['content'])
        self.assertEqual(''.join(fragments),text)
        class Actor(SharedRoleActor):
            def _query(self,messages):
                self.calls+=1
                return (json.dumps({'tool':'read_artifact','arguments':{'artifact':'diagnosis','max_chars':100}}) if self.calls==1 else 'by rfl'),{}
        actor=Actor(None,'fake');actor.calls=0
        result=actor.execute(env=env,adapter=adapter,action='Tactic Synthesizer',role_profile={},task='Prove reflexivity')
        self.assertEqual(actor.calls,2)
        self.assertEqual(result['status'],'success')
        self.assertTrue((env.workspace_dir/'public/Main.lean').read_text().endswith('by rfl\n'))
        self.assertIn('```lean4\nimport Mathlib\n',result['requests'][0]['messages'][1]['content'])
        self.assertIn('executor_payload',result['requests'][0])

    def test_code_real_checkout_patch_and_regression_with_shared_runner(self):
        pack=CODE_PACK
        adapter=ExternalCodeDebuggingAdapter()
        adapter.configure_runtime({'repository_mirror':str(PROJECT/'.cache/shared/QuixBugs')})
        roles=json.loads(CODE_ROLES.read_text(encoding='utf-8'))
        class Actor(SharedRoleActor):
            def execute(self,**kwargs):
                if kwargs['action']=='Patch Author':
                    repository=kwargs['adapter']._repository(kwargs['env'])
                    file=repository/'python_programs/flatten.py'
                    original=file.read_text(encoding='utf-8')
                    replacement=original.replace('yield flatten(x)','yield x')
                    patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),replacement.splitlines(keepends=True),fromfile='a/python_programs/flatten.py',tofile='b/python_programs/flatten.py'))
                    result=kwargs['adapter'].execute_tool(kwargs['env'],'repair_candidate',{'patch':patch})
                    return {'status':result.status,'raw_output':result.observable_message,'tool_events':[result.to_dict()],'responses':[]}
                return super().execute(**kwargs)
        router=SequenceRouter(['Repository Test Reproducer','Patch Author','Regression Verifier','STOP'])
        result=SharedTrajectoryRunner(router=router,actor=Actor(None,'unused')).run(manifest_path=pack,adapter=adapter,
            workspace=self.root/'code-quixbugs-flatten/workspace',output_dir=self.root/'code-quixbugs-flatten',roles=roles)
        self.assertTrue(result['passed'])
        from routing_benchmark.v4.execution.base import EnvironmentHandle
        from routing_benchmark.v4.artifacts.manifest import load_manifest
        env=EnvironmentHandle(load_manifest(pack),pack,self.root/'code-quixbugs-flatten/workspace')
        self.assertTrue(adapter.terminal_status(env)['terminal'])
        candidate=adapter._state(env)['artifacts']['candidate']
        old_hash=candidate['sha256']
        rejected=adapter.execute_tool(env,'repair_candidate',{'patch':'diff --git a/tests/new_test.py b/tests/new_test.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/new_test.py\n@@ -0,0 +1 @@\n+assert True\n'})
        self.assertEqual(rejected.status,'invalid_output')
        self.assertFalse((adapter._repository(env)/'tests/new_test.py').exists())
        context=adapter.executor_context(env,'Patch Author',roles[2])
        self.assertIn('patch_scope',context['state'])
        self.assertEqual(context['recent_tool_feedback'][-1]['status'],'invalid_output')
        self.assertEqual(sha256_file(env.workspace_dir/candidate['path']),old_hash)
        self.assertEqual(adapter.execute_tool(env,'repair_candidate',{'patch':'not a patch'}).status,'invalid_output')
        self.assertEqual(sha256_file(env.workspace_dir/candidate['path']),old_hash)
        self.assertFalse(adapter.public_preconditions(env,'Repository Test Reproducer').executable)
        before=(adapter._repository(env)/'python_programs/flatten.py').read_bytes()
        self.assertEqual(adapter.execute_tool(env,'produce_candidate').status,'rejected')
        self.assertEqual((adapter._repository(env)/'python_programs/flatten.py').read_bytes(),before)
        self.assertTrue(adapter.terminal_status(env)['terminal'])
        file=adapter._repository(env)/'python_programs/flatten.py'
        original=file.read_text(encoding='utf-8'); revised=original+'\n# second repair\n'
        patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),revised.splitlines(keepends=True),fromfile='a/python_programs/flatten.py',tofile='b/python_programs/flatten.py'))
        patch+='diff --git a/python_programs/shared_helper.py b/python_programs/shared_helper.py\nnew file mode 100644\n--- /dev/null\n+++ b/python_programs/shared_helper.py\n@@ -0,0 +1 @@\n+VALUE = 1\n'
        second=adapter.execute_tool(env,'repair_candidate',{'patch':patch})
        self.assertEqual(second.status,'success')
        self.assertEqual(second.output['patch_composition'],'cumulative_repository_diff')
        history=adapter._state(env)['candidate_history'][0]
        self.assertEqual(sha256_file(env.workspace_dir/history['path']),history['sha256'])
        self.assertFalse(adapter.terminal_status(env)['terminal'])
        self.assertEqual(adapter.execute_tool(env,'validate_candidate').status,'success')
        replay=adapter.materialize(pack,self.root/'patch-replay')
        adapter._materialize_checkout(replay)
        saved=env.workspace_dir/adapter._state(env)['artifacts']['candidate']['path']
        apply=adapter._run(['git','apply',str(saved.resolve())],cwd=adapter._repository(replay))
        self.assertEqual(apply.returncode,0,apply.stderr)
        self.assertEqual(adapter._run_tests(replay,{},scope='candidate').status,'success')
        self.assertTrue((adapter._repository(replay)/'python_programs/shared_helper.py').exists())
        from routing_benchmark.v4.execution.final_evaluation import audit_domain
        report=audit_domain(self.root/'code-quixbugs-flatten',{'domain':'code_debugging',
            'packet_root':'data/RoutingBench/v4_beta_4d_82_20261102',
            'runtime':{'repository_mirror':str(PROJECT/'.cache/shared/QuixBugs')}},PROJECT)
        self.assertTrue(report['passed'],str(report))
        file=adapter._repository(env)/'python_programs/flatten.py'
        file.write_text(file.read_text()+'\n# changed after validation\n')
        self.assertFalse(adapter.terminal_status(env)['terminal'])
        self.assertFalse(adapter.patch_path_allowed('tests/test_fix.py'))
        self.assertFalse(adapter.patch_path_allowed('../escape.py'))


if __name__=='__main__': unittest.main()
