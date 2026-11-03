"""Pinned local Lean execution through the shared environment protocol."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from ..base import EnvironmentHandle, PreconditionResult, ToolDefinition, ToolResult
from ...artifacts.manifest import load_manifest
from ...artifacts.schema import sha256_file
from .common import read_json, write_json
from .database import DatabaseMigrationAdapter

ROLES = {'Proof State Inspector':'inspect_proof', 'Premise and Lemma Retriever':'write_premises',
    'Tactic Synthesizer':'write_source','Proof Repair Specialist':'write_source',
    'Lean Compiler Executor':'compile_proof','Kernel and Assumption Verifier':'verify_proof',
    'Formal Failure Diagnostician':'write_diagnosis'}


def uncomment(text,mask_strings=True):
    # Mask comments and strings while keeping offsets into the original source.
    out=list(text); i=0; depth=0; string=False
    while i<len(text):
        if depth:
            if text[i:i+2]=='/-': depth+=1; out[i:i+2]='  '; i+=2
            elif text[i:i+2]=='-/': depth-=1; out[i:i+2]='  '; i+=2
            else:
                if text[i]!='\n': out[i]=' '
                i+=1
        elif string:
            if mask_strings: out[i]='\n' if text[i]=='\n' else ' '
            if text[i]=='\\' and i+1<len(text):
                if mask_strings: out[i+1]=' '
                i+=2
            elif text[i]=='"': string=False; i+=1
            else: i+=1
        elif text[i:i+2]=='/-': depth=1; out[i:i+2]='  '; i+=2
        elif text[i:i+2]=='--':
            end=text.find('\n',i); end=len(text) if end<0 else end
            out[i:end]=' '*(end-i); i=end
        elif text[i]=='"':
            string=True
            if mask_strings: out[i]=' '
            i+=1
        else: i+=1
    return ''.join(out)


def proof_template(source):
    masked=uncomment(source)
    declarations=list(re.finditer(r'\b(?:theorem|lemma)\s+([\w.]+)\b',masked))
    if len(declarations)!=1:
        raise ValueError('This protocol requires exactly one editable theorem per packet')
    start=masked.find(':=',declarations[0].end())
    if start<0: raise ValueError('No proof assignment found')
    return source[:start+2],declarations[0].group(1)


def constrained_source(original,content):
    prefix,_=proof_template(original)
    masked=uncomment(content)
    if re.search(r'\b(?:theorem|lemma|import|def|namespace)\b',masked):
        try:
            candidate_prefix,_=proof_template(content)
        except ValueError:
            raise ValueError('Context and theorem statement are immutable') from None
        def parts(text):
            start=re.search(r'\b(?:theorem|lemma)\b',uncomment(text)).start()
            return text[:start],text[start:]
        def tokens(text):
            return re.findall(r'"(?:\\.|[^"\\])*"|\w+|:=|=>|->|<=|>=|==|!=|[^\s]',uncomment(text,mask_strings=False))
        def header_tokens(text):
            # A repeated declaration names the output, not the proposition. Keep
            # the packet's name while checking every other header token exactly.
            match=re.search(r'\b(?:theorem|lemma)\s+([\w.]+)\b',uncomment(text))
            return tokens(text[:match.start(1)]+'__packet_theorem__'+text[match.end(1):])
        original_context,original_header=parts(prefix)
        candidate_context,candidate_header=parts(candidate_prefix)
        if header_tokens(candidate_header)!=header_tokens(original_header) or (tokens(candidate_context) and tokens(candidate_context)!=tokens(original_context)):
            raise ValueError('Context and theorem statement are immutable')
        body=content[len(candidate_prefix):]
    else: body=content.strip()
    # A proof term may contain local have/let, but cannot introduce commands.
    if re.search(r'\b(?:theorem|lemma|def|axiom|opaque|constant|abbrev|namespace|section|end|import|instance|attribute|set_option|syntax|macro|elab|initialize|run_cmd|variable|universe)\b|#',uncomment(body)):
        raise ValueError('Only a proof term is editable; top-level commands are forbidden')
    return prefix+' '+body.strip()+'\n'


def _blocks(text):
    depth, i, out = 0, 0, []
    while i < len(text):
        if text[i:i+2]=='/-': depth+=1; i+=2
        elif depth and text[i:i+2]=='-/': depth-=1; i+=2
        else:
            if not depth: out.append(text[i])
            i+=1
    return ''.join(out)


class FormalProofAdapter(DatabaseMigrationAdapter):
    adapter_id='formal_proof.lean4.v1'

    def __init__(self):
        self.runtime={}

    def configure_runtime(self, profile):
        self.runtime=dict(profile or {})

    def materialize(self,manifest_path,workspace):
        pack=Path(manifest_path)
        if pack.is_file(): pack=pack.parent
        manifest=load_manifest(pack)
        workspace=Path(workspace)
        workspace.mkdir(parents=True,exist_ok=False)
        for item in manifest.public_files:
            source=(pack/item.path).resolve()
            if not source.is_relative_to(pack.resolve()) or sha256_file(source)!=item.sha256:
                raise ValueError('Invalid public source')
            destination=workspace/item.path
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
        write_json(workspace/'_runtime/state.json',{'artifacts':{},'artifact_history':[],'events':[],'snapshots':{}})
        return EnvironmentHandle(manifest,pack,workspace)

    def versions(self,env):
        candidate=self._state(env)['artifacts'].get('candidate')
        return {'source_hash':sha256_file(env.workspace_dir/'public/Main.lean'), 'candidate_hash':candidate['sha256'] if candidate else None}

    def public_observation(self,env):
        state,versions=self._state(env),self.versions(env)
        artifacts={}
        for key,artifact in state['artifacts'].items():
            artifacts[key]={**artifact,'fresh':self._fresh(artifact,state,versions)}
            if key not in ('candidate','inspection','rejected_candidate'):
                artifacts[key]['content']=(env.workspace_dir/artifact['path']).read_text(encoding='utf-8')
        verification=artifacts.get('verification')
        terminal=bool(verification and verification['fresh'] and json.loads(verification['content'])['passed'])
        return {**versions,'artifacts':artifacts,'terminal':terminal,
            'public_execution_contract':read_json(env.workspace_dir/'public/execution_contract.json'),
            'initial_proof_state':read_json(env.workspace_dir/'public/proof_state.json'),
            'edit_contract':{'editable':'proof term; repeated declaration or full source also accepted if context/statement tokens match; declaration name is normalized to the original; original context is always retained','context_and_statement':'immutable',
                'allowed_axioms':['propext','Classical.choice','Quot.sound']},
            'latest_execution':state.get('latest_execution'),
            'latest_tool_feedback':({'tool_name':state['events'][-1]['tool_name'],'status':state['events'][-1]['status'],
                'error':state['events'][-1]['output'].get('error')} if state['events'] else None)}

    def executor_context(self,env,action,role_profile=None):
        state=self.public_observation(env)
        # The prover has a 32k context while each output may itself be 16k.
        # Keep authoritative files intact and offer explicit bounded reads.
        for artifact in state['artifacts'].values():
            content=artifact.pop('content',None)
            if content is not None: artifact['content_chars']=len(content)
        source=(env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8')
        state['source']=source[:8000]
        state['source_access']={'visible_chars':min(len(source),8000),'total_chars':len(source),'artifact':'candidate' if 'candidate' in state['artifacts'] else None}
        if state.get('latest_execution'):
            execution=json.loads(json.dumps(state['latest_execution']))
            for field in ('stdout','stderr'):
                text=execution['result'].get(field,'')
                execution['result'][field]=text[:8000]
                execution['result'][field+'_total_chars']=len(text)
            execution['full_output_artifact']='compilation'
            state['latest_execution']=execution
        return {'state':state,'artifact_access':'Use read_artifact with an artifact key and character offset to read its exact content. No artifact output is automatically summarized.',
                'history':[{'tool_name':e['tool_name'],'status':e['status'],'error':e['output'].get('error')} for e in self._state(env)['events']]}

    def public_preconditions(self,env,action_id,role_profile=None):
        state=self._state(env)
        if action_id not in ROLES: return PreconditionResult(False,('Unknown role',))
        if action_id=='Lean Compiler Executor' and 'candidate' not in state['artifacts']:
            return PreconditionResult(False,('No candidate proof exists',))
        if action_id=='Kernel and Assumption Verifier':
            execution=state.get('latest_execution')
            if not execution or execution['result']['returncode']!=0 or execution['input_versions']!=self.versions(env):
                return PreconditionResult(False,('No fresh successful compilation',))
        return PreconditionResult(True)

    def tool_spec(self,env,action_id,role_profile=None):
        name=ROLES.get(action_id)
        if not name: return ()
        properties=({'symbols':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8}} if name=='write_premises'
                    else {'content':{'type':'string'}} if name.startswith('write_') else {})
        schema={'type':'object','properties':properties,'additionalProperties':False}
        if properties: schema['required']=['symbols' if name=='write_premises' else 'content']
        tools=(ToolDefinition(name,f'Execute {action_id} using pinned Lean and public artifacts.',schema),)
        if name.startswith('write_'):
            tools+=(ToolDefinition('read_artifact','Read an exact bounded fragment of an existing public artifact; use next_offset for subsequent fragments.',
                {'type':'object','properties':{'artifact':{'type':'string'},'offset':{'type':'integer','minimum':0},
                 'max_chars':{'type':'integer','minimum':1,'maximum':8000}},'required':['artifact'],'additionalProperties':False}),)
        return tools

    def _compile(self,path):
        runtime=self.runtime
        if 'lean_executable' not in runtime or 'project_dir' not in runtime:
            return {'returncode':-1,'stdout':'','stderr':'Pinned Lean runtime is not configured','failure_origin':'tool_environment'}
        environment=dict(os.environ)
        if runtime.get('lean_path'): environment['LEAN_PATH']=runtime['lean_path']
        try:
            run=subprocess.run([runtime['lean_executable'],str(path.resolve())],cwd=runtime['project_dir'],env=environment,
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            return {'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        except (OSError,subprocess.TimeoutExpired) as error:
            return {'returncode':-1,'stdout':'','stderr':type(error).__name__,'failure_origin':'tool_environment'}

    def execute_tool(self,env,tool_name,arguments=None):
        state,inputs=self._state(env),self.versions(env)
        args=dict(arguments or {})
        status,output='success',{}
        if tool_name=='read_artifact':
            artifact=state['artifacts'].get(args.get('artifact'))
            if not artifact: return ToolResult(tool_name,'invalid_output',{},'Unknown public artifact key')
            path=(env.workspace_dir/artifact['path']).resolve()
            if not path.is_relative_to(env.workspace_dir.resolve()) or sha256_file(path)!=artifact['sha256']:
                return ToolResult(tool_name,'invalid_output',{},'Artifact integrity check failed')
            text=path.read_text(encoding='utf-8'); offset=int(args.get('offset',0)); limit=min(int(args.get('max_chars',8000)),8000)
            output={'artifact':args['artifact'],'offset':offset,'content':text[offset:offset+limit],
                    'next_offset':min(offset+limit,len(text)),'total_chars':len(text),'sha256':artifact['sha256']}
        elif tool_name=='write_premises':
            symbols=args.get('symbols',[])
            if not symbols or any(not re.fullmatch(r'[\w.]+',s) for s in symbols):
                return ToolResult(tool_name,'invalid_output',{},'Supply 1–8 Lean symbol names to check.')
            original=(env.pack_dir/'public/Main.lean').read_text(encoding='utf-8')
            prefix,_=proof_template(original)
            # Imports/context come from the pinned task, never an actor-authored command.
            context=prefix[:re.search(r'\b(?:theorem|lemma)\b',uncomment(prefix)).start()]
            probe=env.workspace_dir/'_runtime/premises.lean'
            probe.write_text(context+'\n'+'\n'.join('#check '+s for s in symbols),encoding='utf-8')
            output={'symbols':symbols,'lookup':self._compile(probe),'source':'pinned Lean/Mathlib'}
            self._store(env,state,'premises',output,inputs,tool_name)
        elif tool_name.startswith('write_'):
            content=str(args.get('content') or '')
            if not content.strip():
                status,output='invalid_output',{'error':'Empty work product'}
            elif tool_name=='write_source':
                blocks=re.findall(r'```(?:lean4?)?\s*\n(.*?)```',content,re.S)
                content=blocks[-1] if blocks else content
                submitted_declaration=re.search(r'\b(?:theorem|lemma)\s+([\w.]+)\b',uncomment(content))
                original_source=(env.pack_dir/'public/Main.lean').read_text(encoding='utf-8')
                _,original_name=proof_template(original_source)
                try:
                    content=constrained_source(original_source,content)
                except ValueError as error:
                    status,output='invalid_output',{'error':str(error)}
                    self._store(env,state,'rejected_candidate',{'submitted_content':content,'error':str(error)},inputs,tool_name)
                else:
                    (env.workspace_dir/'public/Main.lean').write_text(content,encoding='utf-8')
                    self._store(env,state,'candidate',content,{},tool_name)
                    output={'artifact':state['artifacts']['candidate'],'input_normalization':{'selected_lean_block':len(blocks)-1 if blocks else None,
                        'submitted_declaration_name':submitted_declaration.group(1) if submitted_declaration else None,
                        'original_declaration_name':original_name,
                        'declaration_name_normalized':bool(submitted_declaration and submitted_declaration.group(1)!=original_name),
                        'original_context_retained':True,'accepted_forms':['proof_term','original_declaration','original_full_source']}}
            else:
                self._store(env,state,tool_name.removeprefix('write_'),content,inputs,tool_name)
        elif tool_name=='inspect_proof':
            content=(env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8')
            self._store(env,state,'inspection',content,inputs,tool_name)
            output={'initial_proof_state':read_json(env.workspace_dir/'public/proof_state.json'),'source_ref':'public/Main.lean',
                'source_hash':inputs['source_hash'],'latest_execution':state.get('latest_execution')}
        elif tool_name=='compile_proof':
            result=self._compile(env.workspace_dir/'public/Main.lean')
            output={'result':result,'input_versions':inputs}
            state['latest_execution']=output
            self._store(env,state,'compilation',output,inputs,tool_name)
            status='success' if result['returncode']==0 else 'failed'
        elif tool_name=='verify_proof':
            original=uncomment((env.pack_dir/'public/Main.lean').read_text(encoding='utf-8'))
            current=uncomment((env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8'))
            declarations=lambda text:[(m.group(1),re.sub(r'\s+',' ',m.group(0)).strip()) for m in re.finditer(r'\b(?:theorem|lemma)\s+([\w.]+)\b.*?:=',text,re.S)]
            originals,candidates=declarations(original),declarations(current)
            raw_original=(env.pack_dir/'public/Main.lean').read_text(encoding='utf-8')
            raw_current=(env.workspace_dir/'public/Main.lean').read_text(encoding='utf-8')
            try:
                preserved=constrained_source(raw_original,raw_current)==raw_current
            except ValueError: preserved=False
            checks={'statement_preserved':preserved and bool(originals) and all(item in candidates for item in originals),
                'no_forbidden_placeholders':not re.search(r'\b(?:sorry|admit|axiom)\b',current),
                'fresh_compilation':self.public_preconditions(env,'Kernel and Assumption Verifier').executable}
            probe=env.workspace_dir/'_runtime/assumptions.lean'
            probe.write_text(raw_current+'\n'+'\n'.join('#print axioms '+name for name,_ in originals)+'\n',encoding='utf-8')
            result=self._compile(probe)
            listed=re.findall(r'depends on axioms:\s*\[([^\]]*)\]',result['stdout'])
            unexpected=[axiom.strip() for group in listed for axiom in group.split(',') if axiom.strip() not in ('','propext','Classical.choice','Quot.sound')]
            verified_names=all(name in result['stdout'] for name,_ in originals)
            checks['kernel_assumptions']=result['returncode']==0 and verified_names and not unexpected
            output={'passed':all(checks.values()),'checks':checks,'kernel_probe':result,'unexpected_axioms':unexpected}
            self._store(env,state,'verification',output,inputs,tool_name)
        else:
            status,output='invalid_output',{'error':'Unknown tool'}
        state['events'].append({'tool_name':tool_name,'status':status,'output':output,'input_versions':inputs,'origin':'selected_agent'})
        self._save_state(env,state)
        return ToolResult(tool_name,status,output,json.dumps(output,ensure_ascii=False))
