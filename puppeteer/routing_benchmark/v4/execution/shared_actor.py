"""Bounded role executor with explicit public context and complete usage metadata."""
from __future__ import annotations
import json
import re
import time
from typing import Any

from .adapters.database import TEXT_TOOLS


class RoleTransportError(RuntimeError):
    def __init__(self,error):
        self.cause_type=type(error).__name__
        self.http_status=getattr(error,'status_code',None)
        super().__init__(self.cause_type)


class SharedRoleActor:
    def __init__(self, client, model, *, max_tokens=16384, transport_retries=5, max_tool_calls=3):
        self.client, self.model = client, model
        self.max_tokens, self.transport_retries, self.max_tool_calls = max_tokens, transport_retries, max_tool_calls

    def _query(self, messages):
        for attempt in range(self.transport_retries+1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                    max_tokens=self.max_tokens)
                choice = response.choices[0]
                return choice.message.content or '', {'model':response.model, 'generation_id':response.id,
                    'finish_reason':choice.finish_reason, 'usage':response.usage.model_dump() if response.usage else {},
                    'latency_ms':int((time.perf_counter()-started)*1000), 'transport_attempt':attempt}
            except Exception as error:
                status = getattr(error,'status_code',None)
                if attempt >= self.transport_retries or status not in (None,408,409,429,500,502,503,504):
                    raise RoleTransportError(error) from None
                time.sleep(min(2**(attempt+1),30))

    def execute(self, *, env, adapter, action, role_profile, task):
        tools = list(adapter.tool_spec(env,action,role_profile))
        if not tools:
            return {'status':'invalid_output','raw_output':'No executable tool for role','tool_events':[], 'responses':[]}
        allowed = {t.name:t for t in tools}
        plain_tool=next((name for name in ('write_source','write_diagnosis') if name in allowed),None) if getattr(adapter,'adapter_id','')=='formal_proof.lean4.v1' else None
        context = adapter.executor_context(env,action,role_profile) if hasattr(adapter,'executor_context') else {'state':adapter.public_observation(env)}
        payload = {'task':task,'role':role_profile,'public_context':context,
            'allowed_tools':[t.to_dict() for t in tools],
            'constraints':{'perform_only_selected_role':True,'private_reference_visible':False,'do_not_choose_next_agent':True}}
        direct = len(tools)==1 and tools[0].name in TEXT_TOOLS
        direct = direct or (len(tools)==1 and tools[0].name.startswith('write_') and set(tools[0].input_schema.get('properties',{}))=={'content'})
        sql = tools[0].name=='write_sql'
        deterministic = len(tools)==1 and not tools[0].input_schema.get('properties')
        instruction = ('Return ONLY the complete SQLite migration SQL. Do not include transaction control statements because the packet runner supplies the transaction.' if sql else
            'Return only the Lean proof term beginning with by, preserving the immutable task context and statement.' if direct and tools[0].name=='write_source' else
            'Return the complete work product for your selected role; do not claim unobserved tool execution.' if direct else
            'Return one JSON object: {"tool": "an allowed tool", "arguments": {its typed arguments}}. Use only evidence in public context. Bounded reads may precede the role work product; successful work product completes your turn. Do not choose another role.')
        if plain_tool:
            instruction=('Return a Lean proof term or the original theorem declaration with its completed proof. Preserve the original statement and context.' if plain_tool=='write_source' else 'Return your diagnosis grounded in the public evidence.')+' If you need an artifact fragment first, return JSON {"tool":"read_artifact","arguments":{"artifact":"key","offset":0,"max_chars":8000}}; otherwise return the work product directly. Do not choose another role.'
        rendered=json.dumps(payload,ensure_ascii=False)
        if getattr(adapter,'adapter_id','')=='formal_proof.lean4.v1':
            # Code layout is evidence. Render it verbatim instead of requiring
            # a proof model to reconstruct Lean indentation from JSON escapes.
            display=json.loads(rendered)
            source=display['public_context']['state'].pop('source','')
            display['public_context']['state']['source_rendering']='Verbatim Lean block below'
            rendered=json.dumps(display,ensure_ascii=False)+'\n\nCURRENT LEAN SOURCE\n```lean4\n'+source+'\n```'
        messages = [{'role':'system','content':'You are a bounded executor in a multi-agent workflow.'},
                    {'role':'user','content':instruction+'\n'+rendered}]
        events, responses, requests = [], [], []
        limit = 1 if direct or deterministic else self.max_tool_calls
        final = ''
        for _ in range(limit):
            if deterministic:
                name, arguments = tools[0].name, {}
            else:
                requests.append({'model':self.model,'messages':list(messages),'max_tokens':self.max_tokens,'temperature':0})
                if getattr(adapter,'adapter_id','')=='formal_proof.lean4.v1': requests[-1]['executor_payload']=payload
                raw, metadata = self._query(messages)
                final = raw
                responses.append({'raw_output':raw,'metadata':metadata})
                if not raw.strip():
                    return {'status':'invalid_output','raw_output':raw,'tool_events':events,'responses':responses,'requests':requests}
                if direct:
                    name, arguments = tools[0].name, {'content':raw}
                else:
                    block = re.search(r'```(?:json)?\s*\n(.*?)```',raw,re.S)
                    try:
                        parsed = json.loads(block.group(1) if block else raw)
                        name, arguments = parsed['tool'],parsed.get('arguments',{})
                        if name not in allowed or not isinstance(arguments,dict):
                            raise ValueError('Unauthorized tool')
                    except (ValueError,KeyError,TypeError):
                        if plain_tool:
                            name,arguments=plain_tool,{'content':raw}
                        else:
                            return {'status':'invalid_output','raw_output':raw,'tool_events':events,'responses':responses,'requests':requests}
            # Typed argument validation is independent of model identity.
            import jsonschema
            try:
                jsonschema.validate(arguments,dict(allowed[name].input_schema))
            except jsonschema.ValidationError:
                return {'status':'invalid_output','raw_output':final,'tool_events':events,'responses':responses,'requests':requests}
            result = adapter.execute_tool(env,name,arguments)
            events.append({'tool_call':{'name':name,'arguments':arguments},'tool_result':result.to_dict()})
            final = result.observable_message
            # Preserve the proven code actor semantics: reads may precede one
            # authoritative role operation, whose success completes this turn.
            completed_work = result.status=='success' and name not in ('inspect_inputs','inspect_repository_file','search_repository','read_artifact')
            if direct or deterministic or completed_work or result.status not in ('success','no_progress') or len(tools)==1:
                break
            messages += [{'role':'assistant','content':raw},{'role':'user','content':json.dumps(result.to_dict(),ensure_ascii=False)}]
        status=events[-1]['tool_result']['status'] if events else 'invalid_output'
        if len(tools)>1 and not any(e['tool_call']['name'] not in ('inspect_inputs','inspect_repository_file','search_repository','read_artifact') and e['tool_result']['status']=='success' for e in events):
            if status=='success': status='no_progress'
        return {'status':status, 'raw_output':final,
                'tool_events':events,'responses':responses,'requests':requests}
