"""Read-only blind state viewer; no behavior-action or proposed-label leakage."""
import argparse
import json
from pathlib import Path

PAGE='''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shared environment · Human review</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1100px;margin:24px auto;padding:0 20px}select{max-width:100%;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:Canvas;border:1px solid GrayText;padding:12px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}#roles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.role{border:1px solid GrayText;padding:12px}small{display:block;margin:12px 0}h1{font-size:1.4rem}
</style>
<h1>Shared environment · Human review</h1>
<label>State <select id="state"></select></label><small id="identity"></small>
<h2>Task</h2><p id="task"></p><h2>Budget</h2><pre id="budget"></pre>
<h2>Current public state</h2><div id="current"></div>
<h2>Role cards</h2><div id="roles"></div>
<details><summary>Public history</summary><pre id="history"></pre></details>
<details><summary>Full router-visible observation</summary><pre id="observation"></pre></details>
<small>Human labels: pending. Lựa chọn hiện tại của router và nhãn đề xuất không được hiển thị.</small>
<script id="data" type="application/json">__DATA__</script>
<script>
const rows=JSON.parse(document.getElementById('data').textContent);
const select=document.getElementById('state');
rows.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=row.decision_id;select.append(option)});
function text(id,value){document.getElementById(id).textContent=typeof value==='string'?value:JSON.stringify(value,null,2)}
function detail(parent,label,value){const node=document.createElement('details');const title=document.createElement('summary');title.textContent=label;const pre=document.createElement('pre');pre.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);node.append(title,pre);parent.append(node)}
function show(){const row=rows[Number(select.value)];if(!row)return;
text('identity',(row.run_id||'')+' · '+row.domain+' · '+(row.evaluation_eligible===false?'Excluded from primary routing metrics: '+(row.evaluation_exclusion_reason||'not eligible'):'Reviewable decision state'));
text('task',row.task);text('budget',row.observation.budget);text('history',row.observation.recent_events||[]);text('observation',row.observation);
const current=document.getElementById('current');current.replaceChildren();const state=row.observation.current_state||{};
Object.entries(state).forEach(([key,value])=>{if(key==='artifacts'){Object.entries(value).forEach(([name,artifact])=>detail(current,'Artifact: '+name+' · fresh='+artifact.fresh,artifact))}else{detail(current,key,value)}});
const roles=document.getElementById('roles');roles.replaceChildren();row.agent_pool.forEach(role=>{const node=document.createElement('div');node.className='role';const title=document.createElement('strong');title.textContent=role.agent_id;const goal=document.createElement('p');goal.textContent=role.role_goal||'';node.append(title,goal);['use_when','avoid_when','expected_outputs'].forEach(key=>detail(node,key,role[key]||[]));roles.append(node)})}
select.addEventListener('change',show);show();
</script></html>'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.run_dir/'human_state_review_queue.jsonl').read_text(encoding='utf-8').splitlines() if line]
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    path=args.run_dir/'human_state_review.html'
    path.write_text(PAGE.replace('__DATA__',payload),encoding='utf-8')
    print(json.dumps({'states':len(rows),'output':str(path)}))

if __name__=='__main__': main()
