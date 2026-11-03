"""Executable SQLite role contracts. Public inputs only; no oracle-based routing."""
from __future__ import annotations

import json
from contextlib import closing, contextmanager
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any

from ..base import EnvironmentHandle, PreconditionResult, ToolDefinition, ToolResult
from .common import ConcreteWorkflowAdapter, read_json, write_json
from ...artifacts.manifest import load_manifest
from ...artifacts.schema import canonical_hash, sha256_file


ROLE_TOOLS = {
    'Schema Inspector': 'inspect_schema', 'Data Profiler': 'profile_data',
    'Constraint Analyzer': 'write_analysis', 'Migration Planner': 'write_plan',
    'Data Repair Planner': 'write_repair_plan', 'SQL Migration Author': 'write_sql',
    'Transactional Dry-run Executor': 'execute_migration', 'Integrity Validator': 'validate_integrity',
    'Query Performance Auditor': 'audit_performance', 'Snapshot & Rollback Operator': 'snapshot_operation',
}
TEXT_TOOLS = {'write_analysis': 'analysis', 'write_plan': 'plan', 'write_repair_plan': 'repair_plan', 'write_sql': 'candidate'}


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def readonly(database: Path):
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    # The authorizer also blocks ATTACH, extension loading, and write PRAGMAs.
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
    pragmas = {'table_info', 'table_xinfo', 'index_list', 'index_info', 'index_xinfo', 'foreign_key_list', 'integrity_check', 'foreign_key_check'}
    def authorize(action, arg1, arg2, *_):
        if action in allowed:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() not in {'load_extension', 'writefile', 'readfile'}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA and str(arg1).lower() in pragmas:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    connection.set_authorizer(authorize)
    try:
        yield connection
    finally:
        connection.close()


class DatabaseMigrationAdapter(ConcreteWorkflowAdapter):
    adapter_id = 'database_migration.sqlite.v1'

    def materialize(self, manifest_path, workspace):
        path, workspace = Path(manifest_path), Path(workspace)
        pack = path if path.is_dir() else path.parent
        manifest = load_manifest(pack)
        if manifest.adapter_id != self.adapter_id:
            raise ValueError('Adapter mismatch')
        workspace.mkdir(parents=True, exist_ok=False)
        for item in manifest.public_files:
            source = (pack / item.path).resolve()
            if not source.is_relative_to(pack.resolve()) or not item.path.startswith('public/'):
                raise ValueError('Invalid public artifact path')
            if sha256_file(source) != item.sha256:
                raise ValueError(f'Public source checksum mismatch: {item.path}')
            destination = workspace / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(workspace / 'public/source.sqlite', workspace / 'working.sqlite')
        state = {'artifacts': {}, 'artifact_history': [], 'events': [], 'snapshots': {}, 'terminal': False}
        write_json(workspace / '_runtime/state.json', state)
        return EnvironmentHandle(manifest, pack, workspace)

    def _database(self, env):
        return env.workspace_dir / 'working.sqlite'

    def versions(self, env):
        state = self._state(env)
        candidate = state['artifacts'].get('candidate')
        return {'database_hash': sha256_file(self._database(env)), 'candidate_hash': candidate['sha256'] if candidate else None}

    def _performance(self, env):
        path = env.workspace_dir / 'public/performance_contract.json'
        if path.exists():
            return read_json(path)
        return read_json(env.workspace_dir / 'public/execution_contract.json').get('performance_contract') or {}

    def public_observation(self, env):
        state, versions = self._state(env), self.versions(env)
        artifacts = {}
        for key, artifact in state['artifacts'].items():
            artifacts[key] = {**artifact, 'fresh': self._fresh(artifact,state,versions),
                              'content': (env.workspace_dir / artifact['path']).read_text(encoding='utf-8')}
        validation = artifacts.get('validation')
        terminal = bool(validation and validation['fresh'] and json.loads(validation['content']).get('passed'))
        if terminal and self._performance(env).get('queries'):
            performance = artifacts.get('performance')
            terminal = bool(performance and performance['fresh'] and json.loads(performance['content']).get('passed'))
        return {'database_artifact': 'working.sqlite', **versions, 'artifacts': artifacts,
                'public_execution_contract': read_json(env.workspace_dir / 'public/execution_contract.json'),
                'target_contract': read_json(env.workspace_dir / 'public/target_contract.json'),
                'performance_contract': self._performance(env), 'snapshots': state['snapshots'],
                'latest_execution': state.get('latest_execution'), 'latest_actor_result': state.get('latest_actor_result'),
                'terminal': terminal}

    @staticmethod
    def _fresh(artifact,state,versions):
        return all(versions.get(k)==v for k,v in artifact['input_versions'].items()) and all(
            state['artifacts'].get(ref['key'],{}).get('sha256')==ref['sha256'] for ref in artifact.get('input_refs',[]))

    def executor_context(self, env, action, role_profile=None):
        # One authoritative state, including failure evidence, for router and executor.
        return {'state': self.public_observation(env), 'history': self._state(env)['events']}

    def public_preconditions(self, env, action_id, role_profile=None):
        state = self._state(env)
        if action_id not in ROLE_TOOLS:
            return PreconditionResult(False, ('Unknown role',))
        if action_id == 'Transactional Dry-run Executor' and 'candidate' not in state['artifacts']:
            return PreconditionResult(False, ('No nonempty candidate SQL artifact exists',))
        if action_id == 'Integrity Validator':
            execution = state.get('latest_execution')
            if not execution or execution['result']['returncode'] != 0 or execution['input_versions']['candidate_hash'] != self.versions(env)['candidate_hash'] or execution['database_after'] != self.versions(env)['database_hash']:
                return PreconditionResult(False, ('No successful execution for the current candidate',))
        if action_id == 'Query Performance Auditor' and not self._performance(env).get('queries'):
            return PreconditionResult(False, ('No explicit public performance query contract',))
        return PreconditionResult(True)

    def tool_spec(self, env, action_id, role_profile=None):
        tool = ROLE_TOOLS.get(action_id)
        if not tool:
            return ()
        schema: dict[str, Any] = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        if tool in TEXT_TOOLS:
            schema.update(properties={'content': {'type': 'string'}}, required=['content'])
        elif tool == 'profile_data':
            schema['properties'] = {'relations': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'child_table': {'type': 'string'}, 'child_columns': {'type': 'array', 'items': {'type': 'string'}},
                'parent_table': {'type': 'string'}, 'parent_columns': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['child_table', 'child_columns', 'parent_table', 'parent_columns'], 'additionalProperties': False}}}
        elif tool == 'snapshot_operation':
            schema.update(properties={'operation': {'enum': ['create', 'restore']}, 'snapshot_id': {'type': 'string'}}, required=['operation'])
        return (ToolDefinition(tool, f'Execute {action_id} against public working artifacts.', schema),)

    def _store(self, env, state, key, content, inputs, tool):
        inputs = {k:v for k,v in inputs.items() if v is not None and not (key=='candidate' and k=='candidate_hash')}
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        path = f'_runtime/artifacts/{len(state["artifact_history"]):04}_{key}.txt'
        file = env.workspace_dir / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding='utf-8')
        artifact = {'artifact_id': f'{key}:{len(state["artifact_history"])}', 'version': len(state['artifact_history']),
                    'sha256': sha256_file(file), 'path': path, 'input_versions': inputs,
                    'producer_tool': tool, 'producer_event': len(state['events']), 'visibility': 'public'}
        dependencies=list(state['artifacts']) if tool.startswith('write_') else ['candidate'] if tool=='execute_migration' else ['candidate','execution'] if tool=='validate_integrity' else []
        artifact['input_refs']=[{'key':name,'artifact_id':state['artifacts'][name]['artifact_id'],'sha256':state['artifacts'][name]['sha256']}
                                for name in dependencies if name!=key and name in state['artifacts']]
        state['artifacts'][key] = artifact
        state['artifact_history'].append(artifact)

    def _runner(self, env, command, sql_file=None, database=None):
        argv = [sys.executable, str(env.workspace_dir / 'public/sqlite_runner.py'), command, '--database', str(database or self._database(env))]
        if sql_file:
            argv += ['--sql-file', str(sql_file)]
        started = time.perf_counter()
        try:
            run = subprocess.run(argv, cwd=env.workspace_dir, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
            return {'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr, 'latency_ms': int((time.perf_counter()-started)*1000)}
        except subprocess.TimeoutExpired:
            return {'returncode': -1, 'stdout': '', 'stderr': 'Tool timeout', 'latency_ms': int((time.perf_counter()-started)*1000), 'failure_origin': 'tool_environment'}

    def _schema(self, db):
        with readonly(db) as connection:
            objects = connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
            tables = {}
            for _, name, _, _ in (row for row in objects if row[0] == 'table'):
                tables[name] = {'columns': connection.execute(f'PRAGMA table_info({quote(name)})').fetchall(),
                                'foreign_keys': connection.execute(f'PRAGMA foreign_key_list({quote(name)})').fetchall(),
                                'indexes': connection.execute(f'PRAGMA index_list({quote(name)})').fetchall()}
                tables[name]['index_details'] = {index[1]: connection.execute(f'PRAGMA index_xinfo({quote(index[1])})').fetchall() for index in tables[name]['indexes']}
            return {'schema_objects': objects, 'tables': tables,
                    'dependencies': [{'child_table': name, 'foreign_keys': spec['foreign_keys']} for name, spec in tables.items()]}

    def _profile(self, db, arguments):
        schema = self._schema(db)
        results, queries, relations = {}, [], list(arguments.get('relations') or [])
        with readonly(db) as connection:
            def query(sql):
                rows = [tuple({'sqlite_type':'blob','hex':value.hex()} if isinstance(value,bytes) else value for value in row)
                        for row in connection.execute(sql).fetchall()]
                queries.append({'sql': sql, 'rows': rows})
                return rows
            for table, spec in schema['tables'].items():
                columns = {}
                for column in spec['columns']:
                    name = column[1]
                    identifier, source = quote(name), quote(table)
                    columns[name] = {'null_count': query(f'SELECT count(*) FROM {source} WHERE {identifier} IS NULL')[0][0],
                        'types': query(f'SELECT typeof({identifier}),count(*) FROM {source} GROUP BY typeof({identifier})'),
                        'value_distribution': query(f'SELECT {identifier},count(*) FROM {source} GROUP BY {identifier} ORDER BY count(*) DESC LIMIT 50'),
                        'duplicates': query(f'SELECT {identifier},count(*) FROM {source} GROUP BY {identifier} HAVING count(*)>1')}
                results[table] = {'row_count': query(f'SELECT count(*) FROM {quote(table)}')[0][0], 'columns': columns}
                groups = {}
                for fk in spec['foreign_keys']:
                    groups.setdefault(fk[0], []).append(fk)
                for group in groups.values():
                    ordered = sorted(group, key=lambda r:r[1])
                    parent = ordered[0][2]
                    if parent not in schema['tables']:
                        continue
                    parent_pk = [c[1] for c in sorted(schema['tables'][parent]['columns'], key=lambda c:c[5]) if c[5]]
                    relations.append({'child_table': table, 'child_columns': [fk[3] for fk in ordered], 'parent_table': parent,
                        'parent_columns': [fk[4] or parent_pk[i] for i,fk in enumerate(ordered)]})
            anomalies = []
            for relation in relations:
                child, parent = relation['child_table'], relation['parent_table']
                left, right = relation['child_columns'], relation['parent_columns']
                if child not in schema['tables'] or parent not in schema['tables'] or not left or len(left) != len(right):
                    raise ValueError('Invalid profiling relation')
                for table, cols in ((child,left),(parent,right)):
                    if not set(cols).issubset(c[1] for c in schema['tables'][table]['columns']):
                        raise ValueError('Unknown profiling column')
                conditions = ' AND '.join(f'c.{quote(a)} = p.{quote(b)}' for a,b in zip(left,right))
                nonnull = ' AND '.join(f'c.{quote(a)} IS NOT NULL' for a in left)
                sql = f'SELECT c.* FROM {quote(child)} c WHERE {nonnull} AND NOT EXISTS (SELECT 1 FROM {quote(parent)} p WHERE {conditions})'
                anomalies.append({'relation': relation, 'orphan_rows': query(sql)})
        return {'tables': results, 'relations': anomalies, 'queries': queries}

    def _audit(self, env):
        contract, reports = self._performance(env), []
        with readonly(self._database(env)) as connection:
            for item in contract['queries']:
                sql = item['sql']
                plan = connection.execute('EXPLAIN QUERY PLAN ' + sql).fetchall()
                measurements = []
                for _ in range(int(contract.get('repetitions', 3))):
                    started = time.perf_counter()
                    rows = connection.execute(sql).fetchall()
                    measurements.append((time.perf_counter()-started)*1000)
                sorted_times = sorted(measurements)
                median = sorted_times[len(sorted_times)//2]
                checks = {}
                if 'max_duration_ms' in item:
                    checks['duration'] = median <= item['max_duration_ms']
                if item.get('forbid_full_scan'):
                    checks['no_full_scan'] = not any(re.search(r'\bSCAN\b', str(row[-1])) for row in plan)
                reports.append({'sql': sql, 'query_plan': plan, 'durations_ms': measurements, 'median_ms': median,
                                'row_count': len(rows), 'checks': checks})
        return {'sqlite_version': sqlite3.sqlite_version, 'queries': reports,
                'passed': all(all(r['checks'].values()) for r in reports)}

    def execute_tool(self, env, tool_name, arguments=None):
        arguments, state = dict(arguments or {}), self._state(env)
        before = self.versions(env)
        inputs = dict(before)
        status, output = 'success', {}
        try:
            if tool_name in TEXT_TOOLS:
                text = str(arguments.get('content') or '')
                if not text.strip():
                    status, output = 'invalid_output', {'error': 'Empty work product; no new artifact materialized'}
                else:
                    if tool_name == 'write_sql':
                        block = re.search(r'```(?:sql|sqlite)?\s*\n(.*?)```', text, re.S | re.I)
                        text = block.group(1).strip() if block else text.strip()
                        if not text:
                            raise ValueError('Empty candidate')
                    self._store(env, state, TEXT_TOOLS[tool_name], text, inputs, tool_name)
                    output = {'artifact': state['artifacts'][TEXT_TOOLS[tool_name]]}
            elif tool_name in ('inspect_schema','profile_data'):
                inputs = {'database_hash': before['database_hash']}
                output = self._schema(self._database(env)) if tool_name == 'inspect_schema' else self._profile(self._database(env), arguments)
                self._store(env, state, 'inspection' if tool_name == 'inspect_schema' else 'profile', output, inputs, tool_name)
            elif tool_name == 'execute_migration':
                candidate = state['artifacts']['candidate']
                result = self._runner(env, 'migrate', env.workspace_dir / candidate['path'])
                after = self.versions(env)
                output = {'result': result, 'input_versions': before, 'database_after': after['database_hash'],
                    'database_changed': before['database_hash'] != after['database_hash'],
                    'transaction_outcome': 'committed' if result['returncode']==0 else 'failed_database_unchanged' if before['database_hash']==after['database_hash'] else 'failed_database_changed',
                    'origin': 'database_transaction'}
                state['latest_execution'] = output
                status = 'success' if result['returncode']==0 else 'failed'
                self._store(env, state, 'execution', output, {'candidate_hash': before['candidate_hash']}, tool_name)
            elif tool_name == 'validate_integrity':
                candidate = state['artifacts']['candidate']
                contract = read_json(env.workspace_dir / 'public/execution_contract.json')
                needs_sql = any('verify' in c and '--sql-file' in c for c in contract['allowed_commands'])
                probe = env.workspace_dir / f'_runtime/validation/{len(state["events"]):04}.sqlite'
                probe.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(self._database(env))) as source, closing(sqlite3.connect(probe)) as target:
                    source.backup(target)
                result = self._runner(env, 'verify', env.workspace_dir / candidate['path'] if needs_sql else None, database=probe)
                output = {'passed': result['returncode']==0, 'result': result, 'input_versions': before}
                if self.versions(env)['database_hash'] != before['database_hash']:
                    output['environment_error'] = 'Validation modified working database'
                    output['passed'] = False
                self._store(env, state, 'validation', output, inputs, tool_name)
            elif tool_name == 'audit_performance':
                output = self._audit(env)
                self._store(env, state, 'performance', output, {'database_hash': before['database_hash']}, tool_name)
            elif tool_name == 'snapshot_operation':
                operation = arguments.get('operation')
                if operation == 'create':
                    identifier = f'snapshot-{len(state["snapshots"]):03}'
                    path = f'_runtime/snapshots/{identifier}.sqlite'
                    destination = env.workspace_dir / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # SQLite backup captures a consistent DB, including WAL content.
                    with closing(sqlite3.connect(self._database(env))) as source, closing(sqlite3.connect(destination)) as target:
                        source.backup(target)
                    state['snapshots'][identifier] = {'path':path, 'sha256':sha256_file(destination), 'database_hash': before['database_hash']}
                    output = {'operation':operation, 'snapshot_id':identifier, 'snapshot':state['snapshots'][identifier]}
                elif operation == 'restore':
                    identifier = arguments['snapshot_id']
                    snapshot = state['snapshots'][identifier]
                    source = env.workspace_dir / snapshot['path']
                    if sha256_file(source) != snapshot['sha256']:
                        raise ValueError('Snapshot checksum mismatch')
                    with closing(sqlite3.connect(source)) as saved, closing(sqlite3.connect(self._database(env))) as target:
                        saved.backup(target)
                    output = {'operation':operation, 'snapshot_id':identifier, 'origin':'selected_agent'}
                else:
                    raise ValueError('Explicit create/restore operation required')
            else:
                raise ValueError('Unknown tool')
        except (ValueError, KeyError, sqlite3.Error) as error:
            status, output = 'invalid_output', {'error_type':type(error).__name__, 'error':str(error)}
        event = {'tool_name':tool_name, 'status':status, 'output':output, 'input_versions':inputs,
                 'versions_after':self.versions(env), 'origin':'selected_agent'}
        state['events'].append(event)
        state['latest_actor_result'] = event
        self._save_state(env, state)
        return ToolResult(tool_name, status, output, json.dumps(output, ensure_ascii=False))

    def terminal_status(self, env):
        return {'terminal':self.public_observation(env)['terminal']}

    def validate_postcondition(self, env):
        return self.terminal_status(env)
