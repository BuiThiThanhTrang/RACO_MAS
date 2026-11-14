import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from routing_benchmark.routers import RoutingDecision
from routing_benchmark.v4.execution.adapters.database import DatabaseMigrationAdapter, readonly
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner
from routing_benchmark.v4.domain_packets.roles import role_cards_for

PROJECT=Path(__file__).resolve().parents[1]
PACK=PROJECT/'data/RoutingBench/v4_beta_4d_82_20261102/packs/database_migration_integrity/db-alembic-batchapplytest-test-add-fk'
VALID_SQL='''CREATE TABLE child_new(id INTEGER PRIMARY KEY,parent_id INTEGER NOT NULL,payload TEXT,
FOREIGN KEY(parent_id) REFERENCES parent(id));
INSERT INTO child_new SELECT * FROM child WHERE parent_id IN (SELECT id FROM parent);
DROP TABLE child;
ALTER TABLE child_new RENAME TO child;'''


class DatabaseEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=PROJECT/'tests')
        self.root=Path(self.temporary.name)
        self.adapter=DatabaseMigrationAdapter()
        self.env=self.adapter.materialize(PACK,self.root/'workspace')

    def tearDown(self):
        self.temporary.cleanup()

    def call(self,tool,**args):
        return self.adapter.execute_tool(self.env,tool,args)

    def test_public_materialization_and_real_schema(self):
        self.assertFalse((self.env.workspace_dir/'private').exists())
        self.assertFalse((self.env.workspace_dir/'fixtures').exists())
        output=self.call('inspect_schema').output
        self.assertEqual(output['tables']['child']['foreign_keys'],[])
        self.assertEqual(output['tables']['child']['columns'][1][1],'parent_id')
        self.assertNotIn('reference.sql',json.dumps(self.adapter.executor_context(self.env,'SQL Migration Author')))

    def test_profiler_executes_statistics_and_orphan_query(self):
        output=self.call('profile_data',relations=[{'child_table':'child','child_columns':['parent_id'],'parent_table':'parent','parent_columns':['id']}]).output
        self.assertEqual(output['tables']['child']['row_count'],2)
        self.assertEqual(output['tables']['child']['columns']['parent_id']['null_count'],0)
        self.assertEqual(output['relations'][0]['orphan_rows'],[(2,99,'orphan')])
        self.assertTrue(output['queries'])

    def test_readonly_profiler_connection_blocks_mutation_and_attach(self):
        database=self.env.workspace_dir/'working.sqlite'
        before=database.read_bytes()
        with readonly(database) as connection:
            for sql in ("DELETE FROM child", "ATTACH DATABASE ':memory:' AS other", 'PRAGMA foreign_keys=OFF'):
                with self.assertRaises(sqlite3.DatabaseError): connection.execute(sql)
        self.assertEqual(before,database.read_bytes())

    def test_failed_execution_feedback_is_in_author_context(self):
        self.call('write_sql',content='ALTER TABLE child ADD CONSTRAINT fk FOREIGN KEY(parent_id) REFERENCES parent(id);')
        self.call('execute_migration')
        context=self.adapter.executor_context(self.env,'SQL Migration Author')
        self.assertIn('syntax error',context['state']['latest_execution']['result']['stdout'])
        self.assertIn('ADD CONSTRAINT',context['state']['artifacts']['candidate']['content'])

    def test_candidate_change_invalidates_validation_and_preserves_history(self):
        self.call('write_sql',content=VALID_SQL)
        self.call('execute_migration')
        self.call('validate_integrity')
        self.assertTrue(self.adapter.terminal_status(self.env)['terminal'])
        self.call('write_sql',content='SELECT 1;')
        state=self.adapter.public_observation(self.env)
        self.assertFalse(state['artifacts']['validation']['fresh'])
        self.assertFalse(state['terminal'])
        self.assertEqual(len(self.adapter._state(self.env)['artifact_history']),4)

    def test_database_change_invalidates_inspection(self):
        self.call('inspect_schema')
        self.call('write_sql',content=VALID_SQL)
        self.call('execute_migration')
        self.assertFalse(self.adapter.public_observation(self.env)['artifacts']['inspection']['fresh'])

    def test_empty_output_does_not_create_or_replace_candidate(self):
        self.assertEqual(self.call('write_sql',content='  ').status,'invalid_output')
        self.assertFalse(self.adapter.public_preconditions(self.env,'Transactional Dry-run Executor').executable)
        self.call('write_sql',content='SELECT 1;')
        prior=self.adapter._state(self.env)['artifacts']['candidate']
        self.call('write_sql',content='')
        self.assertEqual(prior,self.adapter._state(self.env)['artifacts']['candidate'])

    def test_transaction_failure_preserves_prior_commit(self):
        self.call('write_sql',content="UPDATE parent SET label='A committed';")
        self.call('execute_migration')
        self.call('write_sql',content='DELETE FROM child; THIS IS INVALID;')
        output=self.call('execute_migration').output
        with closing(sqlite3.connect(self.env.workspace_dir/'working.sqlite')) as connection:
            self.assertEqual(connection.execute('SELECT label FROM parent').fetchone()[0],'A committed')
            self.assertEqual(connection.execute('SELECT count(*) FROM child').fetchone()[0],2)
        self.assertFalse(output['database_changed'])
        self.assertEqual(output['origin'],'database_transaction')

    def test_explicit_snapshot_restore_returns_selected_state(self):
        identifier=self.call('snapshot_operation',operation='create').output['snapshot_id']
        self.call('write_sql',content="UPDATE parent SET label='changed';")
        self.call('execute_migration')
        output=self.call('snapshot_operation',operation='restore',snapshot_id=identifier).output
        with closing(sqlite3.connect(self.env.workspace_dir/'working.sqlite')) as connection:
            self.assertEqual(connection.execute('SELECT label FROM parent').fetchone()[0],'kept')
        self.assertEqual(output['origin'],'selected_agent')

    def test_performance_probe_is_real_and_uses_only_public_contract(self):
        self.assertFalse(self.adapter.public_preconditions(self.env,'Query Performance Auditor').executable)
        path=self.env.workspace_dir/'public/performance_contract.json'
        path.write_text(json.dumps({'queries':[{'sql':'SELECT * FROM parent WHERE id=1','forbid_full_scan':True,'max_duration_ms':1000}],'repetitions':3}),encoding='utf-8')
        self.assertTrue(self.adapter.public_preconditions(self.env,'Query Performance Auditor').executable)
        output=self.call('audit_performance').output
        self.assertTrue(output['passed'])
        self.assertEqual(len(output['queries'][0]['durations_ms']),3)
        self.assertIn('SEARCH',output['queries'][0]['query_plan'][0][-1])

    def test_usage_metadata_retained_without_token_state(self):
        response=SimpleNamespace(model='qwen/qwen3.5-9b',id='test-generation',
            choices=[SimpleNamespace(message=SimpleNamespace(content=''),finish_reason='length')],
            usage=SimpleNamespace(model_dump=lambda:{'completion_tokens':16384,'completion_tokens_details':{'reasoning_tokens':16384}}))
        client=Mock()
        client.chat.completions.create.return_value=response
        actor=SharedRoleActor(client,'qwen/qwen3.5-9b',transport_retries=0)
        roles={r['agent_id']:r for r in role_cards_for('database_migration_integrity')}
        output=actor.execute(env=self.env,adapter=self.adapter,action='SQL Migration Author',role_profile=roles['SQL Migration Author'],task=self.env.manifest.task)
        self.assertEqual(output['responses'][0]['metadata']['finish_reason'],'length')
        self.assertEqual(client.chat.completions.create.call_args.kwargs['max_tokens'],16384)
        self.assertNotIn('token_limit_reached',json.dumps(self.adapter.public_observation(self.env)))
        self.assertNotIn('candidate',self.adapter._state(self.env)['artifacts'])

    def test_analysis_and_repair_plan_versions_are_visible_and_linked(self):
        self.call('profile_data')
        self.call('write_analysis',content='The public profile must be reconciled with the foreign key requirement.')
        self.call('write_plan',content='Plan operations within the runner transaction.')
        self.call('write_repair_plan',content='Remove only rows without a public parent match.')
        before=self.adapter.public_observation(self.env)
        self.assertTrue(before['artifacts']['plan']['fresh'])
        self.assertIn('analysis',[ref['key'] for ref in before['artifacts']['plan']['input_refs']])
        self.assertIn('Remove only',before['artifacts']['repair_plan']['content'])
        self.call('write_analysis',content='Updated analysis based on newly examined public evidence.')
        self.assertFalse(self.adapter.public_observation(self.env)['artifacts']['plan']['fresh'])

    def test_performance_contract_participates_in_terminal_check(self):
        (self.env.workspace_dir/'public/performance_contract.json').write_text(json.dumps({'queries':[{'sql':'SELECT * FROM parent WHERE id=1','forbid_full_scan':True}]}),encoding='utf-8')
        self.call('write_sql',content=VALID_SQL)
        self.call('execute_migration')
        self.call('validate_integrity')
        self.assertFalse(self.adapter.terminal_status(self.env)['terminal'])
        self.call('audit_performance')
        self.assertTrue(self.adapter.terminal_status(self.env)['terminal'])

    def test_budget_boundary_is_asked_and_classified_without_dispatch(self):
        class Router:
            def reset(self,seed): pass
            def select(self,observation,actions): return RoutingDecision('Schema Inspector')
        runner=SharedTrajectoryRunner(router=Router(),actor=SharedRoleActor(None,'unused'),max_calls=2)
        result=runner.run(manifest_path=PACK,adapter=DatabaseMigrationAdapter(),workspace=self.root/'run-workspace',output_dir=self.root/'run',roles=role_cards_for('database_migration_integrity'))
        self.assertEqual(result['router_decision_count'],3)
        self.assertEqual(result['dispatched'],2)
        self.assertEqual(result['suppressed_by_budget'],1)
        self.assertEqual(result['primary_routing_decisions'],2)

    def test_precondition_rejection_remains_in_routing_denominator(self):
        class Router:
            def reset(self,seed): self.index=0
            def select(self,observation,actions):
                action=['Integrity Validator','STOP'][self.index]
                self.index+=1
                return RoutingDecision(action)
        result=SharedTrajectoryRunner(router=Router(),actor=None,max_calls=2).run(manifest_path=PACK,adapter=DatabaseMigrationAdapter(),workspace=self.root/'rejected-workspace',output_dir=self.root/'rejected',roles=role_cards_for('database_migration_integrity'))
        self.assertEqual(result['precondition_rejected'],1)
        self.assertEqual(result['primary_routing_decisions'],2)
        self.assertEqual(result['terminal_reason'],'PREMATURE_STOP')


if __name__=='__main__': unittest.main()
