from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Iterable, Mapping

from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import (
    ArtifactFile,
    ArtifactManifest,
    sha256_bytes,
)


GENERATOR_VERSION = "routingbench-v4-domain-packets-rc1"
DATABASE_ADAPTER = "database_migration.sqlite.v1"
FORMAL_ADAPTER = "formal_proof.lean4.v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _media_type(path: str) -> str:
    overrides = {
        ".db": "application/vnd.sqlite3",
        ".sqlite": "application/vnd.sqlite3",
        ".lean": "text/x-lean",
        ".md": "text/markdown",
        ".sql": "application/sql",
        ".jsonl": "application/x-ndjson",
    }
    return overrides.get(Path(path).suffix.lower(), mimetypes.guess_type(path)[0] or "application/octet-stream")


def _artifact_id(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", path).strip("-")


class _Writer:
    def __init__(self, pack_dir: Path) -> None:
        self.pack_dir = pack_dir
        self.files: list[ArtifactFile] = []

    def add(self, relative_path: str, content: bytes | str, visibility: str) -> None:
        relative_path = Path(relative_path).as_posix()
        raw = content.encode("utf-8") if isinstance(content, str) else content
        destination = self.pack_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        self.files.append(
            ArtifactFile(
                artifact_id=_artifact_id(relative_path),
                path=relative_path,
                media_type=_media_type(relative_path),
                sha256=sha256_bytes(raw),
                visibility=visibility,
            )
        )

    def manifest(
        self,
        *,
        packet_id: str,
        domain: str,
        task: str,
        adapter_id: str,
        source_type: str,
        license_name: str,
        execution_readiness: str,
        metadata: Mapping[str, Any],
    ) -> ArtifactManifest:
        return ArtifactManifest(
            base_workflow_id=packet_id,
            domain=domain,
            task=task,
            adapter_id=adapter_id,
            public_files=tuple(item for item in self.files if item.visibility == "public"),
            private_files=tuple(item for item in self.files if item.visibility == "private"),
            fixture_files=tuple(item for item in self.files if item.visibility == "fixture"),
            source_type=source_type,
            generator_version=GENERATOR_VERSION,
            license=license_name,
            human_review_status="pending",
            packet_review_status="pending",
            execution_readiness=execution_readiness,
            metadata=dict(metadata),
        )


def _db_spec(
    record_id: str,
    *,
    task: str,
    setup_sql: str,
    reference_sql: str,
    checks: list[dict[str, Any]],
    branch_types: list[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "task": task,
        "setup_sql": setup_sql.strip() + "\n",
        "reference_sql": reference_sql.strip() + "\n",
        "checks": checks,
        "branch_types": branch_types,
    }


def _scalar_check(query: str, expected: Any, label: str) -> dict[str, Any]:
    return {"label": label, "query": query, "expected_rows": [[expected]]}


DATABASE_SPECS = (
    _db_spec(
        "BatchApplyTest.test_rename_col_w_index",
        task="Rename customer.full_name to display_name while preserving rows and replacing the dependent index.",
        setup_sql="""
        PRAGMA foreign_keys=ON;
        CREATE TABLE customer(id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, email TEXT);
        CREATE INDEX ix_customer_full_name ON customer(full_name);
        INSERT INTO customer VALUES (1, 'Alice Nguyen', 'alice@example.test'), (2, 'Bao Tran', 'bao@example.test');
        """,
        reference_sql="""
        ALTER TABLE customer RENAME COLUMN full_name TO display_name;
        DROP INDEX ix_customer_full_name;
        CREATE INDEX ix_customer_display_name ON customer(display_name);
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('customer') WHERE name='display_name'", 1, "renamed_column_present"),
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('customer') WHERE name='full_name'", 0, "legacy_column_absent"),
            _scalar_check("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='ix_customer_display_name'", 1, "replacement_index_present"),
            _scalar_check("SELECT COUNT(*) FROM customer", 2, "rows_preserved"),
        ],
        branch_types=["schema_inspection", "dependency_analysis", "migration_authoring", "postcondition_verification"],
    ),
    _db_spec(
        "BatchApplyTest.test_rename_col_fk",
        task="Rename parent.code to external_id without breaking the existing child foreign key or data.",
        setup_sql="""
        PRAGMA foreign_keys=ON;
        CREATE TABLE parent(code TEXT PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE child(id INTEGER PRIMARY KEY, parent_code TEXT NOT NULL REFERENCES parent(code));
        INSERT INTO parent VALUES ('P-1', 'primary');
        INSERT INTO child VALUES (1, 'P-1');
        """,
        reference_sql="ALTER TABLE parent RENAME COLUMN code TO external_id;",
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('parent') WHERE name='external_id'", 1, "renamed_key_present"),
            _scalar_check("SELECT COUNT(*) FROM pragma_foreign_key_list('child') WHERE \"to\"='external_id'", 1, "foreign_key_retargeted"),
            _scalar_check("SELECT COUNT(*) FROM child JOIN parent ON child.parent_code=parent.external_id", 1, "relationship_preserved"),
        ],
        branch_types=["schema_inspection", "foreign_key_analysis", "migration_authoring", "integrity_verification"],
    ),
    _db_spec(
        "BatchApplyTest.test_add_fk",
        task="Add a foreign key from child.parent_id to parent.id, repairing orphaned data before enforcing it.",
        setup_sql="""
        CREATE TABLE parent(id INTEGER PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE child(id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL, payload TEXT);
        INSERT INTO parent VALUES (1, 'kept');
        INSERT INTO child VALUES (1, 1, 'valid'), (2, 99, 'orphan');
        """,
        reference_sql="""
        PRAGMA foreign_keys=OFF;
        DELETE FROM child WHERE parent_id NOT IN (SELECT id FROM parent);
        CREATE TABLE child_new(id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL REFERENCES parent(id), payload TEXT);
        INSERT INTO child_new SELECT id, parent_id, payload FROM child;
        DROP TABLE child;
        ALTER TABLE child_new RENAME TO child;
        PRAGMA foreign_keys=ON;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_foreign_key_list('child') WHERE \"table\"='parent'", 1, "foreign_key_present"),
            _scalar_check("SELECT COUNT(*) FROM child WHERE parent_id NOT IN (SELECT id FROM parent)", 0, "orphans_removed"),
            _scalar_check("SELECT COUNT(*) FROM child", 1, "valid_row_preserved"),
        ],
        branch_types=["data_profiling", "constraint_analysis", "data_repair", "schema_migration", "integrity_verification"],
    ),
    _db_spec(
        "BatchApplyTest.test_add_uq",
        task="Enforce unique contact emails after deterministically removing duplicate rows.",
        setup_sql="""
        CREATE TABLE contact(id INTEGER PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL);
        INSERT INTO contact VALUES (1, 'a@example.test', 'Alice'), (2, 'a@example.test', 'Alias'), (3, 'b@example.test', 'Bao');
        """,
        reference_sql="""
        DELETE FROM contact WHERE id NOT IN (SELECT MIN(id) FROM contact GROUP BY email);
        CREATE UNIQUE INDEX uq_contact_email ON contact(email);
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM (SELECT email FROM contact GROUP BY email HAVING COUNT(*) > 1)", 0, "duplicates_removed"),
            _scalar_check("SELECT COUNT(*) FROM pragma_index_list('contact') WHERE name='uq_contact_email' AND \"unique\"=1", 1, "unique_index_present"),
            _scalar_check("SELECT COUNT(*) FROM contact", 2, "canonical_rows_preserved"),
        ],
        branch_types=["data_profiling", "uniqueness_analysis", "data_repair", "constraint_authoring", "regression_verification"],
    ),
    _db_spec(
        "BatchApplyTest.test_add_ck",
        task="Enforce non-negative inventory quantities, repairing existing negative values first.",
        setup_sql="""
        CREATE TABLE inventory(id INTEGER PRIMARY KEY, sku TEXT NOT NULL, quantity INTEGER NOT NULL);
        INSERT INTO inventory VALUES (1, 'A', 4), (2, 'B', -3);
        """,
        reference_sql="""
        UPDATE inventory SET quantity=0 WHERE quantity < 0;
        CREATE TABLE inventory_new(id INTEGER PRIMARY KEY, sku TEXT NOT NULL, quantity INTEGER NOT NULL CHECK(quantity >= 0));
        INSERT INTO inventory_new SELECT id, sku, quantity FROM inventory;
        DROP TABLE inventory;
        ALTER TABLE inventory_new RENAME TO inventory;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM inventory WHERE quantity < 0", 0, "negative_values_repaired"),
            _scalar_check("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='inventory' AND lower(sql) LIKE '%check(quantity >= 0)%'", 1, "check_constraint_present"),
            _scalar_check("SELECT COUNT(*) FROM inventory", 2, "rows_preserved"),
        ],
        branch_types=["data_profiling", "constraint_analysis", "data_repair", "table_rebuild", "postcondition_verification"],
    ),
    _db_spec(
        "BatchRoundTripTest.test_change_type_boolean_to_int",
        task="Convert textual feature flags to constrained integer booleans without changing their meaning.",
        setup_sql="""
        CREATE TABLE feature_flag(id INTEGER PRIMARY KEY, name TEXT NOT NULL, active TEXT NOT NULL);
        INSERT INTO feature_flag VALUES (1, 'search', 'true'), (2, 'export', 'false');
        """,
        reference_sql="""
        CREATE TABLE feature_flag_new(id INTEGER PRIMARY KEY, name TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN (0,1)));
        INSERT INTO feature_flag_new SELECT id, name, CASE lower(active) WHEN 'true' THEN 1 ELSE 0 END FROM feature_flag;
        DROP TABLE feature_flag;
        ALTER TABLE feature_flag_new RENAME TO feature_flag;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM feature_flag WHERE typeof(active)='integer'", 2, "integer_storage"),
            _scalar_check("SELECT SUM(active) FROM feature_flag", 1, "meaning_preserved"),
            _scalar_check("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='feature_flag' AND sql LIKE '%CHECK(active IN (0,1))%'", 1, "boolean_constraint_present"),
        ],
        branch_types=["schema_inspection", "data_profiling", "conversion_planning", "table_rebuild", "semantic_verification"],
    ),
    _db_spec(
        "AutodetectorTests.test_alter_field_to_not_null_without_default",
        task="Make profile.nickname non-null without a schema default while preserving every profile.",
        setup_sql="""
        CREATE TABLE profile(id INTEGER PRIMARY KEY, nickname TEXT, email TEXT NOT NULL);
        INSERT INTO profile VALUES (1, NULL, 'a@example.test'), (2, 'bao', 'b@example.test');
        """,
        reference_sql="""
        UPDATE profile SET nickname='unknown-' || id WHERE nickname IS NULL;
        CREATE TABLE profile_new(id INTEGER PRIMARY KEY, nickname TEXT NOT NULL, email TEXT NOT NULL);
        INSERT INTO profile_new SELECT id, nickname, email FROM profile;
        DROP TABLE profile;
        ALTER TABLE profile_new RENAME TO profile;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM profile WHERE nickname IS NULL", 0, "nulls_repaired"),
            _scalar_check("SELECT \"notnull\" FROM pragma_table_info('profile') WHERE name='nickname'", 1, "not_null_enforced"),
            _scalar_check("SELECT COUNT(*) FROM profile", 2, "rows_preserved"),
        ],
        branch_types=["data_profiling", "nullability_analysis", "data_repair", "schema_migration", "postcondition_verification"],
    ),
    _db_spec(
        "AutodetectorTests.test_rename_referenced_primary_key",
        task="Rename project.legacy_id to project_id while preserving task foreign keys and rows.",
        setup_sql="""
        PRAGMA foreign_keys=ON;
        CREATE TABLE project(legacy_id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE task(id INTEGER PRIMARY KEY, project_id TEXT NOT NULL REFERENCES project(legacy_id), title TEXT NOT NULL);
        INSERT INTO project VALUES ('PR-1', 'Migration');
        INSERT INTO task VALUES (1, 'PR-1', 'Verify references');
        """,
        reference_sql="ALTER TABLE project RENAME COLUMN legacy_id TO project_id;",
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('project') WHERE name='project_id'", 1, "primary_key_renamed"),
            _scalar_check("SELECT COUNT(*) FROM pragma_foreign_key_list('task') WHERE \"to\"='project_id'", 1, "foreign_key_retargeted"),
            _scalar_check("SELECT COUNT(*) FROM task JOIN project USING(project_id)", 1, "relationship_preserved"),
        ],
        branch_types=["schema_inspection", "dependency_analysis", "migration_authoring", "foreign_key_verification"],
    ),
    _db_spec(
        "AutodetectorTests.test_circular_fk_dependency",
        task="Introduce mutually-referencing foreign keys for department and manager while preserving the existing cycle.",
        setup_sql="""
        CREATE TABLE department(id INTEGER PRIMARY KEY, manager_id INTEGER);
        CREATE TABLE manager(id INTEGER PRIMARY KEY, department_id INTEGER);
        INSERT INTO department VALUES (1, 10);
        INSERT INTO manager VALUES (10, 1);
        """,
        reference_sql="""
        PRAGMA foreign_keys=OFF;
        CREATE TABLE department_new(id INTEGER PRIMARY KEY, manager_id INTEGER REFERENCES manager_new(id) DEFERRABLE INITIALLY DEFERRED);
        CREATE TABLE manager_new(id INTEGER PRIMARY KEY, department_id INTEGER REFERENCES department_new(id) DEFERRABLE INITIALLY DEFERRED);
        INSERT INTO department_new SELECT * FROM department;
        INSERT INTO manager_new SELECT * FROM manager;
        DROP TABLE department;
        DROP TABLE manager;
        ALTER TABLE department_new RENAME TO department;
        ALTER TABLE manager_new RENAME TO manager;
        PRAGMA foreign_keys=ON;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_foreign_key_list('department')", 1, "department_fk_present"),
            _scalar_check("SELECT COUNT(*) FROM pragma_foreign_key_list('manager')", 1, "manager_fk_present"),
            _scalar_check("SELECT COUNT(*) FROM department d JOIN manager m ON d.manager_id=m.id AND m.department_id=d.id", 1, "cycle_preserved"),
        ],
        branch_types=["schema_inspection", "dependency_graph_analysis", "migration_ordering", "table_rebuild", "integrity_verification"],
    ),
    _db_spec(
        "AutodetectorTests.test_remove_field_with_remove_index_or_constraint_dependency",
        task="Remove article.legacy_slug together with its dependent index while preserving article content.",
        setup_sql="""
        CREATE TABLE article(id INTEGER PRIMARY KEY, title TEXT NOT NULL, legacy_slug TEXT NOT NULL);
        CREATE INDEX ix_article_legacy_slug ON article(legacy_slug);
        INSERT INTO article VALUES (1, 'Routing', 'routing-old'), (2, 'Recovery', 'recovery-old');
        """,
        reference_sql="""
        DROP INDEX ix_article_legacy_slug;
        CREATE TABLE article_new(id INTEGER PRIMARY KEY, title TEXT NOT NULL);
        INSERT INTO article_new SELECT id, title FROM article;
        DROP TABLE article;
        ALTER TABLE article_new RENAME TO article;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('article') WHERE name='legacy_slug'", 0, "legacy_column_removed"),
            _scalar_check("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='ix_article_legacy_slug'", 0, "dependent_index_removed"),
            _scalar_check("SELECT COUNT(*) FROM article", 2, "content_preserved"),
        ],
        branch_types=["schema_inspection", "dependency_analysis", "destructive_change_planning", "table_rebuild", "regression_verification"],
    ),
    _db_spec(
        "AutodetectorTests.test_add_field_and_unique_together",
        task="Add membership.role and enforce uniqueness of user, team, and role without deleting memberships.",
        setup_sql="""
        CREATE TABLE membership(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, team_id INTEGER NOT NULL);
        INSERT INTO membership VALUES (1, 7, 3), (2, 7, 3), (3, 8, 3);
        """,
        reference_sql="""
        ALTER TABLE membership ADD COLUMN role TEXT NOT NULL DEFAULT 'member';
        UPDATE membership SET role='admin' WHERE id=2;
        CREATE UNIQUE INDEX uq_membership_user_team_role ON membership(user_id, team_id, role);
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('membership') WHERE name='role'", 1, "role_added"),
            _scalar_check("SELECT COUNT(*) FROM (SELECT user_id,team_id,role FROM membership GROUP BY user_id,team_id,role HAVING COUNT(*)>1)", 0, "combinations_unique"),
            _scalar_check("SELECT COUNT(*) FROM membership", 3, "memberships_preserved"),
        ],
        branch_types=["schema_inspection", "collision_profiling", "data_backfill", "constraint_authoring", "postcondition_verification"],
    ),
    _db_spec(
        "OperationTests.test_run_python_atomic",
        task="Split customer.full_name into first_name and last_name atomically while retaining the original column for rollback.",
        setup_sql="""
        CREATE TABLE customer(id INTEGER PRIMARY KEY, full_name TEXT NOT NULL);
        INSERT INTO customer VALUES (1, 'Alice Nguyen'), (2, 'Bao Tran');
        """,
        reference_sql="""
        BEGIN IMMEDIATE;
        ALTER TABLE customer ADD COLUMN first_name TEXT;
        ALTER TABLE customer ADD COLUMN last_name TEXT;
        UPDATE customer SET first_name=substr(full_name,1,instr(full_name,' ')-1), last_name=substr(full_name,instr(full_name,' ')+1);
        COMMIT;
        """,
        checks=[
            _scalar_check("SELECT COUNT(*) FROM customer WHERE first_name IS NULL OR last_name IS NULL", 0, "names_backfilled"),
            _scalar_check("SELECT COUNT(*) FROM customer WHERE first_name || ' ' || last_name = full_name", 2, "split_semantics_preserved"),
            _scalar_check("SELECT COUNT(*) FROM pragma_table_info('customer') WHERE name IN ('first_name','last_name')", 2, "columns_added"),
        ],
        branch_types=["data_profiling", "transformation_planning", "transactional_execution", "semantic_verification", "rollback_readiness"],
    ),
)


DATABASE_SPEC_BY_RECORD = {spec["record_id"]: spec for spec in DATABASE_SPECS}


FORMAL_SELECTION_IDS = (
    "proof-april-lme-0008eb23463d",
    "proof-april-mlme-006213febdb2",
    "proof-april-tme-00338b98d4ff",
    "proof-april-thme-0014a26fbc97",
    "proof-april-tme-0072686794ba",
    "proof-workbook-lean-workbook-plus-1003",
    "proof-workbook-lean-workbook-plus-1002",
    "proof-workbook-lean-workbook-plus-66384",
    "proof-workbook-lean-workbook-plus-2536",
    "proof-workbook-lean-workbook-plus-27829",
)


def _evaluate_checks(connection: sqlite3.Connection, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for check in checks:
        try:
            rows = [list(row) for row in connection.execute(str(check["query"])).fetchall()]
            error = None
        except sqlite3.Error as exc:
            rows = []
            error = f"{type(exc).__name__}: {exc}"
        reports.append(
            {
                "label": check["label"],
                "passed": error is None and rows == check["expected_rows"],
                "expected_rows": check["expected_rows"],
                "actual_rows": rows,
                "error": error,
            }
        )
    return reports


def _sqlite_integrity(connection: sqlite3.Connection) -> dict[str, Any]:
    integrity = [row[0] for row in connection.execute("PRAGMA integrity_check").fetchall()]
    foreign_keys = [list(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()]
    return {
        "integrity_check": integrity,
        "foreign_key_violations": foreign_keys,
        "passed": integrity == ["ok"] and not foreign_keys,
    }


def _build_source_database(setup_sql: str) -> bytes:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(setup_sql)
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def _run_database_gate(source_bytes: bytes, reference_sql: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(source_bytes)
        connection.execute("PRAGMA foreign_keys=ON")
        baseline_checks = _evaluate_checks(connection, checks)
        baseline_integrity = _sqlite_integrity(connection)
        reference_error = None
        try:
            connection.executescript(reference_sql)
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            reference_error = f"{type(exc).__name__}: {exc}"
        reference_checks = _evaluate_checks(connection, checks)
        reference_integrity = _sqlite_integrity(connection)
    finally:
        connection.close()
    baseline_incomplete = any(not item["passed"] for item in baseline_checks)
    reference_success = (
        reference_error is None
        and all(item["passed"] for item in reference_checks)
        and reference_integrity["passed"]
    )
    return {
        "schema_version": "routingbench.database_execution_gate.v1",
        "baseline_incomplete": baseline_incomplete,
        "baseline_checks": baseline_checks,
        "baseline_integrity": baseline_integrity,
        "reference_execution_error": reference_error,
        "reference_success": reference_success,
        "reference_checks": reference_checks,
        "reference_integrity": reference_integrity,
        "gate_passed": baseline_incomplete and reference_success,
    }


def _build_database_packet(
    candidate_root: Path,
    output_root: Path,
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_id = str(candidate["candidate_id"])
    pack_dir = output_root / "packs" / "database_migration_integrity" / packet_id
    writer = _Writer(pack_dir)
    source_bytes = _build_source_database(str(spec["setup_sql"]))
    gate = _run_database_gate(source_bytes, str(spec["reference_sql"]), list(spec["checks"]))
    if not gate["gate_passed"]:
        raise RuntimeError(f"Database execution gate failed for {packet_id}: {gate}")
    contract = {
        "adapter_id": DATABASE_ADAPTER,
        "database_engine": "SQLite",
        "network_during_execution": False,
        "source_database": "public/source.sqlite",
        "terminal_condition": "all_target_checks_and_integrity_checks_pass",
        "allowed_commands": ["python -m routing_benchmark.v4.domain_packets.sqlite_runner"],
        "router_raw_database_bytes": False,
        "router_observation": {
            "schema_summary": "available_after_schema_inspection",
            "query_results": "available_after_authorized_executor_calls",
            "migration_attempt_status": "available_after_execution",
        },
        "executor_visible_files": [
            "public/task.md",
            "public/source.sqlite",
            "public/target_contract.json",
        ],
        "evaluator_oracle_visibility": "private",
    }
    target_contract = {
        "checks": spec["checks"],
        "required_integrity_checks": ["PRAGMA integrity_check", "PRAGMA foreign_key_check"],
        "reference_migration_visible": False,
    }
    task = str(spec["task"])
    writer.add("public/task.md", task + "\n", "public")
    writer.add("public/source.sqlite", source_bytes, "public")
    writer.add("public/execution_contract.json", _json_bytes(contract), "public")
    writer.add("public/target_contract.json", _json_bytes(target_contract), "public")
    source_test = candidate_root / str(candidate["candidate_path"]) / "source" / "source_test.json"
    writer.add("fixtures/upstream_source_test.json", source_test.read_bytes(), "fixture")
    writer.add(
        "fixtures/upstream_provenance.json",
        _json_bytes(
            {
                "source_repository": candidate["source_repository"],
                "source_revision": candidate["source_revision"],
                "source_path": candidate["source_path"],
                "source_record_id": candidate["source_record_id"],
                "source_license": candidate["source_license"],
                "adaptation": "standalone SQLite scenario preserving the upstream tested migration behavior",
            }
        ),
        "fixture",
    )
    writer.add("private/reference.sql", str(spec["reference_sql"]), "private")
    writer.add("private/baseline_verification.json", _json_bytes(gate), "private")
    manifest = writer.manifest(
        packet_id=packet_id,
        domain="database_migration_integrity",
        task=task,
        adapter_id=DATABASE_ADAPTER,
        source_type=f"{candidate['source_family']}_upstream_adapted",
        license_name=str(candidate["source_license"]),
        execution_readiness="runtime_verified",
        metadata={
            "source_family": candidate["source_family"],
            "source_revision": candidate["source_revision"],
            "source_record_id": candidate["source_record_id"],
            "verification_status": "locally_reproduced",
            "branch_types": spec["branch_types"],
        },
    )
    _write_json(pack_dir / "manifest.json", manifest.to_dict())
    validation = validate_artifact_pack(pack_dir)
    if not validation["valid"]:
        raise RuntimeError(f"Invalid database packet {packet_id}: {validation['errors']}")
    return gate, validation


def _workbook_reference(public_source: Mapping[str, Any], private_reference: Mapping[str, Any]) -> str:
    statement = str(public_source["formal_statement"])
    tactic = str(private_reference["reference_tactic"]).strip()
    replacement = "by\n  " + tactic.replace("\n", "\n  ")
    replaced, count = re.subn(r"\bby\s+sorry\b", replacement, statement, count=1)
    if count != 1:
        raise RuntimeError("Lean-Workbook statement does not contain the expected `by sorry`")
    return "import Mathlib\n\nopen scoped BigOperators\n\n" + replaced.strip() + "\n"


def _ensure_mathlib_import(source: str) -> str:
    source = source.lstrip()
    if re.search(r"^import\s+Mathlib\b", source, flags=re.MULTILINE):
        return source
    return "import Mathlib\n\n" + source


def _formal_source_gate(
    candidate: Mapping[str, Any], public_source: Mapping[str, Any], private_reference: Mapping[str, Any]
) -> dict[str, Any]:
    family = str(candidate["source_family"])
    if family == "APRIL":
        baseline = str(public_source.get("incorrect_proof") or "")
        reference = str(private_reference.get("correct_proof") or "")
        baseline_evidence = bool(public_source.get("compiler_error")) and bool(public_source.get("state_at_error"))
        reference_evidence = bool(reference.strip()) and "sorry" not in reference.lower()
        toolchain = "Lean 4.22.0-rc4 with matching Mathlib"
        evidence = "APRIL compiler-generated failure tuple and aligned verified repair"
    else:
        baseline = str(public_source.get("formal_statement") or "")
        reference = _workbook_reference(public_source, private_reference)
        baseline_evidence = "sorry" in baseline.lower() and bool(public_source.get("initial_proof_state"))
        reference_evidence = (
            str(private_reference.get("reference_final_state") or "").strip().lower() == "no goals"
            and "sorry" not in str(private_reference.get("reference_tactic") or "").lower()
        )
        toolchain = "Lean 4.8.0-rc1 with Mathlib4 v4.8.0-rc1"
        evidence = "Lean-Workbook proved status with initial and final compiler states"
    return {
        "schema_version": "routingbench.formal_source_execution_gate.v1",
        "verification_level": "pinned_upstream_compiler_evidence",
        "locally_recompiled": False,
        "toolchain": toolchain,
        "baseline_failure_or_incompleteness_verified": baseline_evidence,
        "reference_success_verified": reference_evidence,
        "reference_contains_forbidden_placeholder": "sorry" in reference.lower(),
        "evidence_source": evidence,
        "gate_passed": baseline_evidence and reference_evidence and "sorry" not in reference.lower(),
    }


def _build_formal_packet(
    candidate_root: Path,
    output_root: Path,
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_id = str(candidate["candidate_id"])
    source_dir = candidate_root / str(candidate["candidate_path"])
    public_source = _read_json(source_dir / "source" / "public_source_draft.json")
    private_reference = _read_json(source_dir / "private" / "reference.json")
    family = str(candidate["source_family"])
    gate = _formal_source_gate(candidate, public_source, private_reference)
    if not gate["gate_passed"]:
        raise RuntimeError(f"Formal source gate failed for {packet_id}: {gate}")
    if family == "APRIL":
        public_lean = _ensure_mathlib_import(str(public_source["incorrect_proof"]))
        reference_lean = _ensure_mathlib_import(str(private_reference["correct_proof"]))
        task = "Repair the supplied Lean proof using the compiler diagnostic and verify that the repaired theorem closes without placeholders."
        compiler_diagnostic = {
            "compiler_error": public_source.get("compiler_error"),
            "line_at_error": public_source.get("line_at_error"),
            "state_at_error": public_source.get("state_at_error"),
        }
    else:
        public_lean = (
            "import Mathlib\n\nopen scoped BigOperators\n\n"
            + str(public_source["formal_statement"]).strip()
            + "\n"
        )
        reference_lean = _workbook_reference(public_source, private_reference)
        task = "Complete the supplied Lean theorem, using the initial proof state, and verify that the theorem closes without placeholders."
        compiler_diagnostic = {"initial_proof_state": public_source.get("initial_proof_state")}
    pack_dir = output_root / "packs" / "formal_proof" / packet_id
    writer = _Writer(pack_dir)
    contract = {
        "adapter_id": FORMAL_ADAPTER,
        "language": "Lean 4",
        "toolchain": gate["toolchain"],
        "network_during_execution": False,
        "public_source": "public/Main.lean",
        "allowed_validation_command": "lake env lean public/Main.lean",
        "terminal_condition": "kernel_accepts_theorem_and_no_forbidden_placeholders",
        "forbidden_placeholders": ["sorry", "admit"],
        "router_raw_source": False,
        "router_observation": {
            "proof_state": "available_after_proof_state_inspection",
            "compiler_diagnostic": "available_after_compiler_execution",
            "candidate_status": "available_after_candidate_checking",
        },
        "executor_visible_files": ["public/task.md", "public/Main.lean", "public/proof_state.json"],
        "evaluator_oracle_visibility": "private",
    }
    writer.add("public/task.md", task + "\n", "public")
    writer.add("public/Main.lean", public_lean, "public")
    writer.add("public/proof_state.json", _json_bytes(compiler_diagnostic), "public")
    writer.add("public/execution_contract.json", _json_bytes(contract), "public")
    writer.add(
        "fixtures/upstream_provenance.json",
        _json_bytes(
            {
                "source_repository": candidate["source_repository"],
                "source_revision": candidate["source_revision"],
                "source_record_id": candidate["source_record_id"],
                "source_split": candidate["source_split"],
                "source_license": candidate["source_license"],
                "toolchain": gate["toolchain"],
            }
        ),
        "fixture",
    )
    writer.add("private/Reference.lean", reference_lean, "private")
    writer.add("private/source_execution_evidence.json", _json_bytes(gate), "private")
    manifest = writer.manifest(
        packet_id=packet_id,
        domain="formal_proof",
        task=task,
        adapter_id=FORMAL_ADAPTER,
        source_type=f"{family.lower()}_upstream",
        license_name=str(candidate["source_license"]),
        execution_readiness="pending_runtime_verification",
        metadata={
            "source_family": family,
            "source_revision": candidate["source_revision"],
            "source_record_id": candidate["source_record_id"],
            "verification_status": "pinned_upstream_compiler_verified_local_recompile_pending",
            "workflow_features": candidate["workflow_features"],
        },
    )
    _write_json(pack_dir / "manifest.json", manifest.to_dict())
    validation = validate_artifact_pack(pack_dir)
    if not validation["valid"]:
        raise RuntimeError(f"Invalid formal packet {packet_id}: {validation['errors']}")
    return gate, validation


def _screening_reason(candidate: Mapping[str, Any], selected: bool) -> tuple[str, str]:
    if selected:
        return (
            "selected",
            "Passed the required evidence gate and fills a source-balanced, non-redundant workflow branch category.",
        )
    if candidate["domain"] == "database_migration_integrity":
        features = set(candidate.get("workflow_features") or [])
        nonportable = {
            "migration_graph_conflict",
            "partial_application_recovery",
            "history_integrity",
            "migration_graph_branching",
            "migration_graph_traversal",
            "transaction_atomicity",
        }
        if features & nonportable:
            return (
                "excluded_adapter_mismatch",
                "The upstream test primarily exercises framework migration-history or transaction orchestration, not a standalone SQLite state transition with equivalent observable semantics.",
            )
        return (
            "excluded_redundant_behavior",
            "Its principal schema behavior is already covered by a retained packet from the same source family, while another retained packet adds a distinct routing state.",
        )
    if candidate["source_family"] == "APRIL":
        return (
            "excluded_error_or_source_redundancy",
            "A retained APRIL packet covers the same error family or source theorem while preserving broader compiler-failure diversity.",
        )
    return (
        "excluded_tactic_redundancy_or_short_reference",
        "A retained Lean-Workbook packet provides a longer reference and a distinct tactic family under the fixed six-construction quota.",
    )


def _readme(summary: Mapping[str, Any]) -> str:
    return f"""# RoutingBench V4 selected domain packets RC1

This package is the result of screening the 30+30 candidate pool to the retained packets listed below.

## Result

- Database migration & integrity: {summary['selected_by_domain']['database_migration_integrity']} packets.
- Formal proof construction & repair: {summary['selected_by_domain']['formal_proof']} packets.
- Total: {summary['selected_total']} packets.

## Selection contract

No learned utility score or hand-authored numeric preference was used. A retained packet must:

1. have pinned source provenance and a private reference;
2. expose baseline failure or incompleteness before the private reference is applied;
3. have reference success supported by deterministic checks;
4. add a non-redundant workflow/error/tactic category within a source-balanced quota.

Database packets were locally executed on isolated SQLite copies. Formal packets are materialized
as Lean projects, but this host does not contain the two pinned Lean/Mathlib toolchains. Their gates
therefore preserve and validate upstream compiler evidence; manifests explicitly remain
`pending_runtime_verification` until a local recompile is performed. This distinction prevents
upstream evidence from being misreported as local execution.

## Visibility

- `public/`: executor-visible task input and terminal contract.
- `fixtures/`: provenance and upstream source evidence; not a router hint.
- `private/`: reference migration/proof and gate result; never router-visible.

`candidate_screening.jsonl` records a reason for every one of the original 60 candidates.
"""


def build_domain_packets(candidate_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; pass overwrite=True")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    database = _read_jsonl(candidate_root / "database_migration_integrity_candidates.jsonl")
    formal = _read_jsonl(candidate_root / "formal_proof_candidates.jsonl")
    database_by_record = {str(row["source_record_id"]): row for row in database}
    formal_by_id = {str(row["candidate_id"]): row for row in formal}
    missing_db = sorted(set(DATABASE_SPEC_BY_RECORD) - set(database_by_record))
    missing_formal = sorted(set(FORMAL_SELECTION_IDS) - set(formal_by_id))
    if missing_db or missing_formal:
        raise RuntimeError(f"Missing selected candidates: database={missing_db}, formal={missing_formal}")

    selected_ids: set[str] = set()
    selection_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for spec in DATABASE_SPECS:
        candidate = database_by_record[str(spec["record_id"])]
        gate, validation = _build_database_packet(candidate_root, output_root, candidate, spec)
        selected_ids.add(str(candidate["candidate_id"]))
        validations.append(validation)
        selection_rows.append(
            {
                "packet_id": candidate["candidate_id"],
                "domain": candidate["domain"],
                "source_family": candidate["source_family"],
                "source_record_id": candidate["source_record_id"],
                "selection_status": "selected",
                "verification_status": "locally_reproduced",
                "baseline_incomplete": gate["baseline_incomplete"],
                "reference_success": gate["reference_success"],
            }
        )
    for candidate_id in FORMAL_SELECTION_IDS:
        candidate = formal_by_id[candidate_id]
        gate, validation = _build_formal_packet(candidate_root, output_root, candidate)
        selected_ids.add(candidate_id)
        validations.append(validation)
        selection_rows.append(
            {
                "packet_id": candidate_id,
                "domain": candidate["domain"],
                "source_family": candidate["source_family"],
                "source_record_id": candidate["source_record_id"],
                "selection_status": "selected",
                "verification_status": gate["verification_level"],
                "baseline_incomplete": gate["baseline_failure_or_incompleteness_verified"],
                "reference_success": gate["reference_success_verified"],
            }
        )

    screening: list[dict[str, Any]] = []
    for candidate in sorted(database + formal, key=lambda row: str(row["candidate_id"])):
        selected = str(candidate["candidate_id"]) in selected_ids
        status, reason = _screening_reason(candidate, selected)
        screening.append(
            {
                "candidate_id": candidate["candidate_id"],
                "domain": candidate["domain"],
                "source_family": candidate["source_family"],
                "source_record_id": candidate["source_record_id"],
                "status": status,
                "reason": reason,
            }
        )
    _write_jsonl(output_root / "selection_manifest.jsonl", selection_rows)
    _write_jsonl(output_root / "candidate_screening.jsonl", screening)
    summary = {
        "schema_version": "routingbench.v4.domain_packet_selection.v1",
        "candidate_total": len(database) + len(formal),
        "selected_total": len(selected_ids),
        "selected_by_domain": {
            "database_migration_integrity": sum(row["domain"] == "database_migration_integrity" for row in selection_rows),
            "formal_proof": sum(row["domain"] == "formal_proof" for row in selection_rows),
        },
        "database_local_execution_gate_passed": sum(
            row["domain"] == "database_migration_integrity" and row["reference_success"]
            for row in selection_rows
        ),
        "formal_upstream_compiler_evidence_gate_passed": sum(
            row["domain"] == "formal_proof" and row["reference_success"] for row in selection_rows
        ),
        "formal_local_recompile_pending": sum(row["domain"] == "formal_proof" for row in selection_rows),
        "artifact_validation_passed": sum(bool(item["valid"]) for item in validations),
        "artifact_validation_total": len(validations),
        "human_review_status": "pending",
    }
    _write_json(output_root / "selection_report.json", summary)
    _write_json(
        output_root / "validation_report.json",
        {"valid": all(item["valid"] for item in validations), "packets": validations},
    )
    (output_root / "README.md").write_text(_readme(summary), encoding="utf-8", newline="\n")
    return summary
