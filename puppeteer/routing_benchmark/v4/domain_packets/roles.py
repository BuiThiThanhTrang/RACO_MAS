from __future__ import annotations

from typing import Any


DATABASE_ROLE_CARDS: tuple[dict[str, Any], ...] = (
    {
        "agent_id": "Schema Inspector",
        "role_goal": "Read the current schema, indexes, triggers, foreign keys and migration history without modifying the database.",
        "use_when": [
            "The current schema has not yet been materialized as an observable artifact.",
            "A migration reports a table, column, index or dependency error that is not yet understood.",
            "The schema changed after a migration attempt.",
        ],
        "avoid_when": [
            "A fresh schema snapshot exists and the database has not changed since it was produced.",
            "The remaining operation is authoring, execution or verification.",
        ],
        "expected_outputs": ["A schema snapshot and dependency map tied to public artifacts."],
    },
    {
        "agent_id": "Data Profiler",
        "role_goal": "Measure data properties that affect migration, including nulls, duplicates, orphan records, value types and distributions.",
        "use_when": [
            "The migration depends on properties of the current data.",
            "A new constraint may conflict with existing rows.",
            "A dry run failed because the data did not satisfy the target contract.",
        ],
        "avoid_when": [
            "No database artifact is materialized.",
            "A fresh data profile exists and no data has changed since it was produced.",
        ],
        "expected_outputs": ["A data profile and anomaly summary tied to observable rows."],
    },
    {
        "agent_id": "Constraint Analyzer",
        "role_goal": "Analyze invariants, foreign keys, uniqueness, check constraints and concrete validation failures.",
        "use_when": [
            "An invariant or constraint must be reconciled with the observed schema and data profile.",
            "Integrity validation or a dry run returned a concrete constraint-related failure.",
        ],
        "avoid_when": [
            "No constraint contract or failure evidence is visible.",
            "The remaining operation is only to author already-determined SQL.",
        ],
        "expected_outputs": ["An evidence-grounded constraint analysis and failure explanation."],
    },
    {
        "agent_id": "Migration Planner",
        "role_goal": "Plan schema-migration ordering, transaction boundaries and rollback boundaries.",
        "use_when": [
            "The source schema and target requirements are sufficiently established to plan the migration.",
            "Dependencies or operation ordering remain unresolved.",
            "A dry run shows that the current migration order is infeasible.",
        ],
        "avoid_when": [
            "A plan remains applicable to the current evidence.",
            "Only validation or terminal completion remains.",
        ],
        "expected_outputs": ["A migration plan grounded in visible schema, data and dependency evidence."],
    },
    {
        "agent_id": "Data Repair Planner",
        "role_goal": "Design the smallest justified backfill, deduplication, remapping or orphan-record repair required before constraints can be enforced.",
        "use_when": [
            "The data profile reveals rows that do not satisfy the target schema or integrity contract.",
            "A concrete integrity failure requires data repair rather than only DDL repair.",
        ],
        "avoid_when": [
            "No concrete data anomaly is visible.",
            "The current candidate only needs execution or verification.",
        ],
        "expected_outputs": ["A data-repair plan preserving all data not implicated by the public contract."],
    },
    {
        "agent_id": "SQL Migration Author",
        "role_goal": "Author the smallest SQLite migration justified by the visible schema, evidence and current plan.",
        "use_when": [
            "Enough schema and evidence exist to create a candidate migration.",
            "A previous candidate failed and new feedback supports a materially revised candidate.",
        ],
        "avoid_when": [
            "The current candidate is awaiting dry-run execution or validation.",
            "The same strategy would be repeated without new evidence.",
        ],
        "expected_outputs": ["A candidate SQL migration, not an execution or success claim."],
    },
    {
        "agent_id": "Transactional Dry-run Executor",
        "role_goal": "Execute the current candidate migration on an isolated copy inside the adapter's transaction or savepoint and record the execution trace.",
        "use_when": [
            "A candidate migration exists and has not been run.",
            "The candidate changed after the latest dry run.",
        ],
        "avoid_when": [
            "No candidate migration exists.",
            "The unchanged current candidate already has a fresh dry-run result.",
        ],
        "expected_outputs": ["A dry-run result and migrated database candidate for the current migration."],
    },
    {
        "agent_id": "Integrity Validator",
        "role_goal": "Verify the target schema, invariants, foreign keys, row-level expectations and regression queries on the current executed artifact.",
        "use_when": [
            "A database candidate was produced by a successful dry run and lacks fresh validation.",
            "The candidate or its data changed after the latest validation.",
        ],
        "avoid_when": [
            "The migration has not executed successfully.",
            "Fresh validation already covers the unchanged candidate.",
        ],
        "expected_outputs": ["An integrity report including PRAGMA integrity_check, PRAGMA foreign_key_check and task-specific checks."],
    },
    {
        "agent_id": "Query Performance Auditor",
        "role_goal": "Audit query plans or explicit performance limits stated by the task contract.",
        "use_when": [
            "The task contains an explicit performance or query-plan contract.",
            "An index or query-plan change must be checked against that explicit contract.",
        ],
        "avoid_when": [
            "The task contains no performance requirement.",
            "The candidate has not yet passed basic integrity validation.",
        ],
        "expected_outputs": ["A query-plan or performance report tied to the public contract."],
    },
    {
        "agent_id": "Snapshot & Rollback Operator",
        "role_goal": "Create a database snapshot or restore the working database after a failed or incomplete attempt.",
        "use_when": [
            "A risky migration requires a snapshot before execution.",
            "A dry run leaves observable partial, damaged or incomplete state that must be restored.",
        ],
        "avoid_when": [
            "No database artifact is materialized.",
            "No state change needs rollback, or the current snapshot remains sufficient.",
        ],
        "expected_outputs": ["A snapshot artifact or restored database state with an observable operation result."],
    },
)


FORMAL_ROLE_CARDS: tuple[dict[str, Any], ...] = (
    {
        "agent_id": "Proof State Inspector",
        "role_goal": "Inspect the visible Lean goal, hypotheses, source and compiler diagnostics without changing the proof.",
        "use_when": ["The active goal or compiler state is not yet understood.", "A new compiler result materially changes the proof state."],
        "avoid_when": ["The same state already has sufficient fresh inspection evidence.", "A candidate only needs compilation or final verification."],
        "expected_outputs": ["A precise observable proof-state summary."],
    },
    {
        "agent_id": "Premise and Lemma Retriever",
        "role_goal": "Identify applicable visible hypotheses and Mathlib lemmas without claiming that a proof has compiled.",
        "use_when": ["The goal requires selecting a lemma or transformation not evident from current state.", "A compiler failure indicates a missing or mismatched lemma."],
        "avoid_when": ["The needed lemma and candidate proof are already explicit.", "The next operation is compilation."],
        "expected_outputs": ["Candidate lemmas or premises and why their types match the visible goal."],
    },
    {
        "agent_id": "Tactic Synthesizer",
        "role_goal": "Construct a candidate Lean proof from the current goal and available evidence.",
        "use_when": ["The theorem is incomplete and no candidate proof exists.", "A different proof strategy is justified by new evidence."],
        "avoid_when": ["An unchanged candidate is awaiting compilation.", "A failed candidate first needs diagnosis rather than blind regeneration."],
        "expected_outputs": ["An executor-authored candidate proof, not a success claim."],
    },
    {
        "agent_id": "Proof Repair Specialist",
        "role_goal": "Revise a failed Lean candidate using the latest compiler diagnostic while preserving the theorem statement.",
        "use_when": ["A candidate proof has a concrete compiler failure.", "New feedback supports a materially revised candidate."],
        "avoid_when": ["No failed candidate exists.", "The current candidate has not yet been compiled."],
        "expected_outputs": ["A revised candidate tied to the concrete diagnostic."],
    },
    {
        "agent_id": "Lean Compiler Executor",
        "role_goal": "Compile the current candidate in the packet's pinned Lean and Mathlib environment.",
        "use_when": ["A candidate exists without a fresh compiler result.", "The candidate changed after the previous compilation."],
        "avoid_when": ["No candidate exists.", "The unchanged candidate already has a fresh compiler result."],
        "expected_outputs": ["Kernel/compiler exit status and exact public diagnostics."],
    },
    {
        "agent_id": "Kernel and Assumption Verifier",
        "role_goal": "Verify that a compiled theorem closes without forbidden placeholders or added assumptions.",
        "use_when": ["The candidate compiles and terminal compliance still needs confirmation."],
        "avoid_when": ["Compilation is failing or stale.", "No candidate exists."],
        "expected_outputs": ["Terminal-contract evidence for theorem identity, placeholders and assumptions."],
    },
    {
        "agent_id": "Formal Failure Diagnostician",
        "role_goal": "Explain a concrete Lean compiler or proof failure and identify evidence-grounded repair constraints.",
        "use_when": ["A fresh compiler diagnostic exists.", "Repeated candidates fail without substantive progress."],
        "avoid_when": ["No compiler or proof failure is visible.", "The current candidate only needs compilation."],
        "expected_outputs": ["A diagnosis grounded in the visible goal and compiler feedback."],
    },
)


STRUCTURED_DATA_ROLE_CARDS: tuple[dict[str, Any], ...] = (
    {
        "agent_id": "Workbook Surveyor",
        "role_goal": "Inspect workbook sheets, dimensions, values, formulas and styles without modifying the public input.",
        "use_when": ["The workbook structure or relevant ranges are not yet grounded in evidence.", "A later failure suggests an incorrect structural assumption."],
        "avoid_when": ["A fresh inventory already answers the current question.", "The current candidate only needs execution or validation."],
        "expected_outputs": ["A bounded workbook inventory tied to the immutable public input."],
    },
    {
        "agent_id": "Data Quality Profiler",
        "role_goal": "Profile rows, missing values, key multiplicity and data types relevant to the requested transformation.",
        "use_when": ["The transformation depends on duplicate keys, nulls or value distributions.", "Observed output indicates a data-dependent edge case."],
        "avoid_when": ["No relevant sheet has been identified.", "A fresh profile already covers the unchanged input."],
        "expected_outputs": ["An evidence-grounded data profile and duplicate-group summary."],
    },
    {
        "agent_id": "Formula Dependency Analyst",
        "role_goal": "Inspect formulas and dependencies that may be affected by workbook changes.",
        "use_when": ["The target or surrounding sheets contain formulas.", "A transformation may invalidate formula references."],
        "avoid_when": ["The task and inspection establish that formulas are irrelevant.", "Execution or validation is the only remaining action."],
        "expected_outputs": ["A formula inventory and dependency risks grounded in public cells."],
    },
    {
        "agent_id": "Table Reshaping Specialist",
        "role_goal": "Author a reusable Python/OpenPyXL transformation for grouping, filtering, joining, sorting or reshaping tabular workbook data.",
        "use_when": ["The requested operation is primarily structural or row-oriented and enough evidence exists to author a candidate.", "A materially different structural candidate is justified by new evidence."],
        "avoid_when": ["An unchanged candidate is awaiting execution.", "A concrete failure first needs diagnosis."],
        "expected_outputs": ["A candidate solve(input_path, output_path) implementation, not a success claim."],
    },
    {
        "agent_id": "Formula Computation Specialist",
        "role_goal": "Author a reusable transformation when formulas, calculated fields or spreadsheet computation semantics dominate the task.",
        "use_when": ["The output requires creating or preserving formula-driven computation.", "Formula evidence justifies a new candidate."],
        "avoid_when": ["The task is a pure row transformation with no material formula behavior.", "An unchanged candidate is awaiting execution."],
        "expected_outputs": ["A computation-focused candidate solve implementation."],
    },
    {
        "agent_id": "Cross-sheet Reconciler",
        "role_goal": "Author a reusable transformation that reconciles identifiers or values across multiple sheets while preserving unrelated content.",
        "use_when": ["The task requires matching or propagating information across sheets.", "A mismatch is traced to cross-sheet identity or reference logic."],
        "avoid_when": ["Only one sheet is semantically involved.", "The current candidate only needs execution or validation."],
        "expected_outputs": ["A cross-sheet candidate solve implementation with explicit matching logic."],
    },
    {
        "agent_id": "Formatting & Presentation Specialist",
        "role_goal": "Author a reusable transformation when styles, number formats, layout or presentation are part of the observable contract.",
        "use_when": ["The task explicitly requires formatting or layout preservation/change.", "Validation evidence identifies a presentation defect."],
        "avoid_when": ["Only data values and row semantics matter.", "There is no formatting evidence or requirement."],
        "expected_outputs": ["A presentation-aware candidate solve implementation."],
    },
    {
        "agent_id": "Workbook Repair Specialist",
        "role_goal": "Revise a failed candidate using concrete public execution or validation evidence.",
        "use_when": ["A candidate has a fresh execution or validation failure.", "New evidence supports a materially revised implementation."],
        "avoid_when": ["No concrete failure exists.", "The current candidate has not been executed."],
        "expected_outputs": ["A revised candidate tied to the observed failure evidence."],
    },
    {
        "agent_id": "Workbook Test Executor",
        "role_goal": "Run the current candidate on the public workbook in the restricted Python/OpenPyXL runner.",
        "use_when": ["A candidate exists without a fresh public execution.", "The candidate changed after the latest execution."],
        "avoid_when": ["No candidate exists.", "The unchanged candidate already has a fresh execution."],
        "expected_outputs": ["A public execution result and output workbook for the current candidate hash."],
    },
    {
        "agent_id": "Workbook Integrity Validator",
        "role_goal": "Validate a fresh public output against deterministic public transformation and preservation constraints.",
        "use_when": ["The current candidate has a successful fresh public execution.", "The output has not yet been validated for the current candidate hash."],
        "avoid_when": ["Execution failed or is stale.", "Fresh validation already covers the unchanged candidate."],
        "expected_outputs": ["A deterministic public validation report tied to candidate and output hashes."],
    },
)


DOMAIN_ROLE_CARDS = {
    "database_migration_integrity": DATABASE_ROLE_CARDS,
    "formal_proof": FORMAL_ROLE_CARDS,
    "structured_data_processing": STRUCTURED_DATA_ROLE_CARDS,
}


def role_cards_for(domain: str) -> list[dict[str, Any]]:
    return [dict(item) for item in DOMAIN_ROLE_CARDS[domain]]
