# Thiết kế agent pool và role card cho năm domain mới

Ngày chốt thiết kế: 2026-11-01.

Tài liệu này đặc tả agent pool, public role card và executor binding dự kiến cho năm domain mới của RoutingBench:

1. `structured_data_processing`;
2. `scientific_procedure`;
3. `cross_app_workflow`;
4. `multi_hop_knowledge`;
5. `document_media_processing`.

Mỗi domain có 20 sample trong một cohort chung; không chia development/validation/test và không dùng unseen-domain holdout. Nguồn dữ liệu, số lượng sample và nguyên tắc selection đã được người dùng chấp thuận. Tài liệu này không sửa ba domain đã đóng băng: `code_debugging`, `formal_proof` và `database_migration_integrity`.

## 1. Chuẩn kế thừa từ ba domain đã đóng băng

Ba domain hiện tại dùng cùng năm trường role-card cốt lõi:

```json
{
  "agent_id": "...",
  "role_goal": "...",
  "use_when": ["..."],
  "avoid_when": ["..."],
  "expected_outputs": ["..."]
}
```

Năm domain mới phải dùng đúng semantics này.

- `agent_id`: identity ổn định, đồng thời là action mà router chọn.
- `role_goal`: contribution mà executor thực hiện, không phải hướng dẫn chọn workflow.
- `use_when`: tình huống công khai mà capability có khả năng tạo tiến triển.
- `avoid_when`: tình huống công khai mà contribution thường không cần thiết, stale hoặc lặp lại.
- `expected_outputs`: artifact/evidence mà role có thể tạo; không được viết như một success claim.

Không thêm các trường public sau:

- `weakness`;
- `limitations`;
- `recommended_next_role`;
- `required_previous_role`;
- `workflow_stage`;
- `priority`, `utility` hoặc `capability_score`;
- task-specific solution hint.

Code debugging có một số extension lịch sử như `role_name`, `semantic_family`, `tools` và `supported_inputs`. Các trường đó được giữ nguyên trong source và trace đã đóng băng, nhưng không trở thành required public schema. Release có thể lưu raw frozen card để provenance và materialize năm trường cốt lõi khi cần một canonical view.

## 2. Tách public role card khỏi executor binding

Public role card mô tả capability cho router. Executor binding là implementation contract dành cho actor/adapter và không phải gợi ý workflow.

```json
{
  "agent_id": "Formula Dependency Analyst",
  "tools": ["inspect_formulas", "trace_dependencies", "read_artifact"],
  "read_scope": ["public_workbook", "public_runtime_artifacts"],
  "write_scope": ["formula_analysis"],
  "side_effect_class": "read_only",
  "result_schema": "formula_analysis.v1"
}
```

Binding phải khớp role card và có implementation thật. Không thu nhỏ role card chỉ vì adapter chưa hỗ trợ capability. Nếu capability đã công bố thì adapter phải cung cấp tool, hoặc role phải bị loại khỏi pool trước khi benchmark được đóng băng.

## 3. Runtime semantics chung

Mỗi vòng chạy giữ flow của shared environment đã chốt:

```text
public state
  -> router nhận toàn bộ agent pool + STOP
  -> router chọn đúng một action
  -> adapter kiểm tra hard precondition tối thiểu
  -> selected role thực thi tool contract
  -> tool result và side effect thật được ghi nhận
  -> artifact/state được version hóa
  -> router nhận state mới
```

### 3.1 Router observation

Router được xem:

- task công khai;
- current public state;
- artifact inventory với version, hash và freshness;
- last observable result;
- structured public history theo cấu hình;
- calls remaining;
- toàn bộ role cards và `STOP`.

Router không được xem:

- private reference/gold output;
- evaluator assertions;
- `valid_actions` hoặc action labels;
- source difficulty label nếu label đó không thuộc task contract;
- workflow stage do harness suy ra;
- action mask từ `use_when`/`avoid_when`;
- preferred route hoặc reference trajectory.

### 3.2 Action handling

- Guard mặc định tắt; `use_when` và `avoid_when` không lọc action pool.
- Capability không hữu ích nhưng an toàn nên được thực thi và trả `no_progress` hoặc evidence tương ứng.
- `ACTION_REJECTED` chỉ dùng khi operation không thể hoặc không an toàn: thiếu artifact vật lý bắt buộc, vi phạm write scope, permission hoặc tool contract.
- Không auto-route sang role khác khi action bị reject hoặc fail.
- Không auto-fix output.
- Không auto-rollback. Nếu domain có recovery operation, router phải chọn recovery role.
- Mọi mutation và attempt đều được ghi nhận; candidate mới làm evidence cũ stale.
- Khi budget bằng 0 vẫn giữ flow hỏi router cuối; non-STOP được lưu `suppressed_by_budget`, STOP được xử lý theo terminal state.

### 3.3 Kết quả tách biệt

- `final_task_pass`: independent checker xác nhận artifact/state cuối đạt task contract.
- `workflow_stop_pass`: fresh workflow evidence tồn tại và router chọn STOP hợp lệ.
- `routing_quality`: chỉ có sau human review state-level.
- `infrastructure_validity`: transport/runtime/checker availability, không gộp vào task failure.

Không yêu cầu pass rate tối thiểu và không ép mọi role phải được dùng.

## 4. Thiết kế diversity cho 20 sample/domain

Mỗi domain có năm task archetype, bốn sample/archetype. Archetype chỉ cân bằng nguồn task; không được materialize thành workflow label trong router input.

| Domain | 5 archetype, mỗi loại 4 sample |
|---|---|
| Structured data | formula/dependency; reshape/join; cleaning/normalization; aggregation/calculation; formatting/presentation |
| Scientific procedure | direct analysis; method comparison; anomaly diagnosis; result reproduction; conclusion under uncertainty |
| Cross-app | lookup/update; synchronized records; batch operation; transaction/notification; reconciliation/recovery |
| Multi-hop knowledge | short chain; long chain; branching evidence; ambiguity/conflict; insufficient evidence |
| Document/media | OCR/reading order; table; formula; form/key-value; degraded/mixed-layout |

Selection phải hoàn tất trước live run. Không thay sample dựa trên pass/fail của router hoặc executor.

## 5. Domain `structured_data_processing`

### 5.1 Nguồn và task contract

- Nguồn chính: SpreadsheetBench Verified/V1.
- 20 workbook tasks, ưu tiên instruction nhiều operation và nhiều test workbook.
- Public: instruction, input workbook, execution contract, allowed edit scope.
- Private: golden workbook/evaluator assertions.
- Final checker: recalculation cố định rồi OJ-style comparison trên workbook copy.
- Original workbook không bị ghi đè; candidate workbook và reports được version hóa.

### 5.2 Public artifacts

- `workbook_inventory`;
- `data_profile`;
- `formula_analysis`;
- `candidate_workbook`;
- `test_execution`;
- `validation_report`;
- `repair_report`;
- `presentation_report`.

### 5.3 Role cards

#### Workbook Surveyor

- `role_goal`: Inspect workbook sheets, tables, named ranges, formulas, styles and visible structural relationships without modifying the workbook.
- `use_when`:
  - The workbook structure or target regions are not yet sufficiently understood.
  - A new candidate or execution result indicates that workbook structure may have changed.
- `avoid_when`:
  - A fresh inventory already covers the unchanged workbook and the relevant regions are explicit.
  - The remaining operation is a known transformation, execution or validation.
- `expected_outputs`:
  - A workbook inventory tied to the current workbook version.

#### Data Quality Profiler

- `role_goal`: Measure nulls, duplicates, types, distributions and malformed records that affect the requested workbook transformation.
- `use_when`:
  - The task depends on properties of the current data rather than only workbook layout.
  - A transformation or test result exposes a data-quality inconsistency.
- `avoid_when`:
  - No relevant tabular region is observable.
  - A fresh profile already covers unchanged input data and no new anomaly is visible.
- `expected_outputs`:
  - A data profile and anomaly summary tied to observable cells and ranges.

#### Formula Dependency Analyst

- `role_goal`: Analyze formulas, cross-sheet references and dependency relationships without modifying workbook cells.
- `use_when`:
  - The task depends on formulas, references or calculated values.
  - Execution or validation evidence indicates a formula-related inconsistency.
- `avoid_when`:
  - No formula or dependency requirement is visible.
  - A fresh formula analysis already covers the unchanged candidate.
- `expected_outputs`:
  - A formula dependency analysis identifying relevant references and observable anomalies.

#### Table Reshaping Specialist

- `role_goal`: Produce a candidate workbook transformation involving joins, pivots, splits, normalization or table restructuring.
- `use_when`:
  - The task requires a structural transformation of one or more tables.
  - Observable execution feedback supports a materially revised restructuring approach.
- `avoid_when`:
  - The task only requires formula, formatting or presentation changes.
  - The unchanged candidate is awaiting execution or validation.
- `expected_outputs`:
  - A versioned candidate workbook containing the justified table transformation.

#### Formula Computation Specialist

- `role_goal`: Produce or revise workbook formulas and derived calculations justified by the public task and current evidence.
- `use_when`:
  - The task requires calculated values, aggregation or formula repair.
  - A test or dependency report supports a materially revised formula implementation.
- `avoid_when`:
  - No calculation or formula change is required.
  - The unchanged formula candidate only needs execution or validation.
- `expected_outputs`:
  - A candidate workbook with versioned formula changes, not a validation claim.

#### Cross-sheet Reconciler

- `role_goal`: Compare and reconcile records or calculated results across sheets while preserving unrelated workbook content.
- `use_when`:
  - The task requires consistency across two or more sheets.
  - Observable evidence shows mismatched keys, totals or duplicated cross-sheet records.
- `avoid_when`:
  - The task is confined to one sheet with no cross-sheet invariant.
  - Fresh reconciliation evidence already covers the unchanged workbook.
- `expected_outputs`:
  - A reconciliation artifact or candidate workbook tied to the cross-sheet evidence.

#### Formatting & Presentation Specialist

- `role_goal`: Apply or repair visible formatting, conditional formatting, chart or presentation requirements without altering unrelated values.
- `use_when`:
  - The public contract contains formatting, chart or display requirements.
  - Validation exposes a presentation-specific failure.
- `avoid_when`:
  - The task contains no presentation requirement.
  - The current presentation candidate already has fresh passing evidence.
- `expected_outputs`:
  - A candidate workbook and presentation-change record tied to the public contract.

#### Workbook Repair Specialist

- `role_goal`: Revise the current workbook candidate using concrete test or validation evidence while preserving correct prior work.
- `use_when`:
  - A candidate has a concrete execution or validation failure.
  - New evidence supports a materially revised repair rather than repeating the same edit.
- `avoid_when`:
  - No failed candidate or concrete failure evidence exists.
  - The current candidate has not yet been tested where testing is available.
- `expected_outputs`:
  - A revised candidate workbook linked to the failure evidence.

#### Workbook Test Executor

- `role_goal`: Recalculate and execute the public workbook test contract against the current candidate without silently repairing it.
- `use_when`:
  - A candidate exists without fresh execution evidence.
  - The candidate or calculation backend changed after the latest execution.
- `avoid_when`:
  - No candidate workbook exists.
  - Fresh execution evidence already covers the unchanged candidate and runtime.
- `expected_outputs`:
  - A test execution record containing concrete workbook and calculation outcomes.

#### Workbook Integrity Validator

- `role_goal`: Verify required values, formulas, structure and presentation invariants on the current executed workbook.
- `use_when`:
  - A candidate has fresh execution evidence and terminal compliance remains unresolved.
  - A repaired candidate requires validation against the full public contract.
- `avoid_when`:
  - No current candidate or executable workbook artifact exists.
  - Fresh validation already covers the unchanged candidate.
- `expected_outputs`:
  - A validation report tied to the exact workbook candidate and public contract.

### 5.4 Executor bindings

| Role | Primary tools | Side effects |
|---|---|---|
| Workbook Surveyor | `inspect_workbook`, `read_artifact` | Read-only |
| Data Quality Profiler | `profile_workbook_data`, `read_artifact` | Read-only |
| Formula Dependency Analyst | `inspect_formulas`, `trace_dependencies` | Read-only |
| Table Reshaping Specialist | `transform_tables`, `read_artifact` | Candidate mutation |
| Formula Computation Specialist | `edit_formulas`, `read_artifact` | Candidate mutation |
| Cross-sheet Reconciler | `reconcile_sheets`, `read_artifact` | Candidate mutation or analysis |
| Formatting & Presentation Specialist | `edit_presentation`, `read_artifact` | Candidate mutation |
| Workbook Repair Specialist | `repair_workbook`, `read_artifact` | Candidate mutation |
| Workbook Test Executor | `run_workbook_tests` | Execution evidence |
| Workbook Integrity Validator | `validate_workbook` | Validation copy only |

## 6. Domain `scientific_procedure`

### 6.1 Nguồn và task contract

- Nguồn chính: ScienceAgentBench verified artifacts.
- 20 tasks được cân bằng theo procedure archetype và scientific modality.
- Loại khỏi selection ban đầu các source IDs có license ngoại lệ chưa được xác nhận.
- Public: instruction, domain knowledge được source cho phép, dataset và output schema.
- Private: reference/evaluator data và task-specific assertions.
- Final checker: task-specific evaluator chạy trên output copy; không thay bằng LLM judge khi evaluator gốc thực thi được.

### 6.2 Public artifacts

- `dataset_inventory`;
- `preparation_report`;
- `method_analysis`;
- `candidate_analysis`;
- `computation_result`;
- `assumption_audit`;
- `reproduction_result`;
- `diagnosis`;
- `conclusion_artifact`;
- `verification_report`.

### 6.3 Role cards

#### Scientific Data Inspector

- `role_goal`: Inspect scientific datasets, metadata, variables, units and file structure without modifying source data.
- `use_when`:
  - Dataset structure, units or available variables are not yet sufficiently understood.
  - New execution evidence indicates that an earlier understanding of the data is stale.
- `avoid_when`:
  - A fresh inventory covers the unchanged inputs and the relevant variables are explicit.
  - The remaining operation is a known computation or final verification.
- `expected_outputs`:
  - A scientific dataset inventory tied to public files and metadata.

#### Data Preparation Specialist

- `role_goal`: Produce a versioned prepared dataset through justified filtering, normalization, alignment or missing-value handling.
- `use_when`:
  - The public task or observed data requires preparation before analysis.
  - A computation failure identifies a concrete preparation issue.
- `avoid_when`:
  - The provided data is already suitable for the required method.
  - An unchanged prepared dataset is awaiting computation or validation.
- `expected_outputs`:
  - A prepared dataset and preparation report preserving unrelated observations.

#### Statistical Method Analyst

- `role_goal`: Analyze statistical assumptions and select or compare methods grounded in the task and observable data.
- `use_when`:
  - The task leaves method selection or statistical assumptions unresolved.
  - Results conflict with assumptions or require a materially different method.
- `avoid_when`:
  - The public contract fixes the applicable method and its assumptions are already established.
  - The current method only needs execution.
- `expected_outputs`:
  - A method analysis with explicit assumptions and measurable outputs.

#### Spatial & Signal Analyst

- `role_goal`: Analyze geospatial, temporal or physiological signal data using the domain operations available in the packet.
- `use_when`:
  - The task depends on spatial relationships, time series or signal processing.
  - Observable feedback identifies a spatial, temporal or signal-specific issue.
- `avoid_when`:
  - The task contains no supported spatial, temporal or signal artifact.
  - The unchanged specialist result only needs independent validation.
- `expected_outputs`:
  - A versioned specialist analysis and derived artifacts tied to source data.

#### Molecular & Omics Analyst

- `role_goal`: Analyze molecular, materials or omics artifacts using the pinned scientific tools and public task contract.
- `use_when`:
  - The task contains supported molecular structures, materials data or omics matrices.
  - Domain-specific evidence must be recomputed after a data or method change.
- `avoid_when`:
  - No supported molecular, materials or omics artifact is present.
  - Fresh specialist evidence already covers unchanged inputs and method.
- `expected_outputs`:
  - A domain-specific analysis artifact with reproducible parameters and outputs.

#### Scientific Computation Executor

- `role_goal`: Execute the current scientific analysis candidate in the pinned runtime and record complete observable results.
- `use_when`:
  - A prepared input and executable analysis candidate exist without fresh results.
  - The candidate, input or runtime parameters changed after the latest run.
- `avoid_when`:
  - No executable analysis candidate exists.
  - Fresh execution evidence already covers the unchanged candidate and inputs.
- `expected_outputs`:
  - A computation record, output files and exact observable diagnostics.

#### Assumption and Unit Auditor

- `role_goal`: Audit units, leakage, assumptions, sample alignment and methodological invariants using public evidence.
- `use_when`:
  - The result depends on assumptions or units that have not been checked.
  - An implausible or conflicting result suggests an assumption violation.
- `avoid_when`:
  - The task exposes no applicable assumption or unit requirement.
  - A fresh audit already covers unchanged data, method and result.
- `expected_outputs`:
  - An assumption and unit audit tied to the current analysis version.

#### Result Reproduction Operator

- `role_goal`: Reproduce a specified scientific result or workflow using the pinned source artifacts and runtime.
- `use_when`:
  - The task explicitly requires reproduction of a target result or procedure.
  - A previous reproduction attempt changed or produced concrete failure evidence.
- `avoid_when`:
  - The task is not a reproduction task and no target procedure is supplied.
  - A fresh reproduction result already covers the unchanged artifacts.
- `expected_outputs`:
  - A reproduction execution record and generated scientific artifacts.

#### Scientific Failure Diagnostician

- `role_goal`: Explain a concrete preparation, computation or methodological failure without silently changing the analysis.
- `use_when`:
  - Fresh execution, audit or evaluator evidence exposes a failure whose mechanism is unresolved.
  - Repeated attempts produce no substantive progress and new diagnosis is justified.
- `avoid_when`:
  - No concrete failure or conflicting evidence is visible.
  - The current candidate only requires execution or verification.
- `expected_outputs`:
  - An evidence-grounded diagnosis and repair constraints.

#### Conclusion Verifier

- `role_goal`: Verify that the current scientific conclusion is supported by fresh results, uncertainty and the public task contract.
- `use_when`:
  - Current computation results exist and a conclusion or terminal claim requires checking.
  - Data, method or result changes made prior conclusion evidence stale.
- `avoid_when`:
  - No current result or conclusion artifact exists.
  - Fresh verification already covers the unchanged conclusion and evidence.
- `expected_outputs`:
  - A conclusion verification report linked to exact input and result versions.

### 6.4 Executor bindings

| Role | Primary tools | Side effects |
|---|---|---|
| Scientific Data Inspector | `inspect_scientific_data`, `read_artifact` | Read-only |
| Data Preparation Specialist | `prepare_scientific_data` | Candidate data mutation |
| Statistical Method Analyst | `analyze_statistical_method`, `read_artifact` | Analysis artifact |
| Spatial & Signal Analyst | `analyze_spatial_signal` | Analysis artifacts |
| Molecular & Omics Analyst | `analyze_molecular_omics` | Analysis artifacts |
| Scientific Computation Executor | `run_scientific_computation` | Execution outputs |
| Assumption and Unit Auditor | `audit_scientific_assumptions` | Audit artifact |
| Result Reproduction Operator | `reproduce_scientific_result` | Execution outputs |
| Scientific Failure Diagnostician | `diagnose_scientific_failure`, `read_artifact` | Diagnosis artifact |
| Conclusion Verifier | `verify_scientific_conclusion` | Validation copy only |

## 7. Domain `cross_app_workflow`

### 7.1 Nguồn và task contract

- Nguồn chính: AppWorld.
- Chọn 20 task liên quan ít nhất hai app, có dependency giữa operations và state-based evaluator.
- Release lưu task IDs, upstream revision, hashes và encrypted/protected material theo license; không phát hành protected bundle ở dạng plaintext.
- Public state chỉ chứa information mà app APIs hoặc task contract cho phép quan sát.
- Final checker dùng state-based tests và collateral-damage assertions.

### 7.2 Public artifacts

- `app_state_inventory`;
- `entity_resolution`;
- `policy_audit`;
- `query_result`;
- `mapping_plan`;
- `mutation_result`;
- `batch_result`;
- `communication_result`;
- `consistency_report`;
- `compensation_result`.

### 7.3 Role cards

#### App State Inspector

- `role_goal`: Inspect authorized application state and relevant records without performing state-changing operations.
- `use_when`:
  - Relevant app records or current cross-app state are not yet sufficiently understood.
  - A mutation or external result may have changed previously observed state.
- `avoid_when`:
  - Fresh state evidence already covers unchanged records needed by the task.
  - The remaining operation is an explicit known mutation or validation.
- `expected_outputs`:
  - An authorized app-state inventory tied to observable records.

#### Entity Resolution Analyst

- `role_goal`: Resolve identities and record correspondences across applications using observable attributes.
- `use_when`:
  - The task refers to an entity represented differently across apps.
  - A query or mutation exposes ambiguous, duplicated or conflicting matches.
- `avoid_when`:
  - Entity identifiers are already explicit and consistent across relevant apps.
  - No cross-app entity correspondence is required.
- `expected_outputs`:
  - An entity-resolution artifact with evidence for each correspondence.

#### Policy & Permission Auditor

- `role_goal`: Audit public authorization, consent and operation constraints before or after consequential app actions.
- `use_when`:
  - The task contains explicit authorization, consent or policy conditions.
  - An app response indicates a permission or policy conflict.
- `avoid_when`:
  - No applicable policy condition is visible.
  - A fresh audit already covers unchanged identities, permissions and requested operations.
- `expected_outputs`:
  - A policy and permission report grounded in public task and app evidence.

#### Query & Filter Specialist

- `role_goal`: Retrieve and filter authorized records across app APIs without mutating them.
- `use_when`:
  - The target records must be identified from observable criteria.
  - A previous query was incomplete, overbroad or contradicted by new state.
- `avoid_when`:
  - Exact target record identifiers are already established and fresh.
  - The remaining operation is mutation, communication or validation.
- `expected_outputs`:
  - A query result with target records, criteria and observable evidence.

#### Cross-app Data Mapper

- `role_goal`: Map fields, identifiers and values between app schemas without performing the destination mutation.
- `use_when`:
  - Information must be transferred between apps with different schemas.
  - A previous mutation failed due to field, type or identifier mismatch.
- `avoid_when`:
  - The source and destination schemas align directly and the mapped payload is explicit.
  - No cross-app data transfer is required.
- `expected_outputs`:
  - A mapped payload and field correspondence tied to source records.

#### Transactional Action Executor

- `role_goal`: Perform an authorized state-changing app operation and record all resulting side effects without claiming final completion.
- `use_when`:
  - The target, payload and authorization needed for a mutation are observable.
  - A materially revised mutation is justified by concrete app feedback.
- `avoid_when`:
  - The target or required authorization remains ambiguous.
  - The same mutation would be repeated without new evidence.
- `expected_outputs`:
  - A mutation result containing affected records, statuses and observable side effects.

#### Batch Operation Executor

- `role_goal`: Perform a bounded authorized operation across a selected set of records and report per-record outcomes.
- `use_when`:
  - The task explicitly requires the same operation on multiple identified records.
  - A previous batch partially succeeded and remaining records are observable.
- `avoid_when`:
  - The target set is unresolved or the task concerns only one record.
  - Repeating the batch could duplicate already successful side effects.
- `expected_outputs`:
  - A batch execution report with per-record success, failure and side effects.

#### Communication Coordinator

- `role_goal`: Create task-required messages, notifications or calendar communications using verified public state.
- `use_when`:
  - The task explicitly requires communication or notification.
  - An observable state change makes a prior draft or recipient set stale.
- `avoid_when`:
  - No communication requirement exists.
  - Recipients, content or authorization are unresolved.
- `expected_outputs`:
  - A communication result linked to recipients, content and supporting state.

#### Cross-app Consistency Validator

- `role_goal`: Verify required relationships and final-state invariants across applications without silently repairing them.
- `use_when`:
  - One or more app mutations have occurred and cross-app consistency remains unresolved.
  - A compensation or revised action changed state after the latest validation.
- `avoid_when`:
  - No relevant state-changing action has occurred and no consistency claim is possible.
  - Fresh validation covers the unchanged app state.
- `expected_outputs`:
  - A cross-app consistency report tied to exact observable state versions.

#### Compensation & Recovery Operator

- `role_goal`: Perform an explicit compensating action for observable partial, duplicated or incorrect cross-app state.
- `use_when`:
  - A prior action left a concrete inconsistent or partial state that the public contract permits correcting.
  - Validation identifies specific recoverable collateral damage.
- `avoid_when`:
  - No state change requires compensation.
  - The proposed compensation target or authorization is unresolved.
- `expected_outputs`:
  - A compensation result recording affected state and remaining inconsistencies.

### 7.4 Executor bindings

| Role | Primary tools | Side effects |
|---|---|---|
| App State Inspector | `inspect_app_state` | Read-only |
| Entity Resolution Analyst | `resolve_cross_app_entities` | Analysis artifact |
| Policy & Permission Auditor | `audit_app_policy` | Audit artifact |
| Query & Filter Specialist | `query_app_records` | Read-only |
| Cross-app Data Mapper | `map_cross_app_fields` | Mapping artifact |
| Transactional Action Executor | `execute_app_transaction` | External state mutation |
| Batch Operation Executor | `execute_app_batch` | External state mutation |
| Communication Coordinator | `send_app_communication` | External state mutation |
| Cross-app Consistency Validator | `validate_cross_app_state` | Read-only evaluation |
| Compensation & Recovery Operator | `execute_compensation` | External state mutation |

## 8. Domain `multi_hop_knowledge`

### 8.1 Nguồn và task contract

- Nguồn chính: MuSiQue.
- 20 tasks; đề xuất 16 answerable và 4 insufficient/unanswerable từ cùng source family để STOP/insufficiency không bị đồng nhất với failure.
- Corpus/paragraphs được đóng băng; runtime không dùng web mở.
- Public: question, frozen searchable corpus và retrieved evidence artifacts.
- Private: answer aliases, supporting paragraph IDs và evaluator labels.
- Final checker báo answer score và evidence-support score riêng.

### 8.2 Public artifacts

- `query_analysis`;
- `broad_retrieval`;
- `targeted_retrieval`;
- `entity_resolution`;
- `reasoning_artifact`;
- `entailment_report`;
- `conflict_report`;
- `candidate_answer`;
- `citation_audit`.

### 8.3 Role cards

#### Query Structure Analyst

- `role_goal`: Analyze entities, relations and information dependencies in the question without retrieving or answering it.
- `use_when`:
  - The information dependencies or referents in the question are not sufficiently clear.
  - Retrieval results expose a materially different interpretation of the query.
- `avoid_when`:
  - The required entities and relations are already explicit and fresh.
  - The unresolved operation is evidence retrieval, verification or answer synthesis.
- `expected_outputs`:
  - A query-structure artifact identifying observable dependencies and ambiguities.

#### Broad Evidence Retriever

- `role_goal`: Retrieve a high-recall set of candidate passages from the frozen corpus without asserting that they support the answer.
- `use_when`:
  - Relevant evidence has not yet been located.
  - Existing evidence is too narrow or an observed conflict requires broader search.
- `avoid_when`:
  - The necessary evidence set is already fresh and sufficiently scoped.
  - Repeating the same query would not add new evidence.
- `expected_outputs`:
  - Ranked candidate passages with source identifiers and query terms.

#### Targeted Evidence Retriever

- `role_goal`: Retrieve evidence for a specific unresolved entity, relation or sub-question from the frozen corpus.
- `use_when`:
  - A concrete evidence gap or sub-question is observable.
  - Broad retrieval identified an entity or relation requiring focused lookup.
- `avoid_when`:
  - No specific evidence gap is identified.
  - Fresh targeted evidence already covers the same query and corpus version.
- `expected_outputs`:
  - Focused passages tied to the stated evidence gap.

#### Entity & Coreference Resolver

- `role_goal`: Resolve aliases, entity identity and coreference across retrieved passages.
- `use_when`:
  - Retrieved passages contain ambiguous names, pronouns or competing entity matches.
  - A reasoning conflict may result from unresolved entity identity.
- `avoid_when`:
  - Entity references are explicit and consistent across current evidence.
  - No retrieved evidence is available to resolve.
- `expected_outputs`:
  - An entity-resolution artifact grounded in cited passages.

#### Temporal & Comparison Reasoner

- `role_goal`: Derive time, ordering or comparison relationships from observable evidence without inventing missing facts.
- `use_when`:
  - The question requires temporal ordering, numerical comparison or comparison across entities.
  - New evidence changes a prior comparison or time relation.
- `avoid_when`:
  - The question contains no temporal or comparison requirement.
  - Required evidence is absent and no grounded relation can be derived.
- `expected_outputs`:
  - A comparison or temporal reasoning artifact with evidence references.

#### Bridge Reasoner

- `role_goal`: Connect observable facts across multiple passages to derive an intermediate or final relation.
- `use_when`:
  - The question requires linking two or more evidence hops.
  - Retrieved evidence contains the needed facts but their bridge is unresolved.
- `avoid_when`:
  - A direct passage already answers the question without a bridge.
  - Required intermediate evidence is missing.
- `expected_outputs`:
  - A stepwise bridge reasoning artifact referencing each supporting passage.

#### Evidence Entailment Verifier

- `role_goal`: Verify whether current passages entail a stated intermediate claim or candidate answer.
- `use_when`:
  - A reasoning artifact or candidate answer exists without fresh support verification.
  - New evidence makes previous entailment evidence stale or conflicting.
- `avoid_when`:
  - No claim or candidate answer is available to verify.
  - Fresh entailment evidence already covers the unchanged claim and passages.
- `expected_outputs`:
  - An entailment report distinguishing supported, contradicted and unsupported claims.

#### Conflict & Insufficiency Analyst

- `role_goal`: Identify conflicting evidence, unresolved gaps or insufficient support without fabricating an answer.
- `use_when`:
  - Retrieved passages conflict or fail to support a required reasoning step.
  - The task may be unanswerable from the frozen corpus.
- `avoid_when`:
  - Current evidence is consistent and sufficient for all required claims.
  - No retrieval or reasoning evidence exists to analyze.
- `expected_outputs`:
  - A conflict or insufficiency report with exact missing or contradictory evidence.

#### Answer Synthesizer

- `role_goal`: Produce a concise candidate answer grounded only in observable reasoning and evidence artifacts.
- `use_when`:
  - Current evidence supports an answer or a justified insufficiency conclusion.
  - A previous candidate is contradicted and new evidence supports a materially revised answer.
- `avoid_when`:
  - Required evidence remains unresolved and no justified insufficiency conclusion exists.
  - The unchanged candidate only requires support audit.
- `expected_outputs`:
  - A candidate answer with explicit evidence references, not a terminal validation claim.

#### Citation and Support Auditor

- `role_goal`: Audit whether the candidate answer cites the correct passages and whether each material claim is supported.
- `use_when`:
  - A candidate answer exists and citation/support compliance is unresolved.
  - The candidate answer or supporting evidence changed after the latest audit.
- `avoid_when`:
  - No candidate answer exists.
  - Fresh audit evidence covers the unchanged answer and evidence set.
- `expected_outputs`:
  - A citation and support audit linked to the exact answer and passage versions.

### 8.4 Executor bindings

| Role | Primary tools | Side effects |
|---|---|---|
| Query Structure Analyst | `analyze_query_structure` | Analysis artifact |
| Broad Evidence Retriever | `retrieve_broad_evidence` | Evidence artifact |
| Targeted Evidence Retriever | `retrieve_targeted_evidence` | Evidence artifact |
| Entity & Coreference Resolver | `resolve_evidence_entities` | Analysis artifact |
| Temporal & Comparison Reasoner | `reason_temporal_comparison` | Reasoning artifact |
| Bridge Reasoner | `reason_evidence_bridge` | Reasoning artifact |
| Evidence Entailment Verifier | `verify_evidence_entailment` | Verification artifact |
| Conflict & Insufficiency Analyst | `analyze_evidence_conflict` | Diagnosis artifact |
| Answer Synthesizer | `write_candidate_answer` | Candidate mutation |
| Citation and Support Auditor | `audit_answer_support` | Audit artifact |

## 9. Domain `document_media_processing`

### 9.1 Nguồn và task contract

- Nguồn chính: Omni Parse Bench.
- 20 pages/tasks cân bằng theo năm document archetype.
- Mỗi sample phải có tests thuộc ít nhất hai family và license/provenance riêng.
- Public: page/image/PDF artifact, output format và allowed processing tools.
- Private: unit tests và source-specific evaluation metadata.
- Final checker chạy unit tests trên output copy; model inference không được gọi trong grading.

### 9.2 Public artifacts

- `document_inventory`;
- `ocr_result`;
- `layout_analysis`;
- `table_result`;
- `formula_result`;
- `key_value_result`;
- `language_analysis`;
- `candidate_document_output`;
- `region_repair_result`;
- `document_validation`.

### 9.3 Role cards

#### Document Surveyor

- `role_goal`: Inspect pages, visible regions, languages and structural features without producing the final parsed document.
- `use_when`:
  - Document structure, page regions or processing requirements are not yet sufficiently understood.
  - New processing evidence indicates that the earlier survey is stale.
- `avoid_when`:
  - A fresh survey covers the unchanged document and relevant regions are explicit.
  - The remaining operation is a known extraction, composition or validation.
- `expected_outputs`:
  - A document inventory with pages, regions and observable feature types.

#### OCR Specialist

- `role_goal`: Extract text from selected or full-page image regions while preserving observable location and confidence evidence.
- `use_when`:
  - Required text is unavailable from the current document representation.
  - Validation identifies missing, duplicated or incorrect text regions.
- `avoid_when`:
  - Reliable text extraction already exists for unchanged target regions.
  - The unresolved requirement concerns only tables, formulas or layout structure.
- `expected_outputs`:
  - An OCR artifact with text, regions and observable confidence or diagnostic data.

#### Layout & Reading-order Analyst

- `role_goal`: Identify document blocks, block types and reading order without rewriting recognized content.
- `use_when`:
  - The task requires layout structure or reading order that is not established.
  - Validation exposes block classification or ordering errors.
- `avoid_when`:
  - The document is single-region and no layout or order requirement exists.
  - Fresh layout analysis already covers unchanged regions.
- `expected_outputs`:
  - A layout and reading-order artifact tied to page regions.

#### Table Recognition Specialist

- `role_goal`: Extract table boundaries, headers, rows, columns and cell relationships from relevant regions.
- `use_when`:
  - The document contains tables required by the public output contract.
  - Unit-test evidence identifies a table structure or cell-association failure.
- `avoid_when`:
  - No relevant table region is present.
  - Fresh table extraction already covers unchanged table regions.
- `expected_outputs`:
  - A structured table artifact with region and cell provenance.

#### Formula Recognition Specialist

- `role_goal`: Extract displayed or inline mathematical expressions into the required structured representation.
- `use_when`:
  - The document contains formulas required by the output contract.
  - Validation identifies a formula-recognition or ordering error.
- `avoid_when`:
  - No relevant mathematical expression is present.
  - Fresh formula extraction already covers unchanged regions.
- `expected_outputs`:
  - A formula artifact with source regions and normalized representation.

#### Form & Key-value Specialist

- `role_goal`: Extract labels, values and key-value associations from forms and semi-structured document regions.
- `use_when`:
  - The task requires form fields or key-value relationships.
  - Validation indicates a label/value association error.
- `avoid_when`:
  - No form or key-value structure is present.
  - Fresh extraction already covers unchanged target regions.
- `expected_outputs`:
  - A key-value artifact with region and association evidence.

#### Language & Script Specialist

- `role_goal`: Analyze multilingual or script-specific regions and produce normalized text evidence without changing unrelated content.
- `use_when`:
  - The document contains multiple languages or a script requiring specialized processing.
  - OCR evidence indicates a language or script-specific failure.
- `avoid_when`:
  - Existing extraction is correct and no language-specific issue is visible.
  - The document contains only supported plain text with fresh extraction.
- `expected_outputs`:
  - A language/script analysis and corrected regional extraction where justified.

#### Structured Output Composer

- `role_goal`: Compose current text, layout, table, formula and key-value artifacts into the required Markdown or JSON candidate.
- `use_when`:
  - Sufficient component artifacts exist to create or revise the structured output.
  - New component evidence makes the current composed output stale.
- `avoid_when`:
  - Essential component evidence is absent and no partial output is useful.
  - The unchanged composed candidate only needs validation.
- `expected_outputs`:
  - A versioned structured document candidate referencing component artifacts.

#### Document Region Repair Specialist

- `role_goal`: Reprocess specific failed regions using concrete validation evidence without replacing correct unrelated output.
- `use_when`:
  - Validation identifies specific missing, malformed or incorrectly ordered regions.
  - New evidence supports a materially revised regional repair.
- `avoid_when`:
  - No candidate or region-specific failure evidence exists.
  - The same regional approach would be repeated without new evidence.
- `expected_outputs`:
  - A repaired candidate and region-level repair record tied to failure evidence.

#### Document Quality Validator

- `role_goal`: Run the public document checks against the current structured output without silently modifying it.
- `use_when`:
  - A candidate output exists without fresh validation.
  - Component or composed output changed after the latest validation.
- `avoid_when`:
  - No candidate output exists.
  - Fresh validation already covers the unchanged candidate and tests.
- `expected_outputs`:
  - A document validation report with per-check evidence and candidate identity.

### 9.4 Executor bindings

| Role | Primary tools | Side effects |
|---|---|---|
| Document Surveyor | `survey_document`, `read_artifact` | Read-only |
| OCR Specialist | `run_document_ocr` | Component artifact |
| Layout & Reading-order Analyst | `analyze_document_layout` | Component artifact |
| Table Recognition Specialist | `extract_document_tables` | Component artifact |
| Formula Recognition Specialist | `extract_document_formulas` | Component artifact |
| Form & Key-value Specialist | `extract_document_key_values` | Component artifact |
| Language & Script Specialist | `process_document_language` | Component artifact |
| Structured Output Composer | `compose_document_output` | Candidate mutation |
| Document Region Repair Specialist | `repair_document_regions` | Candidate mutation |
| Document Quality Validator | `validate_document_output` | Validation copy only |

## 10. Tool result contract chung

Mọi domain nên chuẩn hóa tool result ở mức tối thiểu:

```json
{
  "tool_name": "...",
  "status": "success | partial | no_progress | failure | environment_error",
  "output": {},
  "observable_message": "...",
  "input_versions": {},
  "versions_after": {},
  "artifact_refs": [],
  "side_effects": [],
  "cost": 1.0
}
```

- `success` chỉ có nghĩa operation hoàn tất theo tool contract, không đồng nghĩa task pass.
- `partial` giữ mọi side effect và chỉ rõ phần đã/ chưa hoàn thành.
- `no_progress` là outcome benchmark hợp lệ, không phải infrastructure failure.
- `failure` là task/tool failure có evidence.
- `environment_error` dùng cho lỗi runtime/service làm episode có thể không đo được.
- Không xóa side effect khi operation fail.
- Không biến output rỗng thành success.

## 11. Hard preconditions tối thiểu

Hard precondition được phép:

- artifact vật lý bắt buộc không tồn tại;
- operation vượt public read/write scope;
- permission hoặc authorization chưa thỏa;
- tool không hỗ trợ loại artifact hiện tại;
- candidate identity/version không tồn tại cho operation bắt buộc phải gắn với candidate;
- operation có nguy cơ vi phạm sandbox hoặc external-state contract.

Không dùng hard precondition cho:

- role ordering;
- “planner chưa chạy”;
- “inspector chưa chạy” khi role vẫn có thể đọc task/artifact;
- role không phải lựa chọn tối ưu;
- role được `avoid_when` mô tả;
- thiếu một artifact phân tích nhưng role có thể tự quan sát hoặc trả no-progress an toàn.

Precondition rejection vẫn tiêu thụ call giống thiết kế đã đóng băng và được báo riêng.

## 12. Chống subgraph/workflow cố định

### 12.1 Design-time audit

Trước live run, kiểm tra trên 20 task/domain:

- có ít nhất năm tập initial useful capabilities khác nhau;
- không role nào được hard-code là predecessor/successor;
- không task contract yêu cầu tên role;
- không có workflow stage trong public state;
- không role card nhắc tên role khác;
- không role card chứa reference solution hoặc gold artifact;
- có task có thể bỏ qua inspection;
- có task có thể đi trực tiếp tới specialist;
- có task cần recovery và task không cần recovery;
- có task mà early validation hữu ích và task mà early validation chỉ tạo no-progress;
- private design review xác nhận nhiều hơn một strategy khả thi ở mỗi archetype, nhưng strategy không được xuất cho router/evaluator.

Các mục này là capability audit, không phải runtime route constraint.

### 12.2 Post-run diversity report

Sau live trajectory chỉ báo cáo, không dùng để sửa ngược run:

- unique role sequences;
- unique executed-role sets;
- initial-action distribution;
- role and transition coverage;
- path-length distribution;
- repeated-action and loop rate;
- validation-before-mutation rate;
- recovery paths sau observed failure;
- sequence concentration theo domain và archetype.

Không đặt ngưỡng pass rate hoặc sequence entropy làm điều kiện benchmark validity. Nếu router chọn cùng một đường dù environment cho phép nhiều đường, đó là hành vi được đo.

## 13. Human review

Mỗi observed decision state được review từ:

- task và public state;
- role cards canonical;
- public history/evidence;
- budget và full action pool.

Reviewer không xem lựa chọn hiện tại trước khi gán nhãn. `valid_actions` là set-valued và có thể chứa nhiều role. Budget-suppressed decision được giữ trong trace nhưng loại khỏi routing denominator chính. Human labels không được suy từ reference workflow, role-card `use_when`, precondition code hoặc task success.

## 14. Acceptance checklist trước implementation freeze

### Dataset

- [ ] 20 source IDs/domain được chốt trước model run.
- [ ] 5 archetype x 4 sample/domain.
- [ ] Source revision, checksum, license và attribution đầy đủ.
- [ ] Public/private/fixtures boundary được validate.
- [ ] Reference output/evaluator chạy được 20/20.

### Role cards

- [ ] Mỗi role có đúng năm required public fields.
- [ ] Không có `weakness` hoặc `limitations`.
- [ ] `use_when`/`avoid_when` dựa trên observable evidence, không dựa trên role order.
- [ ] Expected output là artifact/evidence, không phải success claim.
- [ ] Không nhắc tên next/previous role.

### Executor bindings

- [ ] Mọi advertised capability có implementation thật.
- [ ] Tool permissions và side effects được khai báo riêng.
- [ ] Tool result giữ partial/failure side effects.
- [ ] Candidate/evidence freshness dùng version/hash.
- [ ] Independent final checker chạy trên copy hoặc isolated state.

### Routing protocol

- [ ] Full pool + STOP ở mọi decision.
- [ ] Guard/action masking tắt.
- [ ] Không auto-route, auto-fix, auto-retry task failure hoặc auto-rollback.
- [ ] Budget-zero flow và suppressed decision giữ như ba domain đã đóng băng.
- [ ] Human labels pending cho tới khi review thật hoàn tất.

## 15. Thứ tự triển khai đề xuất

1. Materialize và preflight 100 task packets.
2. Tạo canonical role-card files và executor-binding files theo tài liệu này.
3. Implement adapter + tool contracts cho từng domain, tái sử dụng shared runner/actor/state versioning.
4. Viết independent final checker trước live model call.
5. Chạy contract tests, leakage tests, side-effect tests và anti-masking tests.
6. Chạy smoke bằng deterministic/fake router để kiểm tra environment, không tối ưu route.
7. Đóng băng source/config/packets.
8. Chạy live trajectories.
9. Validate trace, export human review và báo cáo diversity; không sửa trajectory lịch sử.

Thiết kế này giữ chuẩn public role card của database/formal/code, giữ các implementation đã đóng băng, đồng thời cho năm domain mới agent pool đa dạng mà không mã hóa một workflow chung vào harness.
