# RoutingBench V2 — Thiết kế hoàn chỉnh cho bản phát hành Hugging Face

## 1. Mục tiêu của V2

RoutingBench V2 là bản benchmark công khai ở quy mô vừa, dùng để kiểm tra xem bài toán đánh giá dynamic orchestration có:

- tách được routing error khỏi lỗi reasoning của LLM actor;
- đo được cả quyết định cục bộ và lỗi tích lũy theo trajectory;
- phân biệt router tốt với random/fixed routing;
- vẫn có ý nghĩa khi pool chứa role nhiễu hoặc role có chức năng chồng lấn;
- có schema, documentation và evaluator đủ rõ để người khác sử dụng từ Hugging Face.

V2 là **public research preview**, chưa phải benchmark dùng để đưa ra kết luận cấp hội nghị. Toàn bộ label và simulator specification có thể công khai. V3/V4 mới cần hidden test set, evaluation server, quy mô lớn và statistical validation mạnh.

## 2. Vì sao `49/54 state chỉ có một action hợp lệ` là hạn chế

Trong V1, router luôn nhận toàn bộ agent pool, nhưng oracle transition graph xác định action nào có thể tạo một state transition hợp lệ.

Ví dụ:

```text
Observed state:
  evidence_hop1 = available
  needs_bridge = true

Candidate actions:
  Task Decomposer
  Evidence Retriever
  Bridge Reasoner
  Evidence Verifier
  Answer Integrator
  ...

Hidden valid-action set:
  {Bridge Reasoner}
```

Router vẫn phải chọn giữa nhiều role, nhưng chỉ có một đáp án được chấp nhận. Khi tên state và role khớp trực tiếp như `needs_bridge → Bridge Reasoner`, bài toán gần với role classification hơn là lựa chọn chiến lược.

Trong 54 reachable states của V1:

- 49 state có valid-action set kích thước 1;
- 5 state có valid-action set kích thước 2;
- không có đủ tình huống nhiều action đều hợp lệ nhưng khác nhau về cost, risk hoặc độ dài đường đi;
- Track A không chứa nhiều state sinh ra sau invalid/no-progress action.

V2 phải tăng ambiguity có kiểm soát mà không biến ground truth thành ý kiến chủ quan.

## 3. Phạm vi và quy mô đề xuất

### 3.1 Quy mô tổng

V2 gồm:

- **120 interactive workflows** cho Track B;
- khoảng **900–1.000 decision states** cho Track A;
- **8 workflow families**, mỗi family 15 workflow;
- một catalog khoảng **24 qualitative agent profiles**;
- pool từ 8–18 agent trong split thường và 16–24 agent trong stress split;
- task text hoàn toàn synthetic/original, không sao chép câu hỏi từ MuSiQue, GAIA hoặc dataset có license không tương thích.

Quy mô này đủ để:

- có khoảng sai số tổng thể nhỏ hơn đáng kể so với V1;
- hiển thị và tải thuận tiện trên Hugging Face;
- chạy Jev/Clef với chi phí còn kiểm soát được;
- phát hiện vấn đề schema, metric và evaluator trước khi mở rộng lên V3.

Nó chưa đủ để tuyên bố state-of-the-art hoặc so sánh rất sát hai router mạnh.

### 3.2 Workflow families

| Family | Năng lực routing cần kiểm tra | Ví dụ cấu trúc |
|---|---|---|
| Sequential dependency | Theo đúng dependency 2–4 bước | retrieve → bridge → compose |
| Parallel join | Chọn thứ tự linh hoạt và hội tụ hai nhánh | source A + source B → join |
| Conditional branch | Chọn nhánh dựa trên artifact/state | image → vision; table → table tool |
| Conflict resolution | Chỉ gọi verifier khi evidence/candidate xung đột | conflict → verify → finalize |
| Tool/artifact routing | Chọn đúng tác tử theo loại artifact | file/audio/web/database |
| Failure recovery | Đổi chiến lược sau failure/no-progress | failed query → replan → retry |
| Terminal control | Phân biệt candidate, verified candidate và committed answer | integrate/verify/STOP |
| Budget-sensitive routing | Chọn đường hợp lệ theo budget còn lại | direct safe path hoặc longer verification |

Mỗi family có nhiều topology, task surface form và agent-pool perturbation khác nhau.

## 4. Hai track

### 4.1 Track A — Isolated state routing

Input:

```text
task
observed_state
recent_history
remaining_budget
qualitative_agent_profiles
candidate_actions
```

Output:

```text
one agent_id | STOP
```

Mỗi row có hai lớp ground truth:

- `valid_actions`: mọi action thỏa contract và có thể tiếp tục workflow;
- `efficient_actions`: tập con Pareto-optimal của valid actions theo remaining steps và remaining cost.

Core score dùng `valid_actions`. `efficient_actions` chỉ dùng cho efficiency analysis, không biến thành capability score.

### 4.2 Track B — Interactive deterministic trajectory

Router bắt đầu ở initial state và lặp:

```text
observation
  → router chọn agent
  → deterministic actor contract thực thi
  → environment cập nhật state/status/budget
  → observation mới
```

Track B không gọi LLM actor. Nếu dùng Jev/Clef thì chỉ decision router được gọi thật. Environment và actor outputs là deterministic.

Episode kết thúc khi:

- router chọn `STOP` sau khi goal hoàn thành;
- router chọn `STOP` quá sớm;
- hết step budget;
- hết cost budget;
- rơi vào unrecoverable terminal failure.

## 5. Phân bố độ khó bắt buộc

V2 đặt release constraints trên toàn bộ Track A:

| Loại state | Tỷ lệ mục tiêu |
|---|---:|
| Chỉ một valid action | 35–45% |
| Ít nhất hai valid action tương đương | 20–25% |
| Nhiều valid action nhưng chỉ một/tập con efficient | 20–25% |
| STOP-ready state | 10–15% |

Các constraint bổ sung:

- ít nhất 20% rows thuộc counterfactual groups;
- ít nhất 15% rows là post-failure/post-no-progress states;
- ít nhất 25% rows dùng semantically overlapping decoys;
- không workflow family nào chiếm quá 15% toàn bộ rows;
- không role nào là valid action trong quá 30% nonterminal rows;
- không quá 55% state có thể giải chỉ bằng exact lexical match giữa state key và role name.

## 6. Counterfactual và robustness design

### 6.1 Counterfactual state pairs

Giữ nguyên task và agent pool, chỉ thay đổi một yếu tố state khiến next action đổi.

Ví dụ:

```text
Pair A: evidence source chưa được đọc  → Evidence Retriever
Pair B: evidence đã đủ, còn xung đột   → Evidence Verifier
```

Hai rows phải nằm cùng split và có chung `counterfactual_group_id` để tránh leakage.

Metric yêu cầu router đúng trên **cả hai** rows, không chỉ một row.

### 6.2 History/status counterfactuals

State fingerprint V2 không chỉ hash facts. Nó gồm:

```text
facts
artifact inventory
last action
last status
failure counters
remaining budget bucket
```

Điều này cho phép hai state có cùng facts nhưng khác `last_status` cần route khác nhau, ví dụ:

```text
same facts + SUCCESS           → tiếp tục workflow
same facts + RETRYABLE_FAILURE → Recovery Planner
same facts + NO_PROGRESS       → đổi strategy/tool
```

### 6.3 Pool perturbations

Mỗi workflow template có thể sinh bốn biến thể:

- `base`: pool chức năng tối thiểu;
- `irrelevant_noise`: thêm role khác domain;
- `adjacent_noise`: thêm role có mô tả gần với role đúng;
- `permuted_identity`: shuffle order và thay opaque agent IDs.

Không public trường `distractor=true` cho router. Trường này chỉ nằm trong evaluator labels.

### 6.4 Surface-form perturbations

State và profile có tối đa ba cách diễn đạt:

- canonical structured wording;
- semantic paraphrase;
- compact operational wording.

Tất cả variants của cùng hidden state nằm chung split. Không dùng variant của cùng state ở cả validation và test.

## 7. Agent profile design

Public profile chỉ chứa contract định tính:

```json
{
  "agent_id": "agent_07",
  "role_name": "Evidence Resolver",
  "role_goal": "Resolve one explicit evidence dependency from available sources.",
  "use_when": ["a concrete evidence dependency is unresolved"],
  "avoid_when": ["the remaining issue is conflict resolution or finalization"],
  "supported_inputs": ["text_corpus", "retrieval_target"],
  "expected_outputs": ["grounded_claim", "source_reference"],
  "cost_class": "standard"
}
```

Không có:

- backbone/model identity;
- capability score;
- empirical accuracy;
- uncertainty;
- role ranking;
- test-time success history.

`cost_class` hoặc deterministic action cost chỉ biểu diễn resource budget, không biểu diễn chất lượng agent.

Catalog dự kiến có khoảng 24 profiles thuộc các nhóm:

- planning/decomposition;
- corpus/web/database retrieval;
- media/file/table inspection;
- bridge/composition;
- verification/conflict resolution;
- integration/finalization;
- recovery/replanning;
- semantically adjacent distractors;
- irrelevant-domain distractors.

Agent IDs dùng dạng opaque và có permutation variants để giảm shortcut theo identity/order.

## 8. State và transition schema

### 8.1 Public observed state

Không dùng các key quá trực tiếp như `needs_bridge`. Thay vào đó biểu diễn semantic state:

```json
{
  "available_artifacts": ["claim from source A"],
  "unresolved_dependencies": ["identify relation between the intermediate entity and target"],
  "candidate_status": "absent",
  "evidence_status": "partial",
  "last_event": {
    "action": "agent_03",
    "status": "SUCCESS",
    "effect": "first evidence dependency resolved"
  },
  "remaining_budget": {
    "steps": 4,
    "cost": 5.0
  }
}
```

### 8.2 Hidden simulator state

Evaluator giữ:

```text
typed facts
artifact types
transition guards
transition effects
goal predicate
failure schedule
cost function
milestones
```

Router không nhận transition guards, hidden goal predicate, valid actions hoặc distractor labels.

### 8.3 Transition status vocabulary

- `SUCCESS`
- `NO_PROGRESS`
- `RETRYABLE_FAILURE`
- `INVALID_OUTPUT`
- `CONFLICT_DETECTED`
- `NOT_APPLICABLE`
- `HARD_FAILURE`
- `GOAL_COMPLETE`
- `PREMATURE_STOP`
- `STEP_LIMIT`
- `COST_LIMIT`

Status có semantic contract rõ trong documentation và test.

## 9. Hugging Face data schema

### 9.1 Subset `decision`

Một row tương ứng một Track A decision:

| Field | Type | Public |
|---|---|---|
| `decision_id` | string | yes |
| `scenario_id` | string | yes |
| `split_group_id` | string | yes |
| `counterfactual_group_id` | nullable string | yes |
| `workflow_family` | string | yes |
| `state_provenance` | enum | yes |
| `task` | string | yes |
| `observed_state` | struct | yes |
| `recent_history` | list[struct] | yes |
| `agent_pool` | list[struct] | yes |
| `candidate_actions` | list[string] | yes |
| `valid_actions` | list[string] | label |
| `efficient_actions` | list[string] | label |
| `stop_ready` | bool | label |
| `difficulty_tags` | list[string] | yes |
| `oracle_version` | string | yes |

### 9.2 Subset `interactive`

Một row tương ứng một Track B scenario:

| Field | Type |
|---|---|
| `scenario_id` | string |
| `split_group_id` | string |
| `workflow_family` | string |
| `task` | string |
| `initial_observation` | struct |
| `agent_pool` | list[struct] |
| `simulator_spec` | struct |
| `max_steps` | int |
| `max_cost` | float |
| `oracle_min_steps` | int |
| `oracle_min_cost` | float |
| `difficulty_tags` | list[string] |
| `schema_version` | string |

`simulator_spec` có trong public V2 để reproduce. Evaluation harness phải tách trường này khỏi router input.

### 9.3 Subset `agent_catalog`

Chứa profile templates, profile variants, semantic role family và version. Evaluator có thể join catalog với scenario nhưng mỗi decision row vẫn giữ snapshot của profile được trình bày cho router để audit chính xác.

## 10. Split design

120 workflows được chia theo `split_group_id`, không chia ngẫu nhiên từng state:

| Split | Workflows | Mục đích |
|---|---:|---|
| `validation` | 48 | phát triển prompt/evaluator |
| `test` | 48 | held-out topology parameters và surface templates |
| `stress` | 24 | adjacent noise, failure chains, pool 16–24 roles |

Mỗi family đóng góp:

- 6 workflows vào validation;
- 6 workflows vào test;
- 3 workflows vào stress.

Mọi state, paraphrase, counterfactual và pool perturbation từ cùng workflow template group phải nằm cùng split.

V2 không có training split vì mục tiêu là frozen/no-training routing. Nếu cần examples cho prompt, cung cấp một file `examples.json` riêng và không tính vào benchmark score.

## 11. Metric suite

### 11.1 Track A core metrics

- `valid_next_action_accuracy`
- `nonterminal_valid_action_accuracy`
- `stop_precision`, `stop_recall`, `stop_f1`
- `distractor_selection_rate`
- macro score theo workflow family

### 11.2 Track A diagnostic metrics

- `efficient_action_rate`
- `counterfactual_pair_accuracy`: đúng cả hai state trong một pair
- `post_failure_routing_accuracy`
- `ambiguous_state_accuracy`
- `pool_noise_robustness_drop`
- `permutation_consistency`
- calibration/ECE nếu router trả probability distribution

### 11.3 Track B core metrics

- `goal_completion_rate`
- `valid_action_rate`
- `premature_stop_rate`
- `recovery_success_rate`
- `distractor_selection_rate`

### 11.4 Track B efficiency metrics

- `step_regret`
- `cost_regret`
- `no_progress_action_rate`
- `redundant_action_rate`
- `budget_exhaustion_rate`
- `robustness_drop_base_to_stress`

Không gộp tất cả thành một composite score trong V2. Báo cáo một metric vector để tránh phải chọn trọng số chủ quan.

## 12. Định nghĩa `efficient_actions` không dùng capability score

Với mỗi valid action, simulator tính successor state rồi tìm:

- số bước ít nhất còn lại để đạt goal;
- deterministic resource cost ít nhất còn lại.

Một action là `efficient` nếu không có valid action khác tốt hơn hoặc bằng trên cả hai chiều và tốt hơn nghiêm ngặt trên ít nhất một chiều. Đây là Pareto frontier:

```text
minimize (remaining_steps, remaining_cost)
```

Không dùng weighted sum. Không dùng backbone accuracy. Không ước lượng “agent intelligence”.

Ví dụ:

- Verifier và Integrator đều có thể hợp lệ;
- nếu candidate đã đủ evidence, Integrator có thể nằm trên efficient frontier;
- Verifier vẫn được tính là valid, nhưng tạo step regret nếu được chọn.

## 13. Data construction pipeline

### Bước 1 — Author workflow templates

Viết ít nhất 3 topology templates cho mỗi family. Mỗi template khai báo typed facts, transition guards/effects, goal, budget và allowed perturbations.

### Bước 2 — Parameterized instantiation

Sinh task/entity/artifact descriptions mới, profile variants, opaque IDs, pool order và noise composition từ seed cố định.

### Bước 3 — State-space enumeration

Duyệt cả:

- nominal valid paths;
- alternative valid paths;
- optional-but-inefficient paths;
- scheduled failure paths;
- selected invalid/no-progress paths dùng để tạo recovery states.

### Bước 4 — Balanced state sampler

Lấy khoảng 8 decision rows/workflow theo quota độ khó ở mục 5. Không lấy toàn bộ reachable states nếu điều đó làm các linear singleton states lấn át.

### Bước 5 — Automatic validation

Kiểm tra:

- mọi nonterminal state có ít nhất một valid action;
- mọi scenario có ít nhất một path tới goal;
- oracle đạt 100%;
- `efficient_actions ⊆ valid_actions`;
- STOP chỉ valid khi goal predicate đúng;
- không có ambiguous overlapping transition ngoài chủ đích;
- counterfactual pair thực sự đổi oracle action set;
- không leakage giữa split groups;
- state fingerprint chứa status/budget/artifact state;
- public observation không chứa hidden label;
- không có capability/backbone score field.

### Bước 6 — Human review

Review thủ công:

- 100% workflow templates;
- tối thiểu 20% instantiated workflows;
- toàn bộ states bị lexical-leak detector đánh dấu;
- toàn bộ counterfactual pairs;
- toàn bộ examples trong dataset card.

### Bước 7 — Freeze và checksum

Freeze schema/data version, seed manifests và SHA-256 checksums trước khi chạy baselines.

## 14. Anti-leakage và anti-shortcut checks

- Tách group trước rồi mới sinh paraphrase/state variants.
- Không dùng action ID trong task hoặc public state text.
- Tính token overlap giữa unresolved-state phrases và role names; review outliers.
- Shuffle candidate order khi đánh giá consistency.
- Dùng opaque agent IDs.
- Không để một role luôn nằm cùng vị trí.
- Không để workflow family quyết định duy nhất next action.
- Báo cáo baseline chỉ dùng task và baseline chỉ dùng state; nếu task-only đạt cao, dataset có shortcut.
- Báo cáo baseline chỉ dùng last status; nếu quá cao, failure states chưa đủ đa dạng.

## 15. Baselines bắt buộc trước khi release

- Random over all candidates.
- Fixed role sequence.
- Task-only router.
- State-only router.
- Qualitative rule router không dùng oracle labels.
- Jev decision router.
- Clef decision router nếu budget cho phép.
- Oracle shortest-step/lowest-cost frontier.

Không có capability-greedy baseline.

V2 nên được harden lại nếu:

- Jev hoặc rule router đạt trên 95% ở mọi slice;
- task-only hoặc lexical baseline vượt 60%;
- random/fixed trajectory completion quá cao;
- robustness drop bằng 0 vì distractor quá dễ;
- hơn 55% decision rows chỉ có một valid action.

## 16. Release gates

Chỉ tạo tag `v2.0.0` khi:

- schema validation đạt 100%;
- oracle Track A và Track B đạt 100%;
- oracle step/cost regret bằng 0;
- split leakage bằng 0;
- duplicate public observations bằng 0 ngoài các counterfactual/status pairs có chủ đích;
- tỷ lệ state thỏa các quota ở mục 5;
- deterministic rerun tạo cùng checksum;
- README examples chạy được;
- Dataset Viewer mở được cả ba subsets;
- một clean environment tải được dataset và chạy random/oracle smoke test;
- limitations và intended use được ghi rõ.

## 17. Hugging Face repository layout

```text
RoutingBench-V2/
├── README.md
├── LICENSE_DATA
├── VERSION
├── CHANGELOG.md
├── CITATION.cff
├── checksums.sha256
├── data/
│   ├── decision/
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── stress.parquet
│   ├── interactive/
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── stress.parquet
│   └── agent_catalog/
│       └── train.parquet
├── specs/
│   ├── decision.schema.json
│   ├── interactive.schema.json
│   ├── agent.schema.json
│   └── status_vocabulary.json
├── docs/
│   ├── BENCHMARK.md
│   ├── DATASHEET.md
│   ├── EVALUATION.md
│   ├── REPRODUCIBILITY.md
│   └── LIMITATIONS.md
└── examples/
    ├── minimal_router.py
    └── evaluate_router.py
```

Hugging Face dùng `README.md` làm dataset card và hỗ trợ khai báo subsets/splits trong YAML header. Parquet phù hợp với dữ liệu tabular/nested có nhu cầu filter và được Dataset Viewer hỗ trợ trực tiếp.

YAML header dự kiến:

```yaml
---
license: cc-by-4.0
language:
  - en
pretty_name: RoutingBench V2
tags:
  - multi-agent
  - agent-routing
  - orchestration
  - decision-making
configs:
  - config_name: decision
    default: true
    data_files:
      - split: validation
        path: data/decision/validation.parquet
      - split: test
        path: data/decision/test.parquet
      - split: stress
        path: data/decision/stress.parquet
  - config_name: interactive
    data_files:
      - split: validation
        path: data/interactive/validation.parquet
      - split: test
        path: data/interactive/test.parquet
      - split: stress
        path: data/interactive/stress.parquet
  - config_name: agent_catalog
    data_files:
      - split: train
        path: data/agent_catalog/train.parquet
---
```

## 18. Dataset card bắt buộc có

- benchmark summary;
- motivation và intended use;
- hai track;
- schema và ví dụ row;
- cách load bằng `datasets.load_dataset`;
- evaluator protocol;
- baselines và kết quả;
- data construction;
- split strategy;
- quality controls;
- licensing/provenance;
- limitations;
- ethical considerations;
- versioning/changelog;
- citation;
- liên kết GitHub source/evaluator.

Không quảng cáo V2 như thước đo đầy đủ cho real-world multi-agent performance. Nó chỉ đo routing trong deterministic workflow contracts.

## 19. License và provenance

Khuyến nghị:

- dataset/task text: `CC-BY-4.0`;
- evaluator/code: giữ license của repository hoặc `Apache-2.0` nếu phù hợp với project;
- mọi task text V2 do project tạo mới;
- MuSiQue/GAIA/run logs chỉ dùng để trích xuất abstract failure patterns, không copy question, answer, paragraph hay copyrighted artifact;
- ghi rõ synthetic generation process và human review trong DATASHEET.

License cuối cùng cần được kiểm tra với license hiện tại của repository trước khi upload.

## 20. Upload và verification plan

Sau khi release directory vượt qua release gates:

```powershell
hf auth whoami
hf repos create <namespace>/RoutingBench-V2 --type dataset --exist-ok
hf upload <namespace>/RoutingBench-V2 <release-directory> --type dataset --commit-message "Release RoutingBench V2.0.0"
```

Sau upload:

1. Kiểm tra Dataset Viewer cho `decision`, `interactive`, `agent_catalog`.
2. Kiểm tra đủ `validation`, `test`, `stress` splits.
3. Load sạch bằng `datasets.load_dataset`.
4. Chạy schema validator trên bản tải từ Hub.
5. So SHA-256 với release local.
6. Tạo Hub tag `v2.0.0` sau khi verification hoàn tất.

## 21. Kế hoạch thực hiện V2

### Phase 1 — Specification freeze

Deliverables:

- JSON schemas;
- metric definitions;
- status vocabulary;
- split/group policy;
- release constraints.

### Phase 2 — Template authoring

Deliverables:

- 24+ topology templates, tối thiểu 3/family;
- catalog 24 role profiles;
- pool/noise/surface perturbation rules.

### Phase 3 — Generation và balancing

Deliverables:

- 120 workflows;
- 900–1.000 Track A rows;
- quota report;
- split manifest và seed manifest.

### Phase 4 — Evaluator V2

Deliverables:

- richer state fingerprint;
- valid/efficient action computation;
- Track A counterfactual/robustness metrics;
- Track B step/cost regret;
- audit logs tương thích V1 khi có thể.

### Phase 5 — QA và baselines

Deliverables:

- schema/invariant tests;
- leakage/shortcut report;
- random/fixed/task-only/state-only/Jev/oracle results;
- error taxonomy.

### Phase 6 — Hugging Face packaging

Deliverables:

- three Parquet subsets;
- dataset card;
- DATASHEET/EVALUATION/LIMITATIONS;
- examples và checksums;
- public `v2.0.0` tag.

## 22. Ranh giới V2 với V3/V4

### V2 — Public research preview

- 120 workflows;
- toàn bộ oracle labels công khai;
- deterministic actors;
- exploratory baselines;
- không leaderboard claim;
- mục tiêu chính là xác nhận benchmark contract và thu feedback cộng đồng.

### V3 — Research benchmark candidate

- khoảng 1.000+ workflows;
- hidden test labels hoặc evaluation server;
- external template contributors;
- stronger adversarial pools;
- confidence intervals và multi-seed evaluation;
- frozen versioned evaluator.

### V4 — Conference-ready release

- power analysis và statistical comparison protocol;
- human validation với inter-annotator agreement;
- nhiều router families và ablations;
- contamination/leakage audit;
- hidden/private test suite;
- live-agent track báo cáo riêng với routing-pure track;
- paper, artifact evaluation package và long-term maintenance plan.

## 23. Quyết định thiết kế cần giữ cố định

1. Core benchmark không gọi LLM actor.
2. Agent profile không có capability/backbone score.
3. Ground truth là set-valued.
4. Validity và efficiency là hai khái niệm riêng.
5. Không tạo một composite score tùy ý trong V2.
6. Split theo workflow/template group, không theo state row.
7. Public state không chứa hidden transition/valid-action labels.
8. Task text là synthetic/original để giảm rủi ro license.
9. V2 ưu tiên reproducibility và inspectability hơn kích thước.
10. Live LLM-agent performance, nếu thêm sau, là một track khác và không thay thế routing-pure score.

## 24. Tài liệu tham khảo triển khai Hugging Face

- Dataset Cards: https://huggingface.co/docs/hub/main/datasets-cards
- Manual subset/split configuration: https://huggingface.co/docs/hub/en/datasets-manual-configuration
- Data files configuration: https://huggingface.co/docs/hub/main/datasets-data-files-configuration
- Uploading datasets: https://huggingface.co/docs/hub/datasets-adding
- Dataset Viewer/Parquet: https://huggingface.co/docs/dataset-viewer/en/parquet
