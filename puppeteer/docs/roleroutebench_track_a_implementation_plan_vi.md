# RoleRouteBench — Implementation plan cho Track A bốn domain live

Ngày: 2026-11-05

Trạng thái: **implementation và Terra pre-labeling hoàn tất ngày 2026-11-07**.
Track A đã build đủ 568 primary states; LLM đã gán nhãn ba cohort rời nhau, phủ
568/568 state. Human review đang chờ thực hiện. Frozen
Track B artifacts không bị thay đổi.

## 1. Kết quả cần đạt

Tạo Track A mới để đánh giá một decision model tại **một state định tuyến cô
lập** của môi trường multi-agent live. Model nhận đúng public observation và
toàn bộ role pool của domain, sau đó chọn đúng một role hoặc `STOP`.

Track A mới phải dùng bốn domain đã hoàn thiện:

- `database_migration_integrity`;
- `code_debugging`;
- `formal_proof`;
- `structured_data_processing`.

Track A đo chất lượng quyết định riêng lẻ. Nó không thay thế Track B, nơi đo hậu
quả tích lũy của nhiều quyết định và executor actions.

Mỗi primary Track A sample phải ánh xạ **một-một** tới một primary decision step
đã thực sự xuất hiện trong canonical Track B. Collector phải copy exact stored
router observation; không được dựng lại state từ trajectory summary, final
workspace hoặc template.

## 2. Quyết định thiết kế đã chốt

1. Nguồn publication authority là RoleRouteBench commit
   `b70803c339a19c767fe9f0718c5218f51d4ed130`.
2. Không sửa hoặc ghi đè `4.0.0-beta.2`; Track A mới đi trong release mới. Đề
   xuất version `4.1.0-beta.1` vì đây là thay đổi material đối với một track.
3. Chỉ dùng canonical episodes đã publish trong Track B, không nối tất cả run
   directory rồi coi chúng ngang nhau.
4. Không chia development/validation/test/holdout; chỉ dùng split `full`.
5. Candidate actions luôn là full domain agent pool cộng `STOP`; không masking.
6. Action JEV đã chọn không phải ground truth, preferred action hoặc utility.
7. Ground truth là tập action hợp lệ; có thể có nhiều action hợp lệ ở cùng state.
8. Router không thấy private artifacts, evaluator outcome, action as-run, model
   identity, probability của JEV hoặc valid-action rationale.
9. Quyết định tại zero remaining budget được giữ trong raw/audit data nhưng bị
   loại khỏi primary Track A score.
10. Mọi hậu quả action đã xuất hiện trong public state/history được giữ nguyên;
    không dựng lại state “sạch hơn” để tăng performance.
11. `decision_id` nguồn của Track B được giữ nguyên. `sample_id` bổ sung được
    tính từ public-state fingerprint, nhưng không thay thế trace identity.
12. LLM pre-labeler đã chốt là exact model ID `gpt-6.1-terra`,
    `reasoning.effort=high`, chạy một request độc lập cho mỗi state được lấy mẫu.
    Cohort đầu lấy ngẫu nhiên không hoàn lại 100/568 state với seed `20261105`.
    Cohort thứ hai lấy 100 state từ pool còn lại với seed `20261106` và gọi
    `openai/gpt-6.1-terra` qua OpenRouter. Cohort thứ ba chứa toàn bộ 368 state
    còn lại với seed `20261107`, cùng model/effort/prompt/schema. Ba cohort không
    giao nhau và phủ đúng 568/568 primary state.
13. Chỉ 45 decision có `dispatch_status=suppressed_by_budget` là audit-only.
    Hai decision có `budget.calls_remaining=0` nhưng action `STOP` đã được xử lý
    với `dispatch_status=stop_handled` vẫn là primary states.

## 3. Nguồn dữ liệu cố định

Nguồn episode phải theo đúng canonical packet-to-run map đang dùng bởi
`scripts/package_routingbench_v4_track_b_hf.py`.

| Domain | Canonical run | Primary states | Audit-only zero-budget states |
| --- | --- | ---: | ---: |
| Database | `database_shared_jev_16k_full_20261031` | 84 | 6 |
| Code debugging | `code_shared_jev_gemini_v3_20261101` | 143 | 8 |
| Formal proof | `formal_shared_jev_prover_v4_20261031` và `formal_minif2f_20_jev_live_20261101` | 185 | 13 |
| Structured data | bốn canonical structured-data runs của beta.2 | 156 | 18 |
| **Tổng** | 82 canonical episodes | **568** | **45** |

Tổng raw decision records là 613. Audit hiện tại xác nhận:

- 568/568 primary rows có `evaluation_eligible=true`;
- 45 rows bị loại chỉ vì `suppressed_by_budget`;
- task trong row khớp task trong observation;
- candidate actions khớp `observation.constraints.allowed_actions`;
- decision ID khớp packet ID;
- 568 canonical primary observations đều duy nhất khi hash canonical JSON.
- 82/82 episodes có tập `decision_id` trong `decision_states.jsonl` khớp chính
  xác tập `decision_id` trong `decisions.jsonl`;
- 613/613 state có full observation và candidate actions.

Không dùng `track_b_trajectories.decision_sequence_json` để dựng lại router
input vì bảng này là compact analysis view và không chứa full observation.
Nguồn observation có thẩm quyền là `decision_states.jsonl` hoặc
`human_state_review_queue.jsonl` trong canonical run directory.

### Exact replay boundary

`SharedTrajectoryRunner` tạo observation, deep-copy nó thành immutable JSON
snapshot, persist `decision_states.jsonl`, rồi truyền chính object observation
đó vào `router.select(observation, actions)`. Vì vậy Track A có thể tái hiện
chính xác **logical router input** của từng Track B step:

- task;
- current public state;
- full role cards;
- recent public events và action consequences;
- remaining budget;
- full candidate action list;
- router-visible constraints.

Database run có rejection guard được cấu hình bật ở 90 decisions, nhưng audit
metadata cho thấy `guard_blocked_actions` rỗng ở cả 90, nên không có candidate
action nào thực tế bị mask. Guard tắt ở code, formal proof và structured data.
Do đó full pool cộng `STOP` đã tới decision model ở toàn bộ 613 canonical
decisions.

Giới hạn cần ghi rõ: shared runner không lưu content-addressed workspace snapshot
ở từng step. Vì vậy dữ liệu hiện tại không đảm bảo khôi phục byte-identical
database/workbook/repository/Lean workspace để chạy một alternate executor từ
giữa trajectory. Track A là isolated router evaluation nên không cần khả năng
này. Nếu sau này muốn counterfactual execution từ một Track A state, phải chạy
Track B mới với per-step `SnapshotStore` hoặc replay toàn bộ prefix bằng archived
executor outputs.

Remote decision service cũng không đảm bảo trả lại cùng output khi gọi lại.
Guarantee của Track A là cùng logical input/action space tại cùng step, không
phải cùng stochastic model response.

## 4. Namespace và ranh giới freeze

Không thêm implementation mới vào `routing_benchmark/v4`, vì toàn bộ namespace
đó nằm trong runtime checksum của beta.2. Tạo namespace mới:

```text
routing_benchmark/roleroutebench/track_a/
├── schema.py
├── source_manifest.py
├── collector.py
├── canonicalize.py
├── review.py
├── evaluator.py
├── validation.py
└── packaging.py
```

Namespace mới được phép đọc frozen v4 artifacts nhưng không sửa state, adapter,
role cards hoặc trace nguồn.

Các entry point dự kiến:

```text
scripts/build_roleroutebench_track_a.py
scripts/validate_roleroutebench_track_a.py
scripts/label_roleroutebench_track_a.py
scripts/serve_roleroutebench_track_a_review.py
scripts/merge_roleroutebench_track_a_reviews.py
scripts/run_roleroutebench_track_a.py
scripts/package_roleroutebench_track_a_hf.py
```

## 5. Schema đầu ra

### 5.1 Router-facing state

Mỗi row trong `track_a_states/full.parquet` gồm:

- `sample_id`: ID ổn định từ canonical public fingerprint;
- `decision_id`: exact source Track B decision ID;
- `domain`;
- `packet_id`;
- `router_input_json`: exact public observation;
- `candidate_actions_json`: full domain pool cộng `STOP`;
- `budget_remaining`;
- `primary_eligible`;
- `exclusion_reason`, chỉ có giá trị cho audit-only rows;
- `public_state_sha256`;
- `split="full"`.

`router_input_json` không chứa `valid_actions`, action as-run, router
probabilities, final task result hoặc reviewer information.

### 5.2 Evaluator-only labels

Mỗi row trong `track_a_labels/full.parquet` gồm:

- `decision_id`;
- `valid_actions_json`;
- `unsafe_actions_json`;
- `redundant_actions_json`;
- `terminal_valid`;
- `state_phase`;
- `label_status`;
- `label_source`;
- `adjudication_status`;
- `label_contract_version`;
- `split="full"`.

Action taxonomy dùng trong review:

- `valid_progress`: thực hiện hoặc thu thập evidence cần thiết và có thể tạo
  tiến triển material;
- `valid_recovery`: recovery/repair hợp lệ từ hậu quả đang quan sát được;
- `valid_stop`: terminal contract đã thỏa;
- `redundant_no_progress`: có thể gọi nhưng evidence hiện tại cho thấy không tạo
  tiến triển mới;
- `invalid_precondition`: input/artifact/evidence mà role cần chưa tồn tại;
- `unsafe`: có thể vi phạm immutable scope, stale-artifact contract hoặc làm mất
  hậu quả cần giữ;
- `uncertain`: chưa đủ cơ sở để reviewer kết luận.

`valid_actions` là hợp của `valid_progress`, `valid_recovery` và `valid_stop`.
Không xếp hạng các action hợp lệ và không chọn một preferred action.

### 5.3 Provenance

`track_a_provenance/full.parquet` lưu:

- source run, source decision ID và source decision index;
- source packet/manifest checksum;
- role-pool checksum;
- public observation checksum;
- publication revision `b70803c`;
- frozen Track B data revision `cdc67bad`;
- code/runtime revision và aggregate runtime checksum;
- source dispatch status;
- action as-run chỉ để analysis, không phải label.

### 5.4 Review evidence

`track_a_reviews/full.parquet` chỉ publish review status, pseudonymous reviewer
IDs, disagreement/adjudication status và record checksum. Không publish free-form
reviewer notes nếu chúng chứa private/evaluator-only information.

Bảng này cũng lưu model ID/prompt version của pre-label, human decision
(`accepted`, `corrected`, `rejected`) và correction category để đo LLM label
quality mà không biến pre-label thành ground truth.

## 6. Canonical fingerprint và dedup

Fingerprint phải được tính trên canonical JSON của:

- task và public task contract;
- normalized current public state;
- public artifact inventory và freshness/version fields;
- recent observable events;
- budget;
- ordered full role cards;
- allowed actions và router constraints.

Không đưa `decision_id`, run ID, packet path, JEV action, provider/model hoặc
evaluator result vào fingerprint.

Track A bốn-domain đầu tiên giữ ánh xạ một-một giữa sample và source decision.
Validator phải fail nếu hai primary source steps có cùng fingerprint để buộc
review rõ ràng về weighting trước khi release, thay vì silently merge hoặc nhân
trọng số. Hiện tại không có trường hợp này.

Nếu một release tương lai chủ động cho phép nhiều source rows cùng fingerprint,
phải thiết kế một policy mới có version. Policy đó tối thiểu phải:

1. lưu mọi source occurrence trong provenance;
2. yêu cầu labels giống nhau;
3. khai báo rõ dedup hoặc sample-weighting trong evaluator;
4. fail release nếu cùng public state nhận labels khác nhau.

Hiện tại 568 primary states không trùng, nhưng gate này là bắt buộc cho domain
hoặc trajectory được bổ sung sau này.

## 7. LLM pre-label và human-verification contract

### 7.1 Model đã chọn

Model phải đọc tốt code/database/Lean/spreadsheet states, tuân thủ JSON Schema và
phân loại toàn bộ action thay vì chỉ đoán một action tốt nhất.

Build toàn bộ 568 primary states và chạy pre-label trên ba cohort rời nhau
(100, 100 và 368 state) bằng:

- model: `gpt-6.1-terra`;
- reasoning effort: `high`;
- API: authenticated Codex CLI ở cohort đầu; OpenRouter Chat Completions ở
  cohort thứ hai và thứ ba;
- output: strict structured output theo versioned JSON Schema;
- request granularity: một state/request, không đưa state trước hoặc sau cùng
  trajectory vào hội thoại;
- `max_output_tokens=25000`, bao gồm reasoning và visible output;
- không tự động chuyển sang model khác khi lỗi. Retry transport/schema vẫn dùng
  cùng exact model, effort, prompt và state.

Mức 25,000 này chỉ là output cap của **Track A labeler** để chừa chỗ cho
reasoning `high`; nó không thay đổi executor cap 16K hay bất kỳ thiết kế đã
freeze nào của Track B.

Human vẫn kiểm tra mọi state. Terra chỉ tạo pre-label proposal, không trở thành
ground truth và không thay thế bước human verification.

Không dùng JEV làm labeler vì JEV là router tham chiếu đã tạo action as-run. Nên
tránh dùng cùng executor backbone làm nguồn nhãn duy nhất cho domain tương ứng.

### 7.2 Random 100-state pilot

Pilot hiện tại lấy mẫu uniform không hoàn lại từ 568 primary states:

- sample count: 100;
- seed: `20261105`;
- sampling implementation: Python `random.Random(seed).sample`, sau đó sort theo
  `decision_id` để thứ tự chạy ổn định;
- domain distribution là kết quả ngẫu nhiên, không ép quota;
- lưu full decision-ID list và SHA-256 trong `sample_manifest.json`.

Human gán gold pilot độc lập trước khi xem output LLM. So sánh model bằng:

- exact-set agreement của `valid_actions`;
- per-action macro F1;
- Jaccard của valid-action set;
- `STOP` precision/recall;
- unsafe-action recall;
- schema-valid response rate;
- cost và latency mỗi state.

Pilot dùng để freeze prompt/schema, phát hiện
lỗi visibility, ước lượng reasoning-token thực tế và sắp xếp review queue. Nếu
prompt/schema đổi, mọi attempt cũ vẫn được giữ append-only nhưng không được trộn
vào canonical pre-label run; phải tạo một run version mới.

Không đặt quality gate tự động để biến output Terra thành nhãn. Metrics pilot là
baseline để human reviewer biết loại lỗi cần chú ý, đặc biệt false-valid `STOP`,
unsafe miss và nhầm lẫn recovery/redundant.

### 7.2b OpenRouter 100-state extension

Sau khi người dùng phê duyệt mở rộng, cohort thứ hai lấy 100 state từ 468 state
chưa được chọn bằng seed `20261106`, không giao với cohort đầu. Model được gọi
trực tiếp qua OpenRouter với ID `openai/gpt-6.1-terra`, reasoning `high`, cùng
prompt/schema contract. Kết quả là 100/100 schema-valid pre-label; còn lại 368
primary states chưa có LLM attempt. Human review chạy riêng cohort này nhưng ghi
vào cùng append-only review store.

### 7.2c OpenRouter full remaining-state extension

Ngày 2026-11-07, sau phê duyệt mới của người dùng, toàn bộ 368 state còn lại
được freeze thành `labeling_openrouter_remaining_v1` và gán nhãn bằng cùng exact
model `openai/gpt-6.1-terra`, reasoning `high`, prompt/schema version. Kết quả
368/368 pre-label hợp lệ; ba cohort phủ toàn bộ 568 primary states. Mọi transport
failure được giữ append-only và không được tính là evaluator labels.

### 7.3 Dự toán token và chi phí

Phép đo ngày 2026-11-05 dùng canonical compact JSON của exact
`observation` và tokenizer `o200k_base`. Đây là dự toán cục bộ vì tài liệu model
không công bố tokenizer mapping riêng cho Terra. Khi implement, phải gọi input
token counting endpoint trên **final request payload** và lưu kết quả trước khi
gửi bulk labeling requests.

#### Token của exact state payload

| Domain | States | Observation tokens | Trung bình/state |
| --- | ---: | ---: | ---: |
| Database | 84 | 500,969 | 5,963.9 |
| Code debugging | 143 | 353,732 | 2,473.7 |
| Formal proof | 185 | 1,358,399 | 7,342.7 |
| Structured data | 156 | 593,687 | 3,805.7 |
| **Tổng** | **568** | **2,806,787** | **4,941.5** |

Toàn bộ distribution: median 2,744; p90 13,356; p95 19,810; max 23,612
observation tokens/state. State dài nhất vẫn cách xa context window của Terra.
Formal proof chiếm khoảng 48.4% observation tokens nên phải được theo dõi riêng
trong pilot và cost report.

Final prompt chưa được implement nên phần overhead được lập ngân sách như sau:

- versioned instructions + taxonomy + JSON Schema: 1,300–1,800 token/request;
- request framing/metadata: 100–200 token/request;
- tổng input dự kiến: **3.60–3.94 triệu token**, planning value khoảng
  **3.77 triệu**.

Reasoning token ở `high` là adaptive, không thể suy ra chính xác từ input. Nó
được tính như output token và phải đọc từ
`usage.output_tokens_details.reasoning_tokens`. Dùng ba scenario để reserve:

| Scenario | Reasoning/state | Visible output/state | Tổng output 568 states | Tổng input + output |
| --- | ---: | ---: | ---: | ---: |
| Lean | 4,000 | 600 | 2.61M | 6.21–6.55M |
| Planning | 8,000 | 800 | 5.00M | **8.60–8.94M** |
| High | 16,000 | 1,000 | 9.66M | 13.26–13.60M |
| Hard reservation cap | — | — | 14.20M | 17.80–18.14M |

Con số dùng để lập kế hoạch ban đầu là khoảng **8.8 triệu total tokens**. Khoảng
thận trọng là **6.2–13.6 triệu**; 14.20M output là capacity tối đa nếu cả 568
request đều chạm `max_output_tokens=25000`, không phải expected usage.

Với standard price hiện tại của Terra là USD 1/1M input token và USD 5/1M
output token, cost planning tương ứng khoảng USD 28.6–28.9; lean khoảng USD
16.7–17.0; high khoảng USD 51.9–52.2; hard cap khoảng USD 74.6–74.9. Các số này
chưa gồm retry, regional uplift hoặc thay đổi giá. Không trừ prompt caching khỏi
budget để tránh đánh giá thấp chi phí.

Nguồn vận hành phải được kiểm tra lại tại thời điểm chạy:
[Terra model page](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) và
[API pricing](https://developers.openai.com/api/docs/pricing).

Sau mỗi cohort, thay assumptions bằng mean/p50/p90 thực đo cho input, reasoning,
visible output, latency và USD/state. Mọi retry được ghi riêng và cộng vào actual
usage. Toàn bộ 568 state đã được pre-label; bước tiếp theo là human review.

#### Usage thực đo của pilot 100 state

Pilot hoàn tất ngày 2026-11-06 qua authenticated Codex CLI vì workspace không có
`OPENAI_API_KEY`. Exact model/effort là `gpt-6.1-terra` + `high`, nhưng provider
được ghi thành `codex_exec_openai` để thể hiện rõ CLI wrapper có thêm system
context. Không coi số input token này là direct Responses API baseline.

| Chỉ số | Kết quả |
| --- | ---: |
| Successful pre-labels | 100/100 |
| Total attempts | 114 |
| Transport failures được giữ audit | 14 |
| Input tokens | 2,298,681 |
| Output tokens | 56,680 |
| Reported reasoning tokens | 11,337 |
| Total tokens | 2,355,361 |
| Mean input/state | 22,986.81 |
| Mean output/state | 566.8 |
| Mean latency/state | 15.48 giây |
| Standard API-price equivalent | USD 2.582081 |

14 transport failures gồm hai attempt lỗi Windows encoding trước khi fix UTF-8
và 12 attempt bị usage-limit trong sáu state. Resume sau quota reset hoàn tất cả
sáu state; không có schema-invalid response. Append-only attempt log giữ đầy đủ
các failure này.

### 7.4 LLM visibility và prompt contract

LLM labeler chỉ thấy đúng nội dung human reviewer được phép dùng:

- task và exact public state;
- public history/action consequences;
- budget;
- full role cards;
- full candidate actions.

LLM không thấy:

- JEV selected action, probability hoặc rationale;
- task pass/fail cuối;
- private reference/evaluator artifacts;
- executor model/provider identity;
- human labels;
- nhãn của model khác.

Prompt yêu cầu phân loại **mọi candidate action** theo taxonomy ở mục 5.2, gắn
concise evidence-grounded rationale và xuất JSON Schema. Model không được chọn
một preferred workflow hoặc giả định action as-run là đúng.

### 7.5 LLM label artifact

Mỗi record trong `llm_prelabels.jsonl` gồm:

- `decision_id`, `public_state_sha256`;
- `model_provider`, exact `model_id`, model/version metadata;
- `prompt_version`, `label_schema_version`;
- per-action classification và concise rationale;
- proposed `valid_actions`, `unsafe_actions`, `redundant_actions`;
- proposed `terminal_valid` và `state_phase`;
- self-reported confidence chỉ để triage;
- raw-response hash, token usage, latency, retry count;
- `status`: `proposed`, `schema_failed`, `uncertain` hoặc
  `needs_human_priority`.

Raw hidden reasoning/chain-of-thought không được yêu cầu và không cần publish.
Chỉ lưu structured classification cùng concise rationale kiểm chứng được.

Structured-output validator phải từ chối:

- thiếu candidate action;
- action ngoài full pool;
- một action xuất hiện ở nhiều class;
- `STOP=valid_stop` nhưng `terminal_valid=false`;
- valid/unsafe/redundant sets không khớp per-action classifications;
- state hash hoặc schema version không khớp.

Retry chỉ sửa lỗi schema/transport với cùng observation; không silently thay
prompt semantics. Mọi attempt được ghi append-only.

### 7.6 Human verification workflow

Human review là bước kiểm tra và sửa pre-label, không phải gán lại từ đầu:

1. Reviewer thấy exact public state, full role pool, LLM classification và
   concise rationale cho từng action.
2. Reviewer có thể `accept_all`, sửa classification từng action, sửa terminal
   flag/phase, hoặc `reject_for_adjudication`.
3. Human phải kiểm tra toàn bộ candidate actions, không chỉ `valid_actions` mà
   LLM đề xuất.
4. Một state chỉ có `label_status=human_verified` sau khi reviewer submit và
   automated label validator pass.
5. State có unsafe/STOP correction, reviewer uncertainty hoặc model/schema
   failure đi vào adjudication/second-review queue.
6. Lấy ngẫu nhiên ít nhất 10% state được `accept_all` để reviewer thứ hai audit
   nhằm đo false-accept rate.
7. Mọi lần save là append-only record gắn state hash; không overwrite pre-label
   hoặc review cũ.

Review UI hiện tại chỉ đọc state. Cần thêm local review service bằng Python
standard library hoặc backend sẵn có để ghi append-only JSONL. UI phải có
`accept_all`, action checklist/taxonomy, terminal flag, state phase, LLM
rationale, correction reason và adjudication status.

Human reviewer vẫn không thấy JEV action, final outcome hoặc private artifacts.

### 7.7 State phase

Analysis-only phase gồm:

- `initial`;
- `evidence_collection`;
- `candidate_authoring`;
- `execution`;
- `failure_diagnosis`;
- `repair_or_recovery`;
- `validation`;
- `terminal`.

Phase không được đưa vào router input nếu nó không tồn tại trong observation
gốc.

## 8. Evaluator contract

Evaluator nhận một callable/router adapter và thực hiện độc lập từng row:

1. gửi `router_input_json` và candidate actions;
2. yêu cầu đúng một action;
3. parse/normalize action nhưng không sửa quyết định không hợp lệ;
4. join label bằng `decision_id`;
5. chỉ đưa row `primary_eligible=true` và `label_status=human_verified` vào
   primary metrics;
6. lưu mọi output, parse error và latency trong run artifact.

Primary metrics, báo cáo riêng không composite:

- set-valued valid-action accuracy;
- macro valid-action accuracy theo bốn domain;
- premature `STOP` rate;
- terminal `STOP` precision và recall;
- unsafe-action selection rate;
- redundant/no-progress selection rate;
- invalid-output/parse-failure rate;
- breakdown theo state phase và budget band.

Action agreement với JEV chỉ là diagnostic analysis, không phải accuracy.
Evaluator phải nhóm domain từ evaluator/analysis metadata, không dùng public
`workflow_family=dynamic_workflow`.

Sanity baselines:

- oracle chọn một phần tử trong `valid_actions`: phải đạt 100%;
- random chọn đều từ full pool cộng `STOP`;
- `STOP`-only để kiểm tra premature-stop metrics;
- replay JEV để tạo reference result, không gọi nó là oracle.

## 9. Các pha implementation

### Pha 0 — Freeze source manifest

Tạo `config/roleroutebench/track_a_4d_live_v1.json` chứa:

- publication/data/code/runtime revisions;
- 82 packet IDs;
- canonical packet-to-run mapping;
- expected counts 613/568/45;
- domain role-pool hashes;
- input file checksums.

Acceptance: source manifest resolve đủ 82 packet, không duplicate packet, không
tham chiếu superseded episode.

### Pha 1 — Collector và canonicalizer

Implement collector đọc canonical `decision_states.jsonl`, decision dispatch
record và frozen role cards. Collector tạo raw rows, primary eligibility,
fingerprint và provenance; không đọc private packet directory. Observation phải
được copy trực tiếp từ source row, không gọi adapter để tái dựng.

Acceptance:

- 613 raw rows;
- 568 primary rows;
- 45 budget audit rows;
- domain counts 84/143/185/156;
- full pool cộng `STOP` trên mọi row;
- zero selected-action/probability/final-result fields trong router payload.
- canonical JSON của Track A `router_input_json` bằng chính xác canonical JSON
  của source `observation` cho 613/613 rows;
- `decision_id`, budget, candidate-action order và role-card order được bảo toàn.

### Pha 2 — Schema và automated validator

Implement JSON Schema/Pydantic-free validation để tránh thêm runtime dependency.
Validator kiểm tra field types, hash, subset relation, role identity, no masking,
private leakage, dedup và source count.

Acceptance: corruption tests phải fail khi inject private secret, bỏ một role,
đổi state sau hash, thêm label vào router payload hoặc đưa zero-budget row vào
primary score.

### Pha 3 — LLM pre-label pipeline và calibration

Implement:

- `scripts/label_roleroutebench_track_a.py`;
- Responses API client pin `gpt-6.1-terra` với `reasoning.effort=high`;
- versioned prompt và JSON Schema;
- append-only raw/parsed attempt store;
- validation, same-model retry và human-priority queue;
- exact input-token count dry run;
- pilot calibration/usage report.

Hai cohort đầu, mỗi cohort 100 primary states, và cohort cuối gồm 368 state còn
lại đều dùng seed cố định, Terra `high` và sinh human-review queue. Không gọi LLM
cho 45 `suppressed_by_budget` audit states.

Acceptance:

- đúng 568 primary states có một schema-valid pre-label; 45 audit-only state
  không được đưa vào labeling cohorts;
- 100% candidate actions được phân loại trong mỗi valid response;
- zero JEV action/final outcome/private field trong prompt;
- rerun/resume không ghi đè attempts;
- report có exact-set, F1, Jaccard, STOP/unsafe metrics trên pilot;
- model/prompt/schema/hash/token/cost provenance đầy đủ;
- exact usage tách `input_tokens`, cached input, reasoning tokens và visible
  output tokens cho từng attempt;
- actual token/cost report khớp tổng per-attempt records;
- no silent model fallback và mọi canonical response dùng exact
  `gpt-6.1-terra` + `high`.

### Pha 4 — Human-verification service

Sinh verification queue từ LLM pre-labels. Implement append-only human review,
correction tracking, priority/adjudication queue và 10% second-review audit.

Verify theo domain, ưu tiên:

1. database để kiểm tra schema/evidence/recovery semantics;
2. formal proof để kiểm tra compile/repair/verification boundaries;
3. code debugging;
4. structured data processing.

Sau mỗi domain, chạy label consistency audit trước khi chuyển domain tiếp theo.
Structured-data packet review còn pending phải được hoàn tất hoặc các state liên
quan phải giữ `label_status=pending` và không vào primary score.

Acceptance:

- reviewer không thấy action as-run/final outcome/private artifact;
- mọi candidate action đã được human xác nhận hoặc sửa;
- review record không thể áp vào state hash khác;
- LLM pre-label và human correction đều được giữ, không overwrite;
- state unresolved không được đánh dấu `human_verified`;
- second-review sample đủ tối thiểu 10% `accept_all` states và có false-accept
  report.

### Pha 5 — Evaluator và baselines

Implement isolated evaluator, output parser, per-domain metrics và sanity
baselines. Không gọi executor và không mutate source environment.

Acceptance: oracle 100%, deterministic replay cùng seed cho cùng result, random
baseline chỉ chọn candidate action, domain macro có đúng bốn nhóm.

### Pha 6 — HF packaging

Tạo staging release mới từ commit `b70803c`, giữ nguyên Track B beta.2, sau đó
thêm Track A mới.

Canonical configs đề xuất:

- `track_a_states` — router-facing, `full`;
- `track_a_labels` — evaluator-only, `full`;
- `track_a_provenance` — analysis, `full`;
- `track_a_reviews` — review evidence, `full`.

Track A beta.1 cũ được giữ byte-identical dưới namespace `legacy_beta1_*`; không
để `router_state` cũ tiếp tục là default mà không có cảnh báo migration.

Acceptance: package load được bằng `datasets.load_dataset`, row counts và schema
khớp manifest, checksums pass, old Track B tables/packet tree không đổi byte.

### Pha 7 — Release certification

Chạy:

- schema/count/checksum validation;
- private-isolation sentinel;
- no-masking validator;
- label/review completeness;
- LLM pre-label schema/provenance và human-correction accounting;
- evaluator oracle/random/STOP sanity tests;
- beta.2 Track B regression certificate;
- dataset-card/config validation.

Chỉ publish khi automated gates pass và review coverage được ghi trung thực.
Nếu review chưa đủ, publish research-preview metadata nhưng không công bố primary
Track A leaderboard score.

## 10. Test suite cần bổ sung

```text
tests/test_roleroutebench_track_a_source_manifest.py
tests/test_roleroutebench_track_a_collector.py
tests/test_roleroutebench_track_a_private_isolation.py
tests/test_roleroutebench_track_a_full_pool.py
tests/test_roleroutebench_track_a_fingerprint.py
tests/test_roleroutebench_track_a_prelabels.py
tests/test_roleroutebench_track_a_reviews.py
tests/test_roleroutebench_track_a_evaluator.py
tests/test_roleroutebench_track_a_packaging.py
```

Minimum assertions:

- 82 canonical packets và đúng packet-to-run mapping;
- 613 raw, 568 primary, 45 budget-only;
- counts theo domain đúng;
- candidate action set bằng frozen pool cộng `STOP`;
- không có private secret hoặc evaluator result trong router payload;
- valid/unsafe/redundant actions đều là subset của candidate actions;
- `STOP` hợp lệ khi và chỉ khi terminal contract được reviewer xác nhận;
- label conflict trên cùng fingerprint làm validator fail;
- macro metrics có bốn domain;
- Track B runtime checksum vẫn là
  `779cfc4d9e913c26b8145f7435a983cd200771465707f54360fdec15ece2f16e`.

## 11. Definition of done

Implementation và ba labeling cohort hiện tại được coi là hoàn thành khi:

1. source manifest khóa đúng commit/revision/checksum;
2. 568 primary public states được export từ exact live observations;
3. 568 samples ánh xạ một-một tới 568 primary Track B decision IDs và pass
   exact-observation equality;
4. full pool/no-masking và private isolation pass 100%;
5. đúng 568 primary states thuộc ba cohort rời nhau có LLM pre-label với đầy đủ
   model/prompt/schema provenance;
6. 568 labels set-valued đã qua human verification,
   correction/adjudication provenance và second-review audit;
7. evaluator pass oracle/random/STOP sanity tests;
8. primary metrics loại đúng 45 zero-budget decisions nhưng raw provenance vẫn
   giữ chúng;
9. HF staging package load và validate thành công;
10. Track B beta.2 không đổi byte và release certificate tiếp tục pass;
11. dataset card phân biệt rõ Track A mới với legacy beta.1 Track A;
12. release mới có immutable checksum manifest và không overwrite beta.2.

Hiện tại các mục 1–5 và implementation cho 7–9 đã hoàn tất; mục 6 chờ human
review. Toàn bộ pre-labeling đã hoàn tất; full Track A release tiếp theo cần human
verification coverage và policy adjudication rõ ràng cho 568 states.

## 12. Việc không chặn implementation Track A

`scripts/reproduce_roleroutebench.py` không nằm trên critical path của collector,
reviewer hoặc evaluator. Có thể bổ sung wrapper này ở Pha 7 để hoàn thiện
one-command reproduction trước khi publish, nhưng không cần chờ nó để bắt đầu
Pha 0–5.

## 13. Model references

Tài liệu model cần được kiểm tra lại ngay trước khi chạy pilot và pin release:

- OpenAI current model/pricing table:
  `https://platform.openai.com/pricing`;
- Anthropic active/deprecated model table:
  `https://docs.anthropic.com/en/docs/about-claude/model-deprecations`;
- Gemini 3.1 Pro model card:
  `https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview`;
- Gemini structured outputs:
  `https://ai.google.dev/gemini-api/docs/structured-output`.

Không dùng alias `latest` cho canonical labeling run. Nếu provider không có
immutable snapshot ID, phải lưu response model/version metadata, run date,
request hash và raw-response hash để provenance minh bạch.
