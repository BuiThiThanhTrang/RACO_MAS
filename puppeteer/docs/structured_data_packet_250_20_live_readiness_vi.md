# Review packet và live readiness — `sdp-spreadsheetbench-250-20`

Ngày review: **2026-11-01**.

## 1. Kết luận

Packet đã được chuyển từ contract `candidate/output.xlsx` sang contract OJ gần nguồn gốc hơn:

- artifact chính là Python transformation có entrypoint `solve(input_path, output_path)`;
- case 1 là public development case;
- case 3 là trusted private evaluator case;
- case 2 bị loại trước live vì answer upstream không nhất quán với instruction/input;
- cùng một candidate hash phải chạy trên case 1 và 3;
- metric gồm per-case status, soft score và hard pass.

Packet **đạt static/semantic design gate nhưng chưa live-ready**. Chạy live lúc này không tạo kết quả có ý nghĩa so sánh vì adapter, runtime role implementation, sandbox, independent checker và common runner dispatch chưa tồn tại cho adapter mới.

## 2. Thay đổi đã thực hiện

Builder `scripts/build_five_domain_packets.py` có special source-faithful build cho sample `250-20` và giữ cách build cũ cho 19 packet spreadsheet còn lại. Generator version của packet này là `routingbench-v4-five-domains-v2-spreadsheet-oj`.

Packet hiện có:

```text
public/task.json
public/packet_spec.json
public/execution_contract.json
public/cases/case_1/input.xlsx
private/evaluation_target.json
private/cases/case_1/answer.xlsx
private/cases/case_3/input.xlsx
private/cases/case_3/answer.xlsx
fixtures/upstream_provenance.json
fixtures/case_selection.json
manifest.json
```

Source lock lưu cả Verified-400 archive và full v0.1 OJ archive. Case selection manifest ghi `[1]` public, `[1,3]` evaluator và `[2]` excluded. Bản packet cũ được giữ trong ignored cache `puppeteer/.cache/routingbench_v4/five_domains_20261101/packet_backups/`.

## 3. Evidence đã chạy

### 3.1. Static integrity

`validate_artifact_pack()`:

```json
{
  "valid": true,
  "base_workflow_id": "sdp-spreadsheetbench-250-20",
  "domain": "structured_data_processing",
  "public_file_count": 4,
  "private_file_count": 4,
  "fixture_file_count": 2,
  "errors": []
}
```

`packet_spec.json` có đầy đủ top-level fields của template, 10 role contracts có typed tool schemas và ba artifact contracts: `candidate_solution`, `public_execution`, `public_validation`.

### 3.2. Semantic witness và negative shortcut

Reference transformation được viết độc lập từ instruction, không đọc answer khi tạo output:

| Test | Kết quả |
|---|---|
| Reference transformation trên case 1 | PASS |
| Reference transformation trên case 3 | PASS |
| Copy sheet `Desired outcome` trên case 1 | PASS |
| Copy sheet `Desired outcome` trên case 3 | FAIL |

Như vậy case 3 khôi phục khả năng phân biệt solution tổng quát với shortcut copy. Đây là environment witness, không phải performance của executor hoặc router.

### 3.3. Case 2 exclusion

Case 2 bị loại trước live với hash và evidence lưu trong `fixtures/case_selection.json`:

- transformation theo instruction tạo 14 data rows trong khi answer upstream có 16;
- key `(70000009932150, 8)` xuất hiện bốn lần với `J=1`, nhưng answer upstream giữ `J=1` thay vì `J=4`;
- answer chứa rows không tồn tại trong input case 2.

## 4. Gate matrix theo domain extension standard

> Lưu ý: bảng dưới đây là snapshot trước khi adapter được implement. Kết quả preflight và live run mới hơn được ghi tại mục 8 và thay thế các blocker runtime trong snapshot này.

| Gate | Status | Evidence / blocker |
|---|---|---|
| Source selection fixed before model outcomes | PASS | Case 1/3 included và case 2 excluded được ghi trong packet/root selection metadata trước live |
| Manifest, checksums, paths, media openability | PASS | `validate_artifact_pack()` pass |
| Goal, initial state, edit/immutable scope, invariants | PASS — design | `public/packet_spec.json` |
| Candidate acceptance tách task success | PASS — design | Python parse/entrypoint acceptance khác independent hard pass |
| Source provenance and adaptation log | PASS | Hai archive hashes, member hashes, revision và adaptations được lưu |
| Semantic positive witness | PASS | Reference transformation pass case 1 và 3 |
| Negative shortcut reject | PASS — oracle check | Copy example fail case 3; chưa phải test của production checker |
| Private/fixture isolation | PENDING | Physical separation và fixture allowlist rỗng đã khai báo; public JSON không chứa case-2 input hash hay case-3 answer hash. Public contract có thể khai báo sự tồn tại của evaluator case 3 nhưng không lộ nội dung. Chưa có adapter materialization/reader/prompt isolation test |
| Runtime/toolchain pin | PARTIAL | Python 3.13.9, openpyxl 3.1.5, pandas 2.3.3, pywin32 311, Excel 16.0.20527.20112 đã ghi; chưa chạy sandbox + Excel automation preflight |
| Real adapter | FAIL | `spreadsheetbench.python_oj.v1` chưa đăng ký; registry trả `ValueError` |
| Role identity/tool implementation | FAIL | Packet có contract; runtime `role_cards_for('structured_data_processing')` trả `KeyError`. Chỉ 2/10 tool-name sets của shared pool khớp packet; 8/10 mutation/execution roles vẫn dùng workbook-only tools cũ |
| Candidate sandbox | FAIL | Chưa có isolated candidate runner với input read-only, output-only write, network disabled và timeout/resource policy |
| Artifact handoff/version/freshness | PENDING | Contract có dependency/hash rules; chưa có runtime implementation và stale/rejected-attempt tests |
| Independent final checker | FAIL | `final_evaluation.py` chỉ hỗ trợ formal/code; domain này trả `UNAVAILABLE` |
| Recalculation backend | PENDING | Excel binary có mặt và đã pin; chưa có automated open/save smoke trong trusted checker |
| Common live runner | FAIL | `run_shared_environment.py` chỉ allow database/formal/code adapters |
| Full pool + STOP, guard off | PARTIAL | Root routing contract/pool có full actions + STOP và guard off; role loader/new adapter chưa tích hợp |
| Budget boundary/accounting | PENDING | Shared protocol có rule; chưa test với adapter mới |
| End-to-end smoke | FAIL | Chưa thể chạy adapter → actor → artifact → checker |
| Human packet review | PENDING | Không tự chuyển status từ tool result |
| Freeze | PENDING | Chưa freeze adapter/runtime/checker/config; packet vẫn `pending_runtime_verification` |

## 5. Điều kiện tối thiểu trước live trajectory

1. Implement và đăng ký `SpreadsheetBenchPythonOJAdapter` cho exact adapter ID.
2. Adapter materialize **chỉ public files**; fixture allowlist rỗng; private case 3 không vào episode workspace/context/reader/error.
3. Implement typed tools theo role identity, không suy tool từ keyword role name.
4. Candidate writer lưu immutable source versions; rejected candidate không thay applied pointer.
5. Public execution chạy exact candidate trên clean copy của case 1, lưu stdout/stderr/output/hash/dependencies và phân loại candidate failure khác infrastructure unavailable.
6. Public validation chỉ dùng public contract/evidence, không đọc case 3 hoặc answers.
7. Implement sandbox cho untrusted Python: containment, input read-only, output directory writable, no network, timeout, output-size limit và process cleanup.
8. Implement trusted final checker chạy **cùng candidate hash** trên clean case 1 và 3, recalculate output, rồi OJ compare `RNM!A1:J20`.
9. Checker trả per-case PASS/FAIL/UNAVAILABLE, soft score coverage và hard pass; không phản hồi private result vào episode.
10. Add structured-data role pool vào runtime loader và bảo đảm shared actor codec render/parse Python source + typed arguments đúng.
11. Mở live allowlist/evaluator dispatch chỉ sau positive/negative, stale evidence, rejected candidate, visibility và budget tests.
12. Chạy fixed-sequence environment smoke và native executor I/O smoke; giữ riêng khỏi benchmark performance.
13. Freeze packet, pool, adapter, sandbox image/profile, checker, model revision, budget và projection trước khi so decision models.

## 6. Khi nào kết quả có ý nghĩa đánh giá và so sánh

Một live trajectory trên packet này sau khi gate pass chỉ có giá trị **smoke/qualitative case study**. Nó chưa đủ để kết luận performance của decision model cho domain.

So sánh decision models có ý nghĩa khi:

- dùng cùng environment freeze và predeclared packet cohort;
- cùng executor backbone/revision/decoding, role pool, public projection, max 8 non-STOP calls, 16K output cap và guard off;
- cùng checker/runtime/sandbox và case coverage;
- giữ raw probability/action/dispatch/tool/model metadata;
- loại budget-suppressed final non-STOP khỏi primary routing denominator;
- báo riêng task hard pass, soft score, checker coverage, workflow+STOP, rejected actions, artifact freshness/handoff và human-label coverage;
- không bỏ failed packets hoặc chọn attempt pass sau khi xem outcome.

Để có domain-level estimate, cần migrate và preflight một cohort được chốt trước (lý tưởng là đủ 20 structured-data packets). Một packet không đủ để so accuracy giữa routers.

## 7. Executor backbone đề xuất

Primary baseline: **homogeneous `Qwen/Qwen3.5-9B` cho cả 10 roles**.

Lý do:

- đã dùng trong shared database baseline, giảm confound do đổi executor giữa domains;
- 9B phù hợp mục tiêu SLM executor;
- model card chính thức công bố khả năng coding, instruction following, long context và tool calling;
- cùng backbone cho mọi role khiến thí nghiệm tập trung vào role selection/coordination, không biến router thành model-selector ẩn.

Cấu hình baseline đề xuất:

```yaml
executor:
  model: qwen/qwen3.5-9b
  mode: non-thinking
  max_tokens: 16384
  timeout_seconds: 300
  transport_retries: 5
  max_tool_calls: 3
```

Giữ decoding/provider profile cố định giữa decision models và pin exact model revision trong freeze. Không cấp backbone identity cho router. Native I/O smoke chỉ kiểm tra model có tạo đúng JSON/tool arguments/Python artifact hay không; không dùng smoke pass để lọc packet.

Nếu 9B không đạt representation contract trên smoke, báo đó là executor-backbone limitation. Có thể tạo track riêng với backbone lớn hơn, nhưng không âm thầm thay model theo role hoặc packet trong primary comparison.

## 8. Kết quả implementation, preflight và live pilot

### 8.1. Runtime đã được bổ sung

- exact adapter `spreadsheetbench.python_oj.v1` đã được đăng ký;
- runtime loader có đủ 10 role card của structured-data domain;
- shared live runner đã cho phép exact adapter này với full pool + `STOP`, guard tắt;
- candidate được AST-check và chạy trong subprocess OpenPyXL bị giới hạn đường dẫn input/output, timeout và output size;
- episode materialization chỉ chứa public files;
- final evaluator chạy cùng candidate hash trên case 1 và case 3, không phản hồi private result vào episode;
- config pilot pin Jev `typesafe/jev-1.13`, Qwen `qwen/qwen3.5-9b`, 16K output cap, max 8 non-STOP calls.

Restricted worker là containment profile dành cho packet này, không được mô tả như một general-purpose OS sandbox. Nó cấm import ngoài OpenPyXL, private/dunder access và arbitrary file loading; parent process áp timeout. Các packet structured-data rộng hơn vẫn cần audit theo library/runtime surface riêng.

### 8.2. Deterministic preflight

Preflight: **PASS**.

- manifest: 5 public, 4 private, 2 fixture artifacts, checksum hợp lệ;
- public-only materialization: PASS;
- reference candidate: public PASS, terminal PASS, private case 1 PASS, private case 3 PASS;
- independent final evaluator: PASS, soft score `1.0`, cùng candidate hash cho cả hai case;
- negative control copy `Desired outcome`: case 1 PASS nhưng case 3 FAIL;
- forbidden import candidate: bị từ chối và không thay current candidate.

Do graded region không chứa formula, bước recalculation được ghi `not_required` cho exact packet này; không suy rộng trạng thái đó sang packet có formula.

### 8.3. Live trajectory 2026-11-01

Run ID: `structured_data_250_20_jev_qwen35_9b_live_20261101_a`.

Kết quả:

| Chỉ số | Giá trị |
|---|---:|
| Workflow terminal | FAIL — `CALL_LIMIT` |
| Independent task audit | FAIL — không có candidate được submit |
| Primary routing decisions | 8 |
| Budget-suppressed final decision | 1 |
| Dispatched role calls | 7 |
| Precondition-rejected decisions | 1 |
| Executor outcomes | 1 success, 6 invalid output |
| Infrastructure-affected episodes | 0 |
| Source freeze | PASS — unchanged |

Decision sequence:

1. `Workbook Surveyor` — dispatch, inspection thành công.
2. `Table Reshaping Specialist` — dispatch, candidate bị từ chối vì syntax error.
3. `Table Reshaping Specialist` — dispatch, không tạo accepted tool call.
4. `Table Reshaping Specialist` — dispatch, candidate bị từ chối vì syntax error.
5. `Workbook Test Executor` — precondition rejected vì chưa có accepted candidate.
6. `Table Reshaping Specialist` — dispatch, candidate bị từ chối vì syntax error.
7. `Table Reshaping Specialist` — dispatch, không tạo accepted tool call.
8. `Table Reshaping Specialist` — dispatch, không tạo accepted tool call.
9. `Table Reshaping Specialist` — quyết định cuối khi budget bằng 0, bị suppress và không tính là executed routing step.

Ba candidate đi tới `write_solution` đều chứa lỗi `ws = wb['RNM']n`; chúng còn import `collections.defaultdict` ngoài allowlist đã công bố. Rejected attempts được lưu riêng, không làm xuất hiện current candidate. Vì vậy không có public execution, validation hoặc private case execution nào đối với output của model.

### 8.4. Attribution

Pilot này cho thấy environment có thể ghi nhận và phân biệt đầy đủ router decision, precondition rejection, executor invalid output, rejected candidate, budget suppression và final task FAIL. Thất bại task chủ yếu phát sinh ở **executor representation/candidate generation**: Qwen không tạo được Python artifact hợp lệ dù role selection ban đầu phù hợp. Router có một lỗi riêng ở decision 5 khi chọn test executor trước khi tồn tại candidate; các lần quay lại author role sau đó là recovery hợp lý trong agent pool hiện tại.

Một packet chỉ là integration pilot, chưa phải estimate về performance của Jev hay domain. Không nên dùng run này để kết luận comparative routing quality trước khi cohort 20 packet được migrate, preflight và freeze cùng cấu hình.
