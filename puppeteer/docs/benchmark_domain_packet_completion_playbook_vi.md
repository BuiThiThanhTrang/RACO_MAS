# Playbook hoàn thiện packet và domain cho RoutingBench

Phiên bản: **1.0 — 2026-11-02**  
Chuẩn tham chiếu: **structured data processing / SpreadsheetBench Python OJ**  
Phạm vi: mọi domain được bổ sung vào benchmark dùng shared multi-agent environment.

Tài liệu này chuyển kinh nghiệm hoàn thiện 20 packet structured data thành quy trình vận hành dùng
chung. Nó bổ sung cho `domain_extension_packet_adapter_artifact_standard_vi.md`: tài liệu chuẩn mở rộng
định nghĩa kiến trúc và contract, còn playbook này quy định thứ tự triển khai, evidence cần thu và điều
kiện để một packet/domain được gọi là hoàn thiện.

## 1. Mục tiêu và non-goal

Mục tiêu bất biến là tạo môi trường chung để có thể thay decision model rồi đánh giá khả năng điều phối
nhiều executor giải task nhiều bước. Khi so router, packet, executor backbone, role pool, public state,
budget, adapter và evaluator phải được giữ cố định.

Môi trường phải:

- cung cấp đủ thông tin và công cụ thật để task có thể được giải;
- giữ hậu quả của mọi action và recovery trong state/trace;
- có evaluator độc lập, phân biệt task failure với infrastructure failure;
- không lộ private oracle hoặc routing label;
- không quy định trước một workflow duy nhất.

Môi trường không có nhiệm vụ tối ưu Jev, một executor hay một trajectory cụ thể. Không thêm hint, role
order, auto-repair hoặc hidden action mask chỉ để tăng task pass. Task khó và executor yếu có thể tạo
floor effect; đó là kết quả cần báo, không phải lý do âm thầm sửa benchmark.

## 2. Năm mức readiness phải phân biệt

| Mức | Ý nghĩa | Chưa được suy ra |
|---|---|---|
| `packet-constructed` | Packet có manifest, source, public/private files và contract | Chưa chứng minh checker đúng hoặc runtime chạy được |
| `deterministic-ready` | Positive witness, negative controls, isolation, freshness và final evaluator pass | Không phải performance của router/executor |
| `live-ready` | Shared runner, role codec, provider I/O, trace và budget semantics đã smoke thật | Chưa có human routing labels |
| `domain-admitted` | Cohort cố định đủ độ phủ, state có giá trị routing, live audit hợp lệ và metric policy rõ | Chưa phải leaderboard result |
| `benchmark-frozen` | Packet/config/runtime/adapter/checker/pool đã đóng băng; mọi router chạy lại trên cùng freeze | Không cho phép ghép tùy ý run khác engine version |

Không dùng từ “benchmark-ready” nếu mới chỉ validate manifest. Human review, execution readiness và
routing labels là ba trạng thái độc lập; tool pass không tự đổi nhãn thành `human_verified`.

## 3. Kiến trúc chuẩn

```text
source record
    ↓ audit + adaptation log
packet (public / private / fixtures)
    ↓ materialize public-only
adapter workspace
    ↓ public observation
decision model → role → executor → typed tool
    ↓
immutable artifact + event + observable consequence
    ↓ repeated routing
final submitted artifact
    ↓ clean trusted evaluator
PASS / FAIL / UNAVAILABLE
```

Ba đối tượng phải tách biệt:

1. **Packet:** task, source, input, contract, visibility và oracle material.
2. **Adapter/environment:** state transition, tool execution, artifact lineage và checker bridge.
3. **Decision model:** chỉ chọn role hoặc `STOP`; không được đọc oracle hoặc tự gọi tool domain.

Executor thực hiện role được chọn, không tự chọn role kế tiếp. Final evaluator không tham gia episode và
không phản hồi private result vào router.

## 4. Quy trình hoàn thiện packet

### Bước 1 — Chốt selection trước live outcome

- Xác định domain cohort, số packet và các archetype trước khi chạy router.
- Structured data dùng 20 packet, năm archetype, bốn packet/archetype. Domain khác có thể dùng taxonomy
  riêng nhưng phải có coverage policy tương đương.
- Ghi selection manifest và `selected_before_live_trajectory`.
- Không thay packet fail bằng packet khác sau khi xem model outcome.
- Không bắt buộc global train/test split. Một packet vẫn có thể có public development case và private
  evaluator variants để kiểm tra reusable artifact.

Selection archetype chỉ phục vụ audit/stratification; không đưa vào router observation hoặc role prompt.

### Bước 2 — Khóa source và provenance

Mỗi packet phải lưu:

- dataset/repository và canonical revision;
- record ID, split, license;
- archive/member path và hash;
- instruction gốc và mọi adaptation/metadata repair;
- bằng chứng semantics không bị đổi ngoài phần correction đã công bố.

Nếu hai phiên bản nguồn khác nhau, không chỉ so bytes. Phải so sheet/file/schema/value/formula/test
semantics và phân loại:

1. serialization-only;
2. metadata correction;
3. corrected task semantics;
4. incompatible source version.

Ví dụ `13-1`: Verified-400 sửa task sang sort theo date rồi REF, còn answer archive cũ giữ thứ tự xuất
hiện. Private inputs cũ vẫn dùng được như input variants, nhưng answer cũ không còn là oracle. Checker
phải tự suy expected result theo corrected semantics và provenance phải ghi rõ điều này.

### Bước 3 — Định nghĩa output/candidate contract

Phân biệt ba tầng:

1. **Candidate acceptance:** artifact đúng representation/edit contract.
2. **Public execution/validation:** artifact chạy được và thỏa public semantic contract.
3. **Independent task evaluation:** exact submitted artifact pass mọi trusted checks.

Structured data chuẩn hóa candidate thành:

```python
def solve(input_path: str, output_path: str) -> None:
    ...
```

Domain khác thay bằng artifact tự nhiên của domain, ví dụ SQL migration, source patch hoặc Lean proof.
Không yêu cầu mọi domain dùng Python, nhưng mọi candidate phải có entrypoint/edit scope và acceptance
policy kiểm tra được.

Accepted candidate không đồng nghĩa task solved. Compile thành công cũng không đồng nghĩa proof đúng;
workbook mở được không đồng nghĩa transformation đúng; SQL chạy được không đồng nghĩa invariant được giữ.

### Bước 4 — Viết semantic task contract

Contract phải chỉ rõ:

- observable goal và terminal condition;
- initial public state;
- editable và immutable scope;
- invariants/preservation requirements;
- value, formula, ordering, style hoặc schema semantics;
- runtime, command, timeout, output size và network policy;
- failure classification;
- trusted evaluator và PASS/FAIL/UNAVAILABLE policy.

Không chấm bằng exact reference artifact nếu có nhiều lời giải đúng. Reference implementation là witness
kiểm tra môi trường, không phải output duy nhất model phải sao chép.

### Bước 5 — Chia public/private/fixture

| Lớp | Agent được thấy? | Nội dung |
|---|---|---|
| Public | Theo projection | Task, initial input, execution/validation contract cần thiết |
| Private | Không | Hidden cases, oracle, final checker material |
| Fixture | Chỉ allowlist rõ ràng | Provenance, license, source locator; không mặc định public |
| Runtime artifact | Theo role/reader policy | Candidate, inspection, execution và validation evidence |

Phải test workspace thật sau `materialize`, prompt/context thật và artifact reader. Tên thư mục `private`
không tự bảo đảm isolation nếu inherited materializer hoặc arbitrary-path reader vẫn đọc được nó.

### Bước 6 — Thiết kế role pool không ép workflow

Mỗi role contract có tối thiểu:

- `agent_id` và một responsibility rõ;
- typed tools thực sự được implementation;
- preconditions;
- read/write artifacts;
- allowed mutations;
- observable failure consequences.

Không cần trường `weakness`. Không đưa recommended route, fixed subgraph hoặc selection archetype vào
role card. Full role pool cộng `STOP` phải hiện diện ở mọi decision state; guard và valid-action mask tắt
trong primary dynamic-routing track.

Role có thể dùng chung backbone, nhưng role identity/tool surface phải khác đúng capability. Tên role
không được quảng bá compile/test/repair nếu implementation chỉ sinh prose.

### Bước 7 — Implement state đủ thông tin

Router observation phải có:

- public task và current public state;
- full agent pool + `STOP`;
- budget còn lại;
- artifact summary, version/hash/freshness;
- public failures, rejected-attempt count và latest execution summary;
- recent public events đã projection.

Router không cần raw input lớn nếu role inspection có reader thật. Không đưa private result, reference
workflow, valid-action label hoặc selection archetype vào observation.

Executor context theo role phải có:

- role card và typed tool schemas;
- public task/execution/validation contract;
- public source/input hoặc reader đủ để lấy toàn bộ phần cần thiết;
- current candidate/evidence và immutable artifact refs;
- candidate-authoring constraints;
- recent tool evidence cần cho recovery.

Mọi context phải JSON-serializable trước live. Structured data từng bị `ArrayFormula` object làm
`executor_context` lỗi trước khi model được gọi; state-contract validator sau đó được mở rộng để kiểm
tra serialization trên toàn bộ packet.

Preview có giới hạn phải đi kèm bounded range/file reader. `inspect_workbook` và profiler của structured
data hỗ trợ `sheet`/`cell_range`, nên dữ liệu ngoài preview 200 hàng vẫn đọc được mà không lộ private case.

### Bước 8 — Version artifact và giữ hậu quả action

Artifact cần logical key, immutable version, file hash, producer, input versions và dependency refs.

- Candidate mới không overwrite candidate cũ.
- Candidate bị reject được ghi trong rejected history nhưng không thay current accepted pointer.
- Candidate mới được accept làm execution/validation cũ stale.
- Validation chỉ terminal khi fresh với exact candidate/output/checker version.
- Automatic normalization, cleanup hoặc transaction rollback phải có event/evidence.

Phân biệt transaction rollback với router-selected recovery. Rollback tự động không được tính như một
routing action. Recovery chỉ được tính khi router chọn role và tool recovery thật sự chạy. Không reset
state âm thầm để che hậu quả của action sai.

### Bước 9 — Implement final evaluator độc lập

Evaluator phải:

1. chạy sau episode trong clean workspace;
2. lấy exact current submitted artifact và hash;
3. chạy trên toàn bộ trusted cases/tests/invariants;
4. trả per-case/check evidence;
5. phân biệt `FAIL` với `UNAVAILABLE`;
6. không sửa trajectory hoặc cung cấp hidden feedback cho episode.

Nếu packet có nhiều trusted variants, cùng candidate hash phải chạy trên tất cả. Nếu một upstream answer
sai, loại/correct trước live với evidence; không thay oracle sau khi thấy model output.

### Bước 10 — Materialize packet và validate manifest

Builder phải là nguồn authoritative; không sửa generated packet thủ công mà không cập nhật builder.
Sau build:

- validate checksum/path/file inventory;
- xác nhận adapter ID và generator version;
- xác nhận public/private case inventory;
- cập nhật release-level selection/validation reports;
- giữ backup hoặc release version mới khi thay packet đã materialize.

Manifest validation là gate cấu trúc, không thay semantic preflight.

## 5. Deterministic preflight bắt buộc

Mỗi packet phải có tối thiểu các witness sau:

| Witness | Kết quả mong đợi |
|---|---|
| Positive reference candidate | Public và mọi trusted case PASS |
| Identity/no-op negative | Semantic checker FAIL |
| Shortcut/hard-code negative khi phù hợp | Có thể pass public nhưng phải fail hidden variant |
| Forbidden operation/import/path | Bị reject, current candidate không đổi |
| Rejected candidate pointer | Accepted pointer trước đó còn nguyên |
| Stale evidence | Candidate mới làm execution/validation cũ stale |
| Public-only materialization | Không có private/fixture ngoài allowlist |
| Same-candidate final audit | Cùng hash trên mọi trusted case |
| Role-tool conformance | Tool runtime khớp role contract |
| Context serialization/readability | Executor nhận đủ dữ liệu, context encode được |

Positive witness chứng minh task/environment giải được, không phải điểm executor. Negative witness chứng
minh checker có discriminative power. Chỉ positive pass mà identity cũng pass là evaluator lỗi.

Ở cấp domain, preflight thêm:

- 100% packet manifest valid;
- full pool + `STOP`, guard off, không workflow graph/action mask;
- router/private visibility scan;
- range/file reader vượt được preview mặc định;
- common runner, evaluator dispatch và trace validator nhận adapter mới.

## 6. Live trajectory protocol

### 6.1. Freeze trước run

Pin ít nhất:

- packet cohort và selection manifest;
- adapter/checker/runtime revisions;
- role cards/tool schemas;
- router model/revision/profile;
- executor model/revision/decoding;
- max tokens, timeout, retry và max tool calls;
- max non-STOP calls, seed, guard policy và public projection.

Structured-data pilot dùng Jev, Qwen3.5-9B, 16K output cap, 8 executor calls, full pool và guard tắt.
Đây là một baseline configuration, không phải yêu cầu mọi domain dùng cùng toolchain.

### 6.2. Budget accounting

Runner giữ flow quyết định tới lần hỏi cuối ở budget 0:

- `STOP`: xử lý bình thường;
- non-STOP: ghi `suppressed_by_budget`, không dispatch;
- suppressed decision giữ trong raw trace nhưng loại khỏi primary routing denominator.

Không tạo tool result giả cho decision không thực thi. Precondition rejection sau selection vẫn là primary
routing decision và vẫn tiêu tốn call nếu protocol đã quy định như vậy.

### 6.3. Rerun policy

Được rerun khi có evidence rõ về:

- trace bị mất/hỏng;
- provider/router transport failure trước outcome hợp lệ;
- tool/runtime `UNAVAILABLE`;
- harness bug khiến model không nhận được context hoặc evaluator không chạy.

Rerun phải có attempt/run ID mới, giữ attempt cũ và ghi exclusion reason. Không rerun vì task fail,
route xấu hoặc executor output sai rồi chỉ chọn attempt pass.

Nếu sửa adapter giữa pilot, pilot ghép có thể dùng để quyết định environment readiness nhưng không dùng
làm leaderboard. Official router comparison phải chạy lại mọi router từ đầu trên cùng final freeze.

## 7. Human review

Review ở hai cấp:

### Packet/domain review

- Task có đủ thông tin không?
- Contract/oracle có đúng nguồn và không quá chặt không?
- Role implementation có đúng mô tả không?
- Private isolation và failure attribution có đúng không?
- Task/archetype có tạo quyết định routing thật không?

### Decision-state review

- Action đã chọn có hợp lý trong state đó không?
- Có nhiều action hợp lệ không?
- Recovery có dựa trên fresh evidence không?
- Router test/validate/STOP quá sớm hay quá muộn không?
- Action có tạo semantic progress hay chỉ lặp không tiến triển?

Human labels gắn với exact run ID + decision ID, không chỉ packet ID. Budget-suppressed decision không
đưa vào primary action-accuracy denominator. Không dùng reference workflow làm một gold route duy nhất.

## 8. Metric ladder dùng chung

Mọi domain phải báo riêng từng tầng sau:

| Tầng | Ý nghĩa |
|---|---|
| Packet count/archetype coverage | Độ phủ task đã chốt |
| Accepted candidate | Artifact đúng representation contract |
| Public execution success | Artifact chạy được và tạo output |
| Public validation pass | Output thỏa public semantic contract |
| Correct `STOP` | Router kết thúc khi evidence terminal fresh |
| Trusted case coverage | Số hidden/trusted checks pass |
| Trusted hard pass | Tất cả trusted checks của packet pass |
| Primary routing decisions | Decisions thực sự dùng phân tích, loại budget-suppressed |
| Human action/recovery score | Chất lượng routing theo state |
| Efficiency | Calls tới milestone, rejection, loop và wasted budget |

Không gộp các tầng thành một biến “success”. Candidate accepted nhưng runtime fail khác với không có
candidate; public pass nhưng private fail khác với complete task success.

Khi hard pass bị floor hoặc ceiling, vẫn giữ hard pass nhưng không dùng nó làm metric duy nhất. Dùng
metric bundle gồm trusted outcome, progress milestones, human routing labels và routing efficiency.

## 9. Điều kiện admission cấp domain

Một domain đạt `domain-admitted` khi:

- cohort và difficulty/archetype policy được chốt trước outcome;
- packet deterministic-ready và runtime coverage đầy đủ;
- state đủ thông tin, private isolation pass;
- role pool/tool implementation thật, không fixed workflow;
- final evaluator chạy cho toàn bộ packet hoặc báo `UNAVAILABLE` minh bạch;
- ít nhất một live pilot tạo trace hợp lệ và source freeze pass;
- trajectory có state transitions đủ để đánh giá routing/recovery;
- metric policy xử lý floor/ceiling và infrastructure exclusions;
- hạn chế, human-label coverage và version boundaries được ghi rõ.

Domain không cần baseline router đạt accuracy cao để được admit. Nhưng nếu executor failure làm mọi route
không tạo bất kỳ milestone hoặc state transition nào, domain chưa có giá trị so router và cần xem lại
representation/runtime sufficiency — không sửa task để chiều model.

## 10. Structured data làm reference implementation

### 10.1. Deterministic evidence

- 20 packet, đủ năm archetype;
- toàn bộ dùng exact adapter `spreadsheetbench.python_oj.v1`;
- public case 1 và private source-native variants theo packet contract;
- positive reference, identity negative, forbidden import, rejected-pointer và stale-evidence checks;
- state-contract audit PASS 20/20, gồm JSON serialization và range reader;
- role-tool contract, full pool + `STOP`, guard off và private isolation PASS.

### 10.2. Live evidence

Tổng hợp pilot đủ 20 packet:

| Chỉ số | Kết quả |
|---|---:|
| Accepted candidate | 11/20 |
| Public execution thành công | 6/20 |
| Public validation pass + `STOP` | 2/20 |
| Trusted hard pass | 1/20 |
| Trusted case pass | 3/59 |
| Primary routing decisions | 156 |

Floor effect mạnh xuất phát chủ yếu từ executor protocol/candidate correctness. Tuy vậy trajectory có
inspection, authoring, cross-sheet work, profiling, testing, validation, repair, precondition rejection,
missing STOP và budget exhaustion. Vì thế domain được admit như routing/coordination domain, với điều
kiện official comparison dùng final patched freeze và metric bundle, không dùng hard pass đơn lẻ.

## 11. Ánh xạ sang các domain khác

| Khái niệm chung | Structured data | Database | Code debugging | Formal proof |
|---|---|---|---|---|
| Candidate | Python workbook transform | SQL/migration artifact | Source patch | Lean source/proof |
| Public execution | Chạy transform trên workbook | Apply/run trên DB copy | Build/test subset | Lean compile |
| Public validation | Semantic workbook checker | Schema/data/invariant checks | Public tests/static checks | Kernel/goal checks |
| Private/trusted evaluation | Hidden workbook cases | Hidden invariant/query cases | Trusted test suite | Original theorem context/kernel |
| Preservation | Unrelated cells/styles/sheets | Unrelated schema/data | Unrelated files/behavior | Assumptions/context/no sorry |
| Recovery | Revise solution từ execution evidence | Repair SQL/restore có log | Patch revision | Proof revision |

Domain mới không cần sao chép role names của structured data. Nó phải giữ cùng **contract semantics**:
typed capability, immutable artifact, fresh evidence, observable consequence và independent checker.

## 12. Checklist triển khai

### Packet construction

- [ ] Selection/archetype policy chốt trước live.
- [ ] Source revision/license/member hashes khóa.
- [ ] Instruction, answer/test metadata và version differences được audit.
- [ ] Output/candidate contract rõ và không đổi task semantics.
- [ ] Public/private/fixture inventory và projection khai báo.
- [ ] Builder tái tạo packet; manifest/hash/path validate.

### Adapter và state

- [ ] Exact adapter đăng ký trong registry, runner và evaluator dispatch.
- [ ] Role cards khớp typed tool implementation.
- [ ] Router observation đủ task/state/artifact/failure/budget/pool.
- [ ] Executor context đủ public input/reader/contracts/artifacts.
- [ ] Context JSON-serializable và bounded reader đọc được phần bị truncate.
- [ ] Accepted/rejected/history/freshness semantics được implement.
- [ ] Automatic side effects và rollback có trace.

### Evaluator và preflight

- [ ] Positive witness pass mọi trusted check.
- [ ] Identity/shortcut negative bị reject.
- [ ] Forbidden operation không thay current candidate.
- [ ] Candidate mới làm evidence cũ stale.
- [ ] Private isolation test trên workspace/context/reader/error.
- [ ] Final checker trả PASS/FAIL/UNAVAILABLE và cùng candidate hash.

### Live và admission

- [ ] Config/model/runtime/pool/budget/projection freeze.
- [ ] Trace state-decision-execution join pass.
- [ ] Source unchanged trong run.
- [ ] Infrastructure episodes tách khỏi task failure.
- [ ] Rerun có attempt ID/exclusion reason, không cherry-pick pass.
- [ ] Metric ladder và denominator exclusions được báo.
- [ ] Human review queue được tạo; label coverage báo riêng.
- [ ] Admission decision ghi limitation và version boundary.

## 13. Artifact và tài liệu cần phát hành

Một domain hoàn thiện nên có:

```text
docs/<domain>_packet_completion_guide_vi.md
docs/<domain>_live_trajectory_review_<date>_vi.md
docs/<domain>_freeze_report_vi.md
config/shared_environment/<domain>_<router>_<executor>.yaml
data/RoutingBench/<release>/packs/<domain>/...
data/RoutingBench/<release>/agent_pools/<domain>.json
scripts/preflight_<domain>_packets.py
scripts/validate_<domain>_state_contract.py
runs/<run-id>/
  resolved_config.json
  source_hashes.json
  freeze_verification.json
  execution_validation_report.json
  evaluation_report.json
  human_state_review_queue.jsonl
```

Biên bản nghiệm thu dùng `docs/templates/domain_extension/ACCEPTANCE.md`. Packet design worksheet dùng
`docs/templates/domain_extension/packet_spec.template.json`. Hai template là điểm bắt đầu; evidence và
hash thực tế phải do builder/preflight tạo, không điền PASS thủ công.

## 14. Quy tắc cuối cùng khi so router

1. Chọn một final environment freeze.
2. Chạy mọi decision model từ đầu trên cùng packet cohort và executor configuration.
3. Không dùng pilot ghép khác source hash làm leaderboard result.
4. Không thay sample, role card, token cap hoặc checker sau khi xem router outcome.
5. Báo task outcome, progress milestones, human routing quality và efficiency riêng.
6. Giữ raw trace, probability/action metadata, rejected attempts và infrastructure exclusions.
7. Mọi environment change tạo version/release mới; không ghi đè historical run.

Tuân thủ playbook này giúp domain mới đo đúng năng lực điều phối của decision model trong một môi trường
đủ thông tin, thay vì đo khả năng đoán hidden contract hoặc hưởng lợi từ workflow được thiết kế sẵn.
