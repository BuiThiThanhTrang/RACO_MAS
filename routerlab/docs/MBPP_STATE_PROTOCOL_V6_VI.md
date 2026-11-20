# MBPP+ protocol v6: state, artifact và executor protocol

Trạng thái: **đã triển khai track end-to-end**. Track controlled recovery và frozen
candidate bank tạm hoãn, chưa thuộc implementation hiện hành. Protocol v5 và các run
`full-*-v5-*` vẫn được giữ bất biến để tái lập kết quả.
Đọc cùng [Harness và luồng thông tin v6](MBPP_HARNESS_INFORMATION_FLOW_V6_VI.md).

Thiết kế này tiếp thu các nguyên tắc artifact-first, freshness, role progress và
bounded handoff từ domain formal proof, nhưng không sao chép role hay tool contract
của Lean. Trong MBPP+, Base Tests vẫn là transition tự động của harness; router chỉ
chọn một executor hoặc `STOP`.

## 1. Mục tiêu và bất biến

V6 phải bảo đảm:

- router chỉ chọn `coder`, `logic_reviewer`, `edge_case_analyst`, `repair_agent`
  hoặc `STOP`;
- không action masking và không ép pipeline;
- mọi non-STOP action được chọn đều tốn một role step, kể cả action bị từ chối do
  precondition;
- code thay đổi được chạy Base Tests tự động;
- Plus Tests chỉ chạy trong final evaluation và không đi vào online state;
- không rollback hoặc chọn ngầm code tốt nhất;
- cùng một public state projection và capability card cho mọi router;
- executor dùng chung backbone, checkpoint và decoding trong một comparison track.

## 2. Thay đổi chính so với v5

| V5 | V6 end-to-end |
|---|---|
| Chỉ giữ current code và hai tuple findings | Artifact registry có latest/history, input refs và freshness |
| Findings bị xóa khi code đổi | Giữ history, đánh dấu stale; chỉ evidence fresh được handoff mặc định |
| So sánh source string | Raw hash và Python presentation fingerprint |
| Base feedback chủ yếu là status/count | Base-execution artifact có failure class/signature và diagnostic đã làm sạch |
| Action history phẳng | Recent events và role progress có semantic delta/stagnation |
| Không có input context cap thực thi | Context packer đo token, reserve output và cắt deterministic |
| Lỗi generation chủ yếu nằm trong exception/log | Typed generation/contract/tool failure trong public state phù hợp |

## 3. State authoritative nội bộ

`EpisodeStateV6` là nguồn state authoritative của episode:

```text
task_contract                 bất biến
candidate                     current candidate ref hoặc null
artifacts.latest              artifact mới nhất theo kind
artifacts.history             metadata của mọi artifact đã tạo
latest_execution              Base execution gần nhất
latest_generation_failure     lỗi generation gần nhất
role_progress                 tiến triển và stagnation theo role
events                        event log append-only
budget                        role, Base-execution và wall-time budget
terminal                      trạng thái kết thúc của online workflow
```

Source hiện tại không được suy ra từ raw response gần nhất. Chỉ candidate đã qua parser
và edit-contract validation mới có thể trở thành `candidate` hiện tại.

### 3.1 Task contract

```json
{
  "task_id": "Mbpp/468",
  "prompt": "<canonical EvalPlus prompt>",
  "entry_point": "<required function>",
  "language": "python",
  "dataset_hash": "<pinned hash>",
  "evaluation_policy": "base-online-plus-final-v1"
}
```

Không materialize `base_input`, `plus_input`, expected output, canonical solution hoặc
special-oracle data vào public workspace.

## 4. Artifact envelope

Các kind chuẩn:

- `candidate`;
- `base_execution`;
- `logic_review`;
- `edge_case_analysis`;
- `rejected_candidate`;
- `generation_failure`.

Mỗi artifact có envelope:

```json
{
  "artifact_id": "0007_base_execution",
  "kind": "base_execution",
  "version": 7,
  "sha256": "...",
  "producer_action": "automatic_base_evaluator",
  "producer_step": 3,
  "input_versions": {"candidate": 2},
  "input_refs": {"candidate_sha256": "..."},
  "visibility": "public | private",
  "created_at_event": 12,
  "fresh": true,
  "content": "<kind-specific payload>"
}
```

`artifact_id` mới không tự chứng minh tiến triển. Freshness phụ thuộc exact input refs,
không phụ thuộc việc artifact đang là latest hay không.

## 5. Candidate identity

Mỗi candidate lưu hai identity:

1. `raw_sha256`: thay đổi khi bất kỳ byte nào thay đổi;
2. `presentation_fingerprint`: nhận diện candidate giống nhau về cấu trúc trình bày.

Fingerprint đề xuất:

- nếu parse Python thành công: hash của `ast.dump(tree, include_attributes=False)`;
- nếu parse thất bại: fingerprint dựa trên Python tokenizer;
- nếu tokenizer cũng thất bại: exact-source fallback.

Fingerprint chỉ dùng phát hiện duplicate/stagnation. Nó không chứng minh hai chương
trình tương đương ngữ nghĩa và không được dùng để reuse kết quả Base Tests.

## 6. Freshness và provenance

Quy tắc freshness:

- `base_execution` fresh khi candidate version và raw hash trùng candidate hiện tại;
- `logic_review` và `edge_case_analysis` fresh khi candidate ref trùng, hoặc khi được
  tạo ở trạng thái chưa có candidate và chỉ khai báo `scope=task_specification`;
- candidate mới làm evidence gắn với candidate cũ thành stale nhưng không xóa history;
- evidence stale vẫn có thể hiển thị metadata cho router, nhưng executor không nhận
  nội dung stale mặc định;
- final evaluator không dùng review/analysis để chấm, chỉ dùng submitted source.

Việc giữ stale history giúp audit xem router có lặp lại lỗi cũ hay không, đồng thời
tránh việc v5 xóa mất mọi dấu vết chẩn đoán sau một lần edit.

## 7. Public observation cho router

Router nhận một projection ổn định:

```json
{
  "task": "<public task contract>",
  "current_state": {
    "candidate": {
      "exists": true,
      "version": 2,
      "raw_sha256": "...",
      "presentation_fingerprint": "...",
      "preview": "<bounded source preview>",
      "visible_chars": 1800,
      "total_chars": 1800,
      "truncated": false
    },
    "latest_base_execution": "<public projection or null>",
    "artifacts": "<latest metadata/projection by kind>",
    "latest_generation_failure": "<typed public failure or null>",
    "role_progress": "<per-role counters and last result>",
    "recent_events": ["<bounded semantic events>"],
    "terminal": false
  },
  "agent_profiles": "<canonical capability cards>",
  "budget": "<remaining role/Base/wall-time budget>",
  "constraints": {
    "allowed_actions": [
      "coder", "logic_reviewer", "edge_case_analyst", "repair_agent", "STOP"
    ]
  }
}
```

Router không nhận raw model response, chain-of-thought, private checker output, Plus
result hoặc source của version cũ. `recent_events` giữ action, status, origin,
input/output refs và semantic delta; không nhúng raw executor output.

## 8. Base-execution artifact công khai

Feedback v6 vẫn không công bố test input hay expected/actual output. Public payload:

```json
{
  "candidate_version": 2,
  "candidate_sha256": "...",
  "status": "pass | fail | timeout | runtime_error | invalid",
  "passed": false,
  "passed_tests": 1,
  "total_tests": 3,
  "failure_count": 2,
  "failure_kind": "assertion | exception | timeout | contract | unknown",
  "failure_signature": "<stable hash of sanitized failure class/location>",
  "diagnostic": {
    "exception_class": "TypeError",
    "generated_code_line": 7,
    "message": "<sanitized, bounded message>"
  }
}
```

`diagnostic` không chứa argument values, assertion source, expected/actual values hoặc
Plus evidence. Raw checker output được lưu private để audit. Cần chạy ablation
`counts_only_v1` và `sanitized_diagnostic_v2` để tách đóng góp của richer feedback khỏi
đóng góp của router.

## 9. Executor context và automatic handoff

Executor được chọn nhận:

- role card của chính role đó;
- public task contract;
- current candidate đầy đủ nếu vừa input cap, nếu không là bounded preview có metadata;
- latest fresh Base-execution artifact;
- latest fresh Logic Review và Edge-case Analysis;
- role progress liên quan: duplicate count, consecutive no-change, generation failures;
- source/evidence refs và freshness.

Executor không nhận router probability/confidence, router raw reasoning, remaining
routing budget, Plus evidence hoặc raw response của executor trước.

Handoff được đóng gói deterministic, không dùng LLM tóm tắt. Thứ tự ưu tiên:

1. metadata, critical failure fields và freshness;
2. current candidate;
3. fresh Base diagnostic;
4. fresh review/edge findings;
5. recent progress events;
6. stale-evidence metadata, không tự chuyển nội dung stale.

Với context window 8.192 và output cap 1.024, input cap mặc định:

```text
input_cap = context_limit - max_output_tokens - framing_reserve
          = 8192 - 1024 - 256
          = 6912 token
```

Tokenizer phải khớp checkpoint. Nếu quá cap, giảm prose/evidence preview dài nhất,
sau đó source preview; không loại critical fields. Vẫn quá cap thì trả
`context_budget_exceeded` trước khi gọi model.

## 10. Output contract theo role

| Role | Output | Precondition |
|---|---|---|
| Coder | Complete Python candidate | Không |
| Logic Reviewer | Structured findings | Không; có thể review specification |
| Edge-case Analyst | Structured edge cases | Không |
| Repair Agent | Complete Python candidate | Candidate phải tồn tại |

Finding nên có schema tối thiểu:

```json
{
  "finding_id": "logic-3-1",
  "category": "contract | logic | complexity | edge_case",
  "claim": "<bounded text>",
  "code_span": {"start_line": 4, "end_line": 8},
  "confidence": "low | medium | high"
}
```

Finding vẫn là `agent_assessment`, không phải checker truth. Parser không tự điền field
thiếu hoặc cứu output xung đột thành artifact hợp lệ.

## 11. Role progress và semantic delta

Mỗi role lưu:

- số lần được dispatch;
- lần gần nhất: status, input refs, artifact refs, source/candidate changed;
- `duplicate_candidate` và `consecutive_duplicate_count`;
- generation/format failure liên tiếp;
- Base passed-count delta sau code action;
- failure-signature changed hay không.

`semantic_delta` chỉ ghi nhận thay đổi có ý nghĩa quan sát được:

- candidate fingerprint thay đổi;
- Base passed count hoặc failure signature thay đổi;
- fresh evidence content hash thay đổi;
- terminal thay đổi.

Tăng counter, artifact ID mới hoặc format-only edit không tự được xem là progress.
Harness không tự đổi role hoặc STOP dựa trên delta; router phải quyết định từ state.

## 12. Typed failure taxonomy

| Origin | Ví dụ |
|---|---|
| `router_service` | API exception/refusal/invalid typed choice |
| `executor_service` | HTTP/auth/transport failure |
| `generation` | repetition, truncation, empty output |
| `contract` | parse failure, invalid entry point, repair without candidate |
| `base_tool` | evaluator timeout/process/OSError |
| `context_budget` | packed input vượt cap |

Public event chỉ chứa typed fields phù hợp; raw exception và provider response nằm ở
audit log. Không gom lỗi hạ tầng vào lỗi thuật toán của candidate.

## 13. Recovery và metric contract

Base recovery:

```text
initial Base fail -> final Base pass
```

Phải báo cáo thêm:

- opportunity count và coverage;
- partial Base improvement/regression;
- recovery theo step và theo action sequence;
- recovered task final Plus-pass count;
- duplicate/no-progress rate;
- stale/fresh evidence utilization;
- marginal recovery của mỗi round/step.

Vì opportunity set phụ thuộc candidate đầu tiên, thiết kế phân biệt hai track:

1. **End-to-end routing (đã triển khai):** router bắt đầu từ empty state; recovery set
   theo từng method;
2. **Controlled recovery (tạm hoãn):** mọi policy nhận cùng frozen initial candidate
   và Base artifact; recovery denominator chung, phù hợp để cô lập routing.

Việc chia full manifest thành cohort 124 common-failure và 254 task còn lại chỉ là
phân hoạch run end-to-end. Nó không biến cohort 124 thành controlled recovery vì mỗi
method vẫn tự tạo candidate và evidence ban đầu.

## 14. Invariant phải test

- không field Plus nào xuất hiện trong router/executor payload hoặc public trace;
- evidence fresh luôn tham chiếu đúng candidate hash/version;
- candidate đổi không xóa artifact history;
- stale review không được handoff như fresh review;
- duplicate candidate không chạy lại Base Tests;
- candidate presentation-only change không reuse Base evidence;
- repair không có candidate bị `action_rejected` nhưng vẫn tốn role step;
- router vẫn thấy full action set tại zero budget;
- submitted source là current candidate, không phải best historical candidate;
- context packer deterministic và không làm mất critical fields.
