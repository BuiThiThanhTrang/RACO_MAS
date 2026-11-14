# RoutingBench V4-alpha.2: thiết kế history-aware và recovery-aware dynamic routing

## 1. Mục tiêu

RoutingBench V4-alpha đánh giá orchestrator chọn **agent tiếp theo** từ task, trạng thái
công khai, role card và (tùy cấu hình) lịch sử thực thi. Benchmark không yêu cầu
orchestrator chia task thành các subtasks. Mỗi workflow là một quá trình nhiều bước có
trạng thái thay đổi sau mỗi lần executor chạy.

V4 giữ các nguyên tắc của V3:

- ground truth là tập các action hợp lệ, không có `preferred_actions` hay
  `action_utilities`;
- không đưa capability score, backbone identity, gold answer hoặc nhãn nguồn gốc lỗi
  vào input của router;
- `STOP` là một terminal decision riêng, không chiếm execution budget;
- routing quality, execution quality và end-task success được báo cáo riêng.

V4 bổ sung hai mục tiêu mới:

1. đo giá trị của structured execution history;
2. đo khả năng phát hiện, kiểm tra, sửa và phục hồi sau lỗi executor, kể cả lỗi tích
   lũy do một quyết định sai ở đầu trajectory.

Các counterfactual được cân bằng theo thông tin router quan sát: cùng workflow family
dùng cùng task, initial state, budget, agent pool và tập action hợp lệ ban đầu. Fault
là kết quả hậu thực thi có kiểm soát của một action, không được tạo bằng cách âm thầm
biến alternate executor hợp lệ thành invalid. `scenario_id` router nhận chỉ định
workflow family (`v4-<domain>-case`), không chứa fault type; ID nội bộ đầy đủ chỉ dùng
cho audit và evaluation sidecar.

Đây là **alpha implementation**. Human review vẫn là `pending`; V4-alpha.2 chưa thay thế
V3-beta đã phát hành.

V4-alpha.2 có ba track bổ trợ nhau:

- **Track A — decision-state routing**: chấm một quyết định độc lập tại state có sẵn;
- **Track B — full trajectory**: router bắt đầu từ đầu và phải tự đi qua toàn workflow;
- **Track C — controlled post-fault recovery**: router bắt đầu ngay sau một fault đã
  quan sát được, kèm matched clean control để đo recovery riêng khỏi khả năng đi tới
  fault state.

Track C không thay thế `recovery_success_rate` của Track B. Track B đo recovery tự
nhiên có điều kiện trên những trajectory thực sự gặp fault nên có thể có coverage bằng
0. Track C chủ động bảo đảm coverage và đo phản ứng sau fault, nhưng không đo khả năng
router tránh lỗi từ đầu.

## 2. Ba cấu hình chính

Mỗi orchestrator chỉ cần chạy ba cấu hình chính:

| Tên | Prompt/harness | Memory router nhìn thấy | Mục đích |
|---|---|---|---|
| `naive_state` | naive | current state | baseline tối thiểu |
| `structured_state` | structured | current state | đo giá trị của role contract và harness |
| `structured_history` | structured | current state + structured history | headline history-aware |

`structured_raw_history` chỉ là diagnostic trên development subset. Nó không phải cấu
hình headline vì raw text làm tăng prompt length và phụ thuộc mạnh vào executor
backbone.

So sánh chính:

- `structured_state - naive_state`: hiệu quả prompting/harness;
- `structured_history - structured_state`: giá trị gia tăng của memory;
- so sánh cùng ba cấu hình trên nhiều router (Jev, Luna Decisions, Sol, ...): khả năng
  tổng quát giữa orchestrator.

## 3. Các view quan sát

### State-only

Router nhận task, public state, artifact inventory, last event, budget và role cards.
Không có lịch sử trước last event.

### Structured history

Memory do evaluation harness tạo, không dùng gold và không gọi LLM summarizer:

- `recent_events`: action, intent/role, observed status, state delta, evidence refs;
- `persistent_memory`: attempts by agent, completed contributions, observed failures,
  failed approaches, available artifacts và unresolved public requirements;
- semantic deduplication dựa trên canonical event fields;
- giữ đầy đủ failure event; không biến post-hoc failure-origin attribution thành input.

### Raw history diagnostic

Router nhận structured history và raw executor outputs gần nhất. Raw outputs được lưu
để audit/replay, nhưng view này chỉ dùng ở development. Với controlled workflows, raw
outputs là deterministic fixtures. Với live Track B, raw output là output thật của
executor và phải được lưu trong run artifact, không ghi đè dataset gốc.

## 4. Domain coverage

V4-alpha thêm sáu domain workflow nhiều bước nhưng không đòi hỏi phân chia subtasks:

1. `transactional_service`: kiểm tra điều kiện, thực hiện thay đổi, xác minh và commit;
2. `personal_device_tool`: đọc trạng thái thiết bị, áp dụng thao tác, kiểm tra hậu điều
   kiện;
3. `cross_app_workflow`: chuyển artefact qua nhiều ứng dụng/tool contract;
4. `structured_data_processing`: đọc bảng, biến đổi/tính toán, kiểm tra schema;
5. `code_debugging`: tái hiện lỗi, áp dụng bản sửa, chạy kiểm tra;
6. `scientific_procedure`: xử lý dữ liệu đo, kiểm tra consistency, kết luận.

Mỗi domain dùng cùng một recovery grammar nhưng task, artefact, executor role và lỗi
quan sát được khác nhau. Split được tách theo domain để test cross-domain routing:

- development: transactional service, structured data;
- validation: cross-app workflow, code debugging;
- test: personal device tool, scientific procedure.

Nguồn tham khảo thiết kế (không sao chép sample hoặc gold data): transactional policy
workflow của [tau-bench](https://arxiv.org/abs/2406.12045), stateful tool interaction
của [ToolSandbox](https://arxiv.org/abs/2408.04682), cross-application workflow của
[AppWorld](https://arxiv.org/abs/2407.18901), multi-turn function calling của
[BFCL](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html), và
interactive scientific procedures của
[ScienceWorld](https://github.com/allenai/ScienceWorld). V4-alpha dùng template do
project tự viết để giữ executor deterministic và provenance kiểm tra được.

## 5. Recovery taxonomy

Mỗi domain có một clean workflow và bốn fault variants, cùng `pair_id`:

| Variant | Tín hiệu công khai | Hành vi mong đợi |
|---|---|---|
| `clean` | không có anomaly | tiếp tục/finalize; verification chỉ là optional diagnostic |
| `explicit_invalid` | executor trả invalid output | inspect contract, repair hoặc đổi executor |
| `silent_corruption` | status success nhưng checksum/consistency signal đáng ngờ | verifier trước khi downstream consumption |
| `retryable_failure` | executor chạy nhưng không tiến triển/lỗi tạm thời | tạo revised method rồi retry hoặc dùng alternate executor |
| `conflicting_outputs` | hai output hợp lệ về format nhưng mâu thuẫn | conflict audit rồi repair/resolve |

Benchmark phân biệt:

- **pre-execution rejection**: router chọn role không phù hợp; executor không chạy;
- **post-execution failure**: executor đã chạy và công khai failure status;
- **silent semantic/consistency fault**: executor báo success nhưng state có observable
  anomaly;
- **downstream contamination**: router cho agent khác sử dụng artefact đáng ngờ trước
  khi kiểm tra;
- **recovery**: trajectory loại bỏ/khóa artefact lỗi, tạo artefact thay thế và quay lại
  đường hoàn thành task.

Một role có thể áp dụng về mặt input nhưng không hữu ích. Ví dụ planner được phép chạy
khi thứ tự workflow đã công khai, nhưng kết quả là `EXECUTED_INEFFICIENT`. Trường hợp
này không thuộc safe gold set, song cũng không bị ghi thành `ACTION_REJECTED` và không
được tính là task-executor failure.
`ACTION_REJECTED` chỉ dành cho role không thỏa precondition trước khi executor chạy.

`fault_class`, `fault_origin`, clean counterpart và expected response classes nằm trong
evaluation sidecar. Router chỉ thấy status, effect, artefact và memory tương ứng với
view được chọn.

## 6. Controlled injection và lỗi tích lũy

V4 dùng cặp counterfactual có cùng task:

- clean control;
- fault injected ở early/middle/late checkpoint;
- cùng pool, initial action set, budget policy và terminal contract.

Một số fault workflow chứa executable-but-unsafe downstream transitions. Chúng cho
phép đo chuỗi:

`fault -> missed verification -> contaminated artifact -> extra calls -> terminal failure`

Các transition này không thuộc `valid_actions`, nhưng simulator vẫn thực thi để đo
thiệt hại của quyết định sai. Quy tắc loại action khỏi ground truth dựa trên contract
được khai báo trong recovery sidecar, không dựa vào capability score hoặc magic
utility.

## 7. Metrics

Không tạo composite score. Mọi metric báo riêng, kèm count và denominator.

### Routing và kiểm tra lỗi

- `valid_action_accuracy`: số decision chọn action thuộc set-valued gold / tổng
  decision.
- `recovery_state_accuracy`: action đúng trên các state recovery / tổng recovery state.
- `fault_detection_recall`: fault episode có verifier/inspector hợp lệ trước lần đầu
  consume artefact lỗi / số detectable fault episode.
- `unnecessary_check_rate`: verification call tại clean state không có public check
  requirement / tổng verification call trên clean state.
- `unsafe_consumption_rate`: fault episode có artefact lỗi bị downstream agent consume
  trước khi được kiểm tra / tổng fault episode có hazardous consumer.
- `pre_execution_rejection_rate`: số call bị `ACTION_REJECTED` / tổng non-STOP call.
- `applicable_no_progress_rate`: số call đã thực thi nhưng không tạo tiến triển hữu
  ích / tổng call thực thi được; báo riêng khỏi rejection.
- `inefficient_execution_rate`: số call có status `EXECUTED_INEFFICIENT` / tổng call
  thực thi được. Metric này đo lãng phí routing, không gán lỗi cho executor.

### Recovery

- `recovery_success_rate`: recoverable fault episode hoàn thành goal / tổng recoverable
  fault episode.
- `repair_success_rate`: episode tạo corrected/replacement artifact sau diagnosis / số
  episode bắt đầu repair.
- `median_calls_to_detection`: median số actor calls từ fault event đến valid
  inspector/verifier; episode không detect được báo riêng, không impute.
- `median_calls_to_recovery`: median số actor calls từ fault event đến corrected
  artifact; episode không recover báo riêng.
- `loop_after_failure_rate`: fault episode lặp cùng failed approach mà không có
  substantive state change / tổng fault episode.
- `task_executor_failure_rate`: số event có post-execution failure status / tổng event
  thực thi được; không gồm `ACTION_REJECTED` hay `EXECUTED_INEFFICIENT`.
- `recovery_after_executor_failure_rate`: episode tạo corrected artifact sau một
  task-executor failure quan sát được / số episode có task-executor failure.
- `mean_budget_wasted_before_first_progress`: trung bình non-progress actor calls trước
  lần đầu state tiến triển thực chất.
- `mean_calls_after_goal_path_available`: trung bình actor calls không phải finalizer
  sau khi state đã có đủ điều kiện công khai để đi tới terminal goal.

### Error propagation và paired counterfactual

Với mỗi fault run và clean control cùng `pair_id`:

- `success_degradation = success_clean - success_fault`;
- `excess_calls = calls_fault - calls_clean`;
- `excess_tokens = tokens_fault - tokens_clean` khi live actors báo token;
- `failure_amplification = downstream_fault_events_after_injection / injected_faults`;
- `cascade_containment_rate`: fault episode được detect/recover trước contamination /
  tổng fault episode;
- báo riêng theo injection position (`early`, `middle`, `late`) và fault class.

Không clamp `excess_calls` hoặc `excess_tokens`: giá trị âm vẫn được giữ để tránh làm
đẹp kết quả. Không gán trọng số để cộng các metric này.

### Track C — post-fault checkpoint

Mỗi một trong 24 fault workflow tạo một cặp checkpoint:

- `fault`: replay transition lỗi thật rồi dừng ngay sau tín hiệu hậu thực thi;
- `control`: replay clean transition ở cùng prefix depth, với cùng task, pool và
  remaining execution budget.

Tổng cộng có 48 episode (24 fault + 24 control), chia đều 16 episode cho development,
validation và test. Router chỉ nhận checkpoint ID opaque, task/state/history theo
memory view, budget và role cards. `condition`, `pair_id`, `fault_class`, valid set,
unsafe transition IDs và oracle path length chỉ tồn tại ở private sidecar.

Track C báo riêng:

- `post_fault_goal_completion_rate` và `clean_continuation_success_rate`;
- `first_action_valid_accuracy`, `first_action_unsafe_rate` và
  `post_checkpoint_valid_transition_rate`;
- `fault_detection_recall`, `first_action_detector_rate`, `repair_success_rate`;
- `unsafe_consumption_rate`, `clean_unnecessary_check_rate`;
- loop, budget exhaustion, premature STOP;
- median calls tới detection/repair/goal và excess calls so với shortest safe path;
- paired `success_degradation` và `recovery_overhead_calls`.

Không có composite score. Kết quả recovery và clean specificity phải được đọc cùng
nhau để tránh một router luôn gọi verifier được xem là tốt.

## 8. Dataset artefacts

V4-alpha tạo:

- `agents.jsonl`: qualitative role contracts;
- `scenarios.jsonl`: executable workflow definitions;
- `decision_states.jsonl`: Track A state snapshots và set-valued labels;
- `public_inputs.jsonl`: **nguồn input chính thức cho router**, chứa các view công khai
  được đặt tên riêng (`state`, `structured`, và `raw` chỉ ở development); harness phải
  chọn đúng một view theo cấu hình trước khi gọi router;
- `private_labels.jsonl`: valid action set, rationale, fault/recovery và internal IDs
  dành riêng cho evaluator;
- `executor_outputs.jsonl`: deterministic raw/structured executor fixtures;
- `recovery_annotations.jsonl`: private evaluation sidecar cho fault/recovery;
- `post_fault_inputs.jsonl`: Track C router input leakage-safe;
- `post_fault_runtime.jsonl`: private checkpoint state/history để replay simulator;
- `post_fault_labels.jsonl`: private valid/unsafe/recovery labels và oracle path length;
- `post_fault_pair_manifest.jsonl`: private mapping giữa fault và clean control;
- `post_fault_human_review_queue.jsonl`: checklist review từng cặp checkpoint;
- `provenance.jsonl`: nguồn template và derivation record;
- `split_manifest.json`, `quota_report.json`, `validation_report.json`, schemas và
  checksums.

Mỗi decision row nội bộ có ba view độc lập: state-only, structured-history và
raw-history. `public_inputs.jsonl` lưu các view công khai trong namespace riêng;
evaluator materialize chính xác một view đã chọn rồi join nhãn bằng
`public_decision_id`. Không được truyền nguyên public wrapper chứa nhiều view hoặc row từ
`decision_states.jsonl`, `valid_actions`, action rationales, fault annotations hoặc
clean counterpart cho router.

## 9. Validation gates

Build thất bại nếu:

- public router input chứa gold label, fault-origin label, preferred action, action
  utility, backbone hoặc capability score;
- recovery annotation không có clean counterpart;
- hazardous action bị đưa vào valid set;
- fault signal cần cho gold decision không xuất hiện trong view được đánh giá;
- raw fixture thiếu transition/action/status tương ứng;
- split group hoặc counterfactual pair vượt qua partition;
- workflow không có đường clean/recovery tới terminal goal;
- domain/fault class quota không đạt.
- public/private artifact thiếu row, public ID chứa fault token, hoặc hai public
  observation giống hệt nhau lại có safe-action labels khác nhau.
- Track C checkpoint không replay đúng fingerprint hoặc không còn safe path tới goal;
- fault/control pair khác task, pool, prefix budget hoặc vượt split;
- Track C public input làm lộ condition, pair ID, fault label hay oracle information.

## 10. Trạng thái triển khai alpha

V4-alpha là một controlled benchmark slice để kiểm tra schema, memory ablation và
recovery metrics. Trước khi phát hành V4-beta cần:

1. hai annotator độc lập review valid action sets và observable fault signals;
2. adjudication cho disagreement;
3. chạy random/oracle sanity baselines;
4. chạy ba cấu hình headline trên ít nhất hai orchestrator;
5. thêm live Track B/Track C traces và kiểm tra paired clean/fault token accounting;
6. giữ một domain holdout chưa dùng trong prompt/harness iteration.

## 11. Live Track B với Gemini 2.5 Flash

V4 hỗ trợ thêm Track B live. Router vẫn chọn action từ state và role cards; executor
được gọi thật bằng `gemini-2.5-flash`. Một live output không hợp lệ được tính là lỗi
executor và không làm transition tiến lên. Khi live output đạt contract, controlled
post-execution check của counterfactual vẫn được áp dụng để giữ fault injection có thể
so sánh giữa các router. Vì vậy execution mode được báo rõ là
`live_llm_actor_with_controlled_fault_injection`, không được trình bày như một môi
trường hoàn toàn tự nhiên.

```powershell
python scripts/run_routing_benchmark_v4.py `
  --config config/routingbench_v4/jev_structured_history_live_gemini.yaml `
  --run_id "routingbench_v4a_jev_structured_history_live_gemini_dev"
```

## 12. Lệnh chạy Track C

Track C deterministic là cấu hình chính để so sánh router vì giữ executor outcome cố
định. Ví dụ Jev structured-history:

```powershell
python scripts/run_routing_benchmark_v4.py `
  --config config/routingbench_v4/jev_structured_history.yaml `
  --track c `
  --split development `
  --run_id "routingbench_v4a2_jev_structured_history_track_c_dev"
```

Khi chạy riêng `--track c`, `--limit` giới hạn số **pair**, không phải số checkpoint;
evaluator luôn giữ cả fault và clean checkpoint của mỗi pair được chọn. Khi báo cáo,
không trộn metric Track C vào Track B.
