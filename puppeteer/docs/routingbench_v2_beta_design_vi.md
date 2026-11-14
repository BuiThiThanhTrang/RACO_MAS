# RoutingBench V2-beta — thiết kế đã triển khai

## Mục tiêu

V2-beta đánh giá việc chọn agent tiếp theo từ `task + task state + execution
memory + role cards`. Benchmark không đánh giá khả năng router đoán hidden outcome
của simulator và không sử dụng capability score hay backbone identity.

Track A chấm một quyết định độc lập. Track B cho router tương tác với deterministic
simulator cho đến khi hoàn thành, dừng sớm, hết budget hoặc rơi vào semantic loop.

## Quyết định thiết kế

### Ground truth theo thông tin quan sát được

- `executable_actions`: action có transition phù hợp và đủ immediate budget.
- `admissible_actions`: action hợp lý từ public observation; đây là nhãn chính.
- `progress_actions`: action mà hidden simulator biết vẫn có đường tới goal; chỉ dùng
  để phân tích.
- `oracle_efficient_actions`: Pareto frontier theo hidden steps/cost; không dùng để
  kết luận một lựa chọn epistemically hợp lý là routing error.

Khi vị trí evidence chưa biết, corpus và web đều có thể admissible. Sau khi một
attempt trả `EXECUTED_NO_PROGRESS`, state mới mới được dùng để đánh giá quyết định
tiếp theo.

### Observation

Input chuẩn không chứa toàn bộ raw state snapshots. Nó gồm:

- current canonical state;
- artifact inventory và open requirements;
- compact `execution_memory` theo agent;
- tối đa ba `recent_events`;
- remaining step/cost budget;
- qualitative agent profiles;
- resource và constraint thực sự quan sát được từ task.

Không đưa source location ẩn, valid labels, oracle path, capability score, model
identity hoặc progress signature vào router input.

### Failure taxonomy

- `ACTION_REJECTED`: role không khớp state; actor chưa thực thi.
- `EXECUTED_NO_PROGRESS`: actor chạy nhưng không tạo tiến triển.
- `EXECUTED_RETRYABLE_FAILURE`: actor chạy và phương pháp cần được sửa.
- `EXECUTED_INVALID_OUTPUT`: actor trả artifact không hợp lệ.
- `EXECUTED_SUCCESS`: actor cập nhật state thành công.

Routing rejection và actor failure có counter riêng trong state và metric riêng.

### Semantic loop

Evaluator tính progress từ facts và artifacts, không từ step index, counter hoặc
event wording. Gọi lại cùng action mà progress state không đổi được đánh dấu
`stagnant_repeat`. Ba lần gọi liên tiếp không tiến triển kết thúc bằng
`LOOP_LIMIT`; episode vẫn bị tính là thất bại.

### Ablation naive và structured orchestration

Decision router có hai profile để đo trực tiếp giá trị của orchestration
engineering trong khi giữ nguyên decision model, dataset, simulator, seed và agent
pool:

- `naive`: prompt tối thiểu, role card chỉ có tên và mục tiêu, không có harness
  guard;
- `structured`: role contract đầy đủ (`use_when`, `avoid_when`, supported input,
  expected output), prompt phân biệt pre-execution routing rejection với
  post-execution actor failure, và observable rejection guard.

Guard structured chỉ tạm loại một action đã trả `ACTION_REJECTED` hoặc
`NOT_APPLICABLE` khi chưa có substantive state change sau đó. Guard chỉ đọc
`recent_events` công khai; không đọc applicable/admissible/progress/oracle labels.
Mỗi decision log ghi `orchestration_profile`, `prompt_version`, `role_card_mode`,
`harness_version` và `guard_blocked_actions` để audit.

So sánh trực tiếp `naive` với `structured` chứng minh hiệu quả của toàn bộ gói
prompt + role contract + harness. Nó không tự tách riêng causal contribution của
từng thành phần; muốn làm vậy cần thêm factorial ablation trong phiên bản sau.

## Quy mô phát hành

- 120 workflows, 8 family × 15.
- 809 Track A decision states.
- 24 qualitative profiles.
- Development/validation/test: 24/32/64 workflows.
- Các instance cùng template group luôn ở cùng split.
- 49 counterfactual groups.
- Không có observational collision trong validation.

Phân bố Track A:

| Nhóm | Tỷ lệ |
|---|---:|
| Singleton nonterminal | 42,89% |
| Multi-action equivalent | 18,67% |
| Multi-action có efficiency trade-off | 23,61% |
| STOP-ready | 14,83% |
| Post actor-failure | 6,55% |
| Post routing-rejection | 13,97% |

Post-failure và post-rejection cộng lại chiếm 20,52%. Beta không ép actor failure
lên 15% riêng vì điều đó đòi hỏi cài failure nhân tạo vào gần như mọi workflow và
làm sai phân bố task tự nhiên.

## Workflow families

1. Sequential dependency.
2. Parallel join.
3. Conditional branch.
4. Conflict resolution.
5. Tool/artifact routing.
6. Failure recovery.
7. Terminal control.
8. Budget-sensitive routing.

Pool perturbation gồm base, irrelevant noise, adjacent noise, order permutation
và stress pool. Supported candidate có hai terminal route định tính tương đương để
benchmark không quay lại dạng single-label role classification.

## Validation gates

- deterministic generation;
- oracle Track A admissible accuracy 100%;
- oracle Track B completion 100%;
- observational collision count bằng 0;
- random Track A admissible accuracy 11,74%;
- random Track B completion 5%;
- alpha và beta unit tests cùng phải qua.

Evaluator còn báo cáo macro score theo `split_group_id`/workflow template:

- Track A: `macro_template_admissible_accuracy`;
- Track B: `macro_template_completion_rate`.

Macro score tránh việc nhiều state hoặc nhiều pool variant của cùng một template
làm một failure pattern có trọng số quá lớn.

## Chạy benchmark

Build lại dataset:

```powershell
python scripts/build_routingbench_v2_beta.py
```

Chạy Jev trên development split trước:

```powershell
python scripts/run_routing_benchmark_v2.py `
  --data_dir data/RoutingBench/v2_beta `
  --split development `
  --track all `
  --router decision `
  --decision_model typesafe/jev-1.13 `
  --orchestration_profile structured `
  --run_id "routingbench_v2beta_jev_dev"
```

Chạy ablation naive với cùng model và split:

```powershell
python scripts/run_routing_benchmark_v2.py `
  --data_dir data/RoutingBench/v2_beta `
  --split development `
  --track all `
  --router decision `
  --decision_model typesafe/jev-1.13 `
  --orchestration_profile naive `
  --run_id "routingbench_v2beta_jev_naive_dev"
```

Sau khi so sánh trên development, đóng băng prompt/harness rồi chạy validation.
Không dùng test split để chỉnh prompt hoặc schema. Vì test hiện tại đã được dùng để
phân tích failure và thiết kế structured profile, mọi lần chạy lại trên split đó chỉ
là diagnostic; bản công bố cuối cần một held-out test mới chưa được quan sát.
