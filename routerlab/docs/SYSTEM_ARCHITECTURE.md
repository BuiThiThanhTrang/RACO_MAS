# Tài liệu hệ thống và protocol của RouterLab

Tài liệu này mô tả đầy đủ contract thực thi của codebase RouterLab hiện tại. Đây là
tài liệu tham chiếu chính thức cho các thí nghiệm MBPP+: harness, thông tin router quan
sát được, thông tin executor nhận được, ranh giới đánh giá, budget, logging và quy tắc
chọn code nộp cuối cùng. Mọi phương pháp được so sánh phải sử dụng cùng protocol được
mô tả tại đây.

## 1. Phạm vi nghiên cứu

RouterLab đánh giá bài toán **dynamic executor routing** trong một workflow sinh code
đa tác tử. Expert pool mặc định dùng chung một SLM backbone, trong khi routing policy
được thay đổi giữa các run. Thiết kế này giữ cố định năng lực executor, decoding, task
và harness để khác biệt quan sát được chủ yếu đến từ quyết định routing.

Môi trường đầu tiên là MBPP+ thông qua EvalPlus. Mỗi episode bắt đầu khi chưa có code.
Trong E3, router có thể chọn bất kỳ executor nào hoặc `STOP` ngay từ bước đầu; không có
Coder bootstrap bắt buộc và không có action mask.

Action space cố định gồm:

```text
coder | logic_reviewer | edge_case_analyst | repair_agent | STOP
```

### 1.1 Protocol của các full run v5 đã khóa

Ba full run chính sử dụng protocol:

```text
v5-capability-cards-current-code
```

V5 có các thuộc tính bắt buộc sau:

- router-first, không Coder bootstrap và không action mask;
- năm action cố định tại mọi decision point;
- router nhìn thấy capability card v5 của mọi action;
- Base-pass và Base-fail đều là bằng chứng chưa đầy đủ, không phải routing rule;
- Base Tests tự động chạy sau mỗi code change;
- Plus Tests chỉ chạy ở final evaluation;
- code mới luôn trở thành `current_code`, kể cả khi Base score giảm;
- final submission là chính xác current code, không rollback về Base-best code;
- mỗi non-STOP executor action tốn một trong bảy step; `STOP` miễn phí;
- chọn executor ở zero budget được ghi nhận là bad decision và không được thực thi.

### 1.2 Cấu hình khóa cho ba full run

| Thành phần | Giá trị đang dùng |
|---|---|
| Dataset | MBPP+ qua EvalPlus, 378 task |
| Manifest | `manifests/mbpp_plus_full.json` |
| Dataset hash | `92743def42b30b354a30898e4fa33fb0` |
| Expert | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Qwen revision trên Modal | `d3a7bedd480bc27342f4336c71e86858b25df935` |
| Expert runtime | Modal, một NVIDIA T4, tối đa một container/một request |
| Expert decoding | `temperature=0.0`, `top_p=1.0`, `max_output_tokens=1024` |
| Expert context limit | 8192 token |
| Router actions | `coder`, `logic_reviewer`, `edge_case_analyst`, `repair_agent`, `STOP` |
| Budget | 7 executor step, 20 Base executions, 900 giây/episode |
| JEV | `typesafe/jev-1.13` qua OpenRouter Decisions API |
| Luna | `openai/gpt-6-luna-decisions` qua OpenRouter Decisions API |
| Laya | `convaiinnovations/laya-typed-decisions`, local CPU |
| Laya revision | `989e28febe5886389496fe13a8036c62b2a20ae0` |
| Laya context limit | 1024 token |
| EvalPlus package | `evalplus==0.3.1` với Windows compatibility backend |

Source YAML đặt `dataset.limit: 1` như một mặc định an toàn cho smoke test. Khi chạy
với `--manifest manifests\mbpp_plus_full.json`, CLI thay `dataset.manifest` bằng full
manifest và đặt `dataset.limit: null`; vì vậy toàn bộ 378 task được chạy theo đúng thứ
tự manifest. `seed: 42` vẫn được ghi trong resolved config nhưng không sample lại task.

Environment local được ghi nhận khi bắt đầu full runs:

| Package | Version |
|---|---:|
| Python | 3.11 |
| `dynamic-router-lab` | 0.1.0 |
| `evalplus` | 0.3.1 |
| `laya` | 0.4.31 |
| `torch` | 2.15.0 |
| `transformers` | 5.23.0 |
| `openai` | 3.51.0 |
| `huggingface_hub` | 2.4.0 |
| `psutil` | 7.2.2 |
| `PyYAML` | 6.0.3 |

Các full run trong cùng bảng kết quả phải dùng cùng environment này hoặc ghi rõ thay
đổi. Nên lưu thêm `pip freeze` cùng artifact khi chốt kết quả cuối.

JEV và Luna trong config dùng model alias của OpenRouter. `public_trace.jsonl` ghi
`resolved_model` tại từng decision; cần báo cáo giá trị resolved thực tế của full run
để phát hiện provider-side version drift. Pilot v5 đã quan sát
`typesafe/jev-1.13-20260917` và `openai/gpt-6-luna-decisions-20261006`, nhưng full run
phải được kiểm tra lại từ trace thay vì mặc định chúng không đổi.

### 1.3 Protocol v6 end-to-end hiện hành

V6 được triển khai song song và không diễn giải lại log v5. Router vẫn chỉ chọn bốn
executor hoặc `STOP`, nhưng public state dùng candidate identity, artifact history và
freshness, role progress, recent events và typed failure/no-progress outcomes. Executor
nhận bounded context có request audit; Plus Tests vẫn chỉ chạy ở final evaluation.

V6 hiện chỉ hỗ trợ end-to-end routing. Controlled recovery và frozen candidate bank
được hoãn. Chi tiết normative nằm trong
`MBPP_STATE_PROTOCOL_V6_VI.md` và `MBPP_HARNESS_INFORMATION_FLOW_V6_VI.md`.

Các mục còn lại của tài liệu này mô tả contract chung và chi tiết tái lập v5. Khi
field/state behavior khác nhau, hai tài liệu v6 nói trên là nguồn chuẩn cho run v6.

## 2. Cấu trúc repository

| Đường dẫn | Trách nhiệm |
|---|---|
| `src/routerlab/cli.py` | CLI, tạo run directory, khởi tạo dependency và vòng lặp task |
| `src/routerlab/config.py` | Dataclass cấu hình YAML và kiểm tra cấu hình |
| `src/routerlab/contracts.py` | Kiểu dữ liệu dùng chung cho task, state, action, result, usage và budget |
| `src/routerlab/capabilities.py` | Capability card chuẩn hóa dành cho router |
| `src/routerlab/dataset.py` | Load MBPP+, chọn task xác định và quản lý manifest |
| `src/routerlab/routers.py` | Heuristic router, generative router và typed decision-model router |
| `src/routerlab/decision_models.py` | Registry decision model và provider adapter |
| `src/routerlab/executors.py` | Tạo payload cho expert và chuẩn hóa output |
| `src/routerlab/pool.py` | Registry executor cố định và loại output của từng executor |
| `src/routerlab/prompts/` | Role prompt của executor và prompt của generative router |
| `src/routerlab/workflows.py` | Controller E1/E2/E3 và state transition của episode |
| `src/routerlab/budget.py` | Theo dõi step, lần chạy Base, thời gian, token và chi phí |
| `src/routerlab/evaluator.py` | Tích hợp EvalPlus và tạo Base feedback đã được làm sạch |
| `src/routerlab/windows_evalplus.py` | Worker EvalPlus tương thích Windows, timeout và giới hạn bộ nhớ |
| `src/routerlab/metrics.py` | Tổng hợp metric chất lượng, hiệu quả và routing |
| `src/routerlab/runlog.py` | Public trace, private evaluation và solution log |
| `src/routerlab/protocol_v6.py` | Artifact ledger, freshness, v6 router state và context packer |
| `deploy/` | Cấu hình triển khai dịch vụ inference từ xa |
| `configs/` | Cấu hình có thể tái lập cho từng method/model/provider |
| `manifests/` | Danh sách task có thứ tự cố định dùng chung giữa các phương pháp |
| `tests/` | Test contract, workflow, adapter, logging và Windows evaluator |

## 3. Kiến trúc end-to-end

```text
EvalPlus MBPP+ record
  |
  | chỉ expose task_id, canonical prompt và entry point
  v
Public episode state -----------------------------------------------+
  |                                                                |
  v                                                                |
Router + capability cards                                          |
  |                                                                |
  +--> STOP --------------------------------------------------+     |
  |                                                           |     |
  +--> executor được chọn                                     |     |
         |                                                     |     |
         +--> chỉ tạo findings --> cập nhật public findings ---+-----+
         |
         +--> tạo code --> thay thế current code
                           |
                           +--> tự động chạy Base Tests
                                  |
                                  +--> Base feedback đã làm sạch --+

Episode kết thúc
  |
  +--> chưa có current code: không có submission, task thất bại
  |
  +--> đã có current code: nộp chính xác version hiện tại
           |
           +--> chạy một lần final Base + Plus evaluation riêng tư
```

Router không được quyền lên lịch chạy test. Base Test là một transition xác định của
harness sau khi code thay đổi. Plus Tests là dữ liệu riêng tư và không bao giờ xuất
hiện trong online state transition.

## 4. Dataset contract

`EvalPlusMBPPDataset` chỉ expose mỗi task dưới dạng:

```json
{
  "task_id": "Mbpp/468",
  "prompt": "<canonical EvalPlus prompt>",
  "entry_point": "<tên hàm bắt buộc>"
}
```

Canonical prompt của EvalPlus MBPP+ đã chứa một assertion mẫu. RouterLab không nối
thêm `base_input`, `plus_input`, expected output, canonical solution hoặc dữ liệu của
special oracle vào prompt.

Manifest lưu dataset hash và danh sách task theo đúng thứ tự. Mọi phương pháp được so
sánh phải dùng chung một manifest thay vì tự sample task độc lập.

## 5. Cơ chế của harness

### 5.1 Episode state nội bộ

Session nội bộ quản lý:

- task bất biến;
- `CodeArtifact` hiện tại hoặc trạng thái chưa có code;
- Base feedback đã làm sạch của code version hiện tại;
- findings của Logic Reviewer cho code version hiện tại;
- findings của Edge-case Analyst cho code version hiện tại;
- lịch sử executor action;
- public step records;
- kết quả Base-only của code đầu tiên;
- kết quả Base+Plus của code được nộp cuối cùng;
- các bộ đếm budget và usage.

### 5.2 Một routing transition trong E3

Tại mỗi decision point của E3, controller thực hiện:

1. Tạo `RoutingState` chỉ từ thông tin công khai.
2. Cung cấp cho router toàn bộ năm action và capability card chuẩn hóa.
3. Ghi lại action được chọn, tính hợp lệ, usage, budget state và khả năng thực thi.
4. Nếu action là `STOP`, kết thúc episode mà không tiêu tốn executor step.
5. Nếu không còn executor step nhưng router chọn action khác `STOP`, ghi nhận budget
   violation và kết thúc mà không chạy executor đó.
6. Nếu action hợp lệ để thực thi, chạy chính xác executor được chọn và trừ một step.
7. Áp dụng findings hoặc code output của executor vào session.
8. Khi và chỉ khi source code thay đổi, tăng code version và tự động chạy Base suite.
9. Quay lại bước 1, trừ khi safety limit của controller buộc episode kết thúc.

### 5.3 Findings và tính hợp lệ theo code version

Logic findings và edge-case findings mô tả một code version cụ thể. Mỗi khi một code
artifact mới trở thành current code, hai danh sách findings được xóa để tránh gắn review
cũ vào implementation mới. Nếu executor trả lại code không đổi, harness không tạo code
version mới và không chạy lại Base Tests.

### 5.4 Quy tắc nộp current code

Mỗi code artifact có source thay đổi sẽ lập tức thay thế `current_code`, kể cả khi điểm
Base công khai bằng hoặc thấp hơn version trước. Khi episode kết thúc, harness nộp chính
xác `current_code`.

Harness **không** rollback về version cũ có điểm Base cao nhất. Vì vậy:

- một chỉnh sửa tốt có thể thay đổi final Plus accuracy dù code cũ và code mới đều
  pass Base suite thưa;
- một chỉnh sửa xấu ở cuối episode có thể tạo regression;
- chọn `STOP` sau regression được quan sát như một bad routing decision;
- final quality đo trajectory của router và executor, không đo thêm một policy cứu lỗi
  ngầm của harness.

Các version cũ có thể được giữ trong log để phân tích nhưng không thay thế submission
chính. Hiện tại action space không có action `REVERT` riêng.

### 5.5 Base evaluation tự động và final Plus evaluation

Trong episode:

- output code có thay đổi từ Coder hoặc Repair sẽ kích hoạt Base-only evaluation;
- code không đổi, reviewer findings, analyst findings và `STOP` không kích hoạt một lần
  chạy Base online;
- chỉ status và số lượng test đã được làm sạch trở thành public feedback.

Sau khi episode kết thúc:

- nếu chưa có code, harness không gọi final evaluator và outcome được tính là thất bại;
- nếu có code, EvalPlus chạy đúng một lần với `include_plus=True` trên current code được
  nộp;
- final Base và Plus result chỉ được ghi vào private evaluation log.

Thông tin Plus không bao giờ được trả lại router hoặc executor.

## 6. Chính xác router nhìn thấy gì

### 6.1 Dynamic routing state

Mọi router nhận các public field sau:

| Field | Ý nghĩa |
|---|---|
| `task_id` | Định danh MBPP+ ổn định |
| `task_prompt` | Canonical prompt, bao gồm một assertion mẫu |
| `entry_point` | Tên hàm Python bắt buộc |
| `current_code` | Toàn bộ source hiện tại hoặc `null` trước khi có code |
| `code_version` | `0` khi chưa có code; tăng dần mỗi khi code thực sự thay đổi |
| `base_feedback` | Feedback đã làm sạch của current code hoặc `null` |
| `logic_review` | Logical findings của current code version |
| `edge_cases` | Edge-case findings của current code version |
| `action_history` | Danh sách executor ID đã thực thi theo thời gian |
| `budget` | Step, Base execution, wall time còn lại và action cost |
| `action_space` | Toàn bộ năm action theo thứ tự cố định |

`base_feedback` chỉ chứa:

```json
{
  "code_version": 1,
  "status": "pass | fail | timeout | ...",
  "passed": true,
  "passed_tests": 3,
  "total_tests": 3,
  "failure_count": 0,
  "summary": "Base Tests: pass; 3/3 passed"
}
```

Nó không chứa assertion ẩn, input, expected output, actual output hoặc Plus result.

Budget object có dạng:

```json
{
  "remaining_steps": 7,
  "remaining_base_executions": 20,
  "remaining_wall_time_ms": 899500.0,
  "non_stop_action_cost": 1,
  "stop_cost": 0
}
```

Không có episode-level token limit. Generation của executor chỉ bị giới hạn bởi
`max_output_tokens` cho từng lần gọi; tổng token được đo để báo cáo efficiency.

### 6.2 Router-facing capability card tĩnh

Capability card chuẩn hóa trong `capabilities.py` mô tả cho mỗi action:

- input mà action sử dụng;
- output và ảnh hưởng lên state;
- action có thay đổi code hoặc chạy test hay không;
- availability hoặc precondition;
- step cost.

Card của Reviewer và Analyst ghi rõ chúng có thể được dùng khi có hoặc chưa có current
code, và khi Base Tests pass hoặc fail. Đây là mô tả capability, không phải chỉ dẫn tạo
một route cố định.

Typed decision model nhận các card làm description/criteria của một choice question.
Generative router nhận cùng mapping trong field `action_capabilities` của JSON user
payload. Heuristic router hiện thực policy trực tiếp nên không chạy model inference trên
các card này.

### 6.3 Common typed-decision question

JEV, Laya, Luna/OpenAI Decisions và DREX tương thích nhận một câu hỏi duy nhất:

```text
Which action should the multi-agent code workflow execute next to maximize the
chance of solving the task within the remaining budget?
```

Instruction đầy đủ nêu rằng Base-pass và Base-fail đều chỉ là bằng chứng chưa đầy đủ,
không phải routing rule; review là tùy chọn; và mỗi action chỉ nên được chọn khi có
expected value cao nhất trong state và budget hiện tại.

### 6.4 Thông tin bị ẩn khỏi router

Router không bao giờ nhìn thấy:

- các Base assertion còn lại ngoài assertion mẫu trong canonical prompt;
- Base input, expected output, actual output hoặc evaluator internals;
- Plus input, Plus result hoặc Plus pass count;
- canonical/reference solution;
- output tương lai của executor;
- source của các version đã bị thay thế, ngoài bằng chứng gián tiếp trong action history
  và version counter;
- private evaluation và aggregate result của run hiện tại hoặc run khác, trừ khi một
  thí nghiệm offline chủ động sử dụng chúng.

## 7. Chính xác executor nhìn thấy gì

Mỗi lần gọi executor gồm hai message:

1. Role-specific system prompt từ `src/routerlab/prompts/<executor>.txt`.
2. Một JSON user payload chứa public execution state dùng chung.

Payload dùng chung có dạng:

```json
{
  "task": {
    "task_id": "Mbpp/468",
    "prompt": "<canonical prompt với một assertion mẫu>",
    "entry_point": "<tên hàm>"
  },
  "current_code": "<toàn bộ source hiện tại hoặc null>",
  "code_version": 1,
  "base_feedback": "<object đã làm sạch hoặc null>",
  "logic_review": ["<finding của current version>"],
  "edge_cases": ["<finding của current version>"]
}
```

Executor được chọn **không** nhận:

- probability, confidence, provider hoặc reasoning của router;
- routing budget còn lại hoặc action cost;
- toàn bộ action history;
- Base/Plus test ẩn hoặc expected output;
- private evaluation metric;
- source code của version đã bị thay thế.

### 7.1 Coder

Vai trò: tạo một Python implementation hoàn chỉnh và thực thi được. Coder nhận toàn bộ
field trong common payload và được yêu cầu sử dụng Base feedback cùng reviewer findings.
Nó có thể được gọi trước khi có code hoặc được dùng để thay thế implementation hiện tại.
Output sau khi chuẩn hóa chỉ gồm code.

### 7.2 Logic Reviewer

Vai trò: xác định correctness obligation, logical defect, contract violation,
assumption và material complexity risk mà không viết code. Nó có thể phân tích
specification trước khi có code hoặc audit implementation hiện tại. Output sau khi
chuẩn hóa là tuple findings được lưu trong `logic_review`.

### 7.3 Edge-case Analyst

Vai trò: xác định các trường hợp boundary, empty, singleton, duplicate, negative,
structural, type, large-input và recursion được task hỗ trợ mà không tự đặt thêm yêu
cầu. Output sau khi chuẩn hóa là tuple findings được lưu trong `edge_cases`.

### 7.4 Repair Agent

Vai trò: sửa code hiện tại dựa trên task, Base feedback đã làm sạch và hai danh sách
findings. Nếu chưa có current code, harness đánh dấu đây là một charged no-op và không
nhận code được sinh ra. Nếu đã có code, output sau khi chuẩn hóa là complete code.

### 7.5 Expert backbone dùng chung

`ExecutorPool` gửi mọi expert role qua cùng một `ModelClient`. Role prompt, state và
routing order có thể khác nhau; checkpoint và decoding configuration không khác. Đây
là cơ chế isolation mặc định khi so sánh router.

## 8. Các workflow

### E1: single-agent self-repair

Single agent tạo code, nhận Base feedback đã làm sạch và thử lại trong khi Base fail và
budget còn cho phép. Mỗi version có thay đổi đều được Base-tested. Current version được
nộp khi episode kết thúc.

### E2: fixed multi-agent workflow

E2 bắt đầu bằng Coder. Nếu Base fail, nó thực hiện số round đã cấu hình theo thứ tự:

```text
Logic Reviewer -> Edge-case Analyst -> Repair Agent
```

Workflow dừng sớm khi code sau repair pass Base hoặc khi safety budget cạn.

### E3: dynamic routing

E3 là router-first và luôn expose toàn bộ action space ở mỗi decision point. Router
điều khiển việc chọn executor và `STOP`; harness chỉ điều khiển transition xác định,
logging và safety limit.

Các router implementation hiện có:

- `heuristic`: baseline xác định;
- `model`: generative chat/completion model trả về action ID;
- `laya`: cấu hình tương thích ngược cho local Laya;
- `decision_model`: typed-choice interface dùng chung cho các cặp model/provider đã
  đăng ký.

## 9. Decision-model adapter

Registry ánh xạ một cặp `(name, provider)` thành adapter, default checkpoint, endpoint
và tên biến môi trường chứa API key. Adapter được phép chuyển đổi wire format nhưng
phải giữ nguyên state, choice question, thứ tự capability card và cách chuẩn hóa output.

Local Laya nhận thêm field `revision`, được truyền trực tiếp vào `laya.load()`. Các run
được báo cáo phải pin một commit hash bất biến thay vì theo nhánh `main`, nhờ đó không
tự tải weights khác khi repository trên Hugging Face được cập nhật.

Checkpoint Laya đã pin có một calibration entry `choice:11+` nằm ngoài khoảng runtime
cho phép và bị clamp. RouterLab chỉ có năm action nên Laya dùng bucket hợp lệ
`choice:3-5`; warning này không thay đổi action argmax của thí nghiệm hiện tại. Không
nên diễn giải confidence của bucket bị cảnh báo như xác suất đã calibration.

Các family đang được hỗ trợ gồm Laya, JEV, OpenAI Decisions/Luna và DREX. Cấu hình theo
provider và lệnh chạy được mô tả trong `docs/DECISION_MODELS.md`.

Typed choice không hợp lệ hoặc bị từ chối được chuẩn hóa thành `STOP` và ghi log với
`valid: false`; harness không âm thầm ánh xạ chúng sang một executor khác.

## 10. Budget và termination

Các field budget mặc định:

| Budget | Ý nghĩa |
|---|---|
| `max_steps` | Số executor action khác `STOP` tối đa được thực thi |
| `max_base_executions` | Safety ceiling cho online Base và final evaluation đã được reserve |
| `max_wall_time_seconds` | Giới hạn an toàn về wall time của episode |

Mọi executor action được thực thi đều tốn một step, bao gồm Reviewer và Repair no-op.
Router inference không tiêu tốn executor step, nhưng model call, token, latency và chi
phí được provider báo cáo vẫn được tính. `STOP` tốn 0 step.

Khi `remaining_steps = 0`, E3 vẫn hỏi router. Chọn `STOP` là hợp lệ; chọn executor được
ghi nhận là bad decision `executor_selected_with_zero_steps`, và executor không được gọi.

Các termination reason khác gồm router stop, router stop khi chưa có submission,
fixed-round completion, Base pass trong fixed workflow, hết Base-execution budget và
hết wall-time budget.

## 11. Evaluation và isolation theo nền tảng

EvalPlus tạo executable solution bằng cách nối canonical prompt với generated code.
Trên nền tảng không phải Windows, RouterLab dùng native EvalPlus checker. Trên Windows,
nó giữ nguyên execution/oracle logic của EvalPlus nhưng thay cơ chế Unix-only
`resource`/`SIGALRM` bằng worker process, timer interruption và theo dõi bộ nhớ qua
`psutil`.

Mọi phương pháp được so sánh phải dùng cùng operating system, evaluator mode,
`fast_check` setting và EvalPlus version. `--fast-check` chỉ dùng để debug; kết quả chính
nên chạy complete suite để passed-test count giữa các task có thể so sánh được.

## 12. Log và privacy boundary

Mỗi run ghi:

```text
runs/<run_name>/<run_id>/
  config.json
  manifest.json
  public_trace.jsonl
  private_eval.jsonl
  solutions.jsonl
  metrics.json
  errors.json
```

`public_trace.jsonl` ghi action, code-version transition, Base outcome đã làm sạch,
router metadata, usage, budget và termination. File này không chứa Plus field.

`private_eval.jsonl` ghi Base-only snapshot của code đầu tiên và Base+Plus snapshot của
current code được nộp cuối cùng. File này không được feed vào router hoặc online
executor.

`solutions.jsonl` chứa chính xác current code đã nộp, hash và submission status.
`errors.json` cho phép run dài tiếp tục sau lỗi có thể phục hồi ở từng task, trừ khi
`--fail-fast` được bật.

## 13. Metrics

Aggregate output hiện tại gồm:

- initial và final Base pass rate;
- final Plus pass rate;
- executor step, tổng model call, token, Base execution, wall time và chi phí;
- chi phí trên mỗi task giải đúng;
- executor utilization;
- invalid-router rate và zero-step budget-violation rate;
- no-op rate, no-submission rate và premature-stop rate;
- số routing/model call đến code đầu tiên;
- số lượng theo termination reason.

`total_cost_usd` chỉ cộng cost do model API trả về. OpenRouter router calls có reported
cost; Modal expert endpoint hiện chỉ trả token usage, không trả chi phí GPU. Vì vậy cost
trong `metrics.json` không phải tổng hóa đơn hạ tầng và không được dùng như full-system
cost nếu chưa bổ sung Modal billing offline.

Các field initial Plus, Plus improvement và Plus-based recovery được cố ý đặt là `null`
vì protocol không chạy Plus Tests trên code đầu tiên. Có thể tính trajectory metric chi
tiết hơn bằng phân tích offline từ log, nhưng chúng không được ảnh hưởng đến episode.

## 14. Cấu hình và lệnh full MBPP+

Full manifest đã được tạo và khóa trong repository:

```text
manifests/mbpp_plus_full.json
task_count: 378
dataset_hash: 92743def42b30b354a30898e4fa33fb0
```

### 14.1 Chuẩn bị PowerShell

```powershell
conda activate routerlab
cd D:\SideProject\ChatDev\routerlab

Get-Content D:\SideProject\ChatDev\.env |
  Where-Object { $_ -match '^\s*[^#].*=' } |
  ForEach-Object {
    $name, $value = $_.Split('=', 2)
    Set-Item -Path "Env:$($name.Trim())" `
      -Value $value.Trim().Trim('"').Trim("'")
  }
```

Ba full config đều cần `MODAL_PROXY_TOKEN`. JEV và Luna cần thêm
`OPENROUTER_API_KEY`.

### 14.2 JEV qua OpenRouter, Qwen qua Modal

```powershell
routerlab run `
  --config configs\mbpp_e3_jev_modal.yaml `
  --manifest manifests\mbpp_plus_full.json `
  --run-id full-jev-v5-s42
```

### 14.3 Luna Decisions qua OpenRouter, Qwen qua Modal

```powershell
routerlab run `
  --config configs\mbpp_e3_luna_openrouter_modal.yaml `
  --manifest manifests\mbpp_plus_full.json `
  --run-id full-luna-v5-s42
```

### 14.4 Laya local đã pin, Qwen qua Modal

Đặt Hugging Face cache trên ổ D để tránh tích lũy nhiều snapshot trên ổ hệ thống. Vì
revision đã được tải và pin, full run có thể dùng offline mode:

```powershell
$env:HF_HOME="D:\ModelCache\huggingface"
$env:HF_HUB_CACHE="D:\ModelCache\huggingface\hub"
$env:HF_HUB_OFFLINE="1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"

routerlab run `
  --config configs\mbpp_e3_laya_modal.yaml `
  --manifest manifests\mbpp_plus_full.json `
  --run-id full-laya-typed-v5-pinned-s42
```

### 14.5 Single-agent baseline qua Modal

```powershell
routerlab run `
  --config configs\mbpp_e1_modal.yaml `
  --manifest manifests\mbpp_plus_full.json `
  --run-id full-single-v5-s42
```

### 14.6 Fixed multi-agent baseline qua Modal

`fixed_rounds: 2` tạo tối đa bảy executor step: một lần Coder ban đầu và hai
round `Logic Reviewer -> Edge-case Analyst -> Repair Agent`.

```powershell
routerlab run `
  --config configs\mbpp_e2_modal.yaml `
  --manifest manifests\mbpp_plus_full.json `
  --run-id full-fixed-v5-s42
```

Nên chạy năm thí nghiệm tuần tự. Không dùng `--fast-check` cho kết quả full. Không đổi
capability card, prompt, budget, endpoint hoặc submission rule giữa các run.

Các lệnh hỗ trợ:

```powershell
routerlab decision-models
routerlab check-models --config configs\mbpp_e3_laya_local.yaml
```

Để so sánh công bằng, phải giữ cố định manifest, expert checkpoint/revision và
endpoint, decoding, executor prompt, capability card, budget, evaluator mode, hệ điều
hành, seed và final-submission rule. Với OpenRouter, cần kiểm tra `resolved_model` trong
trace sau khi full run hoàn tất.

### 14.7 Rerun v6 theo hai cohort

Full manifest được phân hoạch thành cohort A gồm 124 task mà JEV và Luna cùng fail
final Plus ở v5, và cohort B gồm 254 task còn lại. Hai manifest rời nhau và hợp lại
đúng 378 task. Đây là phân tích hậu nghiệm; không được báo cáo riêng cohort A như một
test split độc lập. Danh sách run, lệnh chạy, quy tắc ghép kết quả và `--resume` nằm tại
`docs/PROTOCOL_V6_RUNS_VI.md`.

## 15. Kiểm tra trong quá trình phát triển

```powershell
$env:PYTHONPATH='D:\SideProject\ChatDev\routerlab\src'
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Các regression test quan trọng xác minh rằng:

- router không thể nhận Plus information qua public state hoặc public trace;
- toàn bộ action set luôn khả dụng;
- review có thể diễn ra trước khi có code hoặc sau Base pass;
- Repair khi chưa có code là charged no-op ở v5 và `action_rejected` ở v6;
- chọn executor khi không còn step là budget violation;
- `STOP` vẫn miễn phí khi không còn step;
- latest code được nộp kể cả khi Base score bị giảm;
- typed adapter giữ đúng thứ tự capability card chuẩn hóa.

## 16. Các giới hạn cần lưu ý khi diễn giải kết quả

- Base feedback không cung cấp failing assertion hoặc giá trị test. V5 chủ yếu báo cáo
  số lượng; v6 thêm failure class/signature và diagnostic tổng quát đã làm sạch. Vì
  evaluator hiện không xuất traceback chi tiết, expert vẫn có thể lặp lại code cũ.
- Pilot 30 task phù hợp để debug protocol nhưng chưa đủ cho ranking claim mạnh. Cần báo
  cáo confidence interval và dùng manifest lớn hơn hoặc nhiều seed khi phù hợp.
- Decision checkpoint có thể nhạy với wording và thứ tự option. Cần giữ capability card
  cố định trong protocol chính và đánh giá presentation robustness riêng.
- End-to-end accuracy không đủ để giải thích routing behavior. Cần phân tích thêm action
  repetition, review-to-edit utilization, lựa chọn action theo Base state, budget
  violation, recovery và regression cùng với final Plus accuracy.
