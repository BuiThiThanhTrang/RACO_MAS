# Kế hoạch và triển khai RoutingBench V3-beta

## Mục tiêu

V3-beta giữ nguyên phát hiện quan trọng của alpha: structured prompting có thể
kém hơn naive khi candidate pool lớn và role card quá dài. Dataset không được
chỉnh để buộc structured thắng. Thay vào đó, beta bổ sung task thật và tách bốn
đại lượng phải báo cáo độc lập:

1. routing quality;
2. execution success của actor;
3. contract-based workflow completion;
4. semantic end-task accuracy.

Không thành phần nào dùng capability score, backbone ranking,
`preferred_actions` hoặc action utility.

## Thành phần đã triển khai

### 1. Nguồn dữ liệu và provenance

- Giữ 64 workflow provenance-first của V3-alpha.
- Thêm 24 MuSiQue task thật, cân bằng 2/3/4-hop.
- Thêm 28 MMLU-Pro task thật, hai task cho mỗi domain đã chọn.
- Mỗi workflow có source path, source item ID, SHA-256, license, phép biến đổi
  và trạng thái human review.
- MMLU-Pro chia domain rời nhau giữa development, validation và test để tạo
  một phép thử unseen-domain cụ thể.

MuSiQue được phân phối theo CC-BY-4.0 và MMLU-Pro theo MIT. GAIA không được
đóng gói vì dataset card gated cấm tái phân phối validation/test dạng crawlable.
SRDD chưa được dùng vì file local chưa có upstream revision và license record
được pin rõ ràng.

Hiệu quả dự kiến: nguồn gốc từng mẫu có thể kiểm chứng bằng hash và source ID;
không phải suy đoán task được tạo từ run nào hoặc benchmark nào.

### 2. Tách dữ liệu public, actor input và gold

- `decision_states.jsonl`: chỉ chứa input mà router được phép thấy.
- `task_inputs.jsonl`: corpus/options thật, chỉ cấp cho actor sau khi router đã
  chọn action.
- `evaluation.jsonl`: official answer và aliases, không cấp cho router hoặc
  actor.

Hiệu quả dự kiến: router không được lợi từ gold answer hoặc supporting label,
trong khi Track B vẫn có thể cho actor thực thi task thật và chấm end-task.

### 3. Track B với actor thật

Actor output bổ sung `candidate_answer`. Mọi contribution thực tế được truyền
sang agent tiếp theo qua live execution memory. Chỉ candidate answer được tạo ở
bước làm workflow đạt goal mới trở thành committed answer.

Evaluator báo cáo riêng:

- valid-transition rate;
- actor invalid-output rate;
- goal-completion rate;
- end-task exact accuracy và token F1 khi phù hợp;
- router/executor/mixed/indeterminate failure attribution.

Nếu route hợp lệ và workflow hoàn tất nhưng committed answer sai semantic, lỗi
được ghi về executor/reasoning, không đổ cho router.

### 4. Sol router đúng API

`gpt-6-sol-openrouter` dùng structured chat completion, không gửi vào Decisions
endpoint của Jev. Sol và Jev nhận cùng observable state, cùng role-card mode và
cùng rejection guard, nhờ đó phép so sánh router có ý nghĩa.

Hiệu quả dự kiến: kiểm tra được overprompting là giới hạn riêng của Jev hay còn
xuất hiện ở một orchestrator LLM lớn hơn.

### 4b. OpenAI Decisions API

GPT-6 Luna Decisions đã được thêm như router thứ ba qua provider `openrouter`,
endpoint typed Decisions của OpenRouter và model
`openai/gpt-6-luna-decisions`. Nó nhận cùng observable state, candidate set,
role-card mode và rejection guard đang dùng cho Jev.

Hai config naive/structured riêng được cung cấp cho cả V3-alpha và V3-beta.
Nhờ đó có thể so sánh ba kiểu orchestrator mà không thay đổi benchmark:

- Jev qua OpenRouter Decisions;
- GPT-6 Sol qua structured chat completion;
- GPT-6 Luna Decisions qua OpenRouter.

Hiệu quả dự kiến: tách ảnh hưởng của model/router khỏi ảnh hưởng của prompting
và harness, đồng thời so sánh valid-action accuracy, latency, token input và
failure pattern trên đúng cùng decision states.

### 5. Human review

`human_review_queue.jsonl` tạo checklist cho từng workflow. Trạng thái hiện tại
vẫn là `pending`; formal validation không được trình bày như human annotation.

### 5b. Lượt terminal dành riêng cho STOP

`max_steps` chỉ giới hạn các quyết định thực thi hoặc routing tới agent. Sau khi
`final_artifact` được commit, environment dành thêm đúng một quyết định terminal
chi phí 0 cho `STOP`; quyết định này không tăng `steps_used` và không trừ
`calls_remaining`. Nếu router chọn một agent thay vì `STOP` tại state terminal,
episode kết thúc với `TERMINAL_ACTION_ERROR` và lỗi được quy cho router.

Hiệu quả dự kiến: task không còn bị ghi `STEP_LIMIT` chỉ vì finalizer dùng đúng
lượt actor cuối cùng, trong khi khả năng nhận biết lúc cần dừng của orchestrator
vẫn được đánh giá rõ ràng.

Trước khi gọi đây là bản release chính thức cần hai annotator độc lập và một
adjudicator cho validation/test.

### 6. Quy mô bản beta và sanity checks

Build hiện tại tạo 116 workflow, gồm 64 workflow kiểm soát và 52 workflow dựa
trên task thật (24 MuSiQue, 28 MMLU-Pro), tương ứng 1.156 decision state. Tỷ lệ
state không terminal có nhiều hơn một action hợp lệ là 93,75%; con số này cần
được human audit để chắc rằng các action tương đương thực sự có cùng contract,
không phải valid set bị nới quá rộng.

Trên development split, random baseline đạt 12,5% valid-action accuracy còn
oracle đạt 100%. Đây là sanity check cho thấy evaluator có dải phân biệt và
không vô tình cấp đáp án cho router.

Runner hỗ trợ `--source musique`, `--source mmlu-pro` và
`--real_tasks_only`, nhờ đó Track B có thể smoke-test trực tiếp actor thật trên
nguồn mong muốn thay vì lấy các workflow kiểm soát xuất hiện trước trong split.

## Thứ tự chạy đề xuất

1. Build và validate beta.
2. Chạy Track A Jev naive/structured trên development.
3. Chạy Track A Sol naive/structured trên cùng split.
4. Chạy OpenAI Decisions naive/structured trên cùng split.
5. Audit mọi disagreement và toàn bộ state mà structured kém naive.
6. Chạy Track B `--limit 4`, sau đó 8, trước khi chạy toàn split.
7. Bản beta có thể được đóng gói và upload khi mọi record vẫn ghi rõ
   `human_review_status: pending`; không được mô tả bản này là human-validated.
   Chỉ chuyển sang release candidate hoặc dùng để báo cáo kết quả chính thức sau
   khi hai annotator và một adjudicator hoàn tất review.

## Tiêu chí không được đánh đổi

- Không thêm numeric utility không có công thức được công bố.
- Không để gold, official supporting labels hoặc source-only annotation lọt vào
  router input.
- Không dùng simulator để giả output actor trong Track B.
- Không sửa valid set dựa trên model nào đang được đánh giá.
- Không buộc structured phải thắng naive; regression là kết quả benchmark hợp
  lệ nếu annotation đúng.
