# RoutingBench V3-alpha — thiết kế và triển khai

## Phạm vi của bản alpha

V3-alpha là bản triển khai đầu tiên của thiết kế provenance-first cho bài toán
dynamic routing. Bản này có 64 workflow, 584 decision state và tám workflow
family. Đây chưa phải V3 release cuối cùng: human review vẫn có trạng thái
`pending`, và nguồn workflow hiện tại là RoutingBench V2-beta do project tự tạo.

Mục tiêu của alpha là kiểm tra bốn thành phần trước khi mở rộng dữ liệu từ
MuSiQue, GAIA, MMLU-Pro và SRDD:

1. ground truth dạng tập `valid_actions`;
2. rationale được suy ra từ precondition thực thi được;
3. provenance có source item và SHA-256 kiểm chứng được;
4. Track B gọi actor LLM thật thay vì tạo output bằng simulator.

## Ground truth

V3 không có `preferred_actions`, `action_utilities`, capability score hoặc điểm
chất lượng backbone. Một action hợp lệ khi transition contract của nó thỏa mãn
observable state và call budget hiện tại. Mỗi candidate action có:

- `status`: `valid` hoặc `invalid`;
- `satisfied_preconditions`;
- `violated_preconditions`;
- `rationale_source: formal_transition_preconditions`.

Metric Track A chính là:

```text
ValidActionAccuracy = số lựa chọn thuộc valid_actions / tổng số state
```

Các metric được báo cáo riêng; không có weighted overall score.

## Độ khó

V3-alpha thêm bảy role có capability overlap và tạo hai pool mode:

- `overlap`: role chuẩn cộng role có contract tương đương;
- `stress`: toàn bộ 32 role, gồm nhiều role không liên quan.

Phân bố hiện tại:

- 520 non-terminal states;
- 11.35% singleton-valid states;
- 88.65% multi-valid states;
- 48.27% states có ít nhất ba valid actions;
- 120 post-routing-rejection states;
- 27 post-execution-failure states.

Tỷ lệ nhiều valid action cao không làm bài toán tự động dễ hơn vì stress pool có
tới 32 candidates. Release sau cần báo cáo random baseline và tiếp tục kiểm tra
xem valid set có quá rộng hay không.

## Provenance

Mỗi workflow có một record trong `provenance.jsonl` với:

- dataset và version nguồn;
- source split và source item ID;
- đường dẫn file nguồn;
- SHA-256 của file nguồn;
- script và version tạo dữ liệu;
- danh sách transformation steps;
- validation method;
- human-review status.

Validator kiểm tra source file tồn tại, hash khớp và source item ID thực sự có
trong file. V3-alpha không tự nhận đã human-review; tất cả 64 workflow đều ghi
`human_review_status: pending`.

## Track B live

Track B V3 chỉ hỗ trợ `execution.mode: live`. Flow:

```text
router → valid action check → gọi actor model thật → validate output contract
       → cập nhật observable workflow state → router chọn bước kế tiếp
```

Simulator không tạo output actor. Transition specification chỉ xác định workflow
artifact được cập nhật sau khi output thật vượt qua contract validator.

Failure attribution:

- `router`: action không thuộc valid set hoặc STOP sớm;
- `executor`: action hợp lệ nhưng actor trả NO_PROGRESS/INVALID_OUTPUT;
- `mixed`: cả hai loại lỗi xuất hiện trong episode;
- `indeterminate`: task thất bại nhưng không thuộc ba trường hợp trên.

V3-alpha chưa có semantic final-answer gold cho workflow trừu tượng, do đó
`end_task_accuracy` được ghi `null`; Track B báo cáo contract-based completion.
Đây là giới hạn có chủ ý và phải được khắc phục khi nhập các task có answer gold
từ benchmark nguồn vào V3-beta.

## Cấu hình live

Có bốn cấu hình kiểm soát:

- `live_small_naive.yaml`: Jev naive + Qwen 3.5 9B actors;
- `live_small_structured.yaml`: Jev structured + Qwen 3.5 9B actors;
- `live_strong_naive.yaml`: Jev naive + Gemini 2.5 Flash actors;
- `live_strong_structured.yaml`: Jev structured + Gemini 2.5 Flash actors.

Hai cấu hình Track A không gọi actor/API được cung cấp để kiểm tra biên dưới và
biên trên: `track_a_random.yaml` và `track_a_oracle.yaml`.

Các config mặc định chạy Track B trên development split để tránh vô tình gọi
API cho toàn bộ Track A. Có thể thêm `--track a` hoặc `--track all` khi cần.

Runner nạp `.env` trước khi khởi tạo model clients. Khi chạy trong Git worktree,
runner tìm cả repository root chính từ `git --git-common-dir`; nội dung secret
không được in ra log.

## Lệnh build và kiểm tra

```powershell
python scripts/build_routingbench_v3.py
python -m unittest tests.test_routing_benchmark_v3
```

## Việc còn lại trước V3-beta

1. Chuyển task thật từ MuSiQue, GAIA, MMLU-Pro và SRDD, giữ nguyên source ID,
   license và official gold khi được phép.
2. Thực hiện annotation hai người và adjudication cho test set.
3. Thêm end-task semantic accuracy cho live Track B.
4. Chạy random, naive Jev, structured Jev và strong-router baselines.
5. Audit valid set để tránh cả singleton quá nhiều lẫn valid set quá rộng.
6. Tạo unseen-domain, unseen-workflow và unseen-pool splits cho V3-beta.
