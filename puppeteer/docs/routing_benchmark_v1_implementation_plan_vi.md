# Kế hoạch triển khai RoutingBench V1

## 1. Diễn giải chính xác Track B

Trong bản benchmark routing thuần, Track B **không gọi các LLM actor thật**. Router vẫn hoạt động tương tác qua nhiều bước, nhưng agent được chọn sẽ được thực thi bởi một actor contract xác định:

1. Router nhận `task + observed state + budget + qualitative role cards`.
2. Router chọn đúng một `next agent` hoặc `STOP`.
3. Simulator kiểm tra precondition của agent trong state ẩn.
4. Nếu action hợp lệ, simulator áp dụng state transition và trả status/evidence marker mới.
5. Router nhận state mới và tiếp tục cho tới khi tự chọn `STOP` hoặc hết budget.

“Tự giải toàn bộ workflow” vì vậy nghĩa là router phải tự tạo cả trajectory quyết định, không có oracle chọn hộ các bước sau. Nó không có nghĩa là gọi LLM agent và chấm chất lượng văn bản do agent sinh ra.

Việc chạy lại cùng một router trên cùng seed tạo ra cùng môi trường và outcome. Nhờ đó, lỗi reasoning của Qwen/Gemini/GPT không làm nhiễu phép đo routing. Một live-execution track có thể được thêm sau như phép đo end-to-end riêng, nhưng không thuộc core score của V1.

## 2. Nguyên tắc thiết kế

- Input chuẩn: `task`, observed `state`, budget/config và pool role cards.
- Output chuẩn: một `agent_id` hoặc `STOP`.
- Ground truth là **tập action hợp lệ**, vì một state có thể có nhiều bước tiếp theo tương đương.
- Hidden oracle state chứa transition và goal; router không thấy `valid_actions` hay goal condition.
- Profile chỉ chứa `role_goal`, `use_when`, `avoid_when`, `expected_outputs` và action cost.
- Không lưu hoặc tính capability score, backbone score, uncertainty hay empirical rank.
- Backbone không phải một thuộc tính của benchmark actor. Vì vậy router không thể ưu tiên agent chỉ vì model đứng sau role mạnh hơn.
- Distractor label chỉ dùng khi chấm điểm, không xuất hiện trong public role card.

## 3. Phạm vi V1

V1 gồm 10 scenario nhỏ thuộc bảy nhóm hành vi:

- multi-hop evidence;
- parallel join với nhiều next action hợp lệ;
- conflict resolution;
- media/file/tool routing;
- retry sau khi state thay đổi thực chất;
- noisy mixed-domain agent pool;
- terminal/STOP control và tránh over-planning.

Pool có các role chức năng như Decomposer, Retriever, Bridge, Composer, Verifier, Integrator, Media Analyst, File/Table Analyst và Recovery Planner. Ba role không liên quan được thêm làm nhiễu. Tất cả actor đều xác định và không gọi model.

## 4. Hai track và metric

### Track A — Isolated decision evaluation

Benchmark duyệt các state có thể đạt tới trong workflow và yêu cầu router chọn một next action cho từng state độc lập.

Metric chính:

- `valid_next_agent_accuracy`;
- `nonterminal_routing_accuracy`;
- `stop_precision`, `stop_recall`, `stop_f1`;
- `distractor_selection_rate`;
- score theo workflow family.

### Track B — Interactive trajectory evaluation

Router bắt đầu từ initial state, tự chọn toàn bộ chuỗi action, nhận feedback xác định sau mỗi action và tự quyết định lúc dừng.

Metric chính:

- `goal_completion_rate`;
- `valid_action_rate` và `invalid_action_rate`;
- `no_progress_action_rate`;
- `redundant_action_rate`;
- `premature_stop_rate`;
- `distractor_selection_rate`;
- `recovery_success_rate`;
- average steps/cost;
- normalized step regret so với đường hợp lệ ngắn nhất, chỉ tính trên episode thành công.

Các metric này đo chất lượng routing và trajectory. Chúng không dùng correctness của câu trả lời tự nhiên hay chất lượng backbone.

## 5. Baseline

- `random`: chọn ngẫu nhiên từ toàn bộ pool và STOP.
- `fixed`: chuỗi role cố định, dùng để cho thấy hạn chế của topology tĩnh.
- `oracle`: chỉ dùng làm upper-bound và kiểm tra tính hợp lệ của dataset.
- `decision`: router Jev/Clef thật qua Decisions API, chỉ nhìn public observation và qualitative role cards (`jev` vẫn là alias CLI tương thích).

Không có capability-greedy baseline.

## 6. Các pha triển khai và hiệu quả dự kiến

### Pha 1 — Schema và data contract

Tạo schema actor, scenario, transition, event và loader có validation.

Hiệu quả dự kiến: tách hoàn toàn routing correctness khỏi natural-language answer correctness và backbone quality.

### Pha 2 — Deterministic environment

Thực thi action, cập nhật facts, budget, retry status, redundant/no-progress flags và terminal state.

Hiệu quả dự kiến: kết quả có thể reproduce, dễ tạo counterfactual state và phân biệt routing error với reasoning error.

### Pha 3 — Track A/B evaluator

Sinh reachable states cho Track A; chạy full trajectory và oracle shortest path cho Track B.

Hiệu quả dự kiến: vừa đo local decision quality vừa phát hiện error compounding, premature STOP và recovery failure.

### Pha 4 — Router adapter và CLI

Thêm random/fixed/oracle/Jev adapter, audit JSONL và summary JSON.

Hiệu quả dự kiến: cùng một protocol có thể so sánh Jev, Clef hoặc planner khác mà không thay dataset hay metric.

### Pha 5 — Verification

Test invariants, set-valued ground truth, recovery state và oracle đạt 100% trên cả hai track.

Hiệu quả dự kiến: phát hiện scenario bị kẹt, transition sai hoặc goal không thể đạt trước khi benchmark planner thật.

## 7. Tiêu chí nghiệm thu V1

- Loader đọc đủ 10 scenario và từ chối reference không hợp lệ.
- Public observation không chứa valid action, distractor label, backbone hoặc capability score.
- Có state với nhiều action hợp lệ.
- Có episode bắt buộc recovery trước khi retry.
- Chọn STOP sớm bị chấm sai; hoàn thành goal vẫn phải chủ động chọn STOP.
- Oracle đạt 100% Track A và Track B.
- Mỗi run sinh summary cùng audit record/episode để phân tích lại mà không gọi router lần nữa.

## 8. Trạng thái thực hiện

Các pha 1–5 đã được triển khai trong `routing_benchmark/`, dữ liệu nằm tại `data/RoutingBench/v1/`, CLI ở `scripts/run_routing_benchmark.py`, và invariant tests ở `tests/test_routing_benchmark_v1.py`.

V1 cố ý chưa có live LLM actor backend. Khi cần đánh giá end-to-end, nên thêm một track riêng và báo cáo song song `routing-pure` với `live-execution`, không trộn hai loại score.

Ngoài benchmark, hai config MuSiQue dynamic hiện tại cũng đặt `capability_scoring.enabled: false` và `capability_shortlist.enabled: false`. Harness vẫn giữ dataset-scope gate, terminal gate, retry và adaptive branching; nó không tính hoặc gửi `capability_scores` vào state khi công tắc này tắt.
