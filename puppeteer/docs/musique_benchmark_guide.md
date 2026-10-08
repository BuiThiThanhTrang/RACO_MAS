# MuSiQue benchmark trong Puppeteer

## Mục tiêu

MuSiQue được dùng để đánh giá routing động trên câu hỏi nhiều bước nhưng vẫn giữ
môi trường đóng: mọi đoạn văn cần thiết đều nằm trong input. Điều này tách chất
lượng orchestration khỏi lỗi web search và phù hợp để so sánh cùng một planner
trên actor backbone nhỏ và lớn.

Gold answer, aliases, decomposition và supporting paragraph IDs chỉ được đưa vào
evaluator sau khi reasoning kết thúc. Planner và actor chỉ nhận câu hỏi cùng các
candidate paragraphs.

## Agent pool

Hai pool dùng cùng sáu role và cùng role card; chúng chỉ khác actor backbone:

1. **Task Decomposer**: tách câu hỏi thành các sub-question phụ thuộc nhau.
2. **Evidence Retriever**: chọn và trích dẫn evidence bằng ID `[P…]`.
3. **Bridge Entity Reasoner**: giải hop trung gian để tìm bridge entity.
4. **Comparison & Composition Reasoner**: so sánh hoặc kết hợp evidence.
5. **Evidence Verifier**: kiểm tra candidate answer với chuỗi evidence đã có.
6. **Answer Integrator**: tạo short answer cuối từ evidence đã được xác minh.

Stage guard `musique_stage_v1` không cho gọi Bridge Entity Reasoner trước khi có
evidence, và không cho gọi Verifier/Integrator trước khi có candidate answer. Một
role không được lặp trên state không đổi.

## Dữ liệu

Split validation chính thức đã được đặt tại `data/MuSiQue/validation.parquet`.
Tải lại hoặc tải thêm split bằng:

```powershell
python -m scripts.download_musique --split validation
python -m scripts.download_musique --split all
```

Manifest chứa nguồn, số dòng, kích thước và SHA-256 được lưu tại
`data/MuSiQue/dataset_manifest.json`.

## Cấu hình khuyến nghị

Mặc định dùng `W2D5`: hai reasoning path, mỗi path tối đa năm agent activation.
Depth 5 cho phép chuỗi decompose/retrieve → bridge/compose → verify → integrate;
router không được dừng sau Retriever/Verifier khi chưa có final answer hợp lệ.
Dùng cùng seed, data range, planner và aggregator khi
so sánh hai backbone.

Smoke test Qwen 3.5 9B:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_naive_jev_qwen.yaml `
  --policy_mode frozen `
  --data_limit 20 `
  --run_id "musique_jev_qwen_smoke"
```

Đối chứng Gemini 2.5 Flash:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_naive_jev_gemini.yaml `
  --policy_mode frozen `
  --data_limit 20 `
  --run_id "musique_jev_gemini_smoke"
```

Thêm `--hop_count 2`, `--hop_count 3` hoặc `--hop_count 4` để đánh giá riêng theo
số hop. Với thí nghiệm chính, nên chạy toàn bộ validation rồi báo cáo cả overall
và kết quả phân tầng theo hop count.

## Metrics

Mỗi sample ghi các nhóm metric sau vào result JSONL:

- Answer quality chính thức: Exact Match và token F1.
- Semantic outcome: `semantic_correct` và verdict của semantic judge. Đây là
  nhãn riêng, không ghi đè EM/F1 hay trường `official_correct`.
- Router CS hậu task: planning score, handoff communication score, CS raw/100
  và semantic verdict được trả về trong cùng một structured Luna call.
- Evidence quality có hai view:
  - `support_*`: diagnostic union trên toàn bộ trace, dùng để biết hệ thống đã
    từng tìm thấy evidence nào;
  - `paper_compatible_support_*`: chỉ chấm `supporting_paragraph_ids` được emit
    cùng terminal answer của path mà aggregation chọn. Metric này không mượn
    citation từ path khác hoặc step trước, nên phù hợp hơn để so với MuSiQue.
- Collaboration: milestone achievement, useful handoff, collaboration
  effectiveness, recovery success, redundant transition, useful-call ratio và
  contribution theo role.

`collaboration_effectiveness` bằng milestone achievement rate nhân useful handoff
rate. Giá trị này chỉ có khi trace thật sự có handoff; không ép task một-agent
thành điểm 0.

Tổng hợp một run và phân tầng theo hop count:

```powershell
python -m scripts.analyze_musique_run runs/musique_jev_qwen_full `
  --output runs/musique_jev_qwen_full/musique_summary.json
```

Khi thư mục `results` có cả file canonical và `.rejudged.jsonl`, analyzer mặc
định chỉ đọc file canonical để tránh đếm cùng task hai lần. Truyền trực tiếp
đường dẫn `.rejudged.jsonl` nếu muốn phân tích riêng kết quả hậu kiểm.

Các collaboration metrics dùng gold decomposition chỉ ở giai đoạn hậu kiểm. Vì
vậy chúng đo được agent nào tạo milestone hữu ích mà không làm rò rỉ nhãn vào
router hay actor.

## Candidate và final-answer contract

Retriever/Verifier output không còn được coi là terminal chỉ vì nó chứa một
answer được mang từ step trước. Router phân biệt `candidate_answer` với
`final_answer`; khi Verifier chỉ trả `corrected_answer`, candidate được giữ lại và
chỉ Answer Integrator còn eligible. STOP chỉ được mở khi output có
`final_answer` hoặc `FINAL ANSWER:` ngắn, hợp lệ.

Runtime ưu tiên các trường `final_answer`, `corrected_answer`,
`candidate_answer` hoặc marker `FINAL ANSWER:` đã tồn tại. Nó bỏ citation dạng
`(P2, P8)`/`[P2, P8]` và wrapper trình bày. Nếu output của một answer-capable
role vẫn không parse được, một LLM fallback gold-blind được phép trích đúng một
span liên tục đã xuất hiện trong output. Fallback không được giải lại câu hỏi,
sửa đáp án hay tạo text mới. Kết quả được cache trong phạm vi task.

## Official score và semantic outcome

Exact Match chỉ chạy sau khi router, actor và aggregator đã hoàn tất. Các trường
gold bị loại khỏi `GlobalInfo`, vì vậy semantic judge không thể ảnh hưởng route
của task hiện tại. Sau `prediction_committed`, hệ thống lưu hai nhóm nhãn:

- `answer_em`, `answer_f1`, `official_correct`: metric benchmark chính thức;
- `semantic_correct`, `semantic_success`: tương đương ngữ nghĩa do judge xác
  nhận, ví dụ tên đầy đủ kèm acronym.

Khi combined judge được bật, Luna được gọi đúng một lần sau mỗi task. Một phần
input chứa trace routing để chấm planning và handoff; phần còn lại chỉ chứa
original question, committed prediction và accepted answers để trả
`EQUIVALENT`, `NOT_EQUIVALENT` hoặc `UNCERTAIN`. Prompt cấm dùng correctness để
tăng hoặc giảm collaboration score, đồng thời cấm tìm một đáp án khác trong
trace. Exact match đúng luôn được giữ là semantic success kể cả khi judge trả
`UNCERTAIN`. `UNCERTAIN` và lỗi API mặc định fail closed về kết quả EM.

Các consumer được điều khiển bởi `use_for`:

```yaml
global_config:
  musique:
    semantic_outcome:
      enabled: true
      model: gpt-6-luna-openrouter
      reasoning_effort: low
      failure_policy: exact_match
      use_for:
        - reporting
        - route_experience
        - profile_evidence
        - training_reward
```

Trong frozen naive, nhãn semantic chỉ bổ sung báo cáo. Trong evolving hoặc
training, nhãn này ngăn một biến thể text hợp lệ bị ghi thành route failure,
capability failure hoặc negative policy reward. Đây là nhãn outcome cấp task;
nếu dùng cho profile/training thì các path nhận cùng nhãn cuối, không phải credit
attribution riêng cho từng agent. `correct` vẫn dựa trên EM để accuracy chính
thức không bị thay đổi âm thầm.

`final_metrics.router_cs` và result field `router_cs` chứa hai điểm CS, rationale,
failure tags, model và token usage. `semantic_evaluation` chứa cùng judge receipt
cùng answer verdict. Do cả hai đến từ `router_cs_semantic_v2`, không còn một API
call semantic riêng sau CS.

Chấm lại log cũ bằng combined judge:

```powershell
python scripts/evaluate_multiagentbench_cs.py `
  runs/musique_dynamic_jev_gemini `
  --model gpt-6-luna-openrouter `
  --reasoning_effort low `
  --overwrite
```

Sidecar tạo ra chứa cả `official_correct`, `semantic_correct`, `answer_verdict`
và các trường CS. `--dry_run --limit 1` kiểm tra input mà không gọi API.

## LLM fallback trong runtime

Fallback được cấu hình dưới routing guard:

```yaml
policy:
  routing_guard:
    answer_extraction:
      enabled: true
      model: gpt-6-luna-openrouter
      reasoning_effort: low
      max_repair_attempts: 1
      roles:
        - Comparison & Composition Reasoner
        - General Evidence Solver
        - Evidence Verifier
        - Answer Integrator
```

LLM chỉ được gọi nếu structured fields và deterministic parser đều không tìm
được candidate. `FINAL` mở điều kiện STOP, `CANDIDATE` giữ Answer Integrator hoặc
Verifier trong pool, còn `NO_ANSWER` giữ state chưa hoàn tất. Nếu fallback lỗi,
runtime fail closed và tiếp tục dùng kết quả parser rỗng thay vì bịa đáp án.
