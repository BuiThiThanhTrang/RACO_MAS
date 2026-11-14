# RoleRouteBench Track A — Báo cáo implementation và Terra pilot

Ngày cập nhật: 2026-11-07

## Kết quả

Môi trường Track A bốn domain đã được implement bên ngoài frozen V4 namespace.
Collector tái sử dụng đúng canonical packet-to-run map của Track B và copy exact
stored router observation, không dựng lại state từ workspace cuối.

Artifact đã tạo:

- 82 canonical episodes;
- 613 raw decision states;
- 568 primary Track A states;
- 45 `suppressed_by_budget` audit-only states;
- 568 state thuộc ba cohort rời nhau;
- 568/568 primary state có schema-valid Terra pre-label;
- không còn primary state chưa có LLM attempt.

Sample phân bố tự nhiên thành 23 code debugging, 20 database, 28 formal proof và
29 structured data states. Decision-ID list và SHA-256 của sample được freeze
trong `labeling/sample_manifest.json`.

Cohort thứ hai dùng seed `20261106`, lấy 100 state từ 468 state chưa gọi LLM và
freeze trong `labeling_openrouter_100_v1/sample_manifest.json`. Phân bố là 23
code debugging, 18 database, 32 formal proof và 27 structured data; overlap với
cohort đầu bằng 0.

Cohort thứ ba `labeling_openrouter_remaining_v1` chứa toàn bộ 368 state chưa có
trong hai cohort đầu, freeze bằng seed `20261107`. Phân bố là 97 code debugging,
46 database, 125 formal proof và 100 structured data; hợp ba cohort phủ đúng
568/568 primary state và không có overlap.

## LLM run

Exact model là `gpt-6.1-terra`, reasoning effort `high`, một isolated request cho
mỗi state, strict JSON Schema và `max_output_tokens=25000`.

Workspace không có `OPENAI_API_KEY`, nên pilot dùng authenticated Codex CLI.
Provenance ghi `model_provider=codex_exec_openai`; không giả định đây là direct
Responses API. CLI wrapper thêm system context, vì vậy input-token usage cao hơn
payload-only estimate và không nên dùng làm direct-API baseline.

Kết quả usage:

| Metric | Value |
| --- | ---: |
| Successful pre-labels | 100 |
| Attempts | 114 |
| Transport failures | 14 |
| Input tokens | 2,298,681 |
| Output tokens | 56,680 |
| Reported reasoning tokens | 11,337 |
| Total tokens | 2,355,361 |
| Estimated standard API equivalent | USD 2.582081 |
| Mean latency | 15.48 seconds/state |

Hai transport failures đầu do Windows default encoding; implementation đã pin
UTF-8. Mười hai failure còn lại do usage limit ở sáu state. Resume sau reset đã
hoàn tất cả sáu; failed attempts vẫn được giữ append-only. Không có response nào
fail label schema.

### OpenRouter cohort thứ hai

Cohort thứ hai gọi trực tiếp OpenRouter Chat Completions bằng model ID
`openai/gpt-6.1-terra`, reasoning effort `high`, strict JSON Schema và giữ nguyên
prompt/schema version của Track A. OpenRouter response metadata được lưu cho từng
pre-label, gồm response ID, resolved model/provider, token usage và provider cost.

| Metric | Value |
| --- | ---: |
| Successful pre-labels | 100 |
| Attempts | 180 |
| Transport failures | 80 |
| Input tokens | 508,887 |
| Output tokens | 153,408 |
| Reported reasoning tokens | 108,466 |
| Total tokens | 662,295 |
| OpenRouter reported cost | USD 1.40307375 |
| Mean latency | 21.33 seconds/state |

Sau khi 60 state đầu hoàn tất, một sự cố kết nối tạo 80 failed-attempt rows cho
40 state (hai lần thử/state). Lệnh resume giữ nguyên audit log, bỏ qua 60 state
đã xong và hoàn tất 40 state còn lại. Không có schema failure; failed attempts
không được tính là pre-label.

### OpenRouter cohort 368 state còn lại

Cohort cuối giữ nguyên model, reasoning effort, prompt và strict schema của
cohort OpenRouter trước. Kết quả:

| Metric | Value |
| --- | ---: |
| Successful pre-labels | 368/368 |
| Attempts | 1,592 |
| Transport failures | 1,224 |
| Input tokens | 2,002,129 |
| Output tokens | 534,455 |
| Reported reasoning tokens | 372,094 |
| Total tokens | 2,536,584 |
| OpenRouter reported cost | USD 5.16547785 |
| Mean latency | 18.40 seconds/state |

Toàn bộ 368 successful responses dùng exact model ID
`openai/gpt-6.1-terra`; OpenRouter resolved provider là OpenAI cho cả 368.
Transport failures gồm 736 attempt khi sandbox chưa có quyền mạng và 488 attempt
sau một lần mất kết nối giữa run. Chúng không tạo pre-label; resume bỏ qua 124
state đã thành công rồi hoàn tất 244 state còn lại. Không có schema failure.

### Tổng pre-labeling toàn Track A

| Metric | Value |
| --- | ---: |
| Schema-valid pre-labels | 568/568 |
| Recorded attempts | 1,886 |
| Transport failures | 1,318 |
| Input tokens | 4,809,697 |
| Output tokens | 744,543 |
| Reported reasoning tokens | 491,897 |
| Total tokens | 5,554,240 |
| OpenRouter reported cost (468 states) | USD 6.56855160 |

Tất cả 1,318 transport-failure records có zero usage. Chúng được giữ để audit
nhưng không phát sinh pre-label hoặc token cost. Trong 468 successful OpenRouter
responses, resolved provider là OpenAI cho 466 và Azure cho 2; exact model ID
vẫn là `openai/gpt-6.1-terra` cho tất cả.

## Validation

Final validator pass với `valid=true` và kiểm tra:

- exact source observation equality cho 613/613 rows;
- state/decision ID-set equality;
- 568/45 primary/audit split;
- primary domain counts 84/143/185/156;
- full candidate pool cộng `STOP`;
- unique primary fingerprints;
- đúng 568 IDs thuộc ba cohort không giao nhau có pre-labels;
- không có primary state thiếu LLM attempt hoặc pre-label;
- pre-label state hash, full action coverage/order và STOP/terminal consistency.

Unit tests cho deterministic sampling, action coverage/order và terminal/STOP
contract đều pass. Python compile check cũng pass.

## Human review

Khởi động review UI:

```powershell
cd D:\SideProject\RACO_MAS\puppeteer
python scripts\serve_roleroutebench_track_a_review.py
```

Mặc định UI nạp cả ba cohort. Có thể chọn riêng cohort qua sidebar hoặc lệnh,
ví dụ cohort 368 state cuối:

```powershell
python scripts\serve_roleroutebench_track_a_review.py --labeling-dir labeling_openrouter_remaining_v1
```

Mỗi sample được trình bày theo năm tab:

1. `Decision brief`: task contract, budget, terminal status và fact chính có
   đường dẫn về source field;
2. `role cards`: toàn bộ router-visible agent pool đúng source order, không dùng
   Terra để lọc;
3. `Action review`: mọi candidate action cộng `STOP`, classification/rationale,
   phase, terminal flag và confidence;
4. `Evidence`: full artifacts, recent events và full `current_state`;
5. `Audit JSON`: exact stored `router_input_json`, Track A row và Terra
   provenance, không cắt field.

Reviewer có thể accept proposal, sửa từng action hoặc chuyển adjudication. Mỗi submit được append vào
`reviews/human_reviews.jsonl`; không ghi đè LLM output.

Sau review:

```powershell
python scripts\merge_roleroutebench_track_a_reviews.py
python scripts\report_roleroutebench_track_a.py
python scripts\validate_roleroutebench_track_a.py
```

Report human review sẽ tính exact pre-label acceptance, số action-class sửa,
STOP-class corrections và unsafe-set corrections. Chỉ
`label_status=human_verified` mới trở thành evaluator label.

## Thành phần đã implement

- namespace `routing_benchmark/roleroutebench/track_a/` cho collector, schema,
  sampling, labeling, validation, review, reporting, evaluator và packaging;
- source/config manifest có revisions, 82 packet mapping và input checksums;
- build, validate, label, report, review, merge, evaluate và HF packaging scripts;
- Streamlit human-review UI;
- Parquet package gồm 568 states, 613 provenance rows và 568 pre-labels thuộc
  ba cohort;
- direct Responses API client sẵn sàng khi có `OPENAI_API_KEY`, bên cạnh explicit
  Codex CLI fallback dùng trong pilot này.

Track A chưa có ground-truth labels cho tới khi human review hoàn tất. Toàn bộ
568 pre-label vẫn chỉ là proposal, không phải evaluator gold labels.
