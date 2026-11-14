# RoleRouteBench Track A — Sol router evaluation

Ngày chạy: 2026-11-07

## Cấu hình

- router: `openai/gpt-6-sol` qua OpenRouter Chat Completions;
- reasoning effort: `low`;
- orchestration profile: `structured`;
- output: strict JSON Schema gồm đúng một `action` và một `rationale`;
- candidate pool: full domain pool cộng `STOP`, không masking/guard;
- input: exact stored public `router_input_json` của 568 primary Track A states;
- prompt version: `roleroutebench.track_a.sol_router.v1`.

Runner không đọc Terra labels hoặc JEV action. Scorer riêng chỉ nạp nhãn sau khi
predictions đã được ghi. Validation xác nhận 568 unique predictions, exact
decision-ID coverage, state hashes khớp và zero action ngoài candidate pool.

## Kết quả với Terra pre-label làm ground truth tạm thời

| Metric | Sol | JEV | Delta Sol − JEV |
| --- | ---: | ---: | ---: |
| Set-valued accuracy | 534/568 = 94.01% | 486/568 = 85.56% | +8.45 pp |
| Executor-only accuracy | 508/542 = 93.73% | 449/531 = 84.56% | +9.17 pp |
| Singleton exact agreement | 151/174 = 86.78% | 150/174 = 86.21% | +0.57 pp |
| STOP precision | 100% | 100% | 0 pp |
| STOP recall | 52% | 74% | −22 pp |

Theo domain:

| Domain | Sol | JEV | Delta |
| --- | ---: | ---: | ---: |
| Code debugging | 79.72% | 69.93% | +9.79 pp |
| Database | 97.62% | 97.62% | 0 pp |
| Formal proof | 99.46% | 96.76% | +2.70 pp |
| Structured data | 98.72% | 80.13% | +18.59 pp |

Paired comparison: cả hai đúng 474 states; Sol-only đúng 60; JEV-only đúng 12;
cả hai sai 22. Exact two-sided McNemar p-value là `8.06e-09`. Sol và JEV chọn
cùng literal action ở 390/568 states (68.66%).

Sol sai 34 states theo Terra: 32 `redundant_no_progress`, một
`invalid_precondition`, một `unsafe`. Điểm yếu tập trung ở terminal phase
(28/50 = 56%): Sol không chọn sai STOP, nhưng chỉ dừng ở 26/50 terminal states.
JEV dừng đúng 37/50 terminal states.

## Usage và provenance

- 568 successful responses, 12 transport failures do weekly key limit trước khi
  limit được tăng;
- input tokens: 3,735,501;
- output tokens: 46,329;
- reasoning tokens: 11,608;
- total tokens: 3,781,830;
- OpenRouter reported cost: USD 9.8011905;
- resolved model/provider: `openai/gpt-6-sol` / OpenAI cho 568/568 responses.

## Giới hạn diễn giải

Terra labels vẫn là LLM proposals, chưa phải human-verified gold. Vì vậy các số
trên đo agreement với Terra và chỉ trở thành benchmark accuracy chính thức sau
human review/adjudication.
