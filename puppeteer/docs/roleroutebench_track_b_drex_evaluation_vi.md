# RoleRouteBench Track B — Drex v1.5 live evaluation

Ngày chạy: 2026-11-07–2026-11-08 (Asia/Saigon).

## Cấu hình cố định

- Router: `nace-ai/drex-v1.5`, resolved snapshot
  `nace-ai/drex-v1.5-20261009`, OpenRouter Decisions API.
- Orchestration: `structured`, full pool, observable guard tắt.
- Mỗi episode tối đa 8 executor calls.
- Packet, role card, executor, tool contract, runtime profile và evaluator giữ
  nguyên theo Track B beta.2.
- Executor: Qwen 3.5 9B cho database/structured data, Gemini 2.5 Flash cho code,
  Goedel Prover V2 8B cho formal proof.

## Kết quả

| Domain | Episodes | Raw runner pass | Independent end-task pass | Infrastructure affected | Trạng thái |
| --- | ---: | ---: | ---: | ---: | --- |
| Database | 12 | 5 | 6 | 0 | Accepted |
| Code debugging | 20 | 1 | 11 | 0 | Accepted |
| Formal proof | 30 | 0 | 0 | 30 | Rejected |
| Structured data | 20 | 2 | 2 | 0 | Accepted |
| Tổng | 82 | 8 | 19 | 30 | 52 episodes evaluable |

Formal proof không được tính vào performance: cả 30 episode gặp executor-service
HTTP 400 trước khi nhận được response từ Goedel Prover. Trace chứa 239 executor
transport failures, 0 executor LLM responses và chỉ 1 tool call. Đây không phải
bằng chứng rằng Drex thất bại trên formal proof.

Trên 52 episode hợp lệ, independent end-task pass là **19/52 = 36.54%**. JEV
reference trên cùng ba domain là **20/52 = 38.46%** (database 7, code 12,
structured data 1). Drex thấp hơn 1 sample, tương đương 1.92 điểm phần trăm.

## Hành vi routing

Trên ba domain hợp lệ, chỉ 8/52 episode kết thúc bằng STOP và cả 8 đều raw-pass;
44/52 episode chạm call limit. Independent evaluator lại tìm thấy 19 task đã đạt
end-task contract. Như vậy Drex có STOP precision cao nhưng termination recall
thấp: ít nhất 11 task đã có kết quả đạt contract nhưng router vẫn tiếp tục cho tới
hết budget. Sai khác rõ nhất ở code debugging: raw runner chỉ công nhận 1 pass,
trong khi fresh independent checker xác nhận 11 patch đúng.

Database dùng trung bình 3.5 non-STOP role khác nhau mỗi episode và có tỷ lệ lặp
action liên tiếp 8.70%. Code dùng 3.15 role nhưng lặp action liên tiếp 63.57%; các
lựa chọn tập trung vào Regression Verifier và Patch Author. Structured data dùng
3.95 role, lặp liên tiếp 30.15%, nhưng có 36 precondition rejections. Điều này cho
thấy Drex có thể tạo workflow đa role nhưng kiểm soát vòng lặp, precondition và
termination chưa ổn định.

## Usage và integrity

- 711 decision records; 637 primary routing decisions; 74 quyết định cuối bị
  suppressed do hết budget.
- 3,282,067 router input tokens; nominal router cost khoảng USD 0.131 ở mức
  USD 0.04/M input tokens.
- Mean decision latency khoảng 1,009 ms.
- 711/711 decision resolve đúng snapshot; mọi decision có full probability vector.
- `freeze_verification.unchanged=true` cho cả bốn run.
- Code independent evaluation ban đầu bị sandbox chặn network. Artifact lỗi được
  giữ dưới `evaluation_sandbox_failed_20261108`; clean network-enabled evaluation
  sau đó chấm đủ 20/20 episode và không có infrastructure-affected episode.

## Kết luận

Drex là baseline decision model nhỏ có giá trị cho Track B. Trên ba domain hợp
lệ, end-task performance gần JEV hơn nhiều so với raw runner pass, nhưng model
thường không nhận ra lúc cần STOP và dễ lặp role tới hết budget. Formal proof phải
chạy lại sau khi executor endpoint hoạt động; không được điền 0/30 vào bảng so
sánh router chính thức.
