# Báo cáo triển khai shared environment

Ngày 2026-10-31. Implementation và kiểm tra kỹ thuật hoàn tất cho phạm vi đã chốt; human state-label review vẫn pending.

**Cập nhật sau capability audit:** bản đầu đã chạy được nhưng chưa đạt đầy đủ điều kiện benchmark chung. Các negative/failure-path probes mới phát hiện false positive của formal verifier, final evaluator chưa hỗ trợ đủ domain và partial trace bị mất khi API lỗi. Xem `shared_environment_capability_audit_vi.md` và evidence `runs/shared_environment_capability_audit_20261031/probe-001/findings.json`. Các tỷ lệ database bên dưới giữ nguyên phạm vi của run đã lưu; không dùng trạng thái hoàn tất implementation để suy ra benchmark-ready.

## Phạm vi đã triển khai

- Production entry point/config độc lập worktree cá nhân; các execution/artifact modules cần thiết và dataset packets được đưa về canonical checkout hiện tại.
- Một `SharedTrajectoryRunner` dùng cùng interface router cho database, formal proof và external code debugging; adapter/tool/runtime riêng theo domain.
- Database adapter đủ 10 role capabilities: schema introspection, profiling/query evidence, analysis/planning/repair work products, SQL authoring/execution, integrity checks, performance probes, snapshot và restore.
- Router và bounded executor nhận public state từ cùng adapter. Artifact có hash/version/input refs; stale evidence được đánh dấu, không ghi đè lịch sử.
- SQL lỗi và execution feedback được chuyển vào context của agent. Blank output không được materialize thành candidate mới; usage/finish_reason giữ ở metadata để thống kê hậu kỳ.
- Validation dùng bản sao SQLite; transaction failure không làm driver tự restore database nguồn. Snapshot restore cần operation và snapshot ID rõ ràng.
- Giữ nguyên lần hỏi router cuối tại budget 0; evaluator chỉ loại action không được dispatch do budget khỏi primary routing denominator, giữ STOP và rejection semantics.
- Human-review JSONL và read-only HTML không dùng behavior action làm gold. Không có preferred action, tự sửa SQL hoặc buộc chọn trình tự role.

## Xác nhận kỹ thuật

`runs/shared_environment_verification/report.json`: 25 tests, 0 failures, 0 errors.

Các kiểm thử dùng SQLite thật, checkout QuixBugs/reproducer thật, Lean 4.8/Mathlib và kernel/assumption checks thật. Có các case feedback propagation, artifact freshness, read-only permission, transaction giữ commit trước, explicit snapshot restore, query-plan probes, thay đổi theorem và thêm axiom bị từ chối. Code/formal được smoke qua cùng runner bằng candidate fixture; đây không phải full live evaluation hai domain đó.

## Database live run

Run chính: `runs/database_shared_jev_16k_full_20261031`.

Config: Jev `typesafe/jev-1.13`, SLM `qwen/qwen3.5-9b`, max_tokens 16384, 8 executor calls, temperature 0, seed 42. Không pin/tối ưu provider hoặc thay task/reference để tăng success.

| Chỉ số | Kết quả |
|---|---:|
| Packet/trajectory hoàn tất | 12 |
| Pre-decision states / router decisions | 90 |
| Role invocations thực thi | 78 |
| STOP được xử lý | 6 |
| Decisions bị chặn chỉ do budget | 6 |
| Primary routing denominator | 84 |
| LLM responses/context được đối chiếu | 37 |
| Workflow có validation đạt và STOP | 6/12 |
| Artifact cuối đạt kiểm tra evaluator-only | 7/12 |
| Source files được kiểm tra freeze | 289, không đổi |
| Human action-label accuracy | Chưa chấm |

Khác biệt giữa 6 workflow success và 7 end-task success đến từ `db-alembic-batchapplytest-test-rename-col-w-index`: database cuối đạt task checks, nhưng thiếu validation trước khi hết budget. Không chuyển evaluator-only evidence này thành kiến thức router.

Run đã chọn 6 role types: Schema Inspector, Data Profiler, SQL Migration Author, Transactional Dry-run Executor, Integrity Validator và Constraint Analyzer. Bốn role còn lại vẫn có implementation/capability tests; không ép router gọi chúng để làm đẹp coverage. Mở rộng task diversity là đợt sau.

`execution_validation_report.json` xác nhận joins decision/execution, visibility, 37 actor/router context snapshots, 16k cap và source freeze; không có validation errors. Đây là validation cấu trúc/protocol, không xác nhận mọi quyết định Jev là hợp lệ.

## Artifacts để review

- `evaluation_report.json`: kết quả từng packet, usage/latency/coverage và budget accounting.
- `execution_validation_report.json`: structural/context/freeze checks.
- `human_state_review_queue.jsonl`: 90 states, 84 eligible primary decisions; label pending.
- `human_state_review.html`: viewer đọc-only.
- `episodes/<packet>/`: decisions, role executions, model requests/responses, tool evidence và working artifacts.

Review key: `(run_id, decision_id)`.

Live smoke trước full run được lưu riêng tại `database_shared_jev_16k_smoke_20261031`; smoke đạt task và đã xác nhận freeze tại thời điểm kết thúc. Sau smoke có cập nhật xử lý lỗi transport và biểu diễn BLOB trước khi đóng băng full run. Validator lưu rõ smoke thuộc revision trước, không giả định hash checkout hiện tại phải trùng revision lịch sử. Full run dùng strict current-source validation và khớp toàn bộ 289 hashes.

## Giới hạn phạm vi và bước tiếp theo

- Human review chưa hoàn tất; không tuyên bố release-ready hoặc dùng behavior labels làm ground truth.
- Các tỷ lệ success là của cấu hình/run này, không phải phép so sánh có kiểm soát với collector cũ.
- Tập 12 task chưa bao phủ hết role/trajectory patterns. Chỉ bổ sung task sau khi chốt protocol này, theo dependency/data-repair/recovery/performance contracts công khai.
- Resume CLI xác nhận source/config identity và bỏ qua episode đã hoàn tất; không tự khôi phục một workspace bị gián đoạn giữa invocation.
- Toàn bộ code/formal packets chưa được chạy lại live trong đợt này; adapter compatibility và smoke thực tế đã được kiểm tra như plan.
