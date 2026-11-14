# Database benchmark v1 — phạm vi và protocol

Ngày 2026-10-31. Tài liệu này thu hẹp capability audit về domain database hiện tại; formal proof và code debugging được hoàn thiện sau.

## Trạng thái readiness

Sẵn sàng chạy benchmark end-to-end và human review trên 12 SQLite migration/integrity packets hiện có, với raw decision-model choices và protocol xử lý sự cố bên dưới.

Các chỉ số dùng được: workflow success (validation và STOP), final task-contract success từ evaluator-only checker, agent calls, token usage đã ghi nhận, API latency và role/transition coverage. Chỉ số đúng/sai từng action cần human labels; current evaluator CLI chưa nạp các nhãn này nên action accuracy chưa là chỉ số hoàn tất.

Đây là phạm vi database v1, không phải cam kết mọi dạng database task đã được kiểm tra. Các 12 packet hiện tại không có performance contract riêng. Nếu bổ sung task yêu cầu performance, final-task checker phải kiểm tra cả yêu cầu đó trước khi đưa task vào benchmark chính thức.

## Cấu hình cố định

- Entry point: `scripts/run_shared_environment.py`.
- Config database: `config/shared_environment/database_jev_16k.yaml`.
- 10 database agents và STOP; giữ cả action có thể bị public precondition rejection để model tự chọn.
- `observable_rejection_guard_enabled: false`. Shared router factory mặc định false và truyền đúng constructor argument; explicit true chỉ dùng cho protocol riêng được công bố.
- Executor `qwen/qwen3.5-9b`, 16384 tokens, 8 calls, task/role/public terminal contracts giữ nguyên.
- Giữ flow hỏi router cuối khi budget bằng 0; evaluator loại các action bị chặn chỉ do budget, không loại mọi STOP hoặc precondition rejection.
- Không thêm lời giải, tự sửa SQL, tự recovery hoặc yêu cầu pass rate tối thiểu.

## Xác nhận đối với 90 quyết định đã có

Run: `database_shared_jev_16k_full_20261031`.

- 90/90 decisions có đầy đủ probabilities trên tập 10 agents + STOP.
- 0 decisions có `guard_blocked_actions`.
- Replay offline 90 stored observations qua guard bật và tắt cho cùng effective observation/questions/choices ở cả 90 trường hợp.
- Không gọi hosted model khi kiểm tra replay.

Evidence: `runs/database_benchmark_readiness_20261031/masking_confirmation.json`.

Guard được bật về cấu hình trong run lịch sử nhưng không thay đổi action space ở bất kỳ decision nào. Tắt guard cho future runs không yêu cầu chạy lại 90 decisions để sửa masking. Raw trace, resolved config, source hashes và các nhãn pending của run lịch sử được giữ nguyên. Code hiện tại khác revision lịch sử ở phần factory wiring; không viết lại các hash lịch sử để che thay đổi này.

## Xử lý thiếu trace/lỗi hạ tầng bằng rerun

Không sửa partial-trace code trong đợt database này theo lựa chọn của người dùng.

Đối với database hiện tại, mỗi role có đúng một tool capability; role invocation kết thúc sau tool đó. Không có request API thứ hai sau khi tool đã chạy trong cùng invocation. Vì vậy, lỗi mất partial trace được tái hiện trong multi-tool actor không đi qua đường database v1. Run hiện tại cũng có 0 role transport failures và 0 multi-tool invocations.

Nếu vẫn gặp invocation bị gián đoạn, thiếu record hoặc lỗi dịch vụ:

1. Giữ attempt lỗi cùng logs/workspace; đánh dấu không đủ điều kiện đo coordination. Không biến thiếu trace thành nhãn routing sai và không xóa attempt đó.
2. Chạy lại toàn bộ packet từ public source DB trong workspace và run ID mới; không tiếp tục trên DB có side effect không rõ.
3. Giữ task, role cards, actor/router config, token/call budget. Với CLI hiện tại, seed per-packet là base seed + ordinal trong danh sách đã chọn; nếu cần giữ đúng seed của full run thì config của single-packet retry phải đặt base seed bằng seed thực tế đó. Rerun là trajectory mới, không khôi phục chính xác trajectory cũ.
4. Protocol vận hành ban đầu: tối đa một full-episode rerun do sự cố hạ tầng/trace cho mỗi packet, giống nhau cho mọi decision model. Nếu tiếp tục lỗi, báo unmeasurable/service failure và coverage; không gộp thành task fail.
5. Lưu mapping accepted attempt cho từng packet và số attempt hạ tầng bị loại khi tổng hợp cohort. Evaluator CLI hiện chưa tự chọn replacement giữa nhiều run; bước chọn cohort này do người chạy thực hiện rõ ràng, không âm thầm dùng best-of-N.

SQL sai, output rỗng theo token cap, chọn agent không phù hợp, STOP sớm và hết call budget là outcomes của benchmark; không dùng chính sách này để rerun cho tới khi task pass.

## Điều kiện giữ trạng thái ready

- Mọi accepted episode có trace đầy đủ và không có lỗi environment/service chưa được phân loại.
- Cùng public observation/action protocol, actor configuration và budgets cho các decision models được so sánh.
- Các task mới phải có final checker bao phủ đúng public contract; các attempt không chấm được có status và denominator riêng.
- Không đặt điều kiện orchestrator hoặc MAS phải đạt một tỷ lệ success cao.
- Human state labels vẫn pending cho tới khi reviewer thực sự xác nhận.

## Kiểm thử thay đổi guard

Ba regression tests trong `tests/test_shared_router_protocol.py` xác nhận default off, explicit false có tác dụng và explicit true chỉ lọc trong chế độ chủ động bật. 15 database contract tests hiện tại vẫn đạt. Không chạy thêm live trajectory chỉ để xác nhận factory/config change này.
