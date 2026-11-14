# Shared environment: chạy và review trajectory

Entry point chung: `scripts/run_shared_environment.py`. Các domain dùng cùng `SharedTrajectoryRunner`, interface router `reset/select`, role executor và execution accounting. Adapter quản lý công cụ/artifact riêng cho từng domain.

## Cài đặt và database run

Chạy từ thư mục `puppeteer`:

```powershell
python -m pip install -r requirements-shared-environment.txt
python -B scripts/run_shared_environment.py --config config/shared_environment/database_jev_16k.yaml --run-id database_shared_jev_16k_full_20261031
python -B scripts/evaluate_shared_environment.py runs/database_shared_jev_16k_full_20261031
python -B scripts/validate_shared_environment.py runs/database_shared_jev_16k_full_20261031
python -B scripts/export_shared_review.py runs/database_shared_jev_16k_full_20261031
```

API key được lấy từ environment hoặc `.env` của project; không ghi key vào config hay log. Không dùng lại run ID đã tồn tại nếu không có `--resume`. Resume xác nhận source/config identity và bỏ qua episode đã hoàn thành; workspace đang chạy dở bị từ chối thay vì tự xóa/reset.

Config database chốt Jev `typesafe/jev-1.13`, executor `qwen/qwen3.5-9b`, 16384 tokens cho mỗi LLM request, timeout 300 giây, 8 executor calls. Một role invocation có tối đa 3 tool calls trong phạm vi role; requests, tool calls và token usage được ghi riêng. Các role có đúng một công cụ không cần tham số có thể chạy công cụ trực tiếp.

Đổi decision model qua `router.model/provider` hoặc dùng adapter khác triển khai cùng interface. Không có nhánh điều khiển environment theo tên Jev. Shared router factory mặc định tắt rejection guard; database config đặt rõ `observable_rejection_guard_enabled: false`. Nếu chủ động bật guard thì nó là một policy wrapper riêng, được lưu trong decision metadata; không dùng nhãn oracle để lọc action.

## Ba domain

- Database: `database_migration.sqlite.v1`, đủ 10 role contracts; dùng SQLite thật và public packet runner.
- Code debugging: `code_debugging.external.v1`, tái sử dụng external repository adapter. Role cards ở `config/shared_environment/code_roles.json`. `runtime.repository_mirror` là checkout/cache cục bộ tùy chọn; checkout vẫn được pin theo revision của task. Không dùng code adapter tổng hợp có fixture fix cho live entry point này.
- Formal proof: `formal_proof.lean4.v1`. Có thể cấu hình `lean_cache` để dùng Lean/Mathlib pin trong public contract; hoặc cung cấp `runtime.lean_executable`, `project_dir`, `lean_path`. Compiler và kernel/assumption checks chạy thật, giữ private reference proof ngoài executor context.

Packet, output và runtime paths là cấu hình; production code không import từ worktree cá nhân. Canonical checkout chứa code và bản dataset được copy nguyên trạng với checksum. Smoke tích hợp dùng một QuixBugs packet thật và một theorem fixture được Lean/Mathlib kiểm tra; chưa phải live evaluation toàn bộ code/formal datasets.

## State, context và artifact

- Router observation và executor context được dựng từ cùng adapter state. Executor nhận candidate và lỗi execution/validation mới nhất trong public context.
- Artifact lưu hash, version, producer event và input versions/references. `fresh=false` cho biết input hiện tại đã đổi; artifact lịch sử vẫn còn nguyên.
- `fresh` là mô tả dependency của work product, không phải nhãn action đúng và không tự chọn bước tiếp theo. Candidate đã được execute vẫn là artifact cần đối chiếu với execution/validation gắn đúng hash.
- Database SQL và public tool evidence có thể xuất hiện trong router view. Code/formal router view tuân thủ contract không đưa raw source/test contents từ executor vào history; executor vẫn đọc source và kết quả công cụ được phép.
- Output rỗng được giữ trong raw response, không tạo candidate mới chỉ chứa khoảng trắng. Không thêm state token-exhaustion hay tự nâng token; `finish_reason` và usage nằm trong response metadata cho hậu kỳ.
- Validation probes chạy trên bản sao SQLite nhất quán. Việc kiểm tra không làm đổi working DB.

## Transaction và recovery

SQLite runner áp dụng transaction contract của packet. Kết quả execution ghi database versions trước/sau và transaction outcome. Driver không tự restore database nguồn ban đầu sau lỗi.

Snapshot & Rollback Operator có công cụ `snapshot_operation` với operation `create` hoặc `restore`; restore cần snapshot ID cụ thể. Snapshot dùng SQLite backup và kiểm tra checksum. Tool event ghi thao tác của selected agent, khác với transaction rollback tự động.

## Budget và evaluation

Flow được giữ nguyên: hỏi router, xử lý STOP, rồi kiểm tra executor budget trước khi dispatch. Khi budget bằng 0, action không phải STOP vẫn được ghi nhận nhưng không chạy.

Evaluator phân loại:

- `dispatched`: đã dispatch vào executor;
- `precondition_rejected`: không chạy vì precondition công khai, vẫn tính vào mẫu số routing;
- `stop_handled`: STOP được xử lý, kể cả STOP tại budget 0 nếu có;
- `suppressed_by_budget`: chỉ bị chặn do budget, loại khỏi primary routing denominator.

Không loại máy móc mọi decision cuối hoặc mọi record không có tool execution. Raw records giữ nguyên. Lệnh `--legacy` có thể xuất derived accounting cho bank cũ mà không sửa trace gốc.

Workflow success và terminal-contract evidence được báo cáo riêng với action accuracy. Action accuracy để null khi human labels chưa có; không dùng lựa chọn của Jev làm gold. Role/transition coverage chỉ mô tả diversity, không thưởng việc gọi thêm agent.

## Artifacts của một run

- `resolved_config.json`, `source_hashes.json`, `role_cards.json`, `freeze_verification.json`.
- `episodes/<packet>/workspace/`: working artifacts và versioned runtime history.
- `episodes/<packet>/trajectory.jsonl`: actual role executions, requests/responses, metadata và tool results.
- `episodes/<packet>/decisions.jsonl`: mọi lựa chọn router và dispatch status.
- `episodes/<packet>/decision_states.jsonl`: immutable pre-decision public observations.
- `evaluation_report.json`: accounting/coverage/outcome report.
- `human_state_review_queue.jsonl`: state queue không chứa lựa chọn hiện tại của router làm nhãn; human review vẫn pending.
- `human_state_review.html`: viewer đọc-only cho task, role cards, public state/artifacts và history; không gán nhãn tự động.
- `episodes/<packet>/evaluation/final_task_audit.json`: kiểm tra evaluator-only trên bản sao cuối, không truyền lại vào state router. `end_task_passed` có thể khác workflow đã validate và STOP thành công.

Review key là `(run_id, decision_id)` vì các run có thể có cùng local decision ID. Khi yêu cầu review một state, cung cấp cả run ID.

## Kiểm thử

```powershell
python -B -m unittest discover -s tests -p test_shared_environment.py -v
python -B -m unittest discover -s tests -p test_shared_domains.py -v
python -B scripts/check_shared_environment.py
```

Các kiểm thử tích hợp code cần Git materialize checkout cục bộ; formal cần Lean/Mathlib cache đã pin. Chúng kiểm tra tác dụng công cụ thật, truyền feedback, stale artifacts, quyền read-only, rollback/snapshot, compiler/axiom verification và budget accounting. Chúng không yêu cầu Jev đạt một tỷ lệ success tối thiểu.

Config templates cho formal và code nằm cùng thư mục với database config. Code template materialize repository URL/revision trong từng packet; với BugsInPy/SWE-bench phải cung cấp runtime/harness phù hợp qua config riêng. Mirror QuixBugs chỉ được cấu hình cho smoke test cùng repository, không áp dụng cho repository khác. Các template này chưa được chạy live toàn bộ; smoke contract thực tế đã có trong bộ kiểm thử.
