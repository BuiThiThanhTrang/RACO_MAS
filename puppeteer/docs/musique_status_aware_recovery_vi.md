# MuSiQue status-aware recovery và evidence lineage

## Mục tiêu

Thay đổi này ngăn orchestrator xem `null`, `NEEDS_EVIDENCE` hoặc `CONFLICT`
là đáp án cuối, đồng thời cho phép tiếp tục routing động từ thông tin còn thiếu.
Thiết kế không áp đặt thứ tự role cố định và không sử dụng gold answer, gold
decomposition hoặc oracle hop count trong runtime.

## Contract đầu ra

Parser MuSiQue hỗ trợ sáu trạng thái:

- `FINAL`: đáp án hoàn chỉnh;
- `CANDIDATE`: đáp án tiềm năng nhưng chưa được commit;
- `NEEDS_EVIDENCE`: còn thiếu một fact/subquestion;
- `CONFLICT`: evidence hoặc candidate đang mâu thuẫn;
- `NO_ANSWER`: output không chứa đáp án;
- `MALFORMED`: output vi phạm contract.

Structured `status` có quyền ưu tiên hơn các trường answer do adapter tạo ra.
Vì vậy `final_answer: null` hoặc các biến thể `null,`, `null.`, `none;` không
thể trở thành candidate. `missing_subquestion`, conflict description và
supporting paragraph IDs được giữ trong typed blackboard.

Gold-blind Luna extractor chỉ chạy khi deterministic parser không tìm được
answer và output không có explicit non-final status. Luna không được phép biến
`NEEDS_EVIDENCE` thành một đáp án phỏng đoán.

## Evidence lineage

Mỗi evidence event lưu role, action, paragraph IDs trực tiếp, tập paragraph IDs
đã tích lũy, status và answer được tạo. Final answer kế thừa lineage của path.
Terminal gate kiểm tra `evidence_lineage_exists`; final agent không cần lặp lại
paragraph IDs đã được một agent trước đó tìm thấy.

`paper_compatible_support_f1` vẫn chỉ đọc support được phát ra cùng terminal
artifact để so sánh với paper. Metric này không điều khiển STOP, candidate
masking hoặc routing.

## Recovery và branching

| Trạng thái hiện tại | Nhu cầu tiếp theo mặc định | Branch |
|---|---|---|
| `FINAL` hợp lệ | STOP | Không |
| `CANDIDATE` | verify/compose/integrate | Không mặc định |
| `NEEDS_EVIDENCE` | retrieve/resolve bridge/compose theo missing subquestion | Không ở lần đầu |
| `CONFLICT` | verify hoặc thu thập evidence độc lập | Có thể |
| `NO_ANSWER`/`MALFORMED` | recover/replan/retrieve | Chỉ khi stall/no-progress |

Trong config MuSiQue hiện tại, low-progress, stall và chẩn đoán "independent
branch" của planner không còn đủ để fork. Path mới chỉ được mở khi state có
conflict evidence thực sự. Thiết kế bảo thủ này loại bỏ các path song song dư
thừa quan sát được ở run `musique_status_recovery_jev_gemini`. Root luôn bắt
đầu bằng một path;
`max_width: 3` là capacity tối đa chứ không phải số path bắt buộc.

`max_depth: 5` vẫn là budget thường. Nếu `NEEDS_EVIDENCE` xuất hiện ở một trong
hai bước cuối, path được cấp tối đa hai bước recovery (`musique_recovery_depth_credit:
2`): một bước bổ sung evidence và, khi đã có candidate, một bước Answer
Integrator. Credit không kích hoạt cho task bình thường nên không biến toàn bộ
thí nghiệm thành D7.

Cùng role bị chặn khi semantic state không đổi. Role được phép gọi lại sau khi
có paragraph, candidate, intermediate answer, status hoặc missing subquestion
mới. Retriever và Bridge Entity Reasoner vẫn có thể kết thúc task nếu chúng
phát ra một `FINAL` hợp lệ; terminal capability không bị giới hạn theo tên role.

## Aggregation

Mỗi path chỉ commit final answer hợp lệ gần nhất. Nếu một output sau đó có
explicit non-final status, aggregation không hồi sinh final answer cũ. Khi
deterministic parser thất bại, runtime Luna extractor có thể trích một span đã
tồn tại trong output; nó không được giải lại task hoặc tạo answer mới.

Nếu path hết budget với `CANDIDATE`, Luna thực hiện một gold-blind path-end gate:
chỉ được commit một span liên tục đã có sẵn nếu span đó trả lời trực tiếp câu
hỏi gốc. Task aggregation sau đó tin candidate đã được path commit, thay vì
canonicalize lại và vô tình loại một đáp án dài nhưng hợp lệ.

## Metrics mới

`collaboration.routing_diagnostics` và `analyze_musique_run.py` báo cáo:

- `status_counts`;
- `false_terminal_null_count`;
- `explicit_nonfinal_stop_count`;
- `needs_evidence_recovery_rate`;
- `conflict_recovery_rate`;
- `status_recovery_opportunities` và `status_recoveries`;
- `recovery_branch_count`;
- `evidence_lineage_coverage`;
- `final_answer_by_role`.

Các metric EM/F1, semantic correctness, deterministic collaboration,
MultiAgentBench-style CS và paper-compatible support F1 không bị thay đổi.

## Cấu hình

Hai config so sánh dùng cùng Gemini 2.5 Flash actor pool, W3D5 thường và tối đa
hai recovery call có điều kiện:

- Jev: `config/experiments/decision_musique_dynamic_jev_gemini.yaml`;
- Sol: `config/experiments/frozen_musique_dynamic_sol_gemini.yaml`.

`initial_paths: 1` giữ chi phí thấp trên task bình thường. Harness chỉ cho phép
hai continuation cùng một decision khi branch condition được thỏa mãn, dù
global graph còn capacity đến ba path.

## Lệnh chạy

Smoke test Jev:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_dynamic_jev_gemini.yaml `
  --policy_mode frozen `
  --data_limit 5 `
  --run_id "musique_status_recovery_jev_smoke"
```

Full 104-task Jev:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_dynamic_jev_gemini.yaml `
  --policy_mode frozen `
  --data_limit 104 `
  --run_id "musique_status_recovery_jev_gemini"
```

Full 104-task Sol:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/frozen_musique_dynamic_sol_gemini.yaml `
  --policy_mode frozen `
  --data_limit 104 `
  --run_id "musique_status_recovery_sol_gemini"
```

Chỉ retry các mẫu có `correct=false` từ một run trước (không nạp lại các mẫu
`correct=true`):

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_dynamic_jev_gemini.yaml `
  --policy_mode frozen `
  --retry_failed_from runs/musique_status_recovery_jev_gemini `
  --run_id "musique_status_recovery_failed_retry_v2"
```

Khi dùng `--retry_failed_from`, `data_limit` mặc định trong YAML bị bỏ qua để
chạy toàn bộ failed subset. Có thể truyền `--data_limit 1` để smoke test. Mỗi
record mới lưu `retry_failed_only` và `retry_source` để audit.

Tổng hợp kết quả:

```powershell
python scripts/analyze_musique_run.py `
  runs/musique_status_recovery_jev_gemini `
  --output runs/musique_status_recovery_jev_gemini/analysis/summary.json
```

## Xác minh

- 41 focused regression tests cho MuSiQue parser, routing harness, path-end
  Luna gate, recovery credit, committed aggregation và failed-only retry đã pass.
- Full suite chạy 228 tests: 218 pass và 10 lỗi legacy đã tồn tại từ checkpoint
  trước. Các lỗi này thuộc REINFORCE fixtures, split-size lock, evidence scopes
  của role cũ và source-text assertion; không thuộc frozen MuSiQue pipeline.
- `compileall` hoàn tất không có lỗi cú pháp.
- Analyzer mới đọc thành công toàn bộ 104 task của run
  `musique_adaptive_jev_gemini`. Baseline cũ có 3 false-terminal null, 7 path
  kết thúc ở explicit non-final status và `NEEDS_EVIDENCE` recovery rate bằng
  0; đây là các giá trị đối chứng trực tiếp cho run mới.

## Hiệu quả dự kiến cần xác nhận bằng A/B

- `false_terminal_null_count` về 0;
- không STOP sau explicit `NEEDS_EVIDENCE`/`CONFLICT`;
- giảm answer rỗng và tăng recovery success trên task khó;
- giữ mean actor calls trong khoảng không tăng quá 10–15%;
- giữ hoặc tăng semantic accuracy so với `musique_adaptive_jev_gemini`;
- không đánh đổi routing quality chỉ để tối ưu paper-compatible support F1.

Run `musique_status_recovery_jev_gemini` chứa 47 record có `correct=false`.
Loader mới đã được xác nhận nạp đúng 47 task ID này và không nạp record true.
