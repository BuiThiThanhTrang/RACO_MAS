# Môi trường code debugging và formal proof — kết quả triển khai

Hoàn tất ngày 2026-11-01. Database giữ nguyên bản đóng băng `database_v1_20261031`; checksum archive vẫn khớp. Hai domain mới đã chạy lại từ đầu bằng Jev dynamic routing, cùng phiên bản engine cuối. Không dùng trajectory role cố định trong cohort chính.

## Cohort chính

| Domain | Run | Episode | Final task pass | Workflow + STOP pass | State | Quyết định routing chính |
|---|---|---:|---:|---:|---:|---:|
| Code debugging | code_shared_jev_gemini_v3_20261101 | 20 | 12 | 12 | 151 | 143 |
| Formal proof | formal_shared_jev_prover_v4_20261031 | 10 | 2 | 2 | 86 | 78 |

Final task pass do checker độc lập kiểm tra trạng thái cuối theo contract của packet. Workflow pass còn yêu cầu evidence xác minh còn fresh và orchestrator chọn STOP. Trong hai run chốt, hai số này trùng nhau; evaluator vẫn giữ tách biệt.

Cả 30 episode đều chấm được; không có checker UNAVAILABLE hoặc episode bị lỗi hạ tầng. Tổng 237 state khớp 237 quyết định. 16 quyết định non-STOP ở budget 0 được lưu nhưng loại khỏi routing denominator. Còn 221 quyết định routing chính.

Code giữ **Gemini 2.5 Flash**; formal giữ **Goedel Prover V2 8B**. Cả hai giữ 16384 max_tokens, 8 executor calls, Jev 1.13 structured, guard tắt. Tất cả 151 probability vector code có đủ 6 role + STOP; cả 86 vector formal có đủ 7 role + STOP. Không masking hoặc tự thay lựa chọn của router.

## Thay đổi cần thiết đã thực hiện

- Tái sử dụng runtime/checkout/test contract code đã chạy trước; thêm runtime profile theo packet cho runner chung. Preflight 20/20 code chạy được baseline thật, 10/10 formal compile được context gốc bằng toolchain đã pin.
- Map tool theo identity của 6 role code. Câu mô tả có chữ “repair” hoặc “validation” không còn cấp nhầm tool cho role khác. Bounded reads có thể đi trước work product; lượt kết thúc sau operation tạo work product thành công.
- Code executor nhận public state, edit scope và feedback tool gần nhất. Patch bị giới hạn trong implementation Python, bảo vệ tests/harness/config/reference. Fresh validation gắn với source/repository/public contract/runtime.
- Reproducer không được reset hoặc relabel checkout đã áp dụng candidate. Patch Author trước checkout là action rejection, không gán nhầm lỗi hạ tầng. Patch mới bị reject không ghi đè applied candidate; nhiều lần sửa thành công tạo patch tích lũy, giữ history và ghi nhận phép composition. Test thật đã replay cả patch tích lũy và file mới lên checkout sạch, regression pass.
- Code đủ điều kiện STOP khi candidate validation pass còn fresh. Consumer/Finalizer giữ identity và có thể đóng gói deliverable khi được chọn, không phải hai bước metadata bắt buộc.
- Formal giữ context/definitions/imports/statement gốc; chỉ thay proof. Nhận proof term, declaration gốc hoặc full source có context/statement tokens khớp. Không chấp nhận đổi definition thành True, đổi theorem hoặc thêm axiom. Original prefix được giữ và input normalization được ghi trong tool result.
- Lean source được trình bày nguyên văn với indentation. Bản calibration chỉ đưa source trong chuỗi JSON escape và yêu cầu proof-term-only đã gây rào cản định dạng; không dùng bản đó làm điểm benchmark. Smoke bằng chính Goedel đã tạo proof compile và kernel-check pass trước khi thu cohort mới.
- Formal inspector phân biệt initial proof state với compiler feedback mới. Retriever dùng `#check` thật trên Lean/Mathlib đã pin. Artifact lớn giữ nguyên file/hash/ref, có tool đọc từng đoạn; không tự nhét diagnosis dài vào mọi request hoặc dùng thêm model tóm tắt.
- Final evaluator dispatch theo domain. Code tái dựng implementation cuối trên checkout sạch với trusted tests; formal kiểm tra lại source trong context gốc, compilation và kernel/allowed assumptions. Checker chạy trên workspace riêng, không sửa trajectory. Lỗi hạ tầng được tách khỏi lỗi proof/test, kể cả khi nằm trong output lồng của tool.

## Bằng chứng và phiên bản

- `runs/non_database_cohort_20261031.json`: cohort được chấp nhận và các attempt/probe bị loại.
- Mỗi run chính có `resolved_config.json`, `role_cards.json`, `source_hashes.json`, `source_snapshot.zip`, `freeze_verification.json`, `execution_validation_report.json`, `execution_health.json`, `evaluation_report.json`.
- Shared engine source hashes của hai run chính khớp nhau. Cả hai validator xác nhận source hiện tại khớp run, trace đầy đủ và source không đổi trong khi chạy.
- `runs/non_database_release_final_20261101/report.json`: **36 checks pass**, gồm Git/SQLite/Lean thật, goal/context preservation, artifact history/cumulative patch, budget/STOP, raw routing và human-label scorer.
- `runs/non_database_preflight_20261031/`: preflight và bằng chứng lỗi context của prover đã sửa. Prover có context 32768; từng diagnosis 16k đưa tự động vào request sau đã gây HTTP 400. Output budget vẫn giữ 16k; đổi sang artifact refs + bounded reads để không mất work product.
- `runs/code_domain_history_audit_20261031/history_audit.json`: xác nhận lịch sử code là 20 workflow Jev live dynamic routing, 172 state và pre-label Luna. Đã đối chiếu checksum cả bốn nguồn canonical. Luna là judge offline, không phải controller tạo trajectory.

Các run calibration, run code 14/20 trước bản source cuối và smoke một role được giữ riêng. Cohort chính chọn theo phiên bản engine cuối trước khi xem điểm replacement; không chọn attempt theo pass rate. Smoke không tính vào benchmark. Không chuyển nhãn Luna cũ sang state mới.

Runtime Python/Lean tái sử dụng môi trường đã có trên máy; profile ghi đường dẫn và version. Khi chuyển máy cần provision và preflight lại. Các điểm pass là theo public regression/kernel contract của packet, không phải điểm chính thức trên toàn bộ SWE-bench hoặc đánh giá tổng quát mọi task của domain.

## Human review và cắm decision model

Hai trang review chính:

- `runs/code_shared_jev_gemini_v3_20261101/human_state_review.html`
- `runs/formal_shared_jev_prover_v4_20261031/human_state_review.html`

Trang hiển thị task, public state, budget, role cards và lịch sử đã thấy; ẩn lựa chọn hiện tại và nhãn đề xuất. Human labels vẫn pending, action accuracy vẫn null. Lỗi proof dùng sorry/sorryAx đã bị verifier chặn dù compilation thành công.

Sau review, mỗi dòng nhãn JSONL gồm `run_id`, `decision_id`, `valid_actions`, `label_status: human_verified`, `reviewer`. Chạy từ puppeteer:

```powershell
python scripts/score_shared_reviews.py runs/<run_id> --labels <human_labels.jsonl>
```

Output `human_action_evaluation.json` ghi coverage, accuracy trên quyết định có nhãn và đủ điều kiện, cùng hash nhãn/script. Scorer không chạy model/checker hoặc sửa trace; từ chối nhãn pending, sai run, ID trùng và action ngoài pool.

Decision model mới cung cấp `reset(seed)` và `select(observation, candidates) -> RoutingDecision` cho SharedTrajectoryRunner. CLI hiện hỗ trợ adapter decision-service và random; API khác cần adapter mỏng theo cùng interface. Giữ task, actor, role/tool contracts, budget và evaluator khi so sánh model; không sửa môi trường theo routing mà model đã chọn.

Môi trường đã đủ cho pilot end-to-end và bước human review trên cohort hiện tại. Chưa có action accuracy được human xác nhận; kết quả một baseline trên số packet nhỏ chưa xác lập thứ hạng tổng quát giữa decision model. Không cần thêm role hay ép trajectory đi qua nhiều role để công nhận môi trường hợp lệ.
