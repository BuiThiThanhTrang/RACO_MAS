# Implementation plan: code debugging và formal proof

Ngày 2026-10-31. Đã triển khai xong ngày 2026-11-01; xem `code_formal_environment_implementation_report_vi.md` và cohort chốt trong `runs/non_database_cohort_20261031.json`. Mục tiêu: môi trường chung cho decision model điều phối agent giải task nhiều bước; observation đủ thông tin, action có hậu quả thật, kết quả được chấm độc lập. Không tối ưu pass rate bằng can thiệp routing. Database giữ nguyên bản đóng băng `database_v1_20261031`.

## 1. Xác minh lịch sử và điều chỉnh phạm vi

Code debugging đã chạy live với orchestrator, không phải trajectory role cố định. Bộ empirical RC1 nằm ở worktree `C:/Users/23521/.codex/worktrees/4381/RACO_MAS/puppeteer/data/RoutingBench/v4_code_debugging_empirical_rc1`.

| Nguồn canonical | Workflow được chọn | Bước được ghi | Router |
|---|---:|---:|---|
| code_packets_10_jev_structured_gemini_state_history_v3_20261029 | 6 | 45 | Jev 1.13 |
| cp4r_20261029 | 4 | 39 | Jev 1.13 |
| cd10b_v3 | 7 | 71 | Jev 1.13 |
| sw3c_v3 | 3 | 26 | Jev 1.13 |

Đã đối chiếu SHA256 cả bốn episodes.jsonl với trajectory_source_manifest, tất cả khớp. Tổng 181 record thô; importer loại 9 CALL_LIMIT marker không thực thi, giữ 172 decision state và 161 empirical output. Metadata từng bước có model, probabilities và latency của Jev. Guard bật ở 181 record nhưng không record nào có guard_blocked_actions. Lần chạy mới vẫn tắt guard rõ ràng để giữ agent pool đầy đủ.

Luna là judge offline: 172 action pre-annotation và 161 outcome interpretation; không điều khiển trajectory. Báo cáo labeling hiện dùng cache 172 annotation nên primary_api_calls bằng 0 không có nghĩa chưa tạo nhãn. Có 74 output ưu tiên review và 6 disagreement giữa action đã chọn với Luna; tất cả nhãn vẫn pending human, không phải gold. Nguồn thống kê: `runs/code_domain_history_audit_20261031/history_audit.json`.

Điều chỉnh so với audit trước: không làm lại executor/runtime từ đầu. Actor cũ (`execution/actor.py`) đã kết thúc lượt khi tool tạo work product thành công. Lỗi tiếp tục gọi tool thuộc actor chung mới; chỉ cần mang semantics đã chạy được sang đường chung. Runtime 20 packet đã có lịch sử chạy; cần phục hồi profile và preflight hiện tại, không dựng lại một hệ runtime mới.

## 2. Những lựa chọn cố định

- Code executor: **google/gemini-2.5-flash**; không đổi sang Qwen. Formal giữ Goedel Prover V2 8B đang cấu hình.
- Router pilot: Jev 1.13, structured; interface cho phép thay decision model mà không đổi môi trường.
- Executor max_tokens 16384, 8 executor calls, tool-call cap hiện có; không tăng ngân sách để sửa pass rate. Ghi metadata để phân tích sau.
- Giữ nguyên STOP và quyết định ở budget 0 như database; decision không được thực thi vì budget bị loại khỏi routing denominator.
- Guard false, không tự chọn role, không ép mỗi trajectory đi qua đủ role; precondition reject và output rỗng là kết quả quan sát thật.
- Giữ packet, test, role identity và visibility contract đã dùng nếu phù hợp. Không thêm framework, LSP, container orchestration, role mới hoặc sample mới trong đợt này.
- Bản database đóng băng là nguồn chuẩn cho đánh giá database; shared-code sửa sau này có version/hash mới, không thay bundle hoặc hash của run cũ.

## 3. Công việc triển khai theo thứ tự

### P0 — Khôi phục provenance và kiểm tra lại đường chạy code

1. Đọc script collector, runtime profiles và execution-gate reports của bốn nguồn canonical; lập mapping 20 packet → revision/runtime/test command. Chỉ đưa những file cần chạy về workspace hiện tại; giữ nguồn cũ readonly, không copy toàn bộ cache/run workspace.
2. So sánh actor cũ với SharedRoleActor. Tái sử dụng cơ chế bounded reads rồi một work product: kết thúc lượt ngay khi `diagnose_failure`/`repair_candidate` hoặc operation tương ứng thành công. Không cần phát minh protocol finish mới khi semantics cũ đủ dùng. Tool cap mà chưa tạo output phải trả trạng thái rõ, không ngầm sinh diagnosis/patch.
3. Giữ request/response nguyên bản, model/usage và tool result. Với partial-trace infrastructure failure, dùng chính sách rerun episode đã thống nhất; không mở rộng thành dự án sửa tracing.

Nghiệm thu: executor thật có thể đọc source/test và tạo diagnosis/patch; không bị bắt gọi thêm sau work product thành công; không thay behavior đường một-tool của database.

### P1 — Hoàn thiện code debugging bằng các sửa đổi cần thiết

1. **Patch boundary:** cho sửa source theo public contract, bảo vệ regression tests, runners và evaluator. Không khóa vào đúng gold file/hunk; task có thể cần nhiều file. Reject trước apply nếu patch chạm vùng bị bảo vệ hoặc vượt workspace. Kiểm tra actual changed files để không bỏ lọt rename/binary/path trick.
2. **Final checker:** dispatch evaluator theo domain; chạy tests tin cậy với final repository trên bản sao, đúng revision/runtime. Tách PASS/FAIL/UNAVAILABLE, không biến checker không hỗ trợ thành fail. Không dùng private gold patch làm input actor/router.
3. **Freshness:** tái sử dụng invalidation sẵn có sau patch mới; gắn test evidence với repository/source, test contract và runtime version. Validation cũ không làm candidate mới thành verified.
4. **Terminal contract:** public task yêu cầu verified patch; một candidate có validation pass còn fresh đủ điều kiện STOP. Giữ hai role Consumer/Finalizer và artifact format cũ để tránh đổi agent pool, nhưng không bắt gọi chỉ để tạo JSON metadata. Chúng vẫn có thể tạo deliverable nếu router chọn; không tự gọi thay router. Cập nhật role cards/observation/STOP eligibility nhất quán và version thiết kế.
5. **Runtime:** tái sử dụng profile đã chạy 20 packet; preflight từng packet ở máy hiện tại. Không yêu cầu baseline phải fail nếu contract không quy định, nhưng phải phân biệt expected regression failure với import/setup/service failure.

Nghiệm thu: patch đúng pass checker, patch sai fail, sửa protected tests bị từ chối; terminal dựa bằng chứng fresh và task contract; packaging không còn là điều kiện nhân tạo.

### P2 — Hoàn thiện formal proof với thay đổi nhỏ nhất

1. **Khóa bài toán:** xác định proof region của từng packet và công khai editable scope; giữ nguyên context, definitions, imports, theorem signature/binders. Dùng template ghép proof vào source gốc. Nếu một packet cần sửa helper declaration, khai báo vùng đó riêng thay vì cho sửa cả file. Không chỉ so regex header.
2. **Final checker độc lập:** tái dựng source từ context gốc và proof cuối, compile bằng Lean/Mathlib đúng version, kiểm tra không sorry/admit và allowed assumptions công khai. Chạy trên bản sao, tách unavailable với proof fail. Tái sử dụng compiler/kernel implementation hiện tại; không xây prover khác.
3. **Observation đủ dùng:** gọi proof_state.json là initial state, kèm current source ref/version và compiler feedback mới nhất; không trình bày initial goals như goals hiện tại. Chưa cần interactive goal extraction/LSP.
4. **Retriever đúng contract:** thêm lookup readonly có giới hạn trong Mathlib local: symbol/type/source qua công cụ tra cứu hoặc compiler check trong file tạm. Kết quả có đường dẫn/version hoặc diagnostic thật. Giữ role identity; không để retriever chỉ viết lemma theo trí nhớ mà được mô tả là đã tra thư viện.
5. Tái sử dụng single-tool invocation, artifact lineage và runtime pin hiện có. Retriever có thể thực hiện lookup và tạo artifact trong cùng một domain operation; không mở rộng actor chung nếu không cần.

Nghiệm thu: proof hợp lệ pass; đổi definition thành True, đổi statement, sorry/admit hoặc assumption ngoài contract đều không được công nhận. Source mới làm compilation/verification cũ hết hiệu lực.

### P3 — Kiểm chứng, chạy lại từ đầu và tạo review queue

1. Chạy kiểm tra contract có ý nghĩa ở P1/P2 với tool thật, cộng compatibility check rằng bundle database/checksum không đổi. Không thêm test chỉ phản chiếu implementation.
2. Preflight 20 code và 10 formal packet theo manifest. Ghi readiness matrix, giữ nguyên packet chưa chạy được với lý do infrastructure; không chọn lại cohort theo độ dễ hoặc pass rate.
3. **Code:** chạy lại toàn bộ 20 packet từ workspace sạch, Jev dynamic routing + Gemini 2.5 Flash theo config cố định mới; run ID mới, không resume từ artifact cũ. Đây là lần chạy thay thế để đánh giá thiết kế mới, không gộp bước với run cũ.
4. **Formal:** sau negative-contract tests và preflight, chạy 10 packet từ đầu theo cùng shared protocol, Jev + prover hiện tại. Không tái sử dụng script repair loop cố định của domain_live như dynamic trajectory.
5. Episode hỏng vì hạ tầng/mất trace: giữ attempt cũ, tối đa một rerun toàn episode theo protocol database. Không rerun vì model sai, dừng sớm, output rỗng hoặc hết budget. Báo accepted attempt và exclusion reason.
6. Xuất final-task result, workflow/STOP result, decisions dispatched/rejected/suppressed, cost/latency và trace completeness. Label quality/routing accuracy để pending trước human review.
7. Tạo blind review queue cho state mới. Giữ pre-label Luna cũ ở cohort lịch sử; không gán lại theo packet/step index. Nếu chạy Luna cho cohort mới, dùng cùng labeling rubric, version riêng, không để action đã chọn/gold/future output ảnh hưởng valid-action labels.

Chạy model là bước sau khi các sửa đổi và kiểm tra trên hoàn thành; tài liệu này chưa khởi chạy lại model.

## 4. Điều kiện hoàn tất

- Cả hai domain dùng cùng decision interface, budget/STOP semantics và không masking.
- Role thực thi đúng contract, public state có đủ bằng chứng để quyết định; không tự hỗ trợ orchestrator bằng next-role hints hoặc recovery ẩn.
- Không còn false positive đổi đề formal/sửa tests code; final checker độc lập và có trạng thái unavailable.
- Có manifest runtime/source/config, cohort và accepted attempts rõ ràng; full live run và state review queue mới.
- Pass rate thấp vẫn là kết quả hợp lệ nếu môi trường đúng. Không điều chỉnh thiết kế sau khi xem kết quả chỉ để nâng điểm; lỗi môi trường phát hiện sau run phải ghi version và phạm vi rerun.

Phạm vi đợt này không gồm tăng sample, tìm cấu hình model tốt nhất, xây benchmark framework mới hoặc viết lại adapter đã hoạt động. Chỉ sửa các điểm làm sai task contract, thiếu thông tin hoặc sai phép đo.
