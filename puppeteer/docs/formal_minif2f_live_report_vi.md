# Formal proof: gộp domain và chạy 20 miniF2F live trajectory

Hoàn tất ngày 2026-11-01. Task bank canonical có **30 packet formal proof**; đã chạy đủ **20 packet miniF2F** qua Jev dynamic routing, không chạy lại 10 APRIL/Workbook và không thay đổi database đóng băng.

## Gate trước chạy

- 30 manifest hợp lệ; 283 file packet copy nguyên trạng, byte-identical với nguồn. Giữ bank/run lịch sử.
- 20/20 miniF2F preflight ready, context/reference/allowed-axiom gates pass; hai adapter và independent-checker smoke pass.
- Chạy lại 28/28 kiểm tra common runner/router/scorer và toàn bộ formal integration, không lỗi.
- Giữ Goedel-Prover-V2-8B, Jev 1.13 structured, 7 role + STOP, guard false, 8 lượt, cap 16384, tối đa 3 tool calls/lượt actor.
- Kiểm tra credential tồn tại, selection đúng 20 ID, snapshot source và config trước launch. Không dùng private witness trong trajectory.

Gate: `runs/formal_minif2f_live_gate_20261101/`. Thiết kế và migration: `formal_proof_domain_integration_vi.md`.

## Kết quả được chốt

Run: `runs/formal_minif2f_20_jev_live_20261101`.

| Chỉ số | Giá trị |
|---|---:|
| Episode hoàn tất/đủ điều kiện đánh giá | 20/20 |
| Final task pass (checker độc lập) | 16/20 |
| Workflow + STOP pass | 15/20 |
| State và quyết định routing | 112 |
| Quyết định routing chính | 107 |
| Non-STOP ở budget 0 không thực thi | 5 |
| Role executions | 92 |
| Executor model responses | 46 |
| Probability vectors đủ 7 role + STOP | 112/112 |
| Masking/guard bị kích hoạt | 0 |
| Episode lỗi hạ tầng / checker unavailable | 0 / 0 |

| Nhóm | Task pass | Workflow + STOP pass |
|---|---:|---:|
| Easy (10) | 9/10 | 9/10 |
| Medium (10) | 7/10 | 6/10 |

`mathd_algebra_263` có proof cuối được checker độc lập chấp nhận, nhưng không hoàn tất workflow xác minh + STOP trước hết budget. Đây là lý do task pass 16 khác workflow pass 15; không tự bổ sung verifier hoặc STOP vào trajectory. Bốn task fail theo checker cuối là mathd_algebra_113, mathd_algebra_141, mathd_algebra_160, mathd_algebra_188. Không chạy lại task fail để chọn kết quả tốt hơn.

Độ khó là nhãn project đã chốt trước khi chạy, chưa được human xác nhận. Số pass mô tả kết quả task/workflow; chưa phải action accuracy hoặc bằng chứng xếp hạng tổng quát decision model.

## Trace, freeze và human review

Validator xác nhận đủ 112 state/decision join, 92 execution liên kết đúng, metadata/context của 46 model responses, không lỗi. Cả 324 source/data file được runner xác nhận không đổi trong run; checkout hiện tại khớp snapshot. Archive database đóng băng vẫn có SHA256 `be3891b46e0065855af3a29a2ea86346c9d84a91085fa2202584b90e66c97fb4`.

Source snapshot: `source_snapshot.zip`. Kết quả chính: `evaluation_report.json`, `execution_validation_report.json`, `execution_health.json`, `attempt_status.json`, `difficulty_summary.json`. Final task checks dùng hàm audit_domain hiện có trên workspace riêng, chạy song song hai worker rồi dùng evaluator hiện có tổng hợp kết quả trong bộ nhớ. Wrapper và hash được ghi ở attempt_status để phân biệt orchestration evaluator với thay đổi benchmark.

Human review: `human_state_review.html`, `human_state_review_queue.jsonl`. Có 112 state, 107 đủ điều kiện routing chính; năm state budget được đánh dấu loại. Tất cả valid_actions vẫn null/pending; chưa chuyển nhãn cũ, chưa tạo nhãn tự động và chưa tính human action accuracy. Trang review ẩn lựa chọn hiện tại như thiết kế chung.

## Domain canonical

- Bank: `data/RoutingBench/v4_formal_proof_30_20261101`.
- Config mặc định formal: `config/shared_environment/formal_jev_16k.yaml` (30 packet).
- Registry: `runs/formal_proof_domain_cohort_20261101.json`.
- Run mới chọn 20 miniF2F bằng các --packet ID rõ ràng; resolved_config lưu selection.

Registry giữ 10 trajectory cũ cùng snapshot riêng và 20 trajectory mới. Bản cũ chạy trước sửa chuẩn hóa tên theorem; không gộp điểm 30 task thành kết quả của một phiên bản môi trường duy nhất. Khi cần so model trên cả 30 task, chạy cùng bank/config/adapter đã pin cho tất cả model.
