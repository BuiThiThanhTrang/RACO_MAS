# Báo cáo migration V3 → V4-beta

## Kết quả audit nguồn

Audit đọc toàn bộ V3-beta scenarios và provenance trước khi chọn task seed. Quy tắc chính là không xem một counterfactual variant như một task nguồn mới.

- `eligible`: task nguồn thực, có provenance và có thể dùng làm task seed.
- `needs_rewrite`: workflow tổng hợp có ích để tham khảo pattern nhưng không được copy nguyên task/graph.
- `reject`: duplicate counterfactual, thiếu task, ngoài scope hoặc trùng normalized task.

Kết quả cụ thể được ghi tại `data/RoutingBench/v4_beta/v3_audit_report.json` và `v3_audit_manifest.jsonl`.

## Những thành phần được giữ

- Nội dung task seed của các nguồn MuSiQue/MMLU-Pro được chọn.
- Source dataset, item ID, license và hash nguồn.
- Ý tưởng workflow pattern ở mức khái niệm.

## Những thành phần bị loại và dựng lại

- V3 agent pool và role cards.
- Mọi transition/precondition/effect.
- Valid action, preferred action và utility.
- Capability score/backbone identity.
- Split cũ nếu vi phạm source-group isolation.

V4-beta dựng lại role/transition/fault contract theo quy tắc observable-state của V4-alpha.2, sau đó lint và đưa vào two-reviewer queue.

## Giới hạn hiện tại

Task seeds project-authored có provenance tới generator source và được cấp license project-authored. Chúng vẫn cần human review về tính tự nhiên, role alignment và độ khó. Automated audit không thể thay thế đánh giá nội dung của hai reviewer.
