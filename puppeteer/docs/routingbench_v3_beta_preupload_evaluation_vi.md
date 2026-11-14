# Đánh giá RoutingBench V3-beta trước khi upload

Ngày đánh giá: 2026-10-12.

## Kết luận

V3-beta đạt yêu cầu kỹ thuật để đóng gói thành **public research preview**.
Dataset chưa được xem là human-validated: toàn bộ 116 workflow và 116 record
trong review queue vẫn có trạng thái `pending`. Trạng thái này được giữ nguyên
có chủ ý và phải được công bố rõ trên Hugging Face dataset card.

## Thay đổi terminal budget

- `max_steps` chỉ đếm quyết định thực thi/routing tới agent.
- Sau khi final artifact được commit, environment dành một quyết định terminal
  chi phí 0 cho `STOP`.
- Quyết định `STOP` không tăng `steps_used`, không tăng `cost_used` và vẫn được
  lưu thành event riêng trong trajectory.
- Nếu router chọn agent tại state terminal, episode kết thúc bằng
  `TERMINAL_ACTION_ERROR`; vì vậy khả năng nhận biết khi nào cần dừng vẫn được
  đánh giá, nhưng finalizer không bị phạt chỉ vì dùng actor call cuối cùng.

## Kết quả kiểm định tự động

- Formal validator: pass, 0 error.
- 116 workflow, 10 family, 3 split.
- 52 workflow dựa trên task thật: 24 MuSiQue và 28 MMLU-Pro.
- 1.156 isolated decision state.
- 0 observable-state label collision.
- 93,75% non-terminal state có ít nhất hai action hợp lệ.
- Không có `preferred_actions`, action utility, capability score hoặc backbone
  identity trong router input.
- Oracle development Track A: 100% valid-action accuracy và STOP precision/
  recall đều 100%.
- Random development Track A: 11,875% valid-action accuracy, cho thấy evaluator
  có dải phân biệt rõ với oracle.
- 33 unit test liên quan V2/V3 pass; semantics lịch sử của V2 và V3-alpha không
  bị thay đổi vì terminal policy chỉ áp dụng cho schema `3.0-beta`.

## Human review

- `provenance.jsonl`: 116/116 record là `pending`.
- `human_review_queue.jsonl`: 116/116 item là `pending`.
- Formal validation chỉ kiểm tra cấu trúc, provenance, split, label consistency
  và leakage; nó không thay thế đánh giá của con người.
- Task `v3b-musique-3hop-06` nên được ưu tiên review vì supplied corpus có thể
  chưa biểu đạt trực tiếp bridge fact về nơi sinh cần để suy ra gold answer.

## Vấn đề không chặn beta upload

- Gemini đôi khi trả response không phải JSON hợp lệ trên task chứa LaTeX. Đây
  là lỗi protocol của live executor, không phải lỗi oracle label của dataset.
- Các run Track B đã tạo trước terminal-budget fix không được sửa ngược. Muốn có
  số liệu chính thức phải rerun; post-hoc chỉ có thể dùng để chẩn đoán.

## Điều kiện khi upload

Dataset card phải ghi rõ:

1. release stage là beta/public research preview;
2. human review đang pending;
3. test split và labels đang public, không phải sealed leaderboard;
4. MuSiQue và MMLU-Pro giữ provenance/license riêng;
5. Track B dùng actor thật và kết quả phụ thuộc cả router lẫn executor;
6. `STOP` là terminal decision nằm ngoài execution-call budget.

Theo các điều kiện trên, dataset đủ điều kiện **đóng gói để upload beta**, nhưng
chưa đủ điều kiện để tuyên bố annotation hoàn chỉnh hoặc dùng làm bản V3 chính
thức.
