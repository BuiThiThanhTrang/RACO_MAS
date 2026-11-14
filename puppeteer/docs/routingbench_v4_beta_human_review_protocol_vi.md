# Protocol human review RoutingBench V4-beta

## Đối tượng review

Review theo `base_workflow_id`, không review từng biến thể như các task độc lập. Reviewer phải xem task contract, role cards, cả năm variants, transition effects, public observation và bốn Track C fault/control pairs.

## Quy trình hai người

1. Reviewer 1 và Reviewer 2 làm độc lập, không xem kết luận của nhau.
2. Mỗi checklist item được đánh dấu `approved`, `needs_revision` hoặc `reject` và phải có ghi chú khi không approved.
3. Nếu hai người cùng approved, đặt `adjudication_status: not_required`.
4. Nếu có bất đồng, người thứ ba hoặc phiên thảo luận chung quyết định và ghi `adjudication_status: resolved` cùng lý do.
5. Sau khi sửa dữ liệu, cả hai reviewer phải xác nhận lại workflow đã sửa.

## Checklist base workflow

- Task và state quan sát được đủ để biện minh cho mọi safe action.
- Không có action hợp lệ nào chỉ được biết nhờ simulator hoặc private label.
- Role-card `use_when`, `avoid_when`, supported inputs và expected outputs khớp với transition thực tế.
- Alternate executor vẫn hợp lệ ở clean/fault variants nếu precondition công khai giống nhau.
- Fault là post-execution và tín hiệu của nó quan sát được nhưng không nêu tên fault class.
- Task/pool/initial state/budget/terminal contract giống nhau giữa counterfactuals.
- Provenance, source item và license đúng.
- Không sao chép label/utility/role card từ V3.

## Checklist Track C

- Fault/control có cùng task, pool, prefix depth, prefix cost và remaining budget.
- Khác biệt hành vi recovery được biện minh bởi public execution evidence.
- Router input không chứa pair ID, condition, fault label hoặc oracle safe action.
- Cả hai checkpoint đều có ít nhất một continuation thực tế tới goal.
- Unsafe action thực sự lan truyền lỗi quan sát được, không phải invalid tùy ý.

## Điều kiện phát hành

Chỉ đặt `human_review_status: reviewed` khi mọi base workflow có hai approval và mọi bất đồng đã adjudicate. Nếu còn một mục pending, bản build vẫn dùng được cho internal experiments nhưng không được tuyên bố là release-ready.
