# Baseline Structured Table QA

Hai cấu hình dùng cùng 40 packet, Qwen 3.5 9B, budget 12 và một tool call cho mỗi lượt executor.

## Single-agent

Router chỉ dispatch `Table QA Agent` cho đến khi public verification đạt, sau đó chọn `STOP`. Agent luôn nhận đủ bảy tool và tự quyết định thao tác tiếp theo. Tool gọi sai thời điểm trả lỗi công khai qua cùng contract mà các router khác nhận khi chọn một role chưa đủ precondition.

Baseline này đo khả năng Qwen tự lập kế hoạch và sử dụng tool khi không có router phân vai.

## Predefined trajectory

Policy cố định chọn specialist tiếp theo từ state công khai. Luồng mặc định là inspect, resolve header cho bảng phân cấp, select evidence, analyze, compose khi cần, verify và `STOP`. Khi verification thất bại, policy gọi diagnoser rồi quay lại bước được chẩn đoán.

Baseline này đo kết quả của một workflow được xác định trước và không gọi model router.

## So sánh

Hai cấu hình giữ nguyên packet, executor, budget, tool implementation, public state và trusted evaluator. Khác biệt nằm ở quyền chọn thao tác. Single-agent giao quyền chọn tool cho Qwen. Predefined trajectory giao quyền chọn specialist cho policy cố định.
