# Kết quả triển khai routing runtime audit

Ngày xác nhận: 16/09/2026.

## Kết luận

M0–M4 đã có mã và kiểm chứng offline. M5 đã có danh sách dev, preset và công cụ chạy; **chưa có đánh giá bằng model thật**. Chưa đủ bằng chứng để kết luận nên giảm phạt step/token hoặc accuracy sẽ tăng.

## Kiểm tra đã chạy

| Kiểm tra | Kết quả |
|---|---|
| Toàn bộ unittest hiện có + kiểm thử audit mới | **83/83 đạt**, 4,944 giây ở lần xác nhận cuối |
| Smoke orchestration với model giả | 7 kịch bản: 6 hoàn tất, 1 lỗi cố ý được ghi incomplete |
| Validator trên trace smoke | **7 trace, 0 lỗi invariant** |
| Replay legacy/majority trên cùng candidate | 6 task fixture, hash hợp lệ, chạy thành công |
| CLI help | main và 6 script audit đều exit 0 |
| Định dạng diff trong các tệp triển khai | Không có lỗi whitespace |

Bộ suite gốc có 54 test với 2 lỗi do fixture mặc định threshold=1.0 trong khi code người dùng đã là 1.5. Fixture nay chỉ định rõ 1.0; mặc định production vẫn là 1.5. Lần triển khai thêm 29 test.

Các trường hợp đã kiểm chứng:

- Agent rồi STOP: một activation; đạt depth: không gọi router thêm.
- Cùng teammate xen kẽ hai session; fork không lẫn hội thoại/artifact; reset không để lại hội thoại cho task sau.
- Capacity đầy hoặc chỉ còn một chỗ: không có trajectory ma; reward/action/path ID khớp.
- Prefix thực thi một lần dù xuất hiện trong nhiều leaf.
- Liệt kê outcomes K=1/2/3 kiểm tra tổng xác suất; gradient K=1/2 và cây có prefix/sibling khớp phép tính kỳ vọng.
- Checkpoint khác runtime/routing/objective/estimator không được resume optimizer; chuyển trọng số yêu cầu chế độ thích hợp.
- Lỗi thực thi không cập nhật policy; episode chưa hoàn thành không được train một phần.
- Bật/tắt audit không đổi output, số request, trọng số cuối và trạng thái RNG ở fixture.
- Prediction được chốt trước reward; đổi gold không làm đổi prompt/prediction.
- Retry, usage thiếu, HTTP reward scorer và cache được tính đúng số request.
- Snapshot STOP/continue giữ cùng prefix, không đọc gold; nhánh continue thêm đúng một action.
- Candidate thô dùng nhất quán giữa online và replay.
- Tách chuyển đúng/sai do suy luận và do tổng hợp path.

## Artifact đã lưu

Các đường dẫn tính từ thư mục puppeteer:

| Artifact | Đường dẫn |
|---|---|
| Kết quả unittest đầy đủ | runs/audit_validation/unittest_verified.txt |
| Trace mẫu hiện tại | runs/audit_validation/offline_verified/ |
| Smoke report | runs/audit_validation/offline_verified/smoke_report.json |
| Báo cáo audit | runs/audit_validation/offline_verified/analysis.json |
| Replay | runs/audit_validation/offline_verified/replay.json |
| ID dev, preset và preflight | runs/audit_validation/dev_protocol/ |
| Adapter log MMLU lịch sử | runs/audit_validation/legacy_mmlu_report.json |

Các kết quả trong offline_verified là dữ liệu giả, không được đưa vào bảng accuracy benchmark. Thư mục runs có thể bị Git ignore; công cụ và hướng dẫn có thể tạo lại artifact.

## Đối chiếu log cũ

Adapter đọc nhóm logs/MMLU-Pro10-09-train và tái tạo:

- 200 task.
- 116 task có ít nhất một path đúng.
- 83 đáp án cuối đúng.
- 33 task có path đúng nhưng đáp án cuối sai.

Đây là chẩn đoán nhóm log cũ, không phải kết quả dev/final. Trường thiếu giữ unknown/null: không suy ra checkpoint, split hay p(STOP) chỉ từ tên folder. Chưa xác định được run cũ chỉ gọi Domain Reasoner mà người dùng mô tả.

## M5 còn lại

Preflight đã khóa 140 ID dev và 20 ID smoke, split seed 42. Chưa gửi request mạng.

Thiếu cấu hình hiện tại:

- TASK_ANALYZER_BASE_URL và TASK_ANALYZER_API_KEY.
- Truy cập các backbone llama-3.1-8b, llama-3.2-3b, qwen-2.5-14b, qwen-2.5-7b.

Cần chạy tiếp sau khi môi trường có kết nối:

1. 20 câu smoke với checkpoint/pool/profile đã chọn.
2. 140 câu dev frozen; seed 42, rồi ứng viên cuối thêm 43/44.
3. Replay aggregation và báo cáo paired trên cùng task/candidate.
4. STOP/continue từ các snapshot dev theo quy tắc teammate cố định.
5. Đóng băng phương án trước final, rồi mới pilot cost.

Không có thay đổi phạt cost mới trong lần triển khai. Không chạy hoặc sửa bộ chấm GSM-Hard.

## Giới hạn cần giữ trong báo cáo nghiên cứu

- Legacy là selector cũ trên runtime đã sửa; không phải bản tái hiện lỗi hội thoại cũ. Cần đối chứng có provenance để quy nguyên nhân.
- Objective v2 vẫn là tổng discounted leaf return; chưa thay bằng reward đáp án cuối hoặc objective tiết kiệm chi phí vật lý.
- model_cost là proxy, không phải giá tiền; usage không được provider trả về được đánh dấu thiếu.
- SDK retry chuyển sang wrapper có trace; giữ cơ chế retry cố định khi so sánh.
- Không có kiểm chứng độc lập model API hoặc cải thiện accuracy trong lượt này.
- Kiểm tra whitespace toàn workspace vẫn báo một khoảng trắng ở model/model_config.py đã tồn tại trong thay đổi của người dùng; tệp đó không bị sửa trong triển khai này.

Hướng dẫn chạy: [routing_runtime_audit_guide.md](routing_runtime_audit_guide.md).

