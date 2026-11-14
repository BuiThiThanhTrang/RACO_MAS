# Kế hoạch sửa contract Structured Data v2

## Phạm vi

Dùng lại packet Structured Data hiện tại để kiểm tra contract. Dùng JEV và Qwen 9B với ngân sách 12 lượt dispatch. Giữ tiêu chí public validation và cách trusted evaluator chấm cùng một candidate trên các case của packet.

## Thay đổi

1. Tách contract nộp bài khỏi contract artifact. Agent nộp các câu lệnh Python thao tác trên biến `workbook`. Harness tạo `solve(input_path, output_path)`, mở workbook công khai khi test và lưu đầu ra. Agent author và repair có thể trả Python trực tiếp. Harness điền ID của candidate và public evidence hiện hành khi nhận Python trực tiếp từ role repair.
2. Dùng cùng mô tả interface trong public state, prompt, tool schema dành cho v2. Agent vẫn nhận đề bài, snapshot workbook công khai, quy tắc public validation, candidate hiện hành và evidence cần để sửa. Router vẫn nhận role cards, action list, budget, trạng thái và artifact công khai. Phân biệt candidate được chấp nhận, chạy public thành công và public validation đạt.
3. Giữ cùng đường thực thi giới hạn. Báo lỗi syntax hoặc policy để agent có thể thử sửa trong lượt hiện tại. Gắn revision với candidate trước đó để truy vết. Sau khi đổi candidate, router phải thấy rằng execution và validation cũ cần được làm mới.
4. Chạy kiểm tra qua shared runner trên packet thật với executor và router có kịch bản. Kiểm tra budget, payload hai bên, tạo candidate, lỗi thực thi, repair, retest, validate và STOP. Đối chiếu payload với nội dung private và kiểm tra đường nộp `python_source` của v1. Dùng lệnh live riêng để đo tỷ lệ tuân thủ và kết quả của Qwen.


