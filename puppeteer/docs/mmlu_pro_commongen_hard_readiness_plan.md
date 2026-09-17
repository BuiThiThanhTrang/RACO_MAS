# Implementation plan tối thiểu: MMLU-Pro và CW

Mục tiêu: sửa lỗi ảnh hưởng độ đúng và giữ điều kiện đánh giá nhất quán. Cả hai task đã có dữ liệu, split, config và pipeline; 16/16 test pipeline đã pass. Chưa xác minh bằng inference thật.

## 1. Sửa MMLU lấy đáp án cuối cùng

File: `tasks/evaluator.py` — `check_mmlu` và `extract_choice_answer`.

- Giữ các định dạng đáp án hiện được hỗ trợ, đổi từ đáp án đầu tiên sang đáp án cuối cùng. Xét vị trí trong toàn bộ output, kể cả khi các đáp án dùng định dạng khác nhau.
- Chọn prediction trước rồi mới so với gold; không trả đúng sớm vì một đáp án cũ khớp gold.
- Áp dụng nhất quán cho chấm điểm và extraction dùng trong majority vote, để reward/profile và kết quả đánh giá dùng cùng quy tắc.
- Thêm regression test nhỏ: `The answer is A. Correction: the answer is B.` phải lấy B, đúng với gold B và sai với gold A; thêm trường hợp A → (B), (A) → B và câu trả lời chỉ có một đáp án.

Đạt khi: các case trên pass và bộ test pipeline hiện tại vẫn pass.

## 2. Chuẩn bị và kiểm tra lượt chạy bằng luồng có sẵn

- Sau khi sửa parser, tạo probe profile riêng cho MMLU-Pro và CW bằng `--build_probe_profiles`; hai profile cấu hình hiện yêu cầu chưa có. Dùng đúng pool của experiment.
- Chạy smoke 2 item/task để kiểm tra prediction, file kết quả và text artifact của CW; sau đó kiểm tra train/resume và evaluation bằng checkpoint thực.
- Giữ nguyên judge, rubric, coverage, validation và reward của CW. Chạy bằng tên `CW`.

Đạt khi: hai task hoàn tất smoke, output CW tồn tại và có nội dung, train/resume không lặp hoặc mất item, evaluation không cập nhật policy/profile. Chỉ sửa thêm nếu smoke tái hiện lỗi trực tiếp ngăn đánh giá đúng.

## 3. Ghi rõ điều kiện so sánh trong hướng dẫn chạy

Chỉ cập nhật lệnh mẫu README và một ghi chú ngắn:

- Chỉ định đúng `--config` và `--policy_mode` cho từng task.
- Dùng cùng split/seed, tập item, ngân sách suy luận và quyền tool giữa các phương án so sánh; giữ judge/rubric CW cố định. Với thí nghiệm thay pool/model, ghi rõ đó là biến được so sánh.
- Tạo profile từ probe/reference, train trên train, chọn checkpoint trên dev; không dùng final để huấn luyện hoặc chọn cấu hình. Đánh giá với policy/profile đóng băng.
- MMLU `validation` hiện ánh xạ sang research dev 140 câu; `final` là 2.000 câu thuộc official test, không phải toàn bộ official benchmark. Báo cáo đúng tên subset này.

## Thứ tự thực hiện

Sửa parser + regression tests → tạo profile và smoke bằng luồng hiện có → cập nhật hướng dẫn chạy → chạy experiment.

Không thêm preflight framework, dataset registry, alias, provenance/hash, cache judge, schema output mới hay hệ thống báo cáo mới.

## Trạng thái triển khai

- Đã sửa check_mmlu/extract_choice_answer lấy match cuối theo vị trí và dùng cùng prediction để so gold; đã thêm regression tests.
- Đã cập nhật README với lệnh config/policy_mode đúng và điều kiện so sánh.
- Giữ nguyên judge/validation CW; không thêm alias hoặc hạ tầng mới.
- Smoke MMLU-Pro và CW đều dừng trước item đầu do Task Analyzer thiếu cấu hình endpoint/API key trong môi trường hiện tại. Chưa tạo probe profiles, chưa kiểm chứng train/resume hoặc frozen evaluation bằng dịch vụ thật. Cần cung cấp TASK_ANALYZER_BASE_URL và TASK_ANALYZER_API_KEY theo global.yaml rồi tiếp tục bước 2.
