# Implementation plan: RoutingBench Human Review Dashboard

## Mục tiêu

Tạo một dashboard chạy local để hai annotator review độc lập RoutingBench
V3-beta, sau đó cho adjudicator xử lý bất đồng. Dashboard phải trình bày state,
role card và workflow rõ ràng nhưng không làm lộ gold trong lượt blind review.

## Phạm vi triển khai

1. **Data access an toàn**
   - Đọc bundle V3-beta từ JSON/JSONL hiện có.
   - Tạo public decision view theo allow-list; loại `valid_actions`,
     `action_rationales`, transition và evaluation gold.
   - Chỉ mở source audit sau khi reviewer hoàn tất mọi state của workflow.

2. **Annotation storage**
   - SQLite chạy WAL để hai tiến trình local có thể ghi an toàn.
   - Hỗ trợ draft, submitted, revision history và append-only event log.
   - Không sửa trực tiếp dữ liệu benchmark hoặc `human_review_queue.jsonl`.

3. **Giao diện Streamlit**
   - Progress dashboard có filter theo split, family và source.
   - Blind review theo từng decision state.
   - Workflow audit với Graphviz, provenance, actor input và evaluation gold.
   - Adjudication so sánh hai reviewer và lưu valid-action set cuối cùng.

4. **Export và validation**
   - Export các bảng SQLite thành JSONL ổn định.
   - Validator kiểm tra completeness, candidate labels, blind-review status và
     các disagreement chưa adjudicate.
   - Agreement report gồm exact-set agreement và macro Jaccard; không dùng
     ngưỡng tùy ý để tự động tuyên bố dữ liệu đạt chuẩn.

5. **Tài liệu và kiểm thử**
   - Tách dependency UI trong `requirements-review.txt`.
   - Viết hướng dẫn cài đặt/vận hành dashboard.
   - Viết protocol review chi tiết cho annotator và adjudicator.
   - Unit test gold-blind projection, SQLite persistence, disagreement,
     export và workflow graph.

## Cấu trúc đầu ra

```text
review_app/
├── app.py
├── data_loader.py
├── annotation_store.py
└── workflow_graph.py

scripts/
├── export_human_reviews.py
└── validate_human_reviews.py

docs/
├── routingbench_review_dashboard_usage_vi.md
└── routingbench_human_review_protocol_vi.md
```

## Hiệu quả dự kiến

- Giảm lỗi đọc nhầm các trường JSON lồng nhau.
- Ngăn confirmation bias do gold labels bị ẩn ở vòng annotation đầu.
- Có thể resume sau khi đóng trình duyệt và truy vết mọi lần chỉnh sửa.
- So sánh reviewer và adjudicate có cấu trúc thay vì ghép file thủ công.
- Dễ tái sử dụng cho V3/V4 bằng cách thay dataset directory nhưng giữ schema.

## Điều kiện hoàn thành

- Dashboard khởi động từ thư mục `puppeteer`.
- Reviewer không thể nhìn gold qua trang Blind Review.
- Source Audit bị khóa nếu reviewer chưa submit toàn bộ state của workflow.
- Mọi annotation được lưu xuống SQLite ngay khi bấm Save/Submit.
- Export và validator chạy độc lập không cần Streamlit.
- Unit tests mới pass trên dữ liệu mẫu tạm.
