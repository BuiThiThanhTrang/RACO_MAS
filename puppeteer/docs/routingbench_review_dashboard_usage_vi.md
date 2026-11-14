# Hướng dẫn RoutingBench V4-beta Human Review Dashboard

Dashboard phục vụ protocol human review của RoutingBench V4-beta. Giao diện V3 đã
được gỡ khỏi ứng dụng; database và artifact export của V4 dùng tên riêng để không
trộn với review V3 đã hoàn tất.

## 1. Cài đặt và khởi động

Mở PowerShell trong thư mục `puppeteer`, kích hoạt môi trường Python của project và
cài dependency riêng cho dashboard:

```powershell
conda activate puppeteer_env
pip install -r requirements-review.txt
python -m streamlit run review_app/app.py
```

Mặc định dashboard dùng:

```text
Dataset: data/RoutingBench/v4_beta
Database: data/RoutingBench/v4_beta/reviews/routingbench_v4_reviews.sqlite
```

Có thể đổi đường dẫn trong mục **Cấu hình dữ liệu** ở sidebar hoặc bằng biến môi
trường `ROUTINGBENCH_DATA_DIR` và `ROUTINGBENCH_REVIEW_DB`.

Mỗi người dùng một `Reviewer ID` riêng. Hai reviewer độc lập không dùng chung ID;
adjudicator dùng ID thứ ba.

## 2. Đơn vị review

V4-beta có hai queue bắt buộc:

- 124 `base_workflow`: mỗi workflow gồm năm counterfactual variants;
- 496 `track_c_pair`: mỗi pair gồm một post-fault checkpoint và matched clean
  control.

Mỗi checklist item nhận đúng một nhãn:

- `Approved`;
- `Needs revision`;
- `Reject`.

`Needs revision` và `Reject` bắt buộc có ghi chú tại item. Draft có thể chưa gán đủ;
khi submit, mọi item phải có nhãn.

## 3. Base workflow review

Chọn vai trò **Independent reviewer** và trang **Base workflow**. Bộ lọc hỗ trợ split,
domain, workflow pattern và source kind.

Reviewer cần kiểm tra:

1. task seed và contract;
2. cả năm clean/fault variants và formal transitions;
3. official router-visible observation ở từng decision state;
4. evaluator-side safe-action set và action rationales;
5. role cards trong agent pool;
6. row-level provenance, source item và license.

Giao diện kiểm tra tự động các key evaluator-side bị cấm trong public payload. Dấu
kiểm xanh chỉ xác nhận key boundary; reviewer vẫn phải đánh giá liệu nội dung quan sát
được có vô tình tiết lộ fault class hoặc gold action hay không.

## 4. Track C pair review

Trang **Track C pair** hiển thị fault checkpoint và matched control cạnh nhau. Phần
matched-pair contract kiểm tra task, agent pool, prefix depth, prefix cost và remaining
budget.

Ba tab cung cấp:

- router-visible comparison theo `state`, `structured` hoặc `raw` view;
- private labels dùng cho audit;
- replay runtime state và history để xác nhận continuation tới goal.

Tiêu đề “Fault checkpoint” và “Matched control” chỉ là chrome của dashboard dành cho
reviewer, không phải dữ liệu được truyền cho router.

## 5. Quy trình hai reviewer và adjudication

Reviewer 1 và Reviewer 2 làm độc lập; trang review không hiển thị nhãn của người còn
lại. Chọn vai trò **Adjudicator** để xem những object mà hai reviewer đã submit nhưng
khác nhãn.

Adjudicator phải chọn final label cho toàn bộ checklist và ghi rationale. Nếu dataset
được sửa, cả hai reviewer phải mở lại object và submit xác nhận mới. Validator vẫn
coi mọi `Needs revision` hoặc `Reject` là release blocker, kể cả đã adjudicate.

## 6. Export và validation

Trang **Export** hoặc lệnh sau tạo JSONL audit artifacts:

```powershell
python scripts/export_human_reviews.py
```

Output mặc định ở `data/RoutingBench/v4_beta/reviews/exports/`:

```text
base_workflow_reviews.jsonl
track_c_pair_reviews.jsonl
review_adjudications.jsonl
review_events.jsonl
```

Kiểm tra hai reviewer:

```powershell
python scripts/validate_human_reviews.py `
  --reviewer_a reviewer_01 `
  --reviewer_b reviewer_02 `
  --report data/RoutingBench/v4_beta/reviews/exports/validation_report.json
```

Validator trả exit code khác 0 nếu còn object chưa submit, thiếu checklist item, thiếu
ghi chú cho nhãn không approved, còn bất đồng chưa adjudicate, hoặc object chưa được
hai reviewer xác nhận lại sau khi sửa.

## 7. Backup và quyền riêng tư

- SQLite là working store; không commit file `.sqlite`, WAL hoặc backup.
- Chỉ commit JSONL export sau khi validator pass.
- Không sửa trực tiếp generated queue, public input hoặc private label artifacts.
- Reviewer ID nên là mã nội bộ, không chứa tên hoặc email cá nhân nếu không cần thiết.
