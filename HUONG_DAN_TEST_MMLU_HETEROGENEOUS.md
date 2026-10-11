# Chạy test MMLU-Pro 200 mẫu với heterogeneous pool

## Mục tiêu và artifact cố định

- Split: `final` trong `puppeteer/data/splits/mmlu_pro_seed42.json`.
- Số mẫu: 200 mẫu đầu của split `final`, seed 42.
- Pool: `puppeteer/personas/mmlu_pro_heterogeneous_pool.jsonl`.
- Policy mặc định: `puppeteer/checkpoint/MMLU-Pro_train/policy_net_20261011_075840.pt`.
- Policy ở chế độ evaluation: chỉ inference, không cập nhật optimizer hoặc weights.
- Result: `puppeteer/results/MMLU-Pro_test/MMLU-Pro_test.jsonl`.
- Checkpoint để resume test: `puppeteer/checkpoint/MMLU-Pro_test/resume_latest.pt`.

Không dùng `resume_latest.pt` của thư mục train để resume một test đang dở. Script tự dùng
checkpoint trong thư mục `MMLU-Pro_test` khi có cờ `-Resume`.

## Sau khi mở PowerShell mới

~~~powershell
Set-Location "C:\Users\tncn2\Documents\KLTN\RACO_MAS"
.\.venv311\Scripts\Activate.ps1
~~~

Thiết lập lại năm biến môi trường trong đúng terminal này:

~~~powershell
$env:HF_TOKEN = "<HF token>"
$env:OPENROUTER_API_KEY = "<OpenRouter key>"
$env:REWARD_MODEL_URL = "https://tncn218--puppeteer-nemotron-q5-rewardserver-score.modal.run"
$env:MODAL_KEY = "<Modal proxy token ID bắt đầu bằng wk->"
$env:MODAL_SECRET = "<Modal proxy token secret bắt đầu bằng ws->"
~~~

Không lưu token thật vào Git, file Markdown, ảnh chụp hoặc log chia sẻ công khai.

## Preflight một lần

~~~powershell
.\run_mmlu_heterogeneous_test.ps1 `
    -DataLimit 200 `
    -LiveProviders `
    -PreflightOnly
~~~

Preflight kiểm tra Python 3.11, 400 kết quả train, checkpoint policy, unit tests,
Modal reward endpoint, routing của provider và đúng 200 ID duy nhất từ split test.

## Chạy test 200 mẫu

Sau khi preflight thành công:

~~~powershell
.\run_mmlu_heterogeneous_test.ps1 `
    -DataLimit 200 `
    -CheckpointEvery 1 `
    -SkipRewardSmoke
~~~

Không thêm `-Resume` khi bắt đầu một test mới. Script tự backup result/checkpoint test cũ.
Checkpoint train gốc được kiểm tra SHA-256 trước và sau test để bảo đảm không bị sửa.

## Resume khi bị ngắt

Mở lại đúng repo, kích hoạt `.venv311`, thiết lập lại năm biến môi trường rồi chạy:

~~~powershell
.\run_mmlu_heterogeneous_test.ps1 `
    -DataLimit 200 `
    -CheckpointEvery 1 `
    -SkipRewardSmoke `
    -Resume
~~~

Script đếm số dòng hợp lệ hiện có, nạp `MMLU-Pro_test/resume_latest.pt` và chỉ chạy phần
còn lại. Không chạy đồng thời hai terminal test hoặc train.

## Điểm cần lưu ý khi báo cáo

1. Accuracy test mới là kết quả đánh giá; không dùng accuracy train 74.5% làm test score.
2. Không chọn lại checkpoint sau khi nhìn kết quả test. Policy đã được cố định trước test.
3. 200 mẫu test không trùng ID với 400 mẫu train; script kiểm tra lại sau khi hoàn tất.
4. Hugging Face/OpenRouter và Modal là ba nguồn quota/chi phí độc lập.
5. `CheckpointEvery 1` an toàn nhất nhưng ghi đĩa sau mỗi câu. Có thể tăng lên 5, đổi lại
   khi lỗi có thể phải chạy lại tối đa bốn câu.
6. Nếu Modal trả lỗi tạm thời, giữ nguyên result và dùng `-Resume`; không chạy fresh.
7. `policy.json` có thể hiển thị `training=false` trong lúc test; đây là hành vi đúng.
8. Console log, result, summary JSON và checkpoint test cần được lưu cùng nhau để tái lập.
